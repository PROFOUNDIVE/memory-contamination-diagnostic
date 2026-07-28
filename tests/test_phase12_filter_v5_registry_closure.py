from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from memcontam.experiment.phase12.filter_challenge.registry import validate_registry_closure
from memcontam.experiment.phase12.filter_challenge.registry_manifests import (
    OperationalSuiteRegistry,
    ProbeInventoryRegistry,
)
from memcontam.experiment.phase12.filter_challenge.registry_search import SearchConfig


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "phase12" / "filter_v5"


def _search_payload() -> dict[str, object]:
    return yaml.safe_load((FIXTURE_ROOT / "FilterChallengeSearchConfig.yaml").read_text(encoding="utf-8"))


def _reseal(payload: dict[str, object]) -> None:
    hash_payload = payload.copy()
    del hash_payload["search_config_hash"]
    canonical = json.dumps(hash_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["search_config_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _registries() -> tuple[SearchConfig, ProbeInventoryRegistry, OperationalSuiteRegistry]:
    search = SearchConfig.model_validate(_search_payload())
    inventory = ProbeInventoryRegistry.model_validate_json(
        (FIXTURE_ROOT / "probe_inventory_manifest.json").read_text(encoding="utf-8")
    )
    suite = OperationalSuiteRegistry.model_validate_json(
        (FIXTURE_ROOT / "operational_suite_manifest.json").read_text(encoding="utf-8")
    )
    return search, inventory, suite


def test_search_config_rejects_resealed_empty_required_strata() -> None:
    # Given: a resealed search payload whose required and coverage strata are both empty.
    payload = _search_payload()
    payload["required_strata"] = []
    for contract in payload["coverage_contract_candidates"]:
        contract["strata"] = []
    _reseal(payload)

    # When: strict parsing reaches the finite strata boundary.
    # Then: equal empty sets cannot satisfy the coverage contract.
    with pytest.raises(ValidationError, match="REQUIRED_STRATA_EMPTY"):
        SearchConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("order", "reason_code"),
    (([], "CONSTRAINT_ORDER_EMPTY"), (["required_strata", "required_strata"], "DUPLICATE_CONSTRAINT_ORDER_ITEM")),
)
def test_search_config_rejects_resealed_invalid_constraint_order(
    order: list[str], reason_code: str
) -> None:
    # Given: a resealed empty or duplicate constraint order.
    payload = _search_payload()
    payload["constraint_order"] = order
    _reseal(payload)

    # When: the finite search registry is parsed.
    # Then: ordering remains nonempty and unambiguous.
    with pytest.raises(ValidationError, match=reason_code):
        SearchConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("order", "reason_code"),
    (([], "TIE_BREAK_ORDER_EMPTY"), (["lowest_call_cap", "lowest_call_cap"], "DUPLICATE_TIE_BREAK_ORDER_ITEM")),
)
def test_search_config_rejects_resealed_invalid_tie_break_order(
    order: list[str], reason_code: str
) -> None:
    # Given: a resealed empty or duplicate deterministic tie-break order.
    payload = _search_payload()
    payload["deterministic_tie_break_candidates"][0]["order"] = order
    _reseal(payload)

    # When: the finite search registry is parsed.
    # Then: each tie-break policy remains an ordered unique sequence.
    with pytest.raises(ValidationError, match=reason_code):
        SearchConfig.model_validate(payload)


def test_registry_closure_rejects_probe_id_mismatches_in_both_directions() -> None:
    # Given: valid registries and either side missing one calibration probe ID.
    search, inventory, suite = _registries()
    search_missing_probe = search.model_copy(update={"calibration_probe_ids": search.calibration_probe_ids[:-1]})
    inventory_missing_probe = inventory.model_copy(update={"probe_ids": inventory.probe_ids[:-1]})

    # When: closure compares the independent probe ID sets.
    # Then: neither a search omission nor inventory omission is accepted.
    with pytest.raises(ValueError, match="CALIBRATION_PROBE_IDS_MISMATCH"):
        validate_registry_closure(search_missing_probe, inventory, suite)
    with pytest.raises(ValueError, match="CALIBRATION_PROBE_IDS_MISMATCH"):
        validate_registry_closure(search, inventory_missing_probe, suite)


def test_registry_closure_rejects_suite_id_mismatches_in_both_directions() -> None:
    # Given: valid registries and either side missing one operational suite ID.
    search, inventory, suite = _registries()
    search_missing_suite = search.model_copy(update={"suite_candidates": search.suite_candidates[:-1]})
    suite_missing_id = suite.model_copy(update={"suite_ids": suite.suite_ids[:-1]})

    # When: closure compares the independent suite ID sets.
    # Then: neither a search omission nor manifest omission is accepted.
    with pytest.raises(ValueError, match="SUITE_CANDIDATE_IDS_MISMATCH"):
        validate_registry_closure(search_missing_suite, inventory, suite)
    with pytest.raises(ValueError, match="SUITE_CANDIDATE_IDS_MISMATCH"):
        validate_registry_closure(search, inventory, suite_missing_id)
