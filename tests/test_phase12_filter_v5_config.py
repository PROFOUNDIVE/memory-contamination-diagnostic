from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import memcontam.experiment.phase12.filter_challenge.registry as registry
from memcontam.experiment.phase12.filter_challenge.registry import (
    OperationalSuiteRegistry,
    ProbeInventoryRegistry,
    SearchConfig,
    SelectedPolicy,
    validate_stage,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "phase12" / "filter_v5"


def _load_yaml(name: str):
    return yaml.safe_load((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _load_json(name: str):
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _copy_payload(payload):
    return json.loads(json.dumps(payload))


def test_synthetic_fixtures_parse_as_build_only_registries() -> None:
    # Given: the five immutable synthetic fixture payloads.
    search = SearchConfig.model_validate(_load_yaml("FilterChallengeSearchConfig.yaml"))
    policy = SelectedPolicy.model_validate(_load_yaml("FilterChallengeSelectedPolicy.yaml"))
    inventory = ProbeInventoryRegistry.model_validate(_load_json("probe_inventory_manifest.json"))
    suite = OperationalSuiteRegistry.model_validate(_load_json("operational_suite_manifest.json"))
    prerequisites = _load_json("bct_execution_prerequisites.json")

    # When: their build boundary fields are inspected.
    # Then: every fixture is synthetic, non-scientific, and explicitly execution-blocked.
    for fixture in (search, policy, inventory, suite):
        assert fixture.evidence_layer == "build"
        assert fixture.scientific_result is False
        assert fixture.fixture_only is True
        assert "synthetic-build" in fixture.registry_id
    assert prerequisites["evidence_layer"] == "build"
    assert prerequisites["scientific_result"] is False
    assert prerequisites["fixture_only"] is True
    assert "synthetic-build" in prerequisites["prerequisites_id"]
    assert prerequisites["search_config_frozen"] is False
    assert prerequisites["inventory_frozen"] is False
    assert prerequisites["canonical_patch_status"] == "pending_before_provider_backed_pilot_b"
    assert prerequisites["provider_config_enabled"] is False
    assert prerequisites["runtime_authorization_present"] is False


def test_search_config_is_finite_complete_and_has_no_premature_selection() -> None:
    # Given: the synthetic pre-Pilot-B search registry.
    search = SearchConfig.model_validate(_load_yaml("FilterChallengeSearchConfig.yaml"))

    # When: every finite candidate family is inspected.
    # Then: no required family is empty and no selected/final field is present.
    for field_name in (
        "required_strata",
        "suite_candidates",
        "kappa_candidates",
        "coverage_contract_candidates",
        "replicate_retry_candidates",
        "canonicalizer_candidates",
        "tolerance_candidates",
        "paired_evaluability_candidates",
        "inclusion_rate_candidates",
        "ordinary_route_coverage_candidates",
        "budget_cap_candidates",
        "ci_procedure_candidates",
        "constraint_order",
        "deterministic_tie_break_candidates",
    ):
        assert getattr(search, field_name)
    assert not {name for name in SearchConfig.model_fields if "selected" in name or "final" in name}


def test_hash_uses_exact_model_projection_and_ignores_yaml_surface_form() -> None:
    # Given: one Unicode-bearing semantic search registry in two YAML surface forms.
    payload = _load_yaml("FilterChallengeSearchConfig.yaml")
    rewritten = "\n\n" + yaml.safe_dump(payload, allow_unicode=True, sort_keys=True) + "\n"
    first = SearchConfig.model_validate(payload)
    second = SearchConfig.model_validate(yaml.safe_load(rewritten))

    # When: both strict models compute their stable hashes.
    # Then: bytes, key order, whitespace, and path-independent YAML forms do not change it.
    assert first.stable_hash() == second.stable_hash() == first.search_config_hash
    alternate = hashlib.sha256(
        json.dumps(first.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    assert alternate != first.search_config_hash
    with pytest.raises(ValidationError, match="HASH_MISMATCH"):
        SearchConfig.model_validate(
            first.model_dump(mode="json") | {"search_config_hash": alternate}
        )


def test_hash_excludes_only_its_declared_hash_and_detects_semantic_tampering() -> None:
    # Given: a validated search registry and a semantic mutation.
    search = SearchConfig.model_validate(_load_yaml("FilterChallengeSearchConfig.yaml"))
    payload = _copy_payload(search.model_dump(mode="json"))
    payload["constraint_order"] = list(reversed(payload["constraint_order"]))

    # When: the registry is re-parsed without recomputing its own declared hash.
    # Then: semantic changes fail while replacing only that hash does not alter the projection.
    assert search.model_copy(update={"search_config_hash": "0" * 64}).stable_hash() == search.stable_hash()
    with pytest.raises(ValidationError, match="HASH_MISMATCH"):
        SearchConfig.model_validate(payload)


def test_kappa_and_coverage_reject_incoherent_or_missing_required_strata() -> None:
    # Given: mutations beyond the registered κ and coverage contracts.
    kappa_payload = _copy_payload(_load_yaml("FilterChallengeSearchConfig.yaml"))
    kappa_payload["kappa_candidates"][0]["min_distinct_witness_probes"] = 999
    coverage_payload = _copy_payload(_load_yaml("FilterChallengeSearchConfig.yaml"))
    coverage_payload["coverage_contract_candidates"][0]["strata"] = coverage_payload[
        "coverage_contract_candidates"
    ][0]["strata"][:-1]

    # When: each invalid finite registry is parsed.
    # Then: κ coherence and no-renormalization are both enforced.
    with pytest.raises(ValidationError, match="KAPPA_INCOHERENT"):
        SearchConfig.model_validate(kappa_payload)
    with pytest.raises(ValidationError, match="REQUIRED_STRATA_MISSING"):
        SearchConfig.model_validate(coverage_payload)


def test_selected_policy_references_search_members_and_stage_gate_requires_it_only_for_main() -> None:
    # Given: a finite search registry and a policy selected from it.
    search = SearchConfig.model_validate(_load_yaml("FilterChallengeSearchConfig.yaml"))
    policy = SelectedPolicy.model_validate(_load_yaml("FilterChallengeSelectedPolicy.yaml"))
    invalid = _copy_payload(policy.model_dump(mode="json"))
    invalid["operational_probe_suite_id"] = "synthetic-build-unknown-suite"

    # When: build, Pilot-B, and Main readiness are checked.
    # Then: only Main requires the selected policy, and all references are predeclared.
    assert validate_stage(search, None, stage="build").reason_code is None
    assert validate_stage(search, None, stage="pilot_b").reason_code is None
    assert validate_stage(search, None, stage="main").reason_code == "SELECTED_POLICY_REQUIRED"
    assert validate_stage(search, policy, stage="main").reason_code is None
    with pytest.raises(ValidationError, match="HASH_MISMATCH"):
        SelectedPolicy.model_validate(invalid)
    with pytest.raises(ValueError, match="SELECTED_POLICY_REFERENCE_UNKNOWN"):
        validate_stage(search, policy.model_copy(update={"operational_probe_suite_id": invalid["operational_probe_suite_id"]}), stage="main")


@pytest.mark.parametrize(
    ("field", "index", "property_name", "value", "error"),
    (
        ("suite_candidates", 0, "replicates_per_probe", 0, "greater than 0"),
        ("kappa_candidates", 0, "min_total_evaluable_replicates", 0, "greater than 0"),
        ("coverage_contract_candidates", 0, "strict_clean_solvable_probe_count", -1, "greater than or equal to 0"),
        ("replicate_retry_candidates", 0, "control_retry_limit", -1, "greater than or equal to 0"),
        ("budget_cap_candidates", 0, "call_cap", -1, "greater than or equal to 0"),
        ("budget_cap_candidates", 0, "latency_cap_ms", -1, "greater than or equal to 0"),
        ("coverage_contract_candidates", 0, "paired_evaluability_rate", 1.1, "less than or equal to 1"),
        ("tolerance_candidates", 0, "repeatability_tolerance", -0.1, "greater than or equal to 0"),
    ),
)
def test_search_config_rejects_invalid_numeric_candidate_boundaries(
    field: str, index: int, property_name: str, value: int | float, error: str
) -> None:
    # Given: one finite candidate outside its registered numeric domain.
    payload = _copy_payload(_load_yaml("FilterChallengeSearchConfig.yaml"))
    payload[field][index][property_name] = value

    # When: strict search parsing reaches the candidate boundary.
    # Then: invalid counts, retries, caps, rates, and tolerances are rejected before hashing.
    with pytest.raises(ValidationError, match=error):
        SearchConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "identifier", "error"),
    (
        ("suite_candidates", "operational_probe_suite_id", "DUPLICATE_SUITE_CANDIDATE_ID"),
        ("kappa_candidates", "kappa_id", "DUPLICATE_KAPPA_CANDIDATE_ID"),
        ("coverage_contract_candidates", "coverage_contract_id", "DUPLICATE_COVERAGE_CONTRACT_ID"),
        ("replicate_retry_candidates", "replicate_retry_id", "DUPLICATE_REPLICATE_RETRY_ID"),
        ("canonicalizer_candidates", "canonicalizer_id", "DUPLICATE_CANONICALIZER_ID"),
        ("tolerance_candidates", "tolerance_id", "DUPLICATE_TOLERANCE_ID"),
        ("paired_evaluability_candidates", "rate_id", "DUPLICATE_PAIRED_EVALUABILITY_ID"),
        ("inclusion_rate_candidates", "rate_id", "DUPLICATE_INCLUSION_RATE_ID"),
        ("ordinary_route_coverage_candidates", "rate_id", "DUPLICATE_ORDINARY_ROUTE_COVERAGE_ID"),
        ("budget_cap_candidates", "budget_cap_id", "DUPLICATE_BUDGET_CAP_ID"),
        ("ci_procedure_candidates", "ci_procedure_id", "DUPLICATE_CI_PROCEDURE_ID"),
        ("deterministic_tie_break_candidates", "tie_break_id", "DUPLICATE_TIE_BREAK_ID"),
    ),
)
def test_search_config_rejects_duplicate_candidate_ids(
    field: str, identifier: str, error: str
) -> None:
    # Given: a finite candidate family containing a repeated identifier.
    payload = _copy_payload(_load_yaml("FilterChallengeSearchConfig.yaml"))
    duplicate = _copy_payload(payload[field][0])
    duplicate[identifier] = payload[field][0][identifier]
    payload[field].append(duplicate)

    # When: the search registry is parsed.
    # Then: every named candidate family remains an unambiguous finite set.
    with pytest.raises(ValidationError, match=error):
        SearchConfig.model_validate(payload)


def test_search_config_rejects_duplicate_probe_ids_and_coverage_strata() -> None:
    # Given: duplicated calibration probe and coverage-stratum entries.
    probe_payload = _copy_payload(_load_yaml("FilterChallengeSearchConfig.yaml"))
    probe_payload["calibration_probe_ids"].append(probe_payload["calibration_probe_ids"][0])
    stratum_payload = _copy_payload(_load_yaml("FilterChallengeSearchConfig.yaml"))
    stratum_payload["coverage_contract_candidates"][0]["strata"].append(
        stratum_payload["coverage_contract_candidates"][0]["strata"][0]
    )

    # When: each registry keeps the same set while adding a duplicate.
    # Then: no-renormalization still rejects the duplicate representation.
    with pytest.raises(ValidationError, match="DUPLICATE_CALIBRATION_PROBE_ID"):
        SearchConfig.model_validate(probe_payload)
    with pytest.raises(ValidationError, match="DUPLICATE_COVERAGE_STRATUM"):
        SearchConfig.model_validate(stratum_payload)


@pytest.mark.parametrize(
    ("name", "subject", "field", "error"),
    (
        ("probe_inventory_manifest.json", ProbeInventoryRegistry, "probe_ids", "PROBE_IDS_EMPTY"),
        ("operational_suite_manifest.json", OperationalSuiteRegistry, "suite_ids", "SUITE_IDS_EMPTY"),
    ),
)
def test_manifest_registries_reject_empty_and_duplicate_ids(name, subject, field: str, error: str) -> None:
    # Given: one empty manifest sequence and one repeated manifest identifier.
    empty_payload = _copy_payload(_load_json(name))
    empty_payload[field] = []
    duplicate_payload = _copy_payload(_load_json(name))
    duplicate_payload[field].append(duplicate_payload[field][0])

    # When: each immutable manifest is parsed.
    # Then: it cannot describe an empty or ambiguous registry.
    with pytest.raises(ValidationError, match=error):
        subject.model_validate(empty_payload)
    with pytest.raises(ValidationError, match=error.replace("EMPTY", "DUPLICATE")):
        subject.model_validate(duplicate_payload)


@pytest.mark.parametrize("value", ("synthetic-build-probe", {"probe": "synthetic-build-probe"}, 1, b"probe"))
def test_yaml_sequence_boundary_rejects_non_sequences(value) -> None:
    # Given: a scalar or mapping substituted for a YAML list field.
    payload = _copy_payload(_load_yaml("FilterChallengeSearchConfig.yaml"))
    payload["calibration_probe_ids"] = value

    # When: strict sequence normalization runs.
    # Then: only lists and tuples become immutable tuples.
    with pytest.raises(ValidationError, match="SEQUENCE_REQUIRED"):
        SearchConfig.model_validate(payload)


def test_registry_closure_binds_search_to_inventory_and_suite_hashes() -> None:
    # Given: the three independently parsed synthetic registries.
    search = SearchConfig.model_validate(_load_yaml("FilterChallengeSearchConfig.yaml"))
    inventory = ProbeInventoryRegistry.model_validate(_load_json("probe_inventory_manifest.json"))
    suite = OperationalSuiteRegistry.model_validate(_load_json("operational_suite_manifest.json"))

    # When: their typed closure is validated.
    # Then: the search references both manifest identities and canonical hashes exactly.
    closure = registry.validate_registry_closure(search, inventory, suite)
    assert closure.search_config_id == search.registry_id
    assert closure.calibration_probe_inventory_id == inventory.registry_id
    assert closure.operational_probe_suite_manifest_id == suite.registry_id
    with pytest.raises(ValueError, match="INVENTORY_MANIFEST_REFERENCE_MISMATCH"):
        registry.validate_registry_closure(
            search.model_copy(update={"calibration_probe_inventory_id": "synthetic-build-wrong"}),
            inventory,
            suite,
        )
    with pytest.raises(ValueError, match="SUITE_MANIFEST_REFERENCE_MISMATCH"):
        registry.validate_registry_closure(
            search.model_copy(update={"operational_probe_suite_manifest_hash": "0" * 64}),
            inventory,
            suite,
        )
