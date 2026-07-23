import copy
import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase12" / "FX-SCHEMA-001.json"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


BASE_CASES = {
    "pre_route_readiness": ("valid_run_metadata", 0),
    "pre_route_scientific": ("valid_run_metadata", 1),
    "selected_route_main": ("valid_run_metadata", 2),
    "non_scientific_exploratory": ("valid_run_metadata", 3),
    "prefix_metadata": ("valid_run_metadata", 4),
    "scientific_exploratory": ("valid_run_metadata", 5),
    "timing_cell": ("valid_sensitivity_cells", 1),
    "prefix_trial": ("valid_trials", 0),
    "clean_trial": ("valid_trials", 1),
    "contam_trial": ("valid_trials", 2),
    "nomem_trial": ("valid_trials", 3),
}


def _base_case(fixture: dict[str, Any], case_id: str) -> dict[str, Any]:
    container, index = BASE_CASES[case_id]
    return copy.deepcopy(fixture[container][index])


def _invalid_case(fixture: dict[str, Any], case_id: str) -> dict[str, Any]:
    return next(case for case in fixture["invalid"] if case["id"] == case_id)


def _apply_mutation(record: dict[str, Any], mutation: dict[str, Any]) -> dict[str, Any]:
    mutated = copy.deepcopy(record)
    mutated.update(mutation.get("patch", {}))
    for field_name in mutation.get("remove", []):
        mutated.pop(field_name, None)
    return mutated


def _schema_v3() -> Any:
    return importlib.import_module("memcontam.logging.schema_v3")


def test_schema_v3_module_exists() -> None:
    assert importlib.util.find_spec("memcontam.logging.schema_v3") is not None


def test_accepts_applicable_run_and_trial_variants() -> None:
    schema = _schema_v3()
    fixture = _fixture()
    cases = (
        "pre_route_readiness",
        "pre_route_scientific",
        "selected_route_main",
        "non_scientific_exploratory",
        "scientific_exploratory",
        "prefix_metadata",
        "prefix_trial",
        "clean_trial",
        "contam_trial",
        "nomem_trial",
    )

    parsed = [schema.parse_log_record_v3(_base_case(fixture, case_id)) for case_id in cases]

    assert [record.schema_version for record in parsed] == ["logging_v3"] * len(cases)


@pytest.mark.parametrize(
    "case_id",
    [
        "correct-as-contam-protocol",
        "nomem-with-arm",
        "pre-route-with-selection",
        "non-scientific-with-certificates",
        "selected-route-without-manifest",
        "selected-route-without-seed-allocation",
        "scientific-exploratory-without-activation",
        "non-scientific-exploratory-with-activation",
        "clean-with-intervention",
        "contam-without-intervention",
        "prefix-as-clean-arm",
        "v2-backport",
    ],
)
def test_rejects_inapplicable_fields_and_v2_backport(case_id: str) -> None:
    schema = _schema_v3()
    fixture = _fixture()
    mutation = _invalid_case(fixture, case_id)
    base_case_id = mutation.get("base_case", "selected_route_main")
    record = _apply_mutation(_base_case(fixture, base_case_id), mutation)

    with pytest.raises((ValidationError, schema.Phase12SchemaError), match=mutation["reason"]):
        schema.parse_log_record_v3(record)


def test_rejects_unrelated_sensitivity_fields_and_extra_event_fields() -> None:
    schema = _schema_v3()
    fixture = _fixture()
    mutation = _invalid_case(fixture, "timing-with-horizon")
    sensitivity = _apply_mutation(_base_case(fixture, mutation["base_case"]), mutation)

    with pytest.raises(ValidationError, match=mutation["reason"]):
        TypeAdapter(schema.SensitivityCellRef).validate_python(sensitivity)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        schema.FailureEventV3.model_validate(
            {
                "record_type": "failure_event",
                "schema_version": "logging_v3",
                "contract_level": "phase12",
                "event_id": "failure-1",
                "run_id": "run-1",
                "trial_id": "trial-1",
                "event_seq": 1,
                "failure_class": "provider",
                "unexpected": "forbidden",
            }
        )


def test_rejects_invalid_tool_mode_and_mixed_protocol_versions() -> None:
    schema = _schema_v3()
    fixture = _fixture()

    with pytest.raises(ValidationError, match="tool_mode"):
        schema.ToolEvent.model_validate(
            {
                "record_type": "tool_event",
                "event_id": "tool-1",
                "run_id": "run-1",
                "event_seq": 1,
                "tool_mode": "browser",
                "action": "none",
                "status": "skipped",
            }
        )
    mixed_protocol = _base_case(fixture, "pre_route_readiness")
    mixed_protocol["protocol_version"] = "phase12_code_exploratory_v1"
    with pytest.raises(ValidationError, match="MIXED_PROTOCOL_VERSION"):
        schema.parse_log_record_v3(mixed_protocol)


@pytest.mark.parametrize(
    ("base_case_id", "field_name", "value", "error_code"),
    [
        ("selected_route_main", "scientific_admission_ref", None, "SCIENTIFIC_ADMISSION_REQUIRED"),
        (
            "scientific_exploratory",
            "scientific_admission_ref",
            None,
            "SCIENTIFIC_ADMISSION_REQUIRED",
        ),
        (
            "scientific_exploratory",
            "source_route_selection_manifest_id",
            None,
            "ROUTE_SELECTION_REQUIRED",
        ),
        (
            "scientific_exploratory",
            "source_seed_allocation_manifest_id",
            None,
            "SEED_ALLOCATION_REQUIRED",
        ),
        (
            "non_scientific_exploratory",
            "source_route_selection_manifest_id",
            "route-selection-001",
            "SOURCE_ROUTE_SELECTION_FORBIDDEN",
        ),
        (
            "non_scientific_exploratory",
            "source_seed_allocation_manifest_id",
            "seed-allocation-001",
            "SOURCE_SEED_ALLOCATION_FORBIDDEN",
        ),
        ("contam_trial", "candidate_triplet_id_or_none", None, "INTERVENTION_REQUIRED"),
        ("contam_trial", "native_render_id_or_none", None, "INTERVENTION_REQUIRED"),
    ],
)
def test_rejects_reviewed_missing_applicability_references(
    base_case_id: str, field_name: str, value: Any, error_code: str
) -> None:
    schema = _schema_v3()
    record = _base_case(_fixture(), base_case_id)
    record[field_name] = value

    with pytest.raises(ValidationError, match=error_code):
        schema.parse_log_record_v3(record)


def test_version_dispatch_preserves_v1_v2_readers() -> None:
    from memcontam.logging import parse_log_record

    legacy = {
        "trial_id": "legacy-1",
        "run_id": "legacy-run",
        "task_name": "game24",
        "sample_id": "sample-1",
        "baseline": "no_memory",
        "arm": "clean",
        "backbone": "replay",
        "input": {},
        "gold_or_verifier_spec": {},
        "prompt_messages": [],
        "raw_response": "24",
        "verifier_result": {"is_correct": True},
    }

    assert parse_log_record(legacy).schema_version == "legacy"
