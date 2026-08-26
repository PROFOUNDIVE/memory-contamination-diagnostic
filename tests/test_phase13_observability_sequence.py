from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from memcontam.evaluation.phase13_observability_models import Phase13LineageNode


def _module():
    return importlib.import_module("memcontam.evaluation.phase13_observability")


def test_classifier_accepts_only_registered_task_class_pairs() -> None:
    module = _module()

    metric = module.classify_registered_failure(
        "game24",
        0,
        "G24_CANONICAL_FALSE_RULE_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V1",
    )

    assert metric.status == "supported"
    assert metric.value == "G24_CANONICAL_FALSE_RULE_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V1"
    with pytest.raises(module.Phase13ObservabilityError, match="UNKNOWN_TASK_FAILURE_CLASS"):
        module.classify_registered_failure("game24", 0, "UNKNOWN")
    with pytest.raises(module.Phase13ObservabilityError, match="UNKNOWN_TASK_FAILURE_CLASS"):
        module.classify_registered_failure("unknown", 0, None)


def test_recurrence_includes_exact_h10_boundary_and_excludes_h11() -> None:
    module = _module()
    failure = "G24_CANONICAL_FALSE_RULE_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V1"

    at_boundary = module.has_registered_recurrence((failure, *(None for _ in range(9)), failure), 10)
    beyond_boundary = module.has_registered_recurrence(
        (failure, *(None for _ in range(10)), failure), 10
    )

    assert at_boundary is True
    assert beyond_boundary is False


def test_first_episode_does_not_bridge_gaps_and_marks_endpoint_censoring() -> None:
    module = _module()

    completed = module.first_continuous_episode((True, True, False, True, True), "core_prefix_50")
    censored = module.first_continuous_episode((False, True, True), "core_prefix_50")

    assert completed.value == 2
    assert completed.censoring_status == "OBSERVED_END"
    assert censored.value == 2
    assert censored.censoring_status == "RIGHT_CENSORED"
    assert censored.censoring_endpoint_analysis_window_id == "core_prefix_50"


def test_exact_root_eviction_rejects_prompt_only_disappearance() -> None:
    module = _module()

    assert module.is_exact_root_eviction("root-b", ("root-b",), (), ("root-b",)) is False
    assert module.is_exact_root_eviction("root-b", ("root-b",), ("root-b",), ()) is True


def test_registration_packet_rejects_unbound_implementation_keys() -> None:
    registration = importlib.import_module(
        "memcontam.evaluation.phase13_observability_registration"
    )
    packet_path = Path("data/phase13/observability/registration_packet_v1.json")
    payload = json.loads(packet_path.read_text(encoding="utf-8"))
    payload["implementation_identities"]["unbound"] = payload["implementation_identities"][
        "sequence"
    ]

    with pytest.raises(ValueError, match="OBSERVABILITY_REGISTRATION_PACKET_STALE"):
        registration.ObservabilityRegistrationPacket.model_validate(payload)


def test_registration_packet_binds_protocol_applicability_sources() -> None:
    registration = importlib.import_module(
        "memcontam.evaluation.phase13_observability_registration"
    )
    packet = registration.load_registration_packet(
        Path("data/phase13/observability/registration_packet_v1.json")
    )

    assert {
        task: identity.path
        for task, identity in packet.applicability_identities.items()
    } == {
        "game24": "data/phase12/registries/candidate_registry_v1.json",
        "math_equation_balancer": "data/phase12/registries/candidate_registry_v1.json",
        "word_sorting": "data/phase12/registries/candidate_registry_v1.json",
        "mmlu_pro_engineering": "src/memcontam/readiness/phase13_new_mcq_candidate.py",
        "mmlu_pro_physics": "src/memcontam/readiness/phase13_new_mcq_candidate.py",
    }


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        ("row", "concrete_seed_id", "fixture-game24-seed-9"),
        ("trial", "execution_key", {"kind": "memory_arm", "arm": "correct"}),
        ("trial", "checkpoint_id", "different-checkpoint"),
        ("trial", "analysis_inclusion", "excluded"),
    ],
)
def test_sequence_continuity_rejects_identity_and_inclusion_drift(
    location: str,
    field: str,
    value: object,
) -> None:
    sequence = importlib.import_module(
        "memcontam.evaluation.phase13_observability_sequence"
    )
    models = importlib.import_module("memcontam.readiness.phase13_observability_models")
    fixture = models.Phase13ObservabilityFixture.model_validate_json(
        Path("data/phase13/observability/fixture_v1.json").read_bytes()
    )
    rows = list(fixture.trials)
    if location == "trial":
        rows[1] = rows[1].model_copy(
            update={"trial": rows[1].trial.model_copy(update={field: value})}
        )
    else:
        rows[1] = rows[1].model_copy(update={field: value})

    with pytest.raises(sequence.Phase13ObservabilityError, match="CONTINUITY_MISMATCH"):
        sequence._validate_continuity(tuple(rows))


def test_unrelated_self_root_is_excluded_from_target_root_scope() -> None:
    registration = importlib.import_module(
        "memcontam.evaluation.phase13_observability_registration"
    )
    models = importlib.import_module("memcontam.readiness.phase13_observability_models")
    validator = importlib.import_module(
        "memcontam.readiness.phase13_observability_validate"
    )
    fixture = models.Phase13ObservabilityFixture.model_validate_json(
        Path("data/phase13/observability/fixture_v1.json").read_bytes()
    )
    unrelated = Phase13LineageNode(
        entry_id="unrelated-root",
        lineage_status="exact",
        injected_root_ids=("unrelated-root",),
    )
    rows = tuple(
        row.model_copy(update={"lineage": (*row.lineage, unrelated)})
        for row in fixture.trials
    )
    packet = registration.load_registration_packet(
        Path("data/phase13/observability/registration_packet_v1.json")
    )

    reconstruction = validator.reconstruct_fixture(
        fixture.model_copy(update={"trials": rows}), packet
    )

    assert reconstruction.trials[-1].root_retention_duration.value == 3


def test_trial_analysis_excludes_unrelated_roots() -> None:
    helpers = importlib.import_module("tests.phase13_observability_helpers")
    module = _module()
    evidence = helpers.evidence(module, retrieved=False, included=False, verified=1)
    unrelated = Phase13LineageNode(
        entry_id="unrelated-root",
        lineage_status="exact",
        injected_root_ids=("unrelated-root",),
    )

    row = module.reconstruct_phase13_trial(
        evidence.model_copy(update={"lineage": (*evidence.lineage, unrelated)})
    )

    assert row.root_entry_ids == ("root-b",)


def test_derived_target_remains_a_descendant_after_root_eviction() -> None:
    registration = importlib.import_module(
        "memcontam.evaluation.phase13_observability_registration"
    )
    models = importlib.import_module("memcontam.readiness.phase13_observability_models")
    validator = importlib.import_module(
        "memcontam.readiness.phase13_observability_validate"
    )
    fixture = models.Phase13ObservabilityFixture.model_validate_json(
        Path("data/phase13/observability/fixture_v1.json").read_bytes()
    )
    packet = registration.load_registration_packet(
        Path("data/phase13/observability/registration_packet_v1.json")
    )

    reconstruction = validator.reconstruct_fixture(fixture, packet)
    recurrent = reconstruction.trials[-1]

    assert recurrent.descendant_entry_ids == ("child-b1",)
    assert recurrent.descendant_storage_persistence.value is True
    assert recurrent.descendant_prompt_visibility.value is True
    assert all(
        "unrelated-root" not in trial.root_entry_ids for trial in reconstruction.trials
    )


def test_sequence_continuity_rejects_invalid_first_row_status() -> None:
    sequence = importlib.import_module(
        "memcontam.evaluation.phase13_observability_sequence"
    )
    models = importlib.import_module("memcontam.readiness.phase13_observability_models")
    fixture = models.Phase13ObservabilityFixture.model_validate_json(
        Path("data/phase13/observability/fixture_v1.json").read_bytes()
    )
    rows = list(fixture.trials)
    rows[0] = rows[0].model_copy(
        update={
            "trial": rows[0].trial.model_copy(update={"analysis_inclusion": "excluded"})
        }
    )

    with pytest.raises(sequence.Phase13ObservabilityError, match="CONTINUITY_MISMATCH"):
        sequence._validate_continuity(tuple(rows))


def test_bot_style_sequence_diagnostics_enter_fixture_aggregation() -> None:
    registration = importlib.import_module(
        "memcontam.evaluation.phase13_observability_registration"
    )
    models = importlib.import_module("memcontam.readiness.phase13_observability_models")
    validator = importlib.import_module(
        "memcontam.readiness.phase13_observability_validate"
    )
    fixture = models.Phase13ObservabilityFixture.model_validate_json(
        Path("data/phase13/observability/fixture_v1.json").read_bytes()
    )
    packet = registration.load_registration_packet(
        Path("data/phase13/observability/registration_packet_v1.json")
    )

    reconstruction = validator.reconstruct_fixture(fixture, packet)
    cell = next(item for item in reconstruction.aggregate.cells if item.baseline == "bot_style")

    assert cell.observability_rates["contam"]["generic_recurrence"].value == 0.5
    assert cell.observability_rates["contam"]["root_retention_duration"].value == 3.0
    assert cell.observability_rates["clean"]["generic_recurrence"].reason == (
        "PRODUCTION_RUNTIME_METADATA_JOIN_NOT_MATERIALIZED"
    )
