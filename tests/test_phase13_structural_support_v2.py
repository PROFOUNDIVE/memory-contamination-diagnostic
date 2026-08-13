from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from typing import Final

import pytest

from memcontam.memory.checkpoint_v3 import NativeState, Phase12Checkpoint, serialize_checkpoint
from memcontam.readiness.phase13_structural_support import (
    CheckpointFact,
    StructuralSupportError,
    evaluate_structural_support,
    parse_prospective_selector_input,
    select_prospective_checkpoint,
)
from memcontam.readiness.phase13_structural_authority import (
    StructuralAuthorityError,
    registered_checkpoints,
)


BASELINES: Final = ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")
SHA_A: Final = "a" * 64
SHA_B: Final = "b" * 64
ROOT: Final = Path(__file__).resolve().parents[1]


def _selector_payload(
    stream_id: str = "game24-seed-10000",
) -> dict[str, object]:
    return {
        "stream_id": stream_id,
        "ordered_trials": [
            {"trial_index": index, "sample_id": f"sample-{index:02d}"}
            for index in range(1, 12)
        ],
        "minimum_clean_prefix_length": 1,
        "suffix_length": 10,
        "resources": [
            {
                "baseline": baseline,
                "checkpoint_trial_index": 1,
                "checkpoint_serializable": True,
                "suffix_executable": True,
                "route_capacity_available": True,
            }
            for baseline in BASELINES
        ],
    }


def _states() -> dict[str, NativeState]:
    return {
        "fh_bounded": NativeState(
            "fh_bounded",
            (),
            {
                "checkpoint_index": 1,
                "records": [{"id": "trial-1"}],
                "first_eviction_trial_id": None,
            },
        ),
        "rag_frozen": NativeState(
            "rag_frozen",
            (),
            {
                "branch": "clean",
                "checkpoint_index": 1,
                "corpus_id": "corpus-v1",
                "index_id": "index-v1",
                "read_only": True,
            },
        ),
        "bot_style": NativeState(
            "bot_style",
            (),
            {
                "templates": [],
                "checkpoint_index": 1,
                "clean_competitor_ids": [],
                "active_capacity": 8,
            },
        ),
        "reflexion_style": NativeState(
            "reflexion_style",
            (),
            {"checkpoint_index": 1, "reflections": [], "active_capacity": 8},
        ),
    }


def _checkpoint_facts(states: dict[str, NativeState] | None = None) -> tuple[CheckpointFact, ...]:
    checkpoints = _checkpoints(states)
    return tuple(
        CheckpointFact(
            baseline=baseline,
            trial_index=1,
            checkpoint=checkpoints[baseline],
            expected_sha256=checkpoints[baseline].canonical_sha256,
        )
        for baseline in BASELINES
    )


def _selection(stream_id: str = "game24-seed-10000"):
    return select_prospective_checkpoint(
        parse_prospective_selector_input(_selector_payload(stream_id))
    )


def _checkpoints(
    states: dict[str, NativeState] | None = None,
) -> dict[str, Phase12Checkpoint]:
    return {
        baseline: serialize_checkpoint(state) for baseline, state in (states or _states()).items()
    }


def test_selects_trial_two_from_source_order_and_resource_feasibility() -> None:
    selector_input = parse_prospective_selector_input(_selector_payload())

    selection = select_prospective_checkpoint(selector_input)

    assert selection.selected_trial_index == 2
    assert selection.checkpoint_trial_index == 1
    assert selection.suffix_trial_indices == tuple(range(2, 12))
    assert tuple(row.baseline for row in selection.decisions) == BASELINES
    assert all(row.selected_trial_index == 2 for row in selection.decisions)


@pytest.mark.parametrize(
    "forbidden",
    [
        {"outcome": "correct"},
        {"nested": {"verifier_result": True}},
        {"nested": [{"eligibility": 1.0}]},
        {"future_task": "sample-12"},
        {"analysis_window": "accuracy-h5-primary"},
        {"future_horizon": 20},
        {"support_rate": 0.5},
    ],
)
def test_selector_boundary_rejects_outcome_and_future_leakage_recursively(
    forbidden: dict[str, object],
) -> None:
    payload = _selector_payload() | forbidden

    with pytest.raises(StructuralSupportError, match="SELECTOR_FIELD_FORBIDDEN"):
        parse_prospective_selector_input(payload)


def test_trial_two_is_invariant_to_nonselector_outcome_and_richness_mutations() -> None:
    selector_input = parse_prospective_selector_input(_selector_payload())
    external_observations = {
        "correctness": [True, False],
        "richness": [0, 999],
        "outcome": {"verified_accuracy": 0.0},
    }

    before = select_prospective_checkpoint(selector_input)
    external_observations["correctness"] = [False, True]
    external_observations["richness"] = [999, 0]
    external_observations["outcome"] = {"verified_accuracy": 1.0}
    after = select_prospective_checkpoint(selector_input)

    assert before == after
    assert before.selected_trial_index == 2


def test_selector_never_calls_historical_common_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    import memcontam.experiment.phase12.checkpoint_selection as historical

    def forbidden_call(**_kwargs: object) -> None:
        pytest.fail("v2 called select_common_checkpoint")

    monkeypatch.setattr(historical, "select_common_checkpoint", forbidden_call)

    assert _selection().selected_trial_index == 2


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("unavailable_suffix", "SUFFIX_UNAVAILABLE"),
        ("duplicate_trial", "DUPLICATE_TRIAL"),
        ("source_order", "SOURCE_ORDER_INVALID"),
        ("duplicate_baseline", "DUPLICATE_BASELINE_DECISION"),
    ],
)
def test_selector_fails_closed_on_structural_mutations(mutation: str, code: str) -> None:
    payload = _selector_payload()
    ordered_trials = payload["ordered_trials"]
    resources = payload["resources"]
    assert isinstance(ordered_trials, list)
    assert isinstance(resources, list)
    if mutation == "unavailable_suffix":
        payload["ordered_trials"] = ordered_trials[:-1]
    elif mutation == "duplicate_trial":
        ordered_trials[1] = ordered_trials[0]
    elif mutation == "source_order":
        ordered_trials[0], ordered_trials[1] = ordered_trials[1], ordered_trials[0]
    else:
        resources[1] = resources[0]

    with pytest.raises(StructuralSupportError, match=code):
        select_prospective_checkpoint(parse_prospective_selector_input(payload))


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("checkpoint_serializable", "CHECKPOINT_SERIALIZATION_UNAVAILABLE"),
        ("suffix_executable", "SUFFIX_UNAVAILABLE"),
        ("route_capacity_available", "SUFFIX_UNAVAILABLE"),
    ],
)
def test_selector_requires_every_baseline_resource_fact(field: str, code: str) -> None:
    payload = _selector_payload()
    resources = payload["resources"]
    assert isinstance(resources, list)
    resource = resources[0]
    assert isinstance(resource, dict)
    resource[field] = False

    with pytest.raises(StructuralSupportError, match=code):
        select_prospective_checkpoint(parse_prospective_selector_input(payload))


def test_empty_reflexion_is_ready_with_zero_richness_and_support_is_intersected() -> None:
    states = _states()
    report = evaluate_structural_support(_selection(), _checkpoint_facts(states))

    readiness = {row.baseline: row for row in report.readiness}
    assert readiness["reflexion_style"].ready is True
    assert readiness["reflexion_style"].richness == 0
    assert readiness["bot_style"].ready is True
    assert readiness["bot_style"].richness == 0
    assert len(report.baseline_local) == 4
    assert len(report.exact_pairs) == 6
    assert all(row.supported for row in report.baseline_local)
    assert all(row.supported for row in report.exact_pairs)
    assert report.strict_global.supported is True
    assert report.strict_global.route_gating is False
    assert report.nomem.baseline == "nomem"
    assert report.nomem.gate_population is False


def test_support_intersections_are_derived_from_local_readiness() -> None:
    facts = list(_checkpoint_facts())
    rag = facts[1].checkpoint
    facts[1] = replace(
        facts[1],
        checkpoint=serialize_checkpoint(
            NativeState("rag_frozen", (), {**rag.state.native_state, "branch": "contam"})
        ),
    )
    facts[1] = replace(facts[1], expected_sha256=facts[1].checkpoint.canonical_sha256)

    report = evaluate_structural_support(_selection("game24-seed-10001"), tuple(facts))

    local = {row.baseline: row.supported for row in report.baseline_local}
    pairs = {row.pair_id: row.supported for row in report.exact_pairs}
    assert local == {
        "fh_bounded": True,
        "rag_frozen": False,
        "bot_style": True,
        "reflexion_style": True,
    }
    assert pairs == {
        "P01": False,
        "P02": True,
        "P03": True,
        "P04": False,
        "P05": False,
        "P06": True,
    }
    assert report.strict_global.supported is False


def test_changed_checkpoint_hash_and_nomem_support_insertion_are_rejected() -> None:
    facts = list(_checkpoint_facts())
    facts[0] = replace(facts[0], expected_sha256=SHA_A)

    with pytest.raises(StructuralSupportError, match="CHECKPOINT_HASH_MISMATCH"):
        evaluate_structural_support(_selection(), tuple(facts))

    checkpoint = serialize_checkpoint(NativeState("fh_bounded", (), {"records": []}))
    nomem = CheckpointFact("nomem", 1, checkpoint, SHA_B)
    with pytest.raises(StructuralSupportError, match="NOMEM_SUPPORT_FORBIDDEN"):
        evaluate_structural_support(_selection(), (*_checkpoint_facts(), nomem))


def test_native_trial_two_checkpoint_cannot_be_relabelled_as_trial_one() -> None:
    facts = list(_checkpoint_facts())
    source = facts[0].checkpoint.state
    trial_two = serialize_checkpoint(
        NativeState(source.baseline, source.entries, {**source.native_state, "checkpoint_index": 2})
    )
    facts[0] = CheckpointFact("fh_bounded", 1, trial_two, trial_two.canonical_sha256)

    with pytest.raises(StructuralSupportError, match="CHECKPOINT_LINEAGE_INVALID"):
        evaluate_structural_support(_selection(), tuple(facts))


def test_forged_checkpoint_identity_is_rejected_before_readiness() -> None:
    facts = list(_checkpoint_facts())
    checkpoint = facts[0].checkpoint
    forged = replace(
        checkpoint,
        identity=replace(checkpoint.identity, checkpoint_id="forged-checkpoint"),
    )
    facts[0] = replace(facts[0], checkpoint=forged)

    with pytest.raises(StructuralSupportError, match="CHECKPOINT_IDENTITY_MISMATCH"):
        evaluate_structural_support(_selection(), tuple(facts))


def test_caller_trial_relabel_is_rejected_against_registered_checkpoint() -> None:
    facts = list(_checkpoint_facts())
    facts[0] = replace(facts[0], trial_index=2)

    with pytest.raises(StructuralSupportError, match="CHECKPOINT_TRIAL_INVALID"):
        evaluate_structural_support(_selection(), tuple(facts))


def test_forged_canonical_hash_is_rejected_before_readiness() -> None:
    facts = list(_checkpoint_facts())
    checkpoint = facts[0].checkpoint
    facts[0] = replace(
        facts[0],
        checkpoint=replace(checkpoint, canonical_sha256=SHA_A),
        expected_sha256=SHA_A,
    )

    with pytest.raises(StructuralSupportError, match="CHECKPOINT_HASH_MISMATCH"):
        evaluate_structural_support(_selection(), tuple(facts))


def test_self_consistent_unregistered_trial_one_checkpoint_is_rejected() -> None:
    selection = _selection()
    facts = list(_checkpoint_facts())
    source = facts[0].checkpoint.state
    unregistered = serialize_checkpoint(
        NativeState(source.baseline, source.entries, {**source.native_state, "records": []})
    )
    facts[0] = CheckpointFact(
        "fh_bounded",
        1,
        unregistered,
        unregistered.canonical_sha256,
    )

    with pytest.raises(StructuralSupportError, match="UNREGISTERED_CHECKPOINT"):
        evaluate_structural_support(selection, tuple(facts))


def test_coordinated_checkpoint_and_resource_substitution_is_rejected() -> None:
    states = _states()
    states["fh_bounded"] = NativeState(
        "fh_bounded",
        (),
        {**states["fh_bounded"].native_state, "records": []},
    )
    states["rag_frozen"] = NativeState(
        "rag_frozen",
        (),
        {**states["rag_frozen"].native_state, "corpus_id": "substituted-corpus"},
    )
    states["bot_style"] = NativeState(
        "bot_style",
        (),
        {**states["bot_style"].native_state, "templates": ["substituted-template"]},
    )
    states["reflexion_style"] = NativeState(
        "reflexion_style",
        (),
        {**states["reflexion_style"].native_state, "reflections": ["substituted-reflection"]},
    )
    with pytest.raises(StructuralSupportError, match="UNREGISTERED_CHECKPOINT"):
        evaluate_structural_support(_selection(), _checkpoint_facts(states))


def test_direct_selection_decision_substitution_is_rejected() -> None:
    facts = list(_checkpoint_facts())
    source = facts[0].checkpoint.state
    substituted = serialize_checkpoint(
        NativeState(source.baseline, source.entries, {**source.native_state, "records": []})
    )
    facts[0] = CheckpointFact("fh_bounded", 1, substituted, substituted.canonical_sha256)
    selection = _selection()
    decisions = list(selection.decisions)
    decisions[0] = replace(
        decisions[0],
        registered_checkpoint_id=substituted.identity.checkpoint_id,
        registered_checkpoint_sha256=substituted.canonical_sha256,
    )

    with pytest.raises(StructuralSupportError, match="UNREGISTERED_CHECKPOINT"):
        evaluate_structural_support(replace(selection, decisions=tuple(decisions)), tuple(facts))


def test_resigned_checkpoint_registry_substitution_is_rejected(tmp_path: Path) -> None:
    registry_path = tmp_path / "data/phase13/authority/structural_checkpoint_registry_v1.json"
    partition_path = tmp_path / "data/phase13/calibration_v2/seed_partition_registry_v1.json"
    registry_path.parent.mkdir(parents=True)
    partition_path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "data/phase13/calibration_v2/seed_partition_registry_v1.json", partition_path)
    payload = json.loads(
        (ROOT / "data/phase13/authority/structural_checkpoint_registry_v1.json").read_bytes()
    )
    payload["streams"][0]["checkpoints"]["fh_bounded"] = {
        "checkpoint_id": "checkpoint-substituted",
        "sha256": SHA_A,
    }
    unsigned = dict(payload)
    unsigned.pop("registry_hash")
    payload["registry_hash"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    registry_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StructuralAuthorityError, match="CHECKPOINT_REGISTRY_AUTHORITY_MISMATCH"):
        registered_checkpoints("game24-seed-10000", tmp_path)


def test_duplicate_checkpoint_trial_is_rejected_without_mutating_inputs() -> None:
    facts = _checkpoint_facts()

    with pytest.raises(StructuralSupportError, match="DUPLICATE_CHECKPOINT"):
        evaluate_structural_support(_selection(), (*facts, facts[0]))

    assert facts == _checkpoint_facts()
