from __future__ import annotations

import importlib

import pytest

_helpers = importlib.import_module("tests.phase13_observability_helpers")
_evidence = _helpers.evidence
_memory_event = _helpers.memory_event
_module = _helpers.module
_trial = _helpers.trial


@pytest.mark.parametrize(
    ("retrieved", "included", "verified", "expected"),
    [
        (False, False, 1, (True, False, False, False, 1)),
        (True, False, 1, (True, True, False, False, 1)),
        (True, True, 1, (True, True, True, True, 1)),
        (True, True, 0, (True, True, True, True, 0)),
    ],
)
def test_reconstructs_distinct_store_retrieval_context_exposure_states(
    retrieved: bool,
    included: bool,
    verified: int,
    expected: tuple[bool, bool, bool, bool, int],
) -> None:
    module = _module()

    row = module.reconstruct_phase13_trial(
        _evidence(module, retrieved=retrieved, included=included, verified=verified)
    )

    assert (
        row.target_present_in_store_before_answer.value,
        row.target_retrieved.value,
        row.target_final_context_included.value,
        row.theory_exposure.value,
        row.verified_outcome,
    ) == expected
    assert row.operational_use.status == "not_registered"
    assert row.operational_use.reason == "NOT_REGISTERED_FOR_CURRENT_MAIN"


def test_marks_structural_propagation_inapplicable_and_requires_exact_recorded_lineage() -> None:
    module = _module()
    rag = module.reconstruct_phase13_trial(
        _evidence(module, retrieved=True, included=True, verified=0)
    )
    bot_evidence = _evidence(module, retrieved=True, included=True, verified=0).model_copy(
        update={
            "baseline": "bot_style",
            "trial": _trial(retrieved=True, memory_event=True),
            "memory_after_ids": ("root-b", "child-b1"),
            "new_entry_ids": ("child-b1",),
            "memory_events": (
                _memory_event(("root-b",), ("root-b", "child-b1"), ("child-b1",)),
            ),
            "lineage": (
                module.Phase13LineageNode(
                    entry_id="root-b",
                    lineage_status="exact",
                    injected_root_ids=("root-b",),
                ),
                module.Phase13LineageNode(
                    entry_id="child-b1",
                    lineage_status="exact",
                    injected_root_ids=("root-b",),
                    direct_parent_ids=("root-b",),
                ),
            ),
        }
    )
    bot = module.reconstruct_phase13_trial(bot_evidence)

    assert rag.propagation.status == "not_applicable"
    assert rag.propagation.value is None
    assert bot.propagation.status == "supported"
    assert bot.propagation.value is True
    assert bot.propagation.path == ("root-b", "child-b1")

    approximate = bot_evidence.model_copy(
        update={
            "lineage": (
                bot_evidence.lineage[0],
                bot_evidence.lineage[1].model_copy(update={"lineage_status": "approximate"}),
            )
        }
    )
    with pytest.raises(module.Phase13ObservabilityError, match="EXACT_LINEAGE_REQUIRED"):
        module.reconstruct_phase13_trial(approximate)


def test_noncontamination_arms_preserve_not_applicable_status() -> None:
    module = _module()
    for arm in ("clean", "correct", "irrelevant"):
        evidence = _evidence(module, retrieved=False, included=False, verified=1).model_copy(
            update={
                "trial": _trial(arm=arm),
                "target_set": module.Phase13TargetSetEvidence(
                    target_set_id="targets-v1",
                    target_entry_ids=(),
                    answer_call_id="answer-1",
                ),
                "memory_before_ids": (),
                "memory_after_ids": (),
                "lineage": (),
            }
        )

        row = module.reconstruct_phase13_trial(evidence)

        assert row.target_present_in_store_before_answer.status == "not_applicable"
        assert row.theory_exposure.status == "not_applicable"


def test_rejects_foreign_trial_and_answer_call_joins() -> None:
    module = _module()
    evidence = _evidence(module, retrieved=True, included=True, verified=0)
    foreign_context = evidence.context.model_copy(update={"trial_id": "other-trial"})
    with pytest.raises(module.Phase13ObservabilityError, match="TRIAL_EVENT_IDENTITY_MISMATCH"):
        module.reconstruct_phase13_trial(evidence.model_copy(update={"context": foreign_context}))

    foreign_span = evidence.target_set.answer_call_spans[0].model_copy(
        update={"parent_call_id": "other-answer"}
    )
    with pytest.raises(module.Phase13ObservabilityError, match="ANSWER_CALL_IDENTITY_MISMATCH"):
        module.reconstruct_phase13_trial(
            evidence.model_copy(
                update={
                    "target_set": evidence.target_set.model_copy(
                        update={"answer_call_spans": (foreign_span,)}
                    )
                }
            )
        )

    wrong_context_id = evidence.context.model_copy(update={"event_id": "other-context"})
    with pytest.raises(module.Phase13ObservabilityError, match="CONTEXT_EVENT_IDENTITY_MISMATCH"):
        module.reconstruct_phase13_trial(evidence.model_copy(update={"context": wrong_context_id}))

    wrong_retrieval_id = evidence.retrievals[0].model_copy(update={"event_id": "other-retrieval"})
    with pytest.raises(module.Phase13ObservabilityError, match="RETRIEVAL_EVENT_IDENTITY_MISMATCH"):
        module.reconstruct_phase13_trial(
            evidence.model_copy(update={"retrievals": (wrong_retrieval_id,)})
        )

    late_retrieval = evidence.retrievals[0].model_copy(update={"event_seq": 2})
    with pytest.raises(module.Phase13ObservabilityError, match="EVENT_ORDER_MISMATCH"):
        module.reconstruct_phase13_trial(evidence.model_copy(update={"retrievals": (late_retrieval,)}))

    unprovided_writer = evidence.trial.model_copy(update={"memory_event_ids": ["missing-event"]})
    with pytest.raises(module.Phase13ObservabilityError, match="WRITER_EVENT_IDENTITY_MISMATCH"):
        module.reconstruct_phase13_trial(evidence.model_copy(update={"trial": unprovided_writer}))

    wrong_target_set = evidence.target_set.answer_call_spans[0].model_copy(
        update={"target_set_id": "other-targets"}
    )
    with pytest.raises(module.Phase13ObservabilityError, match="TARGET_SET_IDENTITY_MISMATCH"):
        module.reconstruct_phase13_trial(
            evidence.model_copy(
                update={
                    "target_set": evidence.target_set.model_copy(
                        update={"answer_call_spans": (wrong_target_set,)}
                    )
                }
            )
        )


def test_rejects_target_contamination_evidence_on_noncontam_arms() -> None:
    module = _module()
    evidence = _evidence(module, retrieved=True, included=True, verified=1).model_copy(
        update={"trial": _trial(retrieved=True, arm="clean")}
    )

    with pytest.raises(module.Phase13ObservabilityError, match="NONCONTAM_TARGET_EVIDENCE"):
        module.reconstruct_phase13_trial(evidence)


def test_exposes_unregistered_sequential_policies_as_blockers_not_zeroes() -> None:
    module = _module()

    row = module.reconstruct_phase13_trial(
        _evidence(module, retrieved=True, included=True, verified=0)
    )

    assert row.failure_class.status == "unavailable"
    assert row.generic_recurrence.status == "unavailable"
    assert row.exposure_conditioned_recurrence.status == "unavailable"
    assert row.post_eviction_recurrence.status == "unavailable"
    assert row.root_retention_duration.status == "unavailable"
    assert row.prompt_retention_duration.status == "unavailable"
    assert row.descendant_retention_duration.status == "unavailable"
    assert row.root_storage_persistence.status == "supported"
    assert row.root_storage_persistence.value is True
