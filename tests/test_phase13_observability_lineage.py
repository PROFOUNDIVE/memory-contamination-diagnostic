from __future__ import annotations

import importlib

import pytest

_helpers = importlib.import_module("tests.phase13_observability_helpers")
_context = _helpers.context
_evidence = _helpers.evidence
_memory_event = _helpers.memory_event
_module = _helpers.module
_retrieval = _helpers.retrieval
_span = _helpers.span
_trial = _helpers.trial


def test_propagation_requires_exposure_and_fully_exact_recorded_path() -> None:
    module = _module()
    evidence = _evidence(module, retrieved=False, included=False, verified=0).model_copy(
        update={
            "baseline": "bot_style",
            "trial": _trial(memory_event=True),
            "memory_after_ids": ("root-b", "child-b1"),
            "new_entry_ids": ("child-b1",),
            "memory_events": (
                _memory_event(("root-b",), ("root-b", "child-b1"), ("child-b1",)),
            ),
            "lineage": (
                module.Phase13LineageNode(
                    entry_id="root-b", lineage_status="exact", injected_root_ids=("root-b",)
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
    with pytest.raises(module.Phase13ObservabilityError, match="PROPAGATION_REQUIRES_EXPOSURE"):
        module.reconstruct_phase13_trial(evidence)

    exact = _evidence(module, retrieved=True, included=True, verified=0)
    approximate_path = exact.model_copy(
        update={
            "baseline": "bot_style",
            "trial": _trial(retrieved=True, memory_event=True),
            "memory_after_ids": ("root-b", "child-b1"),
            "new_entry_ids": ("child-b1",),
            "memory_events": (
                _memory_event(("root-b",), ("root-b", "child-b1"), ("child-b1",)),
            ),
            "lineage": (
                exact.lineage[0],
                module.Phase13LineageNode(
                    entry_id="middle",
                    lineage_status="approximate",
                    injected_root_ids=("root-b",),
                    direct_parent_ids=("root-b",),
                ),
                module.Phase13LineageNode(
                    entry_id="child-b1",
                    lineage_status="exact",
                    injected_root_ids=("root-b",),
                    direct_parent_ids=("middle",),
                ),
            ),
        }
    )
    with pytest.raises(module.Phase13ObservabilityError, match="EXACT_LINEAGE_REQUIRED"):
        module.reconstruct_phase13_trial(approximate_path)

    detached = exact.model_copy(
        update={
            "baseline": "bot_style",
            "trial": _trial(retrieved=True, memory_event=True),
            "memory_after_ids": ("root-b", "child-b1"),
            "new_entry_ids": ("child-b1",),
            "memory_events": (
                _memory_event(("root-b",), ("root-b", "child-b1"), ("child-b1",)).model_copy(
                    update={"lineage_edges": []}
                ),
            ),
            "lineage": (
                exact.lineage[0],
                module.Phase13LineageNode(
                    entry_id="child-b1",
                    lineage_status="exact",
                    injected_root_ids=("root-b",),
                    direct_parent_ids=("root-b",),
                ),
            ),
        }
    )
    with pytest.raises(module.Phase13ObservabilityError, match="EXACT_LINEAGE_REQUIRED"):
        module.reconstruct_phase13_trial(detached)

    recorded_event = _memory_event(("root-b",), ("root-b", "child-b1"), ("child-b1",))
    bad_relation_event = recorded_event.model_copy(
        update={
            "lineage_edges": [
                recorded_event.lineage_edges[0].model_copy(
                    update={"relation": "retrieval_only", "lineage_basis": "none"}
                )
            ]
        }
    )
    with pytest.raises(module.Phase13ObservabilityError, match="EXACT_LINEAGE_REQUIRED"):
        module.reconstruct_phase13_trial(
            detached.model_copy(update={"memory_events": (bad_relation_event,)})
        )

    reused_id = detached.model_copy(
        update={
            "new_entry_ids": ("root-b",),
            "memory_events": (
                detached.memory_events[0].model_copy(
                    update={"new_entry_ids": ["root-b"], "lineage_edges": []}
                ),
            ),
        }
    )
    with pytest.raises(module.Phase13ObservabilityError, match="MEMORY_MUTATION_SET_MISMATCH"):
        module.reconstruct_phase13_trial(reused_id)

    with pytest.raises(module.Phase13ObservabilityError, match="MUTATION_CONTEXT_REQUIRED"):
        module.reconstruct_phase13_trial(detached.model_copy(update={"context": None}))

    unrelated_event = _memory_event(
        ("root-b", "clean-root"),
        ("root-b", "clean-root", "clean-child"),
        ("clean-child",),
    )
    unrelated_event = unrelated_event.model_copy(
        update={
            "parent_entry_ids": ["clean-root"],
            "source_entry_ids": ["clean-root"],
            "contaminated_source_ids": [],
            "lineage_edges": [
                unrelated_event.lineage_edges[0].model_copy(
                    update={"parent_entry_id": "clean-root", "injected_root_ids": ["clean-root"]}
                )
            ],
        }
    )
    unrelated = exact.model_copy(
        update={
            "baseline": "bot_style",
            "trial": _trial(retrieved=True, memory_event=True),
            "memory_before_ids": ("root-b", "clean-root"),
            "memory_after_ids": ("root-b", "clean-root", "clean-child"),
            "new_entry_ids": ("clean-child",),
            "memory_events": (unrelated_event,),
            "lineage": (
                exact.lineage[0],
                module.Phase13LineageNode(
                    entry_id="clean-root",
                    lineage_status="exact",
                    injected_root_ids=("clean-root",),
                ),
                module.Phase13LineageNode(
                    entry_id="clean-child",
                    lineage_status="exact",
                    injected_root_ids=("clean-root",),
                    direct_parent_ids=("clean-root",),
                ),
            ),
        }
    )

    unrelated_row = module.reconstruct_phase13_trial(unrelated)

    assert unrelated_row.propagation.status == "supported"
    assert unrelated_row.propagation.value is False


def test_propagation_must_descend_from_the_exact_exposed_root() -> None:
    module = _module()
    evidence = _evidence(module, retrieved=True, included=True, verified=0)
    root_a_span = _span("root-a")
    cross_root = evidence.model_copy(
        update={
            "baseline": "bot_style",
            "trial": _trial(retrieved=True, memory_event=True),
            "retrievals": (
                _retrieval().model_copy(update={"retrieved_entry_ids": ["root-a"]}),
            ),
            "context": _context(["root-a"]),
            "target_set": module.Phase13TargetSetEvidence(
                target_set_id="targets-v1",
                target_entry_ids=("root-a", "root-b"),
                answer_call_id="answer-1",
                answer_call_spans=(root_a_span,),
            ),
            "memory_before_ids": ("root-a", "root-b"),
            "memory_after_ids": ("root-a", "root-b", "child-b1"),
            "new_entry_ids": ("child-b1",),
            "memory_events": (
                _memory_event(
                    ("root-a", "root-b"),
                    ("root-a", "root-b", "child-b1"),
                    ("child-b1",),
                ),
            ),
            "lineage": (
                module.Phase13LineageNode(
                    entry_id="root-a", lineage_status="exact", injected_root_ids=("root-a",)
                ),
                module.Phase13LineageNode(
                    entry_id="root-b", lineage_status="exact", injected_root_ids=("root-b",)
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

    row = module.reconstruct_phase13_trial(cross_root)

    assert row.propagation.status == "supported"
    assert row.propagation.value is False
