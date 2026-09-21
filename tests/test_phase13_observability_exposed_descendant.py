from __future__ import annotations

import importlib

_helpers = importlib.import_module("tests.phase13_observability_helpers")
context = _helpers.context
evidence = _helpers.evidence
memory_event = _helpers.memory_event
module = _helpers.module
retrieval = _helpers.retrieval
span = _helpers.span
trial = _helpers.trial


def test_propagation_starts_from_an_exact_exposed_descendant() -> None:
    observability = module()
    exact = evidence(observability, retrieved=True, included=True, verified=0)
    exposed_descendant = span("root-b").model_copy(
        update={
            "entry_id": "child-b1",
            "source_ids": ["child-b1"],
            "direct_parent_ids": ["root-b"],
            "lineage_id": "child-b1",
            "contamination_class": "derived",
            "injected_root_ids": ["root-b"],
            "lineage_basis": "recorded_parent",
        }
    )
    event = memory_event(
        ("root-b", "child-b1"),
        ("root-b", "child-b1", "child-b2"),
        ("child-b2",),
    )
    event = event.model_copy(
        update={
            "baseline": "dc_rs",
            "parent_entry_ids": ["child-b1"],
            "source_entry_ids": ["child-b1"],
            "lineage_edges": [
                event.lineage_edges[0].model_copy(update={"parent_entry_id": "child-b1"})
            ],
        }
    )
    trial_evidence = exact.model_copy(
        update={
            "baseline": "dc_rs",
            "trial": trial(retrieved=True, memory_event=True),
            "retrievals": (
                retrieval().model_copy(update={"retrieved_entry_ids": ["child-b1"]}),
            ),
            "context": context(["child-b1"]),
            "target_set": exact.target_set.model_copy(
                update={"answer_call_spans": (exposed_descendant,)}
            ),
            "memory_before_ids": ("root-b", "child-b1"),
            "memory_after_ids": ("root-b", "child-b1", "child-b2"),
            "new_entry_ids": ("child-b2",),
            "memory_events": (event,),
            "lineage": (
                exact.lineage[0],
                observability.Phase13LineageNode(
                    entry_id="child-b1",
                    lineage_status="exact",
                    injected_root_ids=("root-b",),
                    direct_parent_ids=("root-b",),
                ),
                observability.Phase13LineageNode(
                    entry_id="child-b2",
                    lineage_status="exact",
                    injected_root_ids=("root-b",),
                    direct_parent_ids=("child-b1",),
                ),
            ),
        }
    )

    reconstructed = observability.reconstruct_phase13_trial(trial_evidence)

    assert reconstructed.propagation.value is True
    assert reconstructed.propagation.path == ("child-b1", "child-b2")
