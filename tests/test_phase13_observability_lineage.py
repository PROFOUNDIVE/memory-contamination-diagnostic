from __future__ import annotations

import importlib
import hashlib
from pathlib import Path

import pytest
from pydantic import JsonValue

from memcontam.evaluation.phase13_observability_registration import (
    ObservabilityRegistrationPacket,
)
from memcontam.readiness.phase13_production_observability import (
    ProductionObservabilityArchive,
    ProductionObservabilityError,
    ProductionTrialRecord,
    ProviderRequestRecord,
    validate_production_archive,
)

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


def test_answer_exposure_rejects_descendant_absent_from_exact_lineage() -> None:
    module = _module()
    evidence = _evidence(module, retrieved=False, included=False, verified=0)
    forged_span = _span("root-b").model_copy(
        update={
            "entry_id": "forged-descendant",
            "source_ids": ["forged-descendant"],
            "lineage_id": "forged-descendant",
            "contamination_class": "derived",
            "injected_root_ids": ["root-b"],
            "lineage_basis": "recorded_source",
        }
    )
    forged = evidence.model_copy(
        update={
            "context": _context(["forged-descendant"]),
            "target_set": evidence.target_set.model_copy(
                update={"answer_call_spans": (forged_span,)}
            )
        }
    )

    with pytest.raises(module.Phase13ObservabilityError, match="FABRICATED_LINEAGE"):
        module.reconstruct_phase13_trial(forged)


@pytest.mark.parametrize(
    "span_update",
    (
        {"direct_parent_ids": ["unrelated"]},
        {"contamination_class": "injected", "lineage_basis": "seed"},
        {"lineage_basis": "version_edge"},
    ),
    ids=("parents", "classification", "basis"),
)
def test_archive_rejects_answer_span_that_contradicts_exact_lineage(
    span_update: dict[str, JsonValue],
) -> None:
    module = _module()
    evidence = _evidence(module, retrieved=False, included=False, verified=0)
    descendant = _span("root-b").model_copy(
        update={
            "entry_id": "child-b1",
            "source_ids": ["child-b1"],
            "lineage_id": "child-b1",
            "contamination_class": "derived",
            "injected_root_ids": ["root-b"],
            "lineage_basis": "recorded_source",
            "direct_parent_ids": ["root-b"],
            **span_update,
        }
    )
    contradictory = evidence.model_copy(
        update={
            "context": _context(["child-b1"]),
            "target_set": evidence.target_set.model_copy(
                update={"answer_call_spans": (descendant,)}
            ),
            "lineage": (
                evidence.lineage[0],
                module.Phase13LineageNode(
                    entry_id="child-b1",
                    lineage_status="exact",
                    injected_root_ids=("root-b",),
                    direct_parent_ids=("root-b",),
                ),
            ),
        }
    )
    packet_path = (
        Path(__file__).resolve().parents[1]
        / "data/phase13/observability/registration_packet_v1.json"
    )
    packet_raw = packet_path.read_bytes()
    archive = ProductionObservabilityArchive(
        schema_version="phase13_production_observability_archive_v1",
        registration_packet_sha256=hashlib.sha256(packet_raw).hexdigest(),
        u_t_status="NOT_REGISTERED_FOR_CURRENT_MAIN",
        records=(
            ProductionTrialRecord(
                execution_template_id="adversarial-lineage",
                run_id="run-1",
                session_id="session-1",
                scientific_result=False,
                ordered_sample_ids_sha256="a" * 64,
                request=ProviderRequestRecord(
                    api="OpenAI Responses API",
                    model="gpt-5.6-luna",
                    service_tier="default",
                    reasoning_mode="standard",
                    reasoning_effort="none",
                    reasoning_context="current_turn",
                    previous_response_id=None,
                    store=False,
                    timeout_seconds=180,
                    retries_after_initial_attempt=0,
                    semantic_invalid_generic_retry=False,
                ),
                parsed_answer="fixture",
                method_calls=(),
                evidence=contradictory,
            ),
        ),
    )

    with pytest.raises(ProductionObservabilityError) as raised:
        validate_production_archive(
            archive,
            ObservabilityRegistrationPacket.model_validate_json(packet_raw),
            archive.registration_packet_sha256,
        )

    assert raised.value.code == "PRODUCTION_RECONSTRUCTION_FAILED"
    assert getattr(raised.value.__cause__, "code", None) == "EXACT_LINEAGE_REQUIRED"


def test_historical_exact_hops_remain_traversable_for_a_current_write() -> None:
    module = _module()
    exact = _evidence(module, retrieved=True, included=True, verified=0)
    event = _memory_event(
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
                event.lineage_edges[0].model_copy(
                    update={"parent_entry_id": "child-b1"}
                )
            ],
        }
    )
    evidence = exact.model_copy(
        update={
            "baseline": "dc_rs",
            "trial": _trial(retrieved=True, memory_event=True),
            "memory_before_ids": ("root-b", "child-b1"),
            "memory_after_ids": ("root-b", "child-b1", "child-b2"),
            "new_entry_ids": ("child-b2",),
            "memory_events": (event,),
            "lineage": (
                exact.lineage[0],
                module.Phase13LineageNode(
                    entry_id="child-b1",
                    lineage_status="exact",
                    injected_root_ids=("root-b",),
                    direct_parent_ids=("root-b",),
                ),
                module.Phase13LineageNode(
                    entry_id="child-b2",
                    lineage_status="exact",
                    injected_root_ids=("root-b",),
                    direct_parent_ids=("child-b1",),
                ),
            ),
        }
    )

    row = module.reconstruct_phase13_trial(evidence)

    assert row.descendant_entry_ids == ("child-b1", "child-b2")
    assert row.propagation.value is True
    assert row.propagation.path == ("root-b", "child-b1", "child-b2")
