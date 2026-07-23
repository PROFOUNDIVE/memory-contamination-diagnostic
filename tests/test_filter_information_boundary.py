from __future__ import annotations

import ast
from dataclasses import replace
import importlib
import inspect
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_FILTERS = (
    ROOT / "src" / "memcontam" / "memory" / "admission.py",
    ROOT / "src" / "memcontam" / "memory" / "filtered_state.py",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }


def _envelope(cards):
    content = "ordinary operational write"
    return cards.MemoryCardEnvelopeV3(
        entry_id="ordinary",
        baseline="fh_bounded",
        semantic_kind="full_history_transcript",
        schema_version="memory_card_v3",
        writer_id="fh_appender",
        writer_event_id="event-ordinary",
        writer_stage="full_history_generate",
        created_trial_id="trial-ordinary",
        source_trial_ids=("trial-ordinary",),
        source_outcome=None,
        trial_support_ids=("trial-ordinary",),
        memory_support_ids=(),
        direct_parent_ids=(),
        version_predecessor_id=None,
        order_key=1,
        native_component="history",
        content=content,
        content_hash=cards.canonical_content_hash(content),
    )


def test_audit_labels_and_future_verifier_are_unreachable() -> None:
    admission = importlib.import_module("memcontam.memory.admission")
    cards = importlib.import_module("memcontam.memory.cards_v3")
    envelope = _envelope(cards)
    context = admission.AdmissionContext(
        writer_event_ids=frozenset({envelope.writer_event_id}),
        trial_record_ids=frozenset({"trial-ordinary"}),
        evidence_envelopes=(envelope,),
    )

    before = admission.evaluate_admission(envelope, context)
    object.__setattr__(envelope, "origin_class", "protocol_injected")
    object.__setattr__(envelope, "is_injected", True)
    object.__setattr__(envelope, "audit_labels", {"ordinary": "controlled-root"})
    object.__setattr__(envelope, "future_verifier_result", False)
    after = admission.evaluate_admission(envelope, replace(context, evidence_envelopes=(envelope,)))

    assert before == after
    for function in (
        admission.evaluate_admission,
        importlib.import_module("memcontam.memory.filtered_state").partition_native_checkpoint,
        importlib.import_module("memcontam.memory.filtered_state").route_candidate_write,
    ):
        assert not {
            "audit",
            "origin_class",
            "is_injected",
            "future_verifier",
            "verifier_result",
        } & set(inspect.signature(function).parameters)


def test_phase12_filter_has_no_hidden_label_or_verifier_imports() -> None:
    imports = set().union(*(_imports(path) for path in PRODUCTION_FILTERS))

    assert not any(
        forbidden in imported
        for imported in imports
        for forbidden in ("audit", "verifier", "contamination.phase12.registry")
    )
    for path in PRODUCTION_FILTERS:
        source = path.read_text(encoding="utf-8")
        assert "filter_legacy_replay_entries" not in source
        assert "origin_class" not in source
        assert "is_injected" not in source
