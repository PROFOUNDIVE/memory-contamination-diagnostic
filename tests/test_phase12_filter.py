from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILTER_FIXTURE = ROOT / "tests" / "fixtures" / "phase12" / "FX-FILTER-001.json"
BRANCH_FIXTURE = ROOT / "tests" / "fixtures" / "phase12" / "FX-BRANCH-001.json"

WRITERS = {
    "fh_bounded": ("full_history_transcript", "fh_appender", "full_history_generate", "history", True),
    "rag_frozen": ("rag_document", "rag_corpus_loader", "rag_corpus_load", "corpus", False),
    "bot_style": ("thought_template", "bot_buffer_manager", "bot_thought_distill", "buffer", True),
    "reflexion_style": (
        "verbal_reflection",
        "reflexion_reflector",
        "reflexion_reflect",
        "reflections",
        True,
    ),
}


def _modules():
    return (
        importlib.import_module("memcontam.memory.admission"),
        importlib.import_module("memcontam.memory.cards_v3"),
        importlib.import_module("memcontam.memory.checkpoint_v3"),
        importlib.import_module("memcontam.memory.filtered_state"),
    )


def _fixture(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _envelope(cards, entry_id: str, baseline: str, order_key: int, **overrides):
    semantic_kind, writer_id, writer_stage, native_component, requires_source = WRITERS[baseline]
    trial_id = f"trial-{entry_id}" if requires_source else None
    payload = {
        "entry_id": entry_id,
        "baseline": baseline,
        "semantic_kind": semantic_kind,
        "schema_version": "memory_card_v3",
        "writer_id": writer_id,
        "writer_event_id": f"event-{entry_id}",
        "writer_stage": writer_stage,
        "created_trial_id": trial_id,
        "source_trial_ids": (trial_id,) if trial_id else (),
        "source_outcome": None,
        "trial_support_ids": (trial_id,) if trial_id else (),
        "memory_support_ids": (),
        "direct_parent_ids": (),
        "version_predecessor_id": None,
        "order_key": order_key,
        "native_component": native_component,
        "content": f"content for {entry_id}",
    }
    payload.update(overrides)
    payload["content_hash"] = cards.canonical_content_hash(payload["content"])
    return cards.MemoryCardEnvelopeV3(**payload)


def _entry(checkpoints, cards, envelope):
    return checkpoints.NativeEntry(
        entry_id=envelope.entry_id,
        semantic_kind=envelope.semantic_kind,
        schema_version="phase12_native_entry_v1",
        native_component=envelope.native_component,
        content=envelope.content,
        content_hash=cards.canonical_content_hash(envelope.content),
        direct_parent_ids=envelope.direct_parent_ids,
    )


def _context(admission, envelopes, *, quarantined=(), trial_ids=()):
    return admission.AdmissionContext(
        writer_event_ids=frozenset(envelope.writer_event_id for envelope in envelopes),
        trial_record_ids=frozenset(trial_ids),
        evidence_envelopes=tuple(envelopes),
        quarantined_envelopes=tuple(quarantined),
    )


def _entry_ids(entries):
    return tuple(entry.entry_id if hasattr(entry, "entry_id") else entry for entry in entries)


def test_fixture_routes_by_provenance_and_parent_state() -> None:
    admission, cards, _, _ = _modules()
    fixture = _fixture(FILTER_FIXTURE)

    ordinary = _envelope(cards, "ordinary-ok", "fh_bounded", 1, created_trial_id="trial-1",
                        source_trial_ids=("trial-1",), trial_support_ids=("trial-1",))
    external = _envelope(
        cards,
        "external-root",
        "fh_bounded",
        2,
        writer_id="protocol_injector",
        writer_stage="protocol_inject",
        created_trial_id=None,
        source_trial_ids=(),
        trial_support_ids=(),
    )
    child = _envelope(
        cards,
        "child-of-root",
        "bot_style",
        3,
        direct_parent_ids=("external-root",),
        memory_support_ids=("external-root",),
        created_trial_id="trial-2",
        source_trial_ids=("trial-2",),
        trial_support_ids=("trial-2",),
    )
    context = _context(admission, (ordinary, external, child), trial_ids=("trial-1", "trial-2"))

    decisions = (
        admission.evaluate_admission(ordinary, context),
        admission.evaluate_admission(external, context),
        admission.evaluate_admission(
            child,
            replace(context, quarantined_envelopes=(external,)),
        ),
    )

    observed = [
        (decision.entry_id, "active" if decision.admitted else "quarantine", None if decision.admitted else decision.reason)
        for decision in decisions
    ]
    assert observed == [(entry["id"], *entry["expected"]) for entry in fixture["entries"]]


def test_filter_starts_from_contam_checkpoint_and_routes_later_writes() -> None:
    admission, cards, checkpoints, filtered_state = _modules()
    fixture = _fixture(BRANCH_FIXTURE)

    for baseline, prefix in fixture["baseline_prefixes"].items():
        source = checkpoints.serialize_checkpoint(checkpoints.NativeState.from_mapping(prefix["checkpoint"]))
        source_envelopes = tuple(
            _envelope(cards, entry_id, baseline, order_key)
            for order_key, entry_id in enumerate(prefix["checkpoint"]["entries"], start=1)
        )
        root_envelope = _envelope(
            cards,
            f"{baseline}-controlled-root",
            baseline,
            len(source_envelopes) + 1,
            writer_id="protocol_injector",
            writer_stage="protocol_inject",
            created_trial_id=None,
            source_trial_ids=(),
            trial_support_ids=(),
        )
        contam = checkpoints.append_native_entry(source, _entry(checkpoints, cards, root_envelope))
        context = _context(
            admission,
            (*source_envelopes, root_envelope),
            trial_ids=tuple(
                trial_id
                for envelope in source_envelopes
                for trial_id in envelope.trial_support_ids
            ),
        )

        partition = filtered_state.partition_native_checkpoint(contam, context)

        assert partition.source_checkpoint == contam
        assert partition.active.state.entries == source.state.entries
        assert partition.quarantine.state.entries == (_entry(checkpoints, cards, root_envelope),)
        assert tuple(decision.entry_id for decision in partition.decisions) == tuple(
            [*prefix["checkpoint"]["entries"], root_envelope.entry_id]
        )
        assert _entry_ids(partition.reader_entries) == tuple(prefix["checkpoint"]["entries"])
        assert partition.reader_entries == partition.updater_entries

        ordinary = _envelope(cards, f"{baseline}-ordinary", baseline, len(source_envelopes) + 2)
        next_context = replace(
            context,
            writer_event_ids=context.writer_event_ids | {ordinary.writer_event_id},
            trial_record_ids=context.trial_record_ids | set(ordinary.trial_support_ids),
        )
        transition = filtered_state.route_candidate_write(
            partition,
            filtered_state.CandidateWrite(_entry(checkpoints, cards, ordinary), ordinary),
            next_context,
        )

        assert transition.decision.admitted
        assert transition.state.quarantine.state.entries == partition.quarantine.state.entries
        assert _entry_ids(transition.reader_entries) == tuple(
            [*prefix["checkpoint"]["entries"], ordinary.entry_id]
        )
        assert root_envelope.entry_id not in set(_entry_ids(transition.reader_entries))


def test_rejected_replacement_keeps_the_prior_active_version() -> None:
    admission, cards, checkpoints, filtered_state = _modules()
    original = _envelope(cards, "original", "fh_bounded", 1)
    source = checkpoints.serialize_checkpoint(
        checkpoints.NativeState("fh_bounded", (_entry(checkpoints, cards, original),), {"records": []})
    )
    context = _context(admission, (original,), trial_ids=original.trial_support_ids)
    partition = filtered_state.partition_native_checkpoint(source, context)
    rejected = _envelope(
        cards,
        "replacement",
        "fh_bounded",
        2,
        writer_id="protocol_injector",
        writer_stage="protocol_inject",
        created_trial_id=None,
        source_trial_ids=(),
        trial_support_ids=(),
        version_predecessor_id="original",
    )
    transition = filtered_state.route_candidate_write(
        partition,
        filtered_state.CandidateWrite(_entry(checkpoints, cards, rejected), rejected),
        replace(context, writer_event_ids=context.writer_event_ids | {rejected.writer_event_id}),
    )

    assert transition.decision.reason == "UNREGISTERED_WRITER_EVENT"
    assert list(_entry_ids(transition.state.reader_entries)) == ["original"]
    assert [entry.entry_id for entry in transition.state.quarantine.state.entries] == ["replacement"]


def test_admission_reports_support_parent_and_version_evidence_failures() -> None:
    admission, cards, _, _ = _modules()
    parent = _envelope(cards, "parent", "fh_bounded", 1)
    context = _context(admission, (parent,), trial_ids=parent.trial_support_ids)
    active_context = replace(context, active_envelopes=(parent,))
    missing_parent = _envelope(
        cards,
        "missing-parent",
        "fh_bounded",
        2,
        direct_parent_ids=("absent",),
        memory_support_ids=("absent",),
    )
    missing_support = _envelope(
        cards,
        "missing-support",
        "fh_bounded",
        2,
        direct_parent_ids=("parent",),
        memory_support_ids=("absent",),
    )
    invalid_version = _envelope(
        cards,
        "invalid-version",
        "bot_style",
        2,
        version_predecessor_id="parent",
    )

    assert admission.evaluate_admission(missing_parent, active_context).reason == "MISSING_PARENT_EVIDENCE"
    assert admission.evaluate_admission(missing_support, active_context).reason == "MISSING_SUPPORT_EVIDENCE"
    version_context = replace(
        active_context,
        trial_record_ids=active_context.trial_record_ids | set(invalid_version.trial_support_ids),
    )
    assert admission.evaluate_admission(invalid_version, version_context).reason == "INVALID_VERSION_EVIDENCE"
