from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.contamination.phase12.renderers import RendererRegistry
from memcontam.memory.admission import AdmissionContext
from memcontam.memory.cards_v3 import MemoryCardEnvelopeV3, canonical_content_hash
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "phase12" / "FX-BRANCH-001.json"
REGISTRY_PATH = ROOT / "data" / "phase12" / "registries" / "candidate_registry_v1.json"

WRITERS = {
    "fh_bounded": ("full_history_transcript", "fh_appender", "full_history_generate", "history"),
    "rag_frozen": ("rag_document", "rag_corpus_loader", "rag_corpus_load", "corpus"),
    "bot_style": ("thought_template", "bot_buffer_manager", "bot_thought_distill", "buffer"),
    "reflexion_style": (
        "verbal_reflection",
        "reflexion_reflector",
        "reflexion_reflect",
        "reflections",
    ),
}


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _context(baseline: str, entry_ids: tuple[str, ...]) -> AdmissionContext:
    semantic_kind, writer_id, writer_stage, native_component = WRITERS[baseline]
    envelopes = tuple(
        MemoryCardEnvelopeV3(
            entry_id=entry_id,
            baseline=baseline,
            semantic_kind=semantic_kind,
            schema_version="memory_card_v3",
            writer_id=writer_id,
            writer_event_id=f"event-{entry_id}",
            writer_stage=writer_stage,
            created_trial_id=f"trial-{entry_id}" if baseline != "rag_frozen" else None,
            source_trial_ids=(f"trial-{entry_id}",) if baseline != "rag_frozen" else (),
            source_outcome=None,
            trial_support_ids=(f"trial-{entry_id}",) if baseline != "rag_frozen" else (),
            memory_support_ids=(),
            direct_parent_ids=(),
            version_predecessor_id=None,
            order_key=index,
            native_component=native_component,
            content=f"content for {entry_id}",
            content_hash=canonical_content_hash(f"content for {entry_id}"),
        )
        for index, entry_id in enumerate(entry_ids, start=1)
    )
    return AdmissionContext(
        writer_event_ids=frozenset(envelope.writer_event_id for envelope in envelopes),
        trial_record_ids=frozenset(
            trial_id for envelope in envelopes for trial_id in envelope.trial_support_ids
        ),
        evidence_envelopes=envelopes,
    )


def _entry_ids(entries) -> tuple[str, ...]:
    return tuple(entry.entry_id if hasattr(entry, "entry_id") else entry for entry in entries)


def test_builds_five_matched_branches_for_each_memory_baseline() -> None:
    from memcontam.experiment.phase12.branching import BranchSet, build_matched_branches

    fixture = _fixture()
    triplet = load_candidate_registry(REGISTRY_PATH).triplets[0]

    for baseline, prefix in fixture["baseline_prefixes"].items():
        prefix_checkpoint = serialize_checkpoint(NativeState.from_mapping(prefix["checkpoint"]))
        branches = build_matched_branches(
            prefix_checkpoint,
            triplet,
            RendererRegistry.native(),
            _context(baseline, tuple(prefix["checkpoint"]["entries"])),
        )
        assert isinstance(branches, BranchSet)

        assert branches.source_checkpoint == prefix_checkpoint
        clean_entries = tuple(prefix["checkpoint"]["entries"])
        assert _entry_ids(branches.clean.checkpoint.state.entries) == clean_entries
        assert _entry_ids(branches.correct.checkpoint.state.entries) == (
            *clean_entries,
            triplet.correct_twin.candidate_id,
        )
        assert _entry_ids(branches.irrelevant.checkpoint.state.entries) == (
            *clean_entries,
            triplet.irrelevant_control.candidate_id,
        )
        assert _entry_ids(branches.contam.checkpoint.state.entries) == (
            *clean_entries,
            triplet.false_candidate.candidate_id,
        )
        assert branches.filter.source_checkpoint == branches.contam.checkpoint
        assert _entry_ids(branches.filter.active.state.entries) == clean_entries
        assert _entry_ids(branches.filter.quarantine.state.entries) == (
            triplet.false_candidate.candidate_id,
        )
        assert {branch.source_checkpoint_id for branch in branches.materialized} == {
            prefix_checkpoint.identity.checkpoint_id
        }
        assert {intervention.arm for intervention in branches.interventions} == {
            "correct",
            "irrelevant",
            "contam",
            "filter",
        }
        assert branches.audit_labels[0].candidate_id == triplet.false_candidate.candidate_id
        assert branches.contam.checkpoint.identity.sha256 != prefix_checkpoint.identity.sha256
        assert not triplet.correct_twin.in_b_star
        assert not triplet.irrelevant_control.in_b_star


def test_rejects_unmatched_or_illegal_branch_construction() -> None:
    from memcontam.experiment.phase12.branching import (
        BranchConstructionError,
        BranchSet,
        build_matched_branches,
    )

    fixture = _fixture()
    baseline = "fh_bounded"
    prefix = fixture["baseline_prefixes"][baseline]["checkpoint"]
    checkpoint = serialize_checkpoint(NativeState.from_mapping(prefix))
    triplet = load_candidate_registry(REGISTRY_PATH).triplets[0]
    context = _context(baseline, tuple(prefix["entries"]))

    with pytest.raises(BranchConstructionError, match="CONTROL_IN_B_STAR"):
        build_matched_branches(
            checkpoint,
            replace(triplet, correct_twin=replace(triplet.correct_twin, in_b_star=True)),
            RendererRegistry.native(),
            context,
        )

    branches = build_matched_branches(checkpoint, triplet, RendererRegistry.native(), context)
    assert isinstance(branches, BranchSet)
    with pytest.raises(BranchConstructionError, match="PREFIX_IDENTITY_DRIFT"):
        replace(branches, correct=replace(branches.correct, source_checkpoint_id="other-prefix"))
    with pytest.raises(BranchConstructionError, match="FILTER_SOURCE_MISMATCH"):
        replace(
            branches, filter=replace(branches.filter, source_checkpoint=branches.clean.checkpoint)
        )


def test_nomem_produces_one_alias_record_without_materialized_branches() -> None:
    from memcontam.experiment.phase12.branching import NoMemAliasRecord, build_matched_branches
    from memcontam.memory.checkpoint_v3 import Phase12Checkpoint, Phase12CheckpointIdentity

    prefix = Phase12Checkpoint(
        identity=Phase12CheckpointIdentity("nomem", "no_memory", "invalid-but-unused"),
        state=NativeState("no_memory", (), {}),
        canonical_bytes=b"",
        canonical_sha256="",
    )
    alias = build_matched_branches(
        prefix,
        load_candidate_registry(REGISTRY_PATH).triplets[0],
        RendererRegistry.native(),
        AdmissionContext(),
    )
    assert isinstance(alias, NoMemAliasRecord)

    assert alias.underlying_execution_count == 1
    assert alias.display_alias_count == 5
    assert alias.materialized_branches == ()
