from __future__ import annotations

from dataclasses import dataclass

from memcontam.contamination.phase12.models import CandidateTriplet, CandidateVariant
from memcontam.memory.checkpoint_v3 import CheckpointError, NativeEntry, Phase12Checkpoint, deserialize_checkpoint
from memcontam.memory.serializer_registry import SerializerRegistry


class RendererError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def render_false(beta: str, triplet: CandidateTriplet, checkpoint: Phase12Checkpoint) -> NativeEntry:
    return _render(beta, triplet.false_candidate, checkpoint)


def render_correct(beta: str, triplet: CandidateTriplet, checkpoint: Phase12Checkpoint) -> NativeEntry:
    return _render(beta, triplet.correct_twin, checkpoint)


def render_irrelevant(beta: str, triplet: CandidateTriplet, checkpoint: Phase12Checkpoint) -> NativeEntry:
    return _render(beta, triplet.irrelevant_control, checkpoint)


def _render(beta: str, candidate: CandidateVariant, checkpoint: Phase12Checkpoint) -> NativeEntry:
    if beta == "no_memory":
        raise RendererError("NOMEM_INJECTION_FORBIDDEN")
    try:
        state = deserialize_checkpoint(checkpoint)
    except CheckpointError as error:
        raise RendererError(error.code) from error
    if state.baseline != beta:
        raise RendererError("CHECKPOINT_BASELINE_MISMATCH")
    if candidate.candidate_id in {
        entry.entry_id if isinstance(entry, NativeEntry) else entry for entry in state.entries
    }:
        raise RendererError("DUPLICATE_ROOT")
    try:
        schema = SerializerRegistry.native().schema_for(beta)
    except CheckpointError as error:
        raise RendererError(error.code) from error
    return NativeEntry(
        entry_id=candidate.candidate_id,
        semantic_kind=schema.semantic_kind,
        schema_version="phase12_native_entry_v1",
        native_component=schema.native_component,
        content=candidate.content,
        content_hash=candidate.content_hash,
        direct_parent_ids=(),
        render_id=candidate.render_id,
    )


@dataclass(frozen=True)
class RendererRegistry:
    @classmethod
    def native(cls) -> RendererRegistry:
        return cls()

    def render_false(
        self, beta: str, triplet: CandidateTriplet, checkpoint: Phase12Checkpoint
    ) -> NativeEntry:
        return render_false(beta, triplet, checkpoint)

    def render_correct(
        self, beta: str, triplet: CandidateTriplet, checkpoint: Phase12Checkpoint
    ) -> NativeEntry:
        return render_correct(beta, triplet, checkpoint)

    def render_irrelevant(
        self, beta: str, triplet: CandidateTriplet, checkpoint: Phase12Checkpoint
    ) -> NativeEntry:
        return render_irrelevant(beta, triplet, checkpoint)
