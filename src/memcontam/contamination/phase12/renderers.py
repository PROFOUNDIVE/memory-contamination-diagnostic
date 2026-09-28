from __future__ import annotations

from dataclasses import dataclass

from memcontam.contamination.phase12.models import (
    CandidateRegistry,
    CandidateTriplet,
    CandidateVariant,
    canonical_content_hash,
)
from memcontam.contamination.phase13_legacy_dc_rs import (
    LegacyDcRsRegistry,
    load_legacy_dc_rs_registry,
)
from memcontam.memory.checkpoint_v3 import (
    CheckpointError,
    NativeEntry,
    Phase12Checkpoint,
    deserialize_checkpoint,
)
from memcontam.memory.serializer_registry import NativeSchema, SerializerRegistry


_NATIVE_CONTENT_TEMPLATES = {
    "full_history": "User: Please demonstrate the procedure.\nAssistant: {candidate}",
    "fh_bounded": "User: Please demonstrate the procedure.\nAssistant: {candidate}",
    "bot_style": "Thought template:\n{candidate}",
    "reflexion_style": "Reflection: I should correct my approach.\nLesson: {candidate}",
}


class RendererError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def render_false(
    beta: str, triplet: CandidateTriplet, checkpoint: Phase12Checkpoint
) -> NativeEntry:
    return _render(beta, triplet.false_candidate, checkpoint)


def render_correct(
    beta: str, triplet: CandidateTriplet, checkpoint: Phase12Checkpoint
) -> NativeEntry:
    return _render(beta, triplet.correct_twin, checkpoint)


def render_irrelevant(
    beta: str, triplet: CandidateTriplet, checkpoint: Phase12Checkpoint
) -> NativeEntry:
    return _render(beta, triplet.irrelevant_control, checkpoint)


def _render(beta: str, candidate: CandidateVariant, checkpoint: Phase12Checkpoint) -> NativeEntry:
    schema = _validate_checkpoint(beta, candidate, checkpoint)
    template = _NATIVE_CONTENT_TEMPLATES.get(beta)
    if beta == "dc_rs":
        raise RendererError("DC_RS_INTERVENTION_REGISTRY_REQUIRED")
    content = candidate.content if template is None else template.format(candidate=candidate.content)
    content_hash = canonical_content_hash(content)
    render_id = (
        candidate.render_id
        if content == candidate.content
        else f"{candidate.render_id}-{beta}-{content_hash[:16]}"
    )
    return NativeEntry(
        entry_id=candidate.candidate_id,
        semantic_kind=schema.semantic_kind,
        schema_version="phase12_native_entry_v1",
        native_component=schema.native_component,
        content=content,
        content_hash=content_hash,
        direct_parent_ids=(),
        render_id=render_id,
        retrieval_description=candidate.content if beta == "bot_style" else None,
        template_body=content if beta == "bot_style" else None,
        category="procedure-based" if beta == "bot_style" else None,
    )


def _validate_checkpoint(
    beta: str,
    candidate: CandidateVariant,
    checkpoint: Phase12Checkpoint,
) -> NativeSchema:
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
    return schema


@dataclass(frozen=True)
class RendererRegistry:
    dc_rs: LegacyDcRsRegistry | None = None

    @classmethod
    def native(cls) -> RendererRegistry:
        return cls()

    @classmethod
    def governed(
        cls,
        raw: bytes,
        candidates: CandidateRegistry,
        candidate_registry_sha256: str,
    ) -> RendererRegistry:
        return cls(load_legacy_dc_rs_registry(raw, candidates, candidate_registry_sha256))

    def render_false(
        self, beta: str, triplet: CandidateTriplet, checkpoint: Phase12Checkpoint
    ) -> NativeEntry:
        return self._render(beta, triplet, triplet.false_candidate, checkpoint)

    def render_correct(
        self, beta: str, triplet: CandidateTriplet, checkpoint: Phase12Checkpoint
    ) -> NativeEntry:
        return self._render(beta, triplet, triplet.correct_twin, checkpoint)

    def render_irrelevant(
        self, beta: str, triplet: CandidateTriplet, checkpoint: Phase12Checkpoint
    ) -> NativeEntry:
        return self._render(beta, triplet, triplet.irrelevant_control, checkpoint)

    def _render(
        self,
        beta: str,
        triplet: CandidateTriplet,
        candidate: CandidateVariant,
        checkpoint: Phase12Checkpoint,
    ) -> NativeEntry:
        if beta != "dc_rs":
            return _render(beta, candidate, checkpoint)
        _validate_checkpoint(beta, candidate, checkpoint)
        if self.dc_rs is None:
            raise RendererError("DC_RS_INTERVENTION_REGISTRY_REQUIRED")
        return self.dc_rs.render(triplet.task, candidate)
