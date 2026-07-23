from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from memcontam.contamination.phase12.controls import (
    construct_correct_control,
    construct_irrelevant_control,
)
from memcontam.contamination.phase12.models import CandidateTriplet
from memcontam.contamination.phase12.renderers import RendererRegistry
from memcontam.memory.admission import AdmissionContext
from memcontam.memory.cards_v3 import MemoryCardEnvelopeV3
from memcontam.memory.checkpoint_v3 import (
    NativeEntry,
    Phase12Checkpoint,
    append_native_entry,
    deserialize_checkpoint,
)
from memcontam.memory.filtered_state import FilteredCheckpoint, partition_native_checkpoint


class BranchConstructionError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class BranchIntervention:
    arm: Literal["correct", "irrelevant", "contam", "filter"]
    candidate_triplet_id: str
    native_render_id: str


@dataclass(frozen=True)
class AuditLabelRecord:
    candidate_id: str
    triplet_id: str
    origin_class: Literal["protocol_injected"] = "protocol_injected"
    independent_of_outcomes: Literal[True] = True


@dataclass(frozen=True)
class MaterializedBranch:
    arm: Literal["clean", "correct", "irrelevant", "contam"]
    source_checkpoint_id: str
    checkpoint: Phase12Checkpoint
    inserted_entry_id: str | None = None
    intervention: BranchIntervention | None = None


@dataclass(frozen=True)
class BranchSet:
    source_checkpoint: Phase12Checkpoint
    clean: MaterializedBranch
    correct: MaterializedBranch
    irrelevant: MaterializedBranch
    contam: MaterializedBranch
    filter: FilteredCheckpoint
    audit_labels: tuple[AuditLabelRecord, ...]

    def __post_init__(self) -> None:
        branches = self.materialized
        source_id = self.source_checkpoint.identity.checkpoint_id
        if any(branch.source_checkpoint_id != source_id for branch in branches):
            raise BranchConstructionError("PREFIX_IDENTITY_DRIFT")
        if (
            self.clean.checkpoint != self.source_checkpoint
            or self.clean.inserted_entry_id is not None
        ):
            raise BranchConstructionError("CLEAN_CHECKPOINT_DRIFT")
        if self.filter.source_checkpoint != self.contam.checkpoint:
            raise BranchConstructionError("FILTER_SOURCE_MISMATCH")
        _validate_branch_entries(self.source_checkpoint, self.correct)
        _validate_branch_entries(self.source_checkpoint, self.irrelevant)
        _validate_branch_entries(self.source_checkpoint, self.contam)
        source_entries = _entry_ids(self.contam.checkpoint)
        partition_entries = _entry_ids(self.filter.active) + _entry_ids(self.filter.quarantine)
        if set(source_entries) != set(partition_entries) or len(partition_entries) != len(
            source_entries
        ):
            raise BranchConstructionError("FILTER_PARTITION_MISMATCH")

    @property
    def materialized(self) -> tuple[MaterializedBranch, ...]:
        return self.clean, self.correct, self.irrelevant, self.contam

    @property
    def interventions(self) -> tuple[BranchIntervention, ...]:
        interventions = tuple(
            branch.intervention for branch in (self.correct, self.irrelevant, self.contam)
        )
        if any(intervention is None for intervention in interventions):
            raise BranchConstructionError("MISSING_INTERVENTION")
        correct, irrelevant, contam = interventions
        assert correct is not None and irrelevant is not None and contam is not None
        return (
            correct,
            irrelevant,
            contam,
            BranchIntervention("filter", contam.candidate_triplet_id, contam.native_render_id),
        )


@dataclass(frozen=True)
class NoMemAliasRecord:
    underlying_execution_count: Literal[1] = 1
    display_alias_count: Literal[5] = 5
    materialized_branches: tuple[MaterializedBranch, ...] = ()

    def __post_init__(self) -> None:
        if self.materialized_branches:
            raise BranchConstructionError("NOMEM_BRANCH_FORBIDDEN")


def build_nomem_alias_record() -> NoMemAliasRecord:
    return NoMemAliasRecord()


def build_matched_branches(
    prefix: Phase12Checkpoint,
    triplet: CandidateTriplet,
    renderers: RendererRegistry,
    filter_policy: AdmissionContext,
) -> BranchSet | NoMemAliasRecord:
    if not isinstance(prefix, Phase12Checkpoint):
        raise BranchConstructionError("INVALID_PREFIX_CHECKPOINT")
    if prefix.state.baseline == "no_memory":
        return build_nomem_alias_record()
    try:
        deserialize_checkpoint(prefix)
    except Exception as error:
        code = getattr(error, "code", "INVALID_PREFIX_CHECKPOINT")
        raise BranchConstructionError(code) from error
    _validate_triplet(triplet)
    if not isinstance(renderers, RendererRegistry):
        raise BranchConstructionError("INVALID_RENDERER_REGISTRY")
    if not isinstance(filter_policy, AdmissionContext):
        raise BranchConstructionError("INVALID_FILTER_POLICY")

    baseline = prefix.state.baseline
    false_entry = _single_entry(renderers.render_false(baseline, triplet, prefix))
    correct_entry = _single_entry(construct_correct_control(baseline, triplet, prefix))
    irrelevant_entry = _single_entry(construct_irrelevant_control(baseline, triplet, prefix))
    contam = append_native_entry(prefix, false_entry)
    correct = append_native_entry(prefix, correct_entry)
    irrelevant = append_native_entry(prefix, irrelevant_entry)
    filtered = partition_native_checkpoint(contam, _filter_context(filter_policy, false_entry))
    source_id = prefix.identity.checkpoint_id

    return BranchSet(
        source_checkpoint=prefix,
        clean=MaterializedBranch("clean", source_id, prefix),
        correct=_intervened_branch("correct", source_id, correct, correct_entry, triplet),
        irrelevant=_intervened_branch(
            "irrelevant", source_id, irrelevant, irrelevant_entry, triplet
        ),
        contam=_intervened_branch("contam", source_id, contam, false_entry, triplet),
        filter=filtered,
        audit_labels=(AuditLabelRecord(false_entry.entry_id, triplet.triplet_id),),
    )


def _validate_triplet(triplet: CandidateTriplet) -> None:
    if not isinstance(triplet, CandidateTriplet):
        raise BranchConstructionError("INVALID_TRIPLET")
    candidates = (triplet.false_candidate, triplet.correct_twin, triplet.irrelevant_control)
    if len({candidate.candidate_id for candidate in candidates}) != 3:
        raise BranchConstructionError("ROOT_COUNT_MISMATCH")
    if triplet.false_candidate.role != "false" or any(
        candidate.role != role
        for candidate, role in (
            (triplet.correct_twin, "correct"),
            (triplet.irrelevant_control, "irrelevant"),
        )
    ):
        raise BranchConstructionError("INVALID_CONTROL_ROLE")
    if triplet.correct_twin.in_b_star or triplet.irrelevant_control.in_b_star:
        raise BranchConstructionError("CONTROL_IN_B_STAR")


def _single_entry(entry: object) -> NativeEntry:
    if not isinstance(entry, NativeEntry):
        raise BranchConstructionError("ROOT_COUNT_MISMATCH")
    return entry


def _filter_context(context: AdmissionContext, entry: NativeEntry) -> AdmissionContext:
    root = _injected_envelope(context, entry)
    return replace(context, evidence_envelopes=(*context.evidence_envelopes, root))


def _injected_envelope(context: AdmissionContext, entry: NativeEntry) -> MemoryCardEnvelopeV3:
    if context.evidence_envelopes:
        template = context.evidence_envelopes[0]
        return replace(
            template,
            entry_id=entry.entry_id,
            content=entry.content,
            content_hash=entry.content_hash,
            writer_id="protocol_injector",
            writer_event_id=f"intervention-{entry.entry_id}",
            writer_stage="protocol_inject",
            created_trial_id=None,
            source_trial_ids=(),
            trial_support_ids=(),
            memory_support_ids=(),
            direct_parent_ids=(),
            version_predecessor_id=None,
            order_key=len(context.evidence_envelopes) + 1,
            native_component=entry.native_component,
            semantic_kind=entry.semantic_kind,
        )
    return MemoryCardEnvelopeV3(
        entry_id=entry.entry_id,
        baseline="",
        semantic_kind=entry.semantic_kind,
        schema_version="memory_card_v3",
        writer_id="protocol_injector",
        writer_event_id=f"intervention-{entry.entry_id}",
        writer_stage="protocol_inject",
        created_trial_id=None,
        source_trial_ids=(),
        source_outcome=None,
        trial_support_ids=(),
        memory_support_ids=(),
        direct_parent_ids=(),
        version_predecessor_id=None,
        order_key=1,
        native_component=entry.native_component,
        content=entry.content,
        content_hash=entry.content_hash,
    )


def _intervened_branch(
    arm: Literal["correct", "irrelevant", "contam"],
    source_id: str,
    checkpoint: Phase12Checkpoint,
    entry: NativeEntry,
    triplet: CandidateTriplet,
) -> MaterializedBranch:
    return MaterializedBranch(
        arm=arm,
        source_checkpoint_id=source_id,
        checkpoint=checkpoint,
        inserted_entry_id=entry.entry_id,
        intervention=BranchIntervention(arm, triplet.triplet_id, entry.render_id or ""),
    )


def _validate_branch_entries(source: Phase12Checkpoint, branch: MaterializedBranch) -> None:
    source_entries = _entry_ids(source)
    entries = _entry_ids(branch.checkpoint)
    if (
        branch.inserted_entry_id is None
        or entries[:-1] != source_entries
        or entries[-1:] != (branch.inserted_entry_id,)
    ):
        raise BranchConstructionError("ROOT_COUNT_MISMATCH")


def _entry_ids(checkpoint: Phase12Checkpoint) -> tuple[str, ...]:
    return tuple(
        entry.entry_id if isinstance(entry, NativeEntry) else entry
        for entry in checkpoint.state.entries
    )
