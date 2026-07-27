from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Literal, Mapping, cast

from memcontam.baselines.bot_phase12 import BoTStateV3
from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.baselines.reflexion_phase12 import ReflexionStateV3
from memcontam.baselines.retrieval_rag_phase12 import RagFrozenStateV3
from memcontam.contamination.phase12.models import CandidateRegistry, CandidateTriplet
from memcontam.contamination.phase12.renderers import RendererRegistry
from memcontam.experiment.phase12.branching import BranchSet, build_matched_branches
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext
from memcontam.experiment.phase12.runtime_registry import LIVE_BASELINE_REGISTRY, RuntimeEntry
from memcontam.memory.admission import AdmissionContext
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, Phase12Checkpoint
from memcontam.memory.filtered_state import FilteredCheckpoint
from memcontam.memory.stores import MemoryEntry
from memcontam.rag.branch_index import EmbeddingProvider, build_branch_indices
from memcontam.rag.phase12_corpus import CleanCorpus, build_branch_corpora


Arm = Literal["clean", "contam", "filter"]
_ARMS: tuple[Arm, ...] = ("clean", "contam", "filter")
_INTERVENED_ARMS: tuple[Arm, ...] = ("contam", "filter")


class LiveBranchError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class LiveBranchEvent:
    kind: Literal["branch_constructed", "intervention_applied"]
    arm: Arm
    prefix_identity: str
    source_identity: str
    injected_root_id: str | None
    candidate_triplet_id: str | None
    native_render_id: str | None


@dataclass(frozen=True)
class LiveArmBranch:
    arm: Arm
    prefix_identity: str
    source_identity: str
    checkpoint: Phase12Checkpoint
    state: object
    root_count: Literal[0, 1]
    injected_root_id: str | None = None
    filter_state: FilteredCheckpoint | None = None


@dataclass(frozen=True)
class LiveThreeArmBranches:
    baseline: str
    model: str
    decoding: Mapping[str, object]
    arms: Mapping[Arm, LiveArmBranch]
    events: tuple[LiveBranchEvent, ...]

    def __post_init__(self) -> None:
        if set(self.arms) != set(_ARMS):
            raise LiveBranchError("THREE_ARM_SET_REQUIRED")
        clean, contam, filtered = (self.arms[arm] for arm in _ARMS)
        if (
            clean.root_count,
            contam.root_count,
            filtered.root_count,
        ) != (0, 1, 1):
            raise LiveBranchError("ROOT_COUNT_MISMATCH")
        if len({branch.prefix_identity for branch in self.arms.values()}) != 1:
            raise LiveBranchError("PREFIX_IDENTITY_DRIFT")
        if contam.source_identity != filtered.source_identity:
            raise LiveBranchError("FILTER_SOURCE_MISMATCH")
        if contam.injected_root_id != filtered.injected_root_id or not contam.injected_root_id:
            raise LiveBranchError("FILTER_SOURCE_MISMATCH")
        if filtered.filter_state is None:
            raise LiveBranchError("FILTER_STATE_REQUIRED")
        states = tuple(branch.state for branch in self.arms.values())
        if len({id(state) for state in states}) != len(states):
            raise LiveBranchError("CROSS_ARM_STATE_LEAKAGE")


def build_live_three_arm_branches(
    *,
    prefix: Phase12Checkpoint,
    context: Game24RuntimeContext,
    candidate_registry: CandidateRegistry,
    filter_policy: AdmissionContext,
    registry: Mapping[str, RuntimeEntry] = LIVE_BASELINE_REGISTRY,
    renderers: RendererRegistry | None = None,
) -> LiveThreeArmBranches:
    if context.task.task_name != "game24" or context.branch != "clean":
        raise LiveBranchError("CLEAN_GAME24_PREFIX_REQUIRED")
    baseline = prefix.state.baseline
    entry = registry.get(baseline)
    if entry is None or baseline == "nomem":
        raise LiveBranchError("MEMORY_PREFIX_REQUIRED")
    triplet = _registered_game24_triplet(candidate_registry)
    materialized = build_matched_branches(
        prefix,
        triplet,
        renderers or RendererRegistry.native(),
        filter_policy,
    )
    if not isinstance(materialized, BranchSet):
        raise LiveBranchError("MEMORY_PREFIX_REQUIRED")

    clean_state = deepcopy(entry.restore_state(prefix.state, context))
    root = _root_entry(materialized)
    clean_state, contam_state, filter_state = _branch_states(
        clean_state,
        context,
        triplet,
        root,
        materialized.filter,
        filter_policy,
    )
    prefix_identity = prefix.identity.checkpoint_id
    contaminated_identity = materialized.contam.checkpoint.identity.checkpoint_id
    root_id = root.entry_id
    render_id = root.render_id
    arms: dict[Arm, LiveArmBranch] = {
        "clean": LiveArmBranch(
            "clean",
            prefix_identity,
            prefix_identity,
            materialized.clean.checkpoint,
            clean_state,
            0,
        ),
        "contam": LiveArmBranch(
            "contam",
            prefix_identity,
            contaminated_identity,
            materialized.contam.checkpoint,
            contam_state,
            1,
            root_id,
        ),
        "filter": LiveArmBranch(
            "filter",
            prefix_identity,
            contaminated_identity,
            materialized.filter.active,
            filter_state,
            1,
            root_id,
            materialized.filter,
        ),
    }
    events = tuple(
        LiveBranchEvent("branch_constructed", arm, prefix_identity, arms[arm].source_identity, arms[arm].injected_root_id, None, None)
        for arm in _ARMS
    ) + tuple(
        LiveBranchEvent(
            "intervention_applied",
            arm,
            prefix_identity,
            contaminated_identity,
            root_id,
            triplet.triplet_id,
            render_id,
        )
        for arm in _INTERVENED_ARMS
    )
    return LiveThreeArmBranches(baseline, context.model, dict(context.decoding), arms, events)


def _registered_game24_triplet(registry: CandidateRegistry) -> CandidateTriplet:
    triplets = tuple(triplet for triplet in registry.triplets if triplet.task == "game24")
    if (
        len(triplets) != 1
        or triplets[0].false_candidate.role != "false"
        or not triplets[0].false_candidate.in_b_star
    ):
        raise LiveBranchError("REGISTERED_GAME24_ROOT_REQUIRED")
    return triplets[0]


def _root_entry(branches: BranchSet) -> NativeEntry:
    root = branches.contam.checkpoint.state.entries[-1]
    if not isinstance(root, NativeEntry):
        raise LiveBranchError("ROOT_COUNT_MISMATCH")
    return root


def _branch_states(
    clean_state: object,
    context: Game24RuntimeContext,
    triplet: CandidateTriplet,
    root: NativeEntry,
    filtered: FilteredCheckpoint,
    filter_policy: AdmissionContext,
) -> tuple[object, object, object]:
    if isinstance(clean_state, RagFrozenStateV3):
        return _rag_branch_states(clean_state, context, triplet, filtered)
    if isinstance(clean_state, NativeState):
        return (
            clean_state,
            NativeState(
                clean_state.baseline,
                (*clean_state.entries, root),
                clean_state.native_state,
                clean_state.schema_version,
            ),
            filtered.active.state,
        )

    contam_state = deepcopy(clean_state)
    filter_state = deepcopy(clean_state)
    _inject_root(contam_state, root)
    _inject_root(filter_state, root)
    _attach_filter_state(filter_state, filtered, filter_policy)
    return clean_state, contam_state, filter_state


def _inject_root(state: object, root: NativeEntry) -> None:
    if isinstance(state, FullHistoryStateV3):
        state.records.append(
            MemoryEntry(
                entry_id=root.entry_id,
                content=root.content,
                memory_type=root.semantic_kind,
                clean_or_contaminated="injected",
            )
        )
        state.injected_root_id = root.entry_id
        return
    if isinstance(state, BoTStateV3):
        state.entries.append(root)
        return
    if isinstance(state, ReflexionStateV3):
        state.reflections.append(root)
        state.injected_root_id = root.entry_id
        return
    raise LiveBranchError("LIVE_BRANCH_STATE_UNSUPPORTED")


def _attach_filter_state(
    state: object, filtered: FilteredCheckpoint, filter_policy: AdmissionContext
) -> None:
    if isinstance(state, (FullHistoryStateV3, BoTStateV3, ReflexionStateV3)):
        state.filter_state = filtered
        state.admission_context = filter_policy
        return
    raise LiveBranchError("LIVE_FILTER_STATE_UNSUPPORTED")


def _rag_branch_states(
    clean_state: RagFrozenStateV3,
    context: Game24RuntimeContext,
    triplet: CandidateTriplet,
    filtered: FilteredCheckpoint,
) -> tuple[RagFrozenStateV3, RagFrozenStateV3, RagFrozenStateV3]:
    if (
        clean_state.branch != "clean"
        or clean_state.corpus is None
        or clean_state.index is None
        or context.embedding_provider is None
    ):
        raise LiveBranchError("RAG_CLEAN_STATE_REQUIRED")
    corpus_id, marker, _ = clean_state.corpus.serialization_id.partition("|clean|")
    if not marker or not corpus_id:
        raise LiveBranchError("RAG_CLEAN_STATE_REQUIRED")
    corpora = build_branch_corpora(
        CleanCorpus(corpus_id=corpus_id, documents=clean_state.corpus.documents), triplet
    )
    filter_corpus = corpora.branches["filter"]
    active_entry_ids = {
        decision.entry_id for decision in filtered.decisions if decision.state == "active"
    }
    corpora = replace(
        corpora,
        branches={
            **corpora.branches,
            "filter": replace(
                filter_corpus,
                active_document_ids=tuple(
                    document.document_id
                    for document in filter_corpus.documents
                    if document.document_id in active_entry_ids
                ),
            ),
        },
    )
    indices = build_branch_indices(
        corpora, cast(EmbeddingProvider, context.embedding_provider), filter_policy=None
    )
    return (
        RagFrozenStateV3("clean", corpora.branches["clean"], indices.branches["clean"]),
        RagFrozenStateV3("contam", corpora.branches["contam"], indices.branches["contam"]),
        RagFrozenStateV3("filter", corpora.branches["filter"], indices.branches["filter"]),
    )


__all__ = [
    "LiveArmBranch",
    "LiveBranchError",
    "LiveBranchEvent",
    "LiveThreeArmBranches",
    "build_live_three_arm_branches",
]
