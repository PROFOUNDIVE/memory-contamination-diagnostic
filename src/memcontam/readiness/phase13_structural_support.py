from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from memcontam.experiment.phase12.native_state_facts import inspect_native_state
from memcontam.memory.checkpoint_v3 import Phase12Checkpoint, serialize_checkpoint
from memcontam.readiness.phase13_calibration_v2_authority import (
    AuthorityError,
    reject_forbidden_fields,
)

Baseline = Literal["fh_bounded", "rag_frozen", "bot_style", "reflexion_style"]
BASELINES: Final[tuple[Baseline, ...]] = (
    "fh_bounded",
    "rag_frozen",
    "bot_style",
    "reflexion_style",
)
PAIRS: Final[tuple[tuple[str, Baseline, Baseline], ...]] = (
    ("P01", "fh_bounded", "rag_frozen"),
    ("P02", "fh_bounded", "bot_style"),
    ("P03", "fh_bounded", "reflexion_style"),
    ("P04", "rag_frozen", "bot_style"),
    ("P05", "rag_frozen", "reflexion_style"),
    ("P06", "bot_style", "reflexion_style"),
)
_FUTURE_TOKENS: Final = frozenset({"future", "outcome", "verifier", "eligibility", "rate"})
_FUTURE_FIELDS: Final = frozenset({"analysis_window", "horizon", "task"})


class StructuralSupportError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class OrderedTrial(_StrictModel):
    trial_index: Annotated[int, Field(gt=0)]
    sample_id: Annotated[str, Field(min_length=1)]


class ResourceFact(_StrictModel):
    baseline: Baseline
    checkpoint_trial_index: Annotated[int, Field(gt=0)]
    checkpoint_serializable: bool
    suffix_executable: bool
    route_capacity_available: bool


class ProspectiveSelectorInput(_StrictModel):
    stream_id: Annotated[str, Field(min_length=1)]
    ordered_trials: tuple[OrderedTrial, ...]
    minimum_clean_prefix_length: Annotated[int, Field(gt=0)]
    suffix_length: Annotated[int, Field(gt=0)]
    resources: tuple[ResourceFact, ...]

    @field_validator("ordered_trials", "resources", mode="before")
    @classmethod
    def _tuples(cls, value: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
        return tuple(value)


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    baseline: Baseline
    selected_trial_index: int


@dataclass(frozen=True, slots=True)
class ProspectiveSelection:
    stream_id: str
    selected_trial_index: int
    checkpoint_trial_index: int
    suffix_trial_indices: tuple[int, ...]
    decisions: tuple[SelectionDecision, ...]


@dataclass(frozen=True, slots=True)
class CheckpointFact:
    baseline: str
    trial_index: int
    checkpoint: Phase12Checkpoint
    expected_sha256: str


@dataclass(frozen=True, slots=True)
class ReadinessRow:
    baseline: Baseline
    checkpoint_id: str
    checkpoint_sha256: str
    ready: bool
    reason_code: str | None
    richness: int


@dataclass(frozen=True, slots=True)
class BaselineSupportRow:
    baseline: Baseline
    supported: bool
    population_kind: Literal["baseline_local"] = "baseline_local"


@dataclass(frozen=True, slots=True)
class PairSupportRow:
    pair_id: str
    left_baseline: Baseline
    right_baseline: Baseline
    supported: bool
    population_kind: Literal["exact_pair"] = "exact_pair"


@dataclass(frozen=True, slots=True)
class GlobalSupportRow:
    supported: bool
    route_gating: Literal[False] = False
    population_kind: Literal["strict_global_sensitivity"] = "strict_global_sensitivity"


@dataclass(frozen=True, slots=True)
class NoMemRow:
    baseline: Literal["nomem"] = "nomem"
    persistent_memory: Literal[False] = False
    gate_population: Literal[False] = False


@dataclass(frozen=True, slots=True)
class StructuralSupportReport:
    selection: ProspectiveSelection
    readiness: tuple[ReadinessRow, ...]
    baseline_local: tuple[BaselineSupportRow, ...]
    exact_pairs: tuple[PairSupportRow, ...]
    strict_global: GlobalSupportRow
    nomem: NoMemRow


def parse_prospective_selector_input(payload: dict[str, object]) -> ProspectiveSelectorInput:
    try:
        reject_forbidden_fields(payload)
    except AuthorityError as error:
        raise StructuralSupportError("SELECTOR_FIELD_FORBIDDEN") from error
    if _contains_future_field(payload):
        raise StructuralSupportError("SELECTOR_FIELD_FORBIDDEN")
    try:
        return ProspectiveSelectorInput.model_validate(payload)
    except ValidationError as error:
        raise StructuralSupportError("SELECTOR_INPUT_INVALID") from error


def select_prospective_checkpoint(selector: ProspectiveSelectorInput) -> ProspectiveSelection:
    indices = tuple(row.trial_index for row in selector.ordered_trials)
    if len(indices) != len(set(indices)):
        raise StructuralSupportError("DUPLICATE_TRIAL")
    if indices != tuple(range(1, len(indices) + 1)):
        raise StructuralSupportError("SOURCE_ORDER_INVALID")
    candidate = selector.minimum_clean_prefix_length + 1
    suffix = tuple(range(candidate, candidate + selector.suffix_length))
    if not suffix or suffix[-1] > indices[-1]:
        raise StructuralSupportError("SUFFIX_UNAVAILABLE")
    resources = {row.baseline: row for row in selector.resources}
    if len(resources) != len(selector.resources):
        raise StructuralSupportError("DUPLICATE_BASELINE_DECISION")
    if tuple(row.baseline for row in selector.resources) != BASELINES:
        raise StructuralSupportError("BASELINE_DECISIONS_INVALID")
    for baseline in BASELINES:
        row = resources[baseline]
        if row.checkpoint_trial_index != candidate - 1:
            raise StructuralSupportError("CHECKPOINT_TRIAL_INVALID")
        if not row.checkpoint_serializable:
            raise StructuralSupportError("CHECKPOINT_SERIALIZATION_UNAVAILABLE")
        if not row.suffix_executable or not row.route_capacity_available:
            raise StructuralSupportError("SUFFIX_UNAVAILABLE")
    return ProspectiveSelection(
        selector.stream_id,
        candidate,
        candidate - 1,
        suffix,
        tuple(SelectionDecision(baseline, candidate) for baseline in BASELINES),
    )


def evaluate_structural_support(
    selection: ProspectiveSelection,
    checkpoints: tuple[CheckpointFact, ...],
) -> StructuralSupportReport:
    if any(row.baseline == "nomem" for row in checkpoints):
        raise StructuralSupportError("NOMEM_SUPPORT_FORBIDDEN")
    keys = tuple((row.baseline, row.trial_index) for row in checkpoints)
    if len(keys) != len(set(keys)):
        raise StructuralSupportError("DUPLICATE_CHECKPOINT")
    by_baseline = {row.baseline: row for row in checkpoints}
    if tuple(by_baseline) != BASELINES:
        raise StructuralSupportError("CHECKPOINT_PANEL_INVALID")
    readiness = tuple(
        _readiness(selection, baseline, by_baseline[baseline]) for baseline in BASELINES
    )
    support = {row.baseline: row.ready for row in readiness}
    local = tuple(BaselineSupportRow(baseline, support[baseline]) for baseline in BASELINES)
    pairs = tuple(
        PairSupportRow(pair_id, left, right, support[left] and support[right])
        for pair_id, left, right in PAIRS
    )
    return StructuralSupportReport(
        selection,
        readiness,
        local,
        pairs,
        GlobalSupportRow(all(support.values())),
        NoMemRow(),
    )


def _readiness(
    selection: ProspectiveSelection,
    baseline: Baseline,
    fact: CheckpointFact,
) -> ReadinessRow:
    if fact.trial_index != selection.checkpoint_trial_index:
        raise StructuralSupportError("CHECKPOINT_TRIAL_INVALID")
    serialized = serialize_checkpoint(fact.checkpoint.state)
    if (
        serialized.canonical_bytes != fact.checkpoint.canonical_bytes
        or serialized.canonical_sha256 != fact.checkpoint.canonical_sha256
        or serialized.canonical_sha256 != fact.expected_sha256
        or fact.checkpoint.identity.baseline != fact.baseline
    ):
        raise StructuralSupportError("CHECKPOINT_HASH_MISMATCH")
    facts = inspect_native_state(serialized)
    ready, reason, richness = _native_readiness(facts, len(serialized.state.entries))
    return ReadinessRow(
        baseline,
        serialized.identity.checkpoint_id,
        serialized.canonical_sha256,
        ready,
        reason,
        richness,
    )


def _native_readiness(facts, entry_count: int) -> tuple[bool, str | None, int]:
    match facts.baseline:
        case "fh_bounded":
            ready = facts.history_count is not None
            return ready, None if ready else "FH_NATIVE_STATE_INVALID", facts.history_count or 0
        case "rag_frozen":
            ready = all((facts.corpus_id, facts.index_id, facts.read_only, facts.branch == "clean"))
            return ready, None if ready else "RAG_NATIVE_STATE_INVALID", entry_count
        case "bot_style":
            ready = facts.template_count is not None and facts.active_capacity_present
            return ready, None if ready else "BOT_NATIVE_STATE_INVALID", facts.template_count or 0
        case "reflexion_style":
            ready = facts.reflection_count is not None and facts.active_capacity_present
            return ready, None if ready else "REFLEXION_NATIVE_STATE_INVALID", facts.reflection_count or 0
        case _:
            raise StructuralSupportError("BASELINE_UNSUPPORTED")


def _contains_future_field(value: object) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            tokens = frozenset(part for part in key.lower().replace("-", "_").split("_") if part)
            if tokens & _FUTURE_TOKENS or key.lower() in _FUTURE_FIELDS:
                return True
            if _contains_future_field(nested):
                return True
    elif isinstance(value, list):
        return any(_contains_future_field(item) for item in value)
    return False


__all__ = (
    "CheckpointFact",
    "ProspectiveSelectorInput",
    "StructuralSupportError",
    "StructuralSupportReport",
    "evaluate_structural_support",
    "parse_prospective_selector_input",
    "select_prospective_checkpoint",
)
