from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from memcontam.experiment.phase12.contracts import BaselineConditionSpec
from memcontam.memory.checkpoint_v3 import Phase12Checkpoint


class EligibilityError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class MaturityDecision:
    condition_id: str
    baseline_family: str
    checkpoint_id: str
    checkpoint_index: int
    horizon: int
    eligible: bool
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.condition_id or not self.baseline_family or not self.checkpoint_id:
            raise EligibilityError("INVALID_MATURITY_DECISION")
        if type(self.checkpoint_index) is not int or self.checkpoint_index < 1:
            raise EligibilityError("INVALID_MATURITY_DECISION")
        _validate_horizon(self.horizon)
        if self.eligible and self.reason_codes:
            raise EligibilityError("INVALID_MATURITY_DECISION")


def evaluate_maturity(
    condition: BaselineConditionSpec, checkpoint: Phase12Checkpoint, horizon: int
) -> MaturityDecision:
    _validate_horizon(horizon)
    index = _checkpoint_index(checkpoint)
    state = checkpoint.state.native_state
    baseline = checkpoint.state.baseline
    reason_codes = _baseline_reasons(condition, baseline, state, horizon)
    return MaturityDecision(
        condition_id=condition.condition_id,
        baseline_family=condition.baseline_family,
        checkpoint_id=checkpoint.identity.checkpoint_id,
        checkpoint_index=index,
        horizon=horizon,
        eligible=not reason_codes,
        reason_codes=tuple(reason_codes),
    )


def evaluate_optional_dc_maturity(
    checkpoint: Phase12Checkpoint, horizon: int, *, condition_id: str = "dc_optional"
) -> MaturityDecision:
    _validate_horizon(horizon)
    index = _checkpoint_index(checkpoint)
    state = checkpoint.state.native_state
    reasons = _horizon_reasons(state, horizon)
    if checkpoint.state.baseline != "dynamic_cheatsheet_rs_optional":
        reasons.append("CHECKPOINT_BASELINE_MISMATCH")
    if not _nonempty_sequence(state.get("archive")):
        reasons.append("DC_ARCHIVE_UNAVAILABLE")
    if not isinstance(state.get("strategy"), str) or not state["strategy"].strip():
        reasons.append("DC_STRATEGY_UNAVAILABLE")
    return MaturityDecision(
        condition_id=condition_id,
        baseline_family="dc",
        checkpoint_id=checkpoint.identity.checkpoint_id,
        checkpoint_index=index,
        horizon=horizon,
        eligible=not reasons,
        reason_codes=tuple(reasons),
    )


def _baseline_reasons(
    condition: BaselineConditionSpec,
    checkpoint_baseline: str,
    state: Mapping[str, Any],
    horizon: int,
) -> list[str]:
    reasons = _horizon_reasons(state, horizon)
    family = condition.baseline_family
    if checkpoint_baseline not in _baseline_names(family):
        reasons.append("CHECKPOINT_BASELINE_MISMATCH")
        return reasons
    if family == "full_history":
        if not _nonempty_sequence(state.get("records")):
            reasons.append("FH_RECORDS_UNAVAILABLE")
        if condition.fh_mode == "exact" and state.get("full_fit") is not True:
            reasons.append("FH_FULL_FIT_UNPROVEN")
    elif family == "rag":
        if state.get("read_only") is not True:
            reasons.append("RAG_NOT_READ_ONLY")
        if not _nonempty_string(state.get("corpus_id")) or not _nonempty_string(
            state.get("index_id")
        ):
            reasons.append("RAG_INDEX_UNAVAILABLE")
    elif family == "bot":
        if not _at_least(state.get("templates"), 2):
            reasons.append("BOT_COMPETITORS_UNAVAILABLE")
    elif family == "reflexion":
        reflections = state.get("reflections")
        capacity = state.get("active_capacity", 3)
        if not isinstance(reflections, (list, tuple)):
            reasons.append("REFLEXION_STATE_UNAVAILABLE")
        elif type(capacity) is not int or capacity < 1 or len(reflections) >= capacity:
            reasons.append("REFLEXION_CAPACITY_UNAVAILABLE")
    else:
        reasons.append("UNSUPPORTED_BASELINE")
    return reasons


def _baseline_names(family: str) -> set[str]:
    return {
        "full_history": {"full_history", "fh_bounded"},
        "rag": {"rag_frozen", "retrieval_rag"},
        "bot": {"bot_style"},
        "reflexion": {"reflexion_style"},
    }.get(family, set())


def _horizon_reasons(state: Mapping[str, Any], horizon: int) -> list[str]:
    registered_horizon = state.get("maturity_horizon")
    if registered_horizon is None:
        return []
    if type(registered_horizon) is not int or registered_horizon < horizon:
        return ["INSUFFICIENT_HORIZON"]
    return []


def _checkpoint_index(checkpoint: Phase12Checkpoint) -> int:
    index = checkpoint.state.native_state.get("checkpoint_index")
    if type(index) is int and index > 0:
        return index
    match = re.search(r"-t(\d+)$", checkpoint.identity.checkpoint_id)
    if match is not None:
        return int(match.group(1))
    raise EligibilityError("CHECKPOINT_INDEX_MISSING")


def _at_least(value: object, count: int) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= count


def _nonempty_sequence(value: object) -> bool:
    return _at_least(value, 1)


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_horizon(horizon: int) -> None:
    if type(horizon) is not int or horizon < 1:
        raise EligibilityError("INVALID_HORIZON")
