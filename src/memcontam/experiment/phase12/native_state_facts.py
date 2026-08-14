from __future__ import annotations

from dataclasses import dataclass

from memcontam.memory.checkpoint_v3 import Phase12Checkpoint


@dataclass(frozen=True, slots=True)
class NativeStateFacts:
    baseline: str
    checkpoint_id: str
    checkpoint_index: int | None
    maturity_horizon: int | None
    maturity_horizon_present: bool
    maturity_horizon_valid: bool
    history_count: int | None
    full_fit: bool | None
    first_eviction_trial_id: str | None
    first_eviction_present: bool
    corpus_id: str | None
    index_id: str | None
    read_only: bool | None
    branch: str | None
    template_count: int | None
    template_ids: tuple[str, ...]
    clean_competitor_ids: tuple[str, ...] | None
    active_capacity: int | None
    active_capacity_present: bool
    reflection_count: int | None


def inspect_native_state(checkpoint: Phase12Checkpoint) -> NativeStateFacts:
    state = checkpoint.state.native_state
    checkpoint_index = state.get("checkpoint_index")
    maturity_horizon = state.get("maturity_horizon")
    records = state.get("records")
    templates = state.get("templates")
    competitors = state.get("clean_competitor_ids")
    reflections = state.get("reflections")
    capacity = state.get("active_capacity")
    return NativeStateFacts(
        baseline=checkpoint.state.baseline,
        checkpoint_id=checkpoint.identity.checkpoint_id,
        checkpoint_index=checkpoint_index if type(checkpoint_index) is int else None,
        maturity_horizon=maturity_horizon if type(maturity_horizon) is int else None,
        maturity_horizon_present="maturity_horizon" in state,
        maturity_horizon_valid=maturity_horizon is None or type(maturity_horizon) is int,
        history_count=len(records) if isinstance(records, (list, tuple)) else None,
        full_fit=True if state.get("full_fit") is True else None,
        first_eviction_trial_id=(
            state.get("first_eviction_trial_id")
            if isinstance(state.get("first_eviction_trial_id"), str)
            else None
        ),
        first_eviction_present=state.get("first_eviction_trial_id") is not None,
        corpus_id=state.get("corpus_id") if isinstance(state.get("corpus_id"), str) else None,
        index_id=state.get("index_id") if isinstance(state.get("index_id"), str) else None,
        read_only=True if state.get("read_only") is True else None,
        branch=state.get("branch") if isinstance(state.get("branch"), str) else None,
        template_count=len(templates) if isinstance(templates, (list, tuple)) else None,
        template_ids=_entry_ids(templates),
        clean_competitor_ids=_string_sequence(competitors),
        active_capacity=capacity if type(capacity) is int else None,
        active_capacity_present="active_capacity" in state,
        reflection_count=len(reflections) if isinstance(reflections, (list, tuple)) else None,
    )


def _entry_ids(value) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(
        item if isinstance(item, str) else item["id"]
        for item in value
        if isinstance(item, str)
        or (isinstance(item, dict) and isinstance(item.get("id"), str))
    )


def _string_sequence(value) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        return None
    return tuple(value)


__all__ = ["NativeStateFacts", "inspect_native_state"]
