from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


SupportStatus = Literal["supported", "not_applicable", "unavailable", "approximate"]
LineageEvidenceStatus = Literal["exact", "approximate", "unavailable"]
_GENERIC_FAILURES = frozenset({"error", "generic_error", "incorrect_answer", "wrong_answer"})
_RETENTION_BASELINES = frozenset({"full_history", "fh_bounded", "reflexion_style"})


class SequentialOutcomeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class EvidenceOutcome:
    status: SupportStatus
    value: bool | None
    path: tuple[str, ...] = ()


@dataclass(frozen=True)
class SequentialTrialOutcome:
    trial_id: str
    order_key: int
    failure_class: str | None
    generic_recurrence: bool
    same_root_exact_lineage_recurrence: bool
    propagation: EvidenceOutcome
    root_storage_persistence: bool | None
    descendant_storage_persistence: bool | None
    root_prompt_visibility: bool | None
    descendant_prompt_visibility: bool | None
    retention: EvidenceOutcome
    eviction: EvidenceOutcome


@dataclass(frozen=True)
class SequentialOutcomeSet:
    window: int
    trials: tuple[SequentialTrialOutcome, ...]
    summary: Mapping[str, Any]


def compute_sequential_outcomes(
    trials: Sequence[Mapping[str, Any]],
    memory_events: Sequence[Mapping[str, Any]],
    lineage: Mapping[str, Mapping[str, Any]],
    window: int | None,
) -> SequentialOutcomeSet:
    """Compute finite-window outcomes using only final-context and recorded-lineage evidence."""
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise SequentialOutcomeError("WINDOW_REQUIRED")
    nodes = _nodes(lineage)
    ordered_trials = _ordered_trials(trials)
    events_by_trial = _events_by_trial(memory_events, nodes)
    known_roots = _known_roots(nodes)
    outcomes: list[SequentialTrialOutcome] = []

    for trial in ordered_trials:
        trial_id = _required_id(trial, "trial_id")
        order_key = _order_key(trial)
        baseline = _required_id(trial, "baseline")
        final_ids = _ids(trial.get("final_context_entry_ids", ()))
        final_roots, final_status = _final_roots(final_ids, nodes)
        failure_class = _failure_class(trial.get("failure_class"))
        generic = _generic_recurrence(outcomes, order_key, failure_class, window)
        exact = generic and bool(
            final_status == "exact"
            and final_roots
            and any(
                _trial_roots(previous, ordered_trials, nodes) & final_roots
                for previous in outcomes
                if order_key - previous.order_key <= window
                and previous.failure_class == failure_class
            )
        )
        event = events_by_trial.get(trial_id)
        propagation = _propagation(final_ids, final_roots, final_status, event, nodes)
        retention = _retention(trial, baseline)
        eviction = _eviction(trial, baseline)
        storage_ids = _storage_ids(event)
        root_storage = _persisted(known_roots, storage_ids)
        descendant_storage = _persisted(set(nodes) - known_roots, storage_ids)
        root_visible, descendant_visible = _visibility(
            final_ids, known_roots, nodes, retention, trial
        )
        outcomes.append(
            SequentialTrialOutcome(
                trial_id=trial_id,
                order_key=order_key,
                failure_class=failure_class,
                generic_recurrence=generic,
                same_root_exact_lineage_recurrence=exact,
                propagation=propagation,
                root_storage_persistence=root_storage,
                descendant_storage_persistence=descendant_storage,
                root_prompt_visibility=root_visible,
                descendant_prompt_visibility=descendant_visible,
                retention=retention,
                eviction=eviction,
            )
        )
    return SequentialOutcomeSet(window, tuple(outcomes), _summary(outcomes, window))


def _nodes(lineage: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    if not isinstance(lineage, Mapping):
        raise SequentialOutcomeError("FABRICATED_LINEAGE")
    nodes: dict[str, Mapping[str, Any]] = {}
    for entry_id, node in lineage.items():
        if not isinstance(entry_id, str) or not entry_id or not isinstance(node, Mapping):
            raise SequentialOutcomeError("FABRICATED_LINEAGE")
        parents = _ids(node.get("direct_parent_ids", ()))
        predecessor = node.get("version_predecessor_id")
        if predecessor is not None and (not isinstance(predecessor, str) or not predecessor):
            raise SequentialOutcomeError("FABRICATED_LINEAGE")
        if node.get("lineage_status") == "exact" and any(
            parent not in lineage for parent in parents
        ):
            raise SequentialOutcomeError("FABRICATED_LINEAGE")
        if isinstance(predecessor, str) and predecessor not in lineage:
            raise SequentialOutcomeError("FABRICATED_LINEAGE")
        nodes[entry_id] = node
    for entry_id, node in nodes.items():
        if node.get("lineage_status") != "exact":
            continue
        for root_id in _ids(node.get("injected_root_ids", ())):
            root = nodes.get(root_id)
            if root is None or root.get("lineage_status") != "exact":
                raise SequentialOutcomeError("FABRICATED_LINEAGE")
    return nodes


def _ordered_trials(trials: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not isinstance(trials, Sequence) or isinstance(trials, (str, bytes)):
        raise SequentialOutcomeError("WINDOW_REQUIRED")
    ordered = sorted(trials, key=_order_key)
    ids = [_required_id(trial, "trial_id") for trial in ordered]
    orders = [_order_key(trial) for trial in ordered]
    if len(ids) != len(set(ids)) or len(orders) != len(set(orders)):
        raise SequentialOutcomeError("WINDOW_REQUIRED")
    return ordered


def _events_by_trial(
    events: Sequence[Mapping[str, Any]], nodes: Mapping[str, Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for event in events:
        if not isinstance(event, Mapping):
            raise SequentialOutcomeError("FABRICATED_LINEAGE")
        trial_id = _required_id(event, "trial_id")
        if trial_id in result:
            raise SequentialOutcomeError("FABRICATED_LINEAGE")
        changed_ids = _ids(event.get("new_entry_ids", ())) + _ids(
            event.get("updated_entry_ids", ())
        )
        if any(entry_id not in nodes for entry_id in changed_ids):
            raise SequentialOutcomeError("FABRICATED_LINEAGE")
        result[trial_id] = event
    return result


def _final_roots(
    entry_ids: tuple[str, ...], nodes: Mapping[str, Mapping[str, Any]]
) -> tuple[set[str], LineageEvidenceStatus]:
    roots: set[str] = set()
    status: LineageEvidenceStatus = "exact"
    for entry_id in entry_ids:
        node = nodes.get(entry_id)
        if node is None:
            continue
        node_status = node.get("lineage_status")
        if node_status == "approximate":
            status = "approximate"
            continue
        if node_status != "exact":
            status = "unavailable"
            continue
        roots.update(_ids(node.get("injected_root_ids", ())))
    return roots, status


def _generic_recurrence(
    outcomes: Sequence[SequentialTrialOutcome],
    order_key: int,
    failure_class: str | None,
    window: int,
) -> bool:
    return failure_class is not None and any(
        previous.failure_class == failure_class and order_key - previous.order_key <= window
        for previous in outcomes
    )


def _trial_roots(
    outcome: SequentialTrialOutcome,
    trials: Sequence[Mapping[str, Any]],
    nodes: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    trial = next(trial for trial in trials if trial["trial_id"] == outcome.trial_id)
    roots, status = _final_roots(_ids(trial.get("final_context_entry_ids", ())), nodes)
    return roots if status == "exact" else set()


def _propagation(
    final_ids: tuple[str, ...],
    final_roots: set[str],
    final_status: LineageEvidenceStatus,
    event: Mapping[str, Any] | None,
    nodes: Mapping[str, Mapping[str, Any]],
) -> EvidenceOutcome:
    if final_status == "approximate":
        return EvidenceOutcome("approximate", None)
    if not final_ids or not final_roots:
        return EvidenceOutcome("unavailable", None)
    if event is None:
        return EvidenceOutcome("supported", False)
    changed_ids = _ids(event.get("new_entry_ids", ())) + _ids(event.get("updated_entry_ids", ()))
    approximate = False
    for entry_id in changed_ids:
        node = nodes[entry_id]
        if node.get("lineage_status") == "approximate":
            approximate = True
            continue
        path = _recorded_path(entry_id, set(final_ids), nodes)
        if path and set(_ids(node.get("injected_root_ids", ())) or path[:1]) & final_roots:
            return EvidenceOutcome("supported", True, path)
    return (
        EvidenceOutcome("approximate", None) if approximate else EvidenceOutcome("supported", False)
    )


def _recorded_path(
    entry_id: str, final_ids: set[str], nodes: Mapping[str, Mapping[str, Any]]
) -> tuple[str, ...]:
    def visit(current: str, seen: set[str]) -> tuple[str, ...]:
        if current in seen:
            raise SequentialOutcomeError("FABRICATED_LINEAGE")
        if current in final_ids:
            return (current,)
        node = nodes[current]
        if node.get("lineage_status") != "exact":
            return ()
        parents = _ids(node.get("direct_parent_ids", ()))
        predecessor = node.get("version_predecessor_id")
        if isinstance(predecessor, str):
            parents += (predecessor,)
        for parent in parents:
            if parent not in nodes:
                raise SequentialOutcomeError("FABRICATED_LINEAGE")
            path = visit(parent, seen | {current})
            if path:
                return (*path, current)
        return ()

    return visit(entry_id, set())


def _retention(trial: Mapping[str, Any], baseline: str) -> EvidenceOutcome:
    retention = trial.get("retention")
    if retention is None:
        return (
            EvidenceOutcome("unavailable", None)
            if baseline in _RETENTION_BASELINES
            else EvidenceOutcome("not_applicable", None)
        )
    if baseline not in _RETENTION_BASELINES or not isinstance(retention, Mapping):
        raise SequentialOutcomeError("UNSUPPORTED_BASELINE_OPERATION")
    persisted = retention.get(
        "root_persists_in_store", retention.get("injected_root_persists_in_store")
    )
    if not isinstance(persisted, bool):
        raise SequentialOutcomeError("UNSUPPORTED_BASELINE_OPERATION")
    return EvidenceOutcome("supported", persisted)


def _eviction(trial: Mapping[str, Any], baseline: str) -> EvidenceOutcome:
    telemetry = trial.get("retention")
    events = trial.get("eviction_events")
    if telemetry is not None and baseline not in _RETENTION_BASELINES:
        raise SequentialOutcomeError("UNSUPPORTED_BASELINE_OPERATION")
    if events is not None and baseline != "reflexion_style":
        raise SequentialOutcomeError("UNSUPPORTED_BASELINE_OPERATION")
    if isinstance(telemetry, Mapping):
        evicted = telemetry.get("first_eviction_trial_id")
        if evicted is not None and not isinstance(evicted, str):
            raise SequentialOutcomeError("UNSUPPORTED_BASELINE_OPERATION")
        return EvidenceOutcome("supported", bool(evicted))
    if events is not None:
        if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
            raise SequentialOutcomeError("UNSUPPORTED_BASELINE_OPERATION")
        for event in events:
            if not isinstance(event, Mapping) or not isinstance(event.get("entry_id"), str):
                raise SequentialOutcomeError("UNSUPPORTED_BASELINE_OPERATION")
        return EvidenceOutcome("supported", bool(events))
    return (
        EvidenceOutcome("unavailable", None)
        if baseline in _RETENTION_BASELINES
        else EvidenceOutcome("not_applicable", None)
    )


def _visibility(
    final_ids: tuple[str, ...],
    roots: set[str],
    nodes: Mapping[str, Mapping[str, Any]],
    retention: EvidenceOutcome,
    trial: Mapping[str, Any],
) -> tuple[bool | None, bool | None]:
    telemetry = trial.get("retention")
    if isinstance(telemetry, Mapping) and isinstance(telemetry.get("root_visible_in_prompt"), bool):
        root_visible = telemetry["root_visible_in_prompt"]
    elif "final_context_entry_ids" in trial:
        root_visible = bool(set(final_ids) & roots)
    else:
        root_visible = None
    descendants = set(nodes) - roots
    descendant_visible = (
        bool(set(final_ids) & descendants) if "final_context_entry_ids" in trial else None
    )
    return root_visible, descendant_visible


def _known_roots(nodes: Mapping[str, Mapping[str, Any]]) -> set[str]:
    return {
        root_id
        for node in nodes.values()
        if node.get("lineage_status") == "exact"
        for root_id in _ids(node.get("injected_root_ids", ()))
    }


def _storage_ids(event: Mapping[str, Any] | None) -> set[str] | None:
    if event is None or "after_entry_ids" not in event:
        return None
    return set(_ids(event.get("after_entry_ids", ())))


def _persisted(entry_ids: set[str], storage_ids: set[str] | None) -> bool | None:
    return None if storage_ids is None else bool(entry_ids & storage_ids)


def _failure_class(value: Any) -> str | None:
    return value if isinstance(value, str) and value and value not in _GENERIC_FAILURES else None


def _summary(outcomes: Sequence[SequentialTrialOutcome], window: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "collapse_like": {"label": "exploratory", "window": window},
        "generic_recurrence_count": sum(outcome.generic_recurrence for outcome in outcomes),
        "same_root_exact_lineage_recurrence_count": sum(
            outcome.same_root_exact_lineage_recurrence for outcome in outcomes
        ),
        "propagation_count": sum(outcome.propagation.value is True for outcome in outcomes),
    }
    for outcome in outcomes:
        order = outcome.order_key
        if outcome.generic_recurrence:
            summary[f"generic_recurrence_at_{order}"] = 1
        if outcome.same_root_exact_lineage_recurrence:
            summary[f"exact_lineage_recurrence_at_{order}"] = 1
        if outcome.propagation.value is True:
            summary[f"propagation_transition_{order}"] = 1
        if outcome.root_storage_persistence:
            summary[f"root_persistence_at_{order}"] = 1
    return summary


def _required_id(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise SequentialOutcomeError("WINDOW_REQUIRED")
    return value


def _order_key(record: Mapping[str, Any]) -> int:
    value = record.get("order_key", record.get("trial"))
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SequentialOutcomeError("WINDOW_REQUIRED")
    return value


def _ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SequentialOutcomeError("FABRICATED_LINEAGE")
    ids = tuple(value)
    if any(not isinstance(entry_id, str) or not entry_id for entry_id in ids) or len(ids) != len(
        set(ids)
    ):
        raise SequentialOutcomeError("FABRICATED_LINEAGE")
    return ids


__all__ = [
    "EvidenceOutcome",
    "SequentialOutcomeError",
    "SequentialOutcomeSet",
    "SequentialTrialOutcome",
    "compute_sequential_outcomes",
]
