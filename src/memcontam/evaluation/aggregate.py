from __future__ import annotations

import json
from collections.abc import Sequence
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import ValidationError

from memcontam.logging.schema import (
    CallEvent,
    FailureEvent,
    FilterEvent,
    MemoryEvent,
    MethodCall,
    RunMetadata,
    TrialLog,
)


NOT_COMPUTED = "not_computed"


def _load_trials(trials_path: Path) -> list[TrialLog]:
    if not trials_path.exists():
        raise SystemExit(f"trials.jsonl not found: {trials_path}")

    trials: list[TrialLog] = []
    with trials_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"malformed trials.jsonl: {trials_path}") from exc
            try:
                trials.append(TrialLog.model_validate(row))
            except ValidationError as exc:
                raise SystemExit(f"invalid trial log row in {trials_path}") from exc
    return trials


def _is_strict_run_dir(run_dir: Path) -> bool:
    return (run_dir / "run.json").exists()


def _load_jsonl_stream(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"stream file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"malformed JSONL: {path}") from exc
    return rows


def _load_strict_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.exists():
        raise SystemExit(f"run.json not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"malformed run.json: {path}") from exc
    return manifest


def _load_strict_calls(run_dir: Path) -> list[CallEvent]:
    rows = _load_jsonl_stream(run_dir / "calls.jsonl")
    events: list[CallEvent] = []
    for row in rows:
        try:
            events.append(CallEvent.model_validate(row))
        except Exception as exc:
            raise SystemExit(f"invalid call event in {run_dir / 'calls.jsonl'}") from exc
    return events


def _load_strict_failures(run_dir: Path) -> list[FailureEvent]:
    rows = _load_jsonl_stream(run_dir / "failures.jsonl")
    events: list[FailureEvent] = []
    for row in rows:
        try:
            events.append(FailureEvent.model_validate(row))
        except Exception as exc:
            raise SystemExit(f"invalid failure event in {run_dir / 'failures.jsonl'}") from exc
    return events


def _load_strict_filters(run_dir: Path) -> list[FilterEvent]:
    rows = _load_jsonl_stream(run_dir / "filter_events.jsonl")
    events: list[FilterEvent] = []
    for row in rows:
        try:
            events.append(FilterEvent.model_validate(row))
        except Exception as exc:
            raise SystemExit(f"invalid filter event in {run_dir / 'filter_events.jsonl'}") from exc
    return events


def _load_strict_memory_events(run_dir: Path) -> list[MemoryEvent]:
    rows = _load_jsonl_stream(run_dir / "memory_events.jsonl")
    events: list[MemoryEvent] = []
    for row in rows:
        try:
            events.append(MemoryEvent.model_validate(row))
        except Exception as exc:
            raise SystemExit(f"invalid memory event in {run_dir / 'memory_events.jsonl'}") from exc
    return events


def _validate_strict_consistency(
    run_metadata: RunMetadata,
    trials: list[TrialLog],
    calls: list[CallEvent],
    failures: list[FailureEvent],
    filters: list[FilterEvent],
    memory_events: list[MemoryEvent],
    expected_stage: str | None,
) -> None:
    # ponytail: one-pass cross-stream validation; O(n) scans are fine for QA-scale runs
    all_trial_ids = {trial.trial_id for trial in trials}
    failed_trial_ids = {trial.trial_id for trial in trials if trial.status == "failed"}
    if len(all_trial_ids) != len(trials):
        raise SystemExit("duplicate trial_id in trials.jsonl")

    def _require_homogeneous(field: str, values: set[str]) -> None:
        if len(values) > 1:
            raise SystemExit(f"mixed {field}: {sorted(values)}")

    _require_homogeneous(
        "run_metadata_id",
        {run_metadata.run_metadata_id}
        | {event.run_metadata_id for event in calls}
        | {event.run_metadata_id for event in failures}
        | {event.run_metadata_id for event in filters}
        | {event.run_metadata_id for event in memory_events}
        | {trial.run_metadata_id for trial in trials if trial.run_metadata_id},
    )
    _require_homogeneous(
        "run_id",
        {run_metadata.run_id}
        | {event.run_id for event in calls}
        | {event.run_id for event in failures}
        | {event.run_id for event in filters}
        | {event.run_id for event in memory_events}
        | {trial.run_id for trial in trials},
    )
    _require_homogeneous(
        "stage",
        {run_metadata.stage}
        | {event.stage for event in calls}
        | {event.stage for event in failures}
        | {event.stage for event in filters}
        | {event.stage for event in memory_events}
        | {trial.stage for trial in trials if trial.stage},
    )
    _require_homogeneous(
        "schema_version",
        {run_metadata.schema_version}
        | {trial.schema_version for trial in trials},
    )

    if expected_stage is not None and run_metadata.stage != expected_stage:
        raise SystemExit(
            f"stage mismatch: expected {expected_stage}, found {run_metadata.stage}"
        )

    seen_event_seq: set[int] = set()
    for stream_name, events in (
        ("calls", calls),
        ("failures", failures),
        ("filters", filters),
        ("memory_events", memory_events),
        ("trials", trials),
    ):
        for event in events:
            seq = event.event_seq
            if seq is None:
                raise SystemExit(f"missing event_seq in {stream_name}")
            if seq in seen_event_seq:
                raise SystemExit(f"duplicate event_seq {seq} across streams")
            seen_event_seq.add(seq)

    seen_ids: dict[str, set[str]] = defaultdict(set)
    for event in calls:
        if event.call_id in seen_ids["call_id"]:
            raise SystemExit(f"duplicate call_id: {event.call_id}")
        seen_ids["call_id"].add(event.call_id)
    for event in failures:
        if event.failure_id in seen_ids["failure_id"]:
            raise SystemExit(f"duplicate failure_id: {event.failure_id}")
        seen_ids["failure_id"].add(event.failure_id)
    for event in filters:
        if event.filter_id in seen_ids["filter_id"]:
            raise SystemExit(f"duplicate filter_id: {event.filter_id}")
        seen_ids["filter_id"].add(event.filter_id)
    for event in memory_events:
        if event.memory_id in seen_ids["memory_id"]:
            raise SystemExit(f"duplicate memory_id: {event.memory_id}")
        seen_ids["memory_id"].add(event.memory_id)

    calls_by_id = {call.call_id: call for call in calls}
    filters_by_trial: dict[str, list[FilterEvent]] = defaultdict(list)
    memory_events_by_trial: dict[str, list[MemoryEvent]] = defaultdict(list)
    for event in filters:
        filters_by_trial[event.trial_id].append(event)
    for event in memory_events:
        memory_events_by_trial[event.trial_id].append(event)

    for trial in trials:
        if trial.status == "failed":
            continue
        if not trial.answer_call_id:
            raise SystemExit(f"missing answer_call_id for trial: {trial.trial_id}")
        if trial.answer_call_id not in calls_by_id:
            raise SystemExit(f"answer_call_id {trial.answer_call_id} not found in calls.jsonl")

    for event in calls + failures + filters + memory_events:
        if event.trial_id not in all_trial_ids:
            raise SystemExit(f"event references unknown trial: {event.trial_id}")

    for trial in trials:
        trial_filters = filters_by_trial.get(trial.trial_id, [])
        if trial.filter_decision is None and trial_filters:
            raise SystemExit(f"filter event without trial filter_decision: {trial.trial_id}")
        if trial.filter_decision is not None:
            actions = [event.action for event in trial_filters]
            for action in {"apply", "outcome"}:
                count = actions.count(action)
                if count > 1:
                    raise SystemExit(f"duplicate filter {action} for trial: {trial.trial_id}")
                if trial.status == "succeeded" and count == 0:
                    raise SystemExit(f"missing filter {action} for trial: {trial.trial_id}")
            for event in trial_filters:
                if event.baseline != trial.baseline or event.arm != trial.arm:
                    raise SystemExit(f"filter join mismatch for trial: {trial.trial_id}")
                if event.action == "outcome" and trial.verifier_result is not None:
                    expected_verdict = str(trial.verifier_result.is_correct).lower()
                    if event.verdict != expected_verdict:
                        raise SystemExit(f"filter outcome verdict mismatch for trial: {trial.trial_id}")

        trial_memory_events = memory_events_by_trial.get(trial.trial_id, [])
        needs_memory_event = (
            trial.memory_write_event is not None
            and trial.baseline not in {"no_memory", "retrieval_rag"}
            and bool(trial.memory_write_event.get("type") or trial.memory_write_event.get("event_type"))
        )
        if needs_memory_event and not trial_memory_events:
            raise SystemExit(f"missing memory event for trial: {trial.trial_id}")
        if not needs_memory_event and trial_memory_events:
            raise SystemExit(f"memory event without trial memory_write_event: {trial.trial_id}")
        for event in trial_memory_events:
            if event.baseline != trial.baseline:
                raise SystemExit(f"memory event baseline mismatch for trial: {trial.trial_id}")
            if event.source_trial_id is not None and event.source_trial_id != trial.trial_id:
                raise SystemExit(f"memory event source_trial_id mismatch for trial: {trial.trial_id}")

    for failure in failures:
        if failure.trial_id in failed_trial_ids:
            continue
        if not any(trial.failure_id == failure.failure_id for trial in trials):
            raise SystemExit(f"failure {failure.failure_id} not linked to a trial")

    if run_metadata.stage in {"pilot", "main", "benchmark"}:
        for trial in trials:
            if trial.arm == "clean":
                continue
            if trial.contamination_exposure.status != "supported":
                raise SystemExit(
                    f"unsupported exposure in {run_metadata.stage} trial: {trial.trial_id}"
                )


def _rate(numerator: int, denominator: int) -> float | str:
    return NOT_COMPUTED if denominator == 0 else numerator / denominator


def _is_evaluable_uptake_label(label: str | None) -> bool:
    return label in {"uptake_detected", "no_uptake_detected"}


def _is_evaluable_repeated_failure_label(label: str | None) -> bool:
    return label in {"first_failure", "repeated_failure"}


def _descendant_link_present(trial: TrialLog) -> bool:
    if not trial.memory_write_event:
        return False
    parent_trial_id = trial.memory_write_event.get("parent_trial_id")
    source_entry_ids = trial.memory_write_event.get("source_entry_ids")
    return bool(parent_trial_id) and bool(source_entry_ids)


def _aggregate_call_metrics(calls: Sequence[MethodCall | CallEvent]) -> dict[str, Any]:
    if not calls:
        return {
            "method_call_count": NOT_COMPUTED,
            "method_call_error_count": NOT_COMPUTED,
            "prompt_token_total": NOT_COMPUTED,
            "completion_token_total": NOT_COMPUTED,
            "total_token_total": NOT_COMPUTED,
            "latency_ms_total": NOT_COMPUTED,
            "stage_histogram": NOT_COMPUTED,
        }

    histogram: dict[str, int] = defaultdict(int)
    error_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    latency_total = 0
    for call in calls:
        stage = call.stage if isinstance(call, MethodCall) else call.method_stage
        histogram[stage] += 1
        if call.error_type is not None:
            error_count += 1
        usage = call.token_usage or {}
        prompt_tokens += int(usage.get("prompt_tokens", 0))
        completion_tokens += int(usage.get("completion_tokens", 0))
        total_tokens += int(usage.get("total_tokens", 0))
        if call.latency_ms is not None:
            latency_total += call.latency_ms

    return {
        "method_call_count": len(calls),
        "method_call_error_count": error_count,
        "prompt_token_total": prompt_tokens,
        "completion_token_total": completion_tokens,
        "total_token_total": total_tokens,
        "latency_ms_total": latency_total,
        "stage_histogram": dict(histogram),
    }


def _method_call_metrics(
    trials: list[TrialLog],
    strict_calls: list[CallEvent] | None = None,
) -> dict[str, Any]:
    if strict_calls is not None:
        return _aggregate_call_metrics(strict_calls)
    return _aggregate_call_metrics([call for trial in trials for call in trial.method_calls])


def _bot_lineage_metrics(trials: list[TrialLog]) -> dict[str, int | str]:
    events = [t.memory_write_event for t in trials if t.memory_write_event]
    if not events:
        return {
            "bot_update_accepted_count": NOT_COMPUTED,
            "bot_update_rejected_count": NOT_COMPUTED,
            "bot_update_incomplete_count": NOT_COMPUTED,
            "bot_update_reused_count": NOT_COMPUTED,
        }

    accepted = 0
    rejected = 0
    reused = 0
    incomplete = 0
    for event in events:
        status = event.get("status")
        if status in {"accepted", "rejected", "reused", "incomplete"}:
            if status == "accepted":
                accepted += 1
            elif status == "rejected":
                rejected += 1
            elif status == "reused":
                reused += 1
            else:
                incomplete += 1
            continue

        flags = {key: event.get(key) for key in ("accepted", "rejected", "reused", "incomplete") if key in event}
        if flags and sum(bool(value) for value in flags.values()) == 1:
            if flags.get("accepted"):
                accepted += 1
            elif flags.get("rejected"):
                rejected += 1
            elif flags.get("reused"):
                reused += 1
            elif flags.get("incomplete"):
                incomplete += 1
            continue

        return {
            "bot_update_accepted_count": NOT_COMPUTED,
            "bot_update_rejected_count": NOT_COMPUTED,
            "bot_update_incomplete_count": NOT_COMPUTED,
            "bot_update_reused_count": NOT_COMPUTED,
        }

    return {
        "bot_update_accepted_count": accepted,
        "bot_update_rejected_count": rejected,
        "bot_update_incomplete_count": incomplete,
        "bot_update_reused_count": reused,
    }


def _validate_trial_calls_against_strict_calls(
    trials: list[TrialLog],
    strict_calls: list[CallEvent],
) -> None:
    calls_by_trial: dict[str, list[CallEvent]] = defaultdict(list)
    for call in strict_calls:
        calls_by_trial[call.trial_id].append(call)

    for trial in trials:
        nested = trial.method_calls
        if not nested:
            continue
        strict = calls_by_trial.get(trial.trial_id, [])
        if len(nested) != len(strict):
            raise SystemExit(
                f"call count mismatch for {trial.trial_id}: "
                f"method_calls={len(nested)}, calls.jsonl={len(strict)}"
            )
        for nested_call, strict_call in zip(nested, strict):
            if nested_call.call_id != strict_call.call_id:
                raise SystemExit(
                    f"call_id mismatch for {trial.trial_id}: "
                    f"{nested_call.call_id} != {strict_call.call_id}"
                )


def _metric_group(
    trials: list[TrialLog],
    strict_calls: list[CallEvent] | None = None,
    failures: list[FailureEvent] | None = None,
) -> dict[str, Any]:
    n_trials = len(trials)
    succeeded_trials = [trial for trial in trials if trial.status != "failed"]
    failed_trials = [trial for trial in trials if trial.status == "failed"]
    n_failed = len(failed_trials)
    n_evaluable = len(succeeded_trials)

    verified_success_count = sum(
        1 for trial in succeeded_trials if trial.verifier_result and trial.verifier_result.is_correct
    )
    contaminated_condition_count = sum(1 for trial in trials if trial.arm != "clean")
    controlled_exposure_count = sum(
        1
        for trial in trials
        if trial.contamination_exposure.status == "supported" and trial.contamination_exposure.is_exposed
    )
    filter_drop_count = sum(
        int(trial.filter_decision.get("dropped", 0)) if trial.filter_decision else 0 for trial in trials
    )
    token_usage_total = sum(int(trial.token_usage.get("total_tokens", 0)) for trial in trials)

    latencies = [trial.latency_ms for trial in trials if trial.latency_ms is not None]
    if latencies:
        latency_ms_min: float | int | str = min(latencies)
        latency_ms_mean: float | int | str = mean(latencies)
        latency_ms_max: float | int | str = max(latencies)
    else:
        latency_ms_min = latency_ms_mean = latency_ms_max = NOT_COMPUTED

    uptake_evaluable = [
        trial for trial in trials if _is_evaluable_uptake_label(trial.bad_memory_uptake_label)
    ]
    repeated_failure_evaluable = [
        trial for trial in trials if _is_evaluable_repeated_failure_label(trial.repeated_failure_label)
    ]
    descendant_evaluable = [trial for trial in trials if _descendant_link_present(trial)]

    uptake_count: int | str = (
        sum(1 for trial in uptake_evaluable if trial.bad_memory_uptake_label == "uptake_detected")
        if uptake_evaluable
        else NOT_COMPUTED
    )
    repeated_failure_count: int | str = (
        sum(1 for trial in repeated_failure_evaluable if trial.repeated_failure_label == "repeated_failure")
        if repeated_failure_evaluable
        else NOT_COMPUTED
    )
    descendant_count: int | str = (
        len(descendant_evaluable) if descendant_evaluable else NOT_COMPUTED
    )

    verified_success_rate = _rate(verified_success_count, n_evaluable)
    contaminated_condition_rate = _rate(contaminated_condition_count, n_trials)
    controlled_exposure_rate = _rate(controlled_exposure_count, n_trials)
    trial_level_uptake_rate = _rate(
        sum(1 for trial in uptake_evaluable if trial.bad_memory_uptake_label == "uptake_detected"),
        len(uptake_evaluable),
    )
    repeated_failure_rate = _rate(
        sum(1 for trial in repeated_failure_evaluable if trial.repeated_failure_label == "repeated_failure"),
        len(repeated_failure_evaluable),
    )
    descendant_rate = _rate(len(descendant_evaluable), len(descendant_evaluable))

    failure_origin_histogram: dict[str, int] = defaultdict(int)
    trial_failure_map: dict[str, FailureEvent] = {}
    if failures is not None:
        trial_failure_map = {failure.trial_id: failure for failure in failures}
    for trial in failed_trials:
        failure = trial_failure_map.get(trial.trial_id)
        origin = failure.origin if failure else "runner"
        failure_origin_histogram[origin] += 1

    metrics = {
        "n_trials": n_trials,
        "n_failed": n_failed,
        "n_evaluable": n_evaluable,
        "verified_success_count": verified_success_count,
        "verified_success_rate": verified_success_rate,
        "contaminated_condition_count": contaminated_condition_count,
        "contaminated_condition_rate": contaminated_condition_rate,
        "controlled_exposure_count": controlled_exposure_count,
        "controlled_exposure_rate": controlled_exposure_rate,
        "contamination_exposure_rate": controlled_exposure_rate,
        "trial_level_uptake_count": uptake_count,
        "trial_level_uptake_rate": trial_level_uptake_rate,
        "contamination_uptake_rate": trial_level_uptake_rate,
        "contaminated_descendant_count": descendant_count,
        "contaminated_descendant_rate": descendant_rate,
        "filter_drop_count": filter_drop_count,
        "token_usage_total": token_usage_total,
        "latency_ms_min": latency_ms_min,
        "latency_ms_mean": latency_ms_mean,
        "latency_ms_max": latency_ms_max,
        "repeated_failure_count": repeated_failure_count,
        "repeated_failure_rate": repeated_failure_rate,
        "failure_origin_histogram": dict(failure_origin_histogram),
    }
    metrics.update(_method_call_metrics(trials, strict_calls=strict_calls))
    metrics.update(_bot_lineage_metrics(trials))
    return metrics


def _paired_degradation(trials_by_combo: dict[tuple[str, str, str], dict[str, list[TrialLog]]]) -> dict[tuple[str, str, str], float | str]:
    degradation_by_combo: dict[tuple[str, str, str], float | str] = {}
    for combo, arm_groups in trials_by_combo.items():
        clean_trials = arm_groups.get("clean", [])
        contaminated_trials = arm_groups.get("contaminated", [])
        clean_sample_ids = {trial.sample_id for trial in clean_trials}
        contaminated_sample_ids = {trial.sample_id for trial in contaminated_trials}
        paired_sample_ids = clean_sample_ids & contaminated_sample_ids
        if not paired_sample_ids:
            degradation_by_combo[combo] = NOT_COMPUTED
            continue

        paired_clean = [
            trial
            for trial in clean_trials
            if trial.sample_id in paired_sample_ids and trial.status != "failed" and trial.verifier_result
        ]
        paired_contaminated = [
            trial
            for trial in contaminated_trials
            if trial.sample_id in paired_sample_ids and trial.status != "failed" and trial.verifier_result
        ]
        if not paired_clean or not paired_contaminated:
            degradation_by_combo[combo] = NOT_COMPUTED
            continue

        clean_rate = sum(
            1
            for trial in paired_clean
            if trial.verifier_result is not None and trial.verifier_result.is_correct
        ) / len(paired_clean)
        contaminated_rate = (
            sum(
                1
                for trial in paired_contaminated
                if trial.verifier_result is not None and trial.verifier_result.is_correct
            )
            / len(paired_contaminated)
        )
        degradation_by_combo[combo] = clean_rate - contaminated_rate
    return degradation_by_combo


def aggregate_run(
    run_dir: Path,
    stage: str | None = None,
    *,
    allow_legacy: bool = False,
) -> dict:
    trials_path = run_dir / "trials.jsonl"
    if not trials_path.exists():
        raise SystemExit(f"trials.jsonl not found: {trials_path}")

    strict_mode = _is_strict_run_dir(run_dir)
    if strict_mode:
        if stage is None:
            raise SystemExit("strict run requires --stage")
        manifest = _load_strict_manifest(run_dir)
        status = manifest.get("status")
        if status not in {"completed", "failed"}:
            raise SystemExit(f"strict run has unexpected status: {status}")
        run_metadata = RunMetadata.model_validate(manifest.get("run_metadata"))
        calls = _load_strict_calls(run_dir)
        failures = _load_strict_failures(run_dir)
        filters = _load_strict_filters(run_dir)
        memory_events = _load_strict_memory_events(run_dir)
        trials = _load_trials(trials_path)
        _validate_strict_consistency(
            run_metadata, trials, calls, failures, filters, memory_events, stage
        )
        _validate_trial_calls_against_strict_calls(trials, calls)
        strict_calls_for_metrics = calls
        failures_for_metrics = failures
        run_status = status
    else:
        if stage is not None:
            raise SystemExit("--stage requires a strict run with run.json")
        if not allow_legacy:
            raise SystemExit(
                "legacy run directory detected; use --allow-legacy to aggregate legacy trials.jsonl-only runs"
            )
        trials = _load_trials(trials_path)
        strict_calls_for_metrics = None
        failures_for_metrics = None
        run_status = "legacy"

    grouped: dict[tuple[str, str, str, str], list[TrialLog]] = defaultdict(list)
    combos: dict[tuple[str, str, str], dict[str, list[TrialLog]]] = defaultdict(lambda: defaultdict(list))
    for trial in trials:
        if trial.metadata.get("exclude_from_aggregate") or trial.metadata.get("phase") == "warmup":
            continue
        key = (trial.task_name, trial.baseline, trial.arm, trial.backbone)
        grouped[key].append(trial)
        combos[(trial.task_name, trial.baseline, trial.backbone)][trial.arm].append(trial)

    degradation_by_combo = _paired_degradation(combos)
    groups: list[dict[str, Any]] = []
    for key in sorted(grouped):
        task_name, baseline, arm, backbone = key
        combo = (task_name, baseline, backbone)
        group: dict[str, Any] = {
            "task_name": task_name,
            "baseline": baseline,
            "arm": arm,
            "backbone": backbone,
        }
        group.update(
            _metric_group(
                grouped[key],
                strict_calls=(
                    [
                        call
                        for call in strict_calls_for_metrics
                        if call.trial_id in {trial.trial_id for trial in grouped[key]}
                    ]
                    if strict_calls_for_metrics is not None
                    else None
                ),
                failures=failures_for_metrics,
            )
        )
        group["vanilla_to_contamination_degradation_rate"] = degradation_by_combo.get(combo, NOT_COMPUTED)
        groups.append(group)

    return {
        "run_dir": str(run_dir),
        "status": run_status,
        "n_trials": len(trials),
        "groups": groups,
    }
