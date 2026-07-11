from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import ValidationError

from memcontam.logging.schema import TrialLog


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


def _method_call_metrics(trials: list[TrialLog]) -> dict[str, Any]:
    calls = [call for trial in trials for call in trial.method_calls]
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
        histogram[call.stage] += 1
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


def _metric_group(trials: list[TrialLog]) -> dict[str, Any]:
    n_trials = len(trials)
    verified_success_count = sum(1 for trial in trials if trial.verifier_result.is_correct)
    contaminated_condition_count = sum(1 for trial in trials if trial.arm != "clean")
    controlled_exposure_count = sum(1 for trial in trials if trial.contamination_exposure.is_exposed)
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

    verified_success_rate = _rate(verified_success_count, n_trials)
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

    metrics = {
        "n_trials": n_trials,
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
    }
    metrics.update(_method_call_metrics(trials))
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

        paired_clean = [trial for trial in clean_trials if trial.sample_id in paired_sample_ids]
        paired_contaminated = [trial for trial in contaminated_trials if trial.sample_id in paired_sample_ids]
        if not paired_clean or not paired_contaminated:
            degradation_by_combo[combo] = NOT_COMPUTED
            continue

        clean_rate = sum(1 for trial in paired_clean if trial.verifier_result.is_correct) / len(paired_clean)
        contaminated_rate = (
            sum(1 for trial in paired_contaminated if trial.verifier_result.is_correct)
            / len(paired_contaminated)
        )
        degradation_by_combo[combo] = clean_rate - contaminated_rate
    return degradation_by_combo


def aggregate_run(run_dir: Path) -> dict:
    trials = _load_trials(run_dir / "trials.jsonl")
    grouped: dict[tuple[str, str, str, str], list[TrialLog]] = defaultdict(list)
    combos: dict[tuple[str, str, str], dict[str, list[TrialLog]]] = defaultdict(lambda: defaultdict(list))
    for trial in trials:
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
        group.update(_metric_group(grouped[key]))
        group["vanilla_to_contamination_degradation_rate"] = degradation_by_combo.get(combo, NOT_COMPUTED)
        groups.append(group)

    return {"run_dir": str(run_dir), "n_trials": len(trials), "groups": groups}
