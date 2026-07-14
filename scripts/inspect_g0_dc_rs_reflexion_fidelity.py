from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from memcontam.logging.schema import TrialLog


DC_RS = "dynamic_cheatsheet_rs_optional"
REFLEXION = "reflexion_style"
TASKS = {
    "game24": {"game24_pilot_001", "game24_pilot_002", "game24_pilot_003"},
    "math_equation_balancer": {"meb_pilot_001", "meb_pilot_002", "meb_pilot_003"},
    "word_sorting": {"word_sorting_pilot_001", "word_sorting_pilot_002", "word_sorting_pilot_003"},
}
ARMS = {"clean", "contaminated", "contaminated_filter"}
MODELS = {"gpt4o", "frontier_reasoning"}


def _load_trials(run_dir: Path) -> list[TrialLog]:
    path = run_dir / "trials.jsonl"
    if not path.exists():
        raise SystemExit(f"trials.jsonl not found: {path}")
    trials: list[TrialLog] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            trials.append(TrialLog.model_validate(json.loads(line)))
        except Exception as exc:
            raise SystemExit(f"line {line_number}: invalid TrialLog: {exc}") from exc
    return trials


def _load_aggregate(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "aggregate.json"
    if not path.exists():
        raise SystemExit(f"aggregate.json not found: {path}")
    try:
        aggregate = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"malformed aggregate.json: {path}") from exc
    if not isinstance(aggregate, dict):
        raise SystemExit(f"aggregate.json must contain an object: {path}")
    return aggregate


def _result(reasons: list[str], **counts: Any) -> dict[str, Any]:
    return {"pass": "pass" if not reasons else "fail", **counts, "reasons": reasons}


def _trial_identity(trial: TrialLog) -> tuple[str, str, str, str, str, str]:
    return (
        trial.run_id,
        trial.task_name,
        trial.sample_id,
        trial.baseline,
        trial.arm,
        trial.backbone,
    )


def _state_identity_from_trial_id(trial_id: str) -> tuple[str, str, str, str, str] | None:
    parts = trial_id.split(":")
    if len(parts) != 6:
        return None
    run_id, task_name, _sample_id, baseline, arm, backbone = parts
    return run_id, task_name, baseline, arm, backbone


def _state_identity(trial: TrialLog) -> tuple[str, str, str, str, str]:
    return (trial.run_id, trial.task_name, trial.baseline, trial.arm, trial.backbone)


def _check_shape(trials: list[TrialLog]) -> dict[str, Any]:
    reasons: list[str] = []
    expected_identities = {
        (task, sample, baseline, arm, model)
        for task, samples in TASKS.items()
        for sample in samples
        for baseline in {DC_RS, REFLEXION}
        for arm in ARMS
        for model in MODELS
    }
    actual_identities = {
        (trial.task_name, trial.sample_id, trial.baseline, trial.arm, trial.backbone) for trial in trials
    }
    counts = Counter(trial.baseline for trial in trials)
    if len(trials) != 108:
        reasons.append(f"expected 108 trial rows, got {len(trials)}")
    if len({_trial_identity(trial) for trial in trials}) != 108:
        reasons.append("expected 108 unique trial identities")
    if actual_identities != expected_identities:
        reasons.append("trial identity matrix does not match locked tasks/samples/arms/models")
    for baseline in (DC_RS, REFLEXION):
        if counts[baseline] != 54:
            reasons.append(f"{baseline}: expected 54 rows, got {counts[baseline]}")
    return _result(reasons, trials=len(trials), unique_identities=len(actual_identities))


def _check_dc_rs(trials: list[TrialLog]) -> dict[str, Any]:
    reasons: list[str] = []
    dc_trials = [trial for trial in trials if trial.baseline == DC_RS]
    for trial in dc_trials:
        stages = [call.stage for call in trial.method_calls]
        if stages != ["dc_rs_synthesize", "dc_rs_generate"]:
            reasons.append(f"{trial.trial_id}: DC-RS stages must be synthesize -> generate, got {stages}")
            continue
        synthesize, generate = trial.method_calls
        if generate.retrieved_records:
            reasons.append(f"{trial.trial_id}: DC-RS generate must not carry retrieved records")
        if not _nonempty_cheatsheet(synthesize.raw_response):
            reasons.append(f"{trial.trial_id}: DC-RS synthesis cheatsheet is empty or missing")
        event: dict[str, Any] = trial.memory_write_event or {}
        if event.get("type") != "dynamic_cheatsheet_rs_update" or event.get("status") != "accepted":
            reasons.append(f"{trial.trial_id}: DC-RS synthesis status is not accepted")
        pair: dict[str, Any] = {}
        pair_value = event.get("pair_appended")
        if isinstance(pair_value, dict):
            pair = pair_value
        pair_id = pair.get("entry_id")
        before_ids = {entry.get("entry_id") for entry in trial.memory_before}
        after_ids = {entry.get("entry_id") for entry in trial.memory_after}
        if not isinstance(pair_id, str) or pair_id in before_ids or pair_id not in after_ids:
            reasons.append(f"{trial.trial_id}: current DC-RS pair must first appear in memory_after")

        pair_entries = {
            entry.get("entry_id"): entry
            for entry in trial.memory_before
            if entry.get("memory_type") == "dc_rs_io_pair" and isinstance(entry.get("entry_id"), str)
        }
        records = synthesize.retrieved_records
        expected_count = min(3, len(pair_entries))
        if len(records) != expected_count:
            reasons.append(
                f"{trial.trial_id}: expected {expected_count} DC-RS retrieved records, got {len(records)}"
            )
        if [record.rank for record in records] != list(range(1, len(records) + 1)):
            reasons.append(f"{trial.trial_id}: DC-RS retrieved ranks must be contiguous from 1")
        record_ids = [record.document_id for record in records]
        if len(record_ids) != len(set(record_ids)):
            reasons.append(f"{trial.trial_id}: DC-RS retrieved records contain duplicate IDs")
        for record in records:
            entry = pair_entries.get(record.document_id)
            if entry is None:
                reasons.append(f"{trial.trial_id}: DC-RS retrieved current/future or foreign pair {record.document_id}")
                continue
            if record.text != entry.get("content"):
                reasons.append(f"{trial.trial_id}: DC-RS retrieval provenance is not input-only")

        for entry in trial.memory_before + trial.memory_after:
            source_trial_id = entry.get("source_trial_id")
            if isinstance(source_trial_id, str) and source_trial_id.startswith(trial.run_id + ":"):
                if _state_identity_from_trial_id(source_trial_id) != _state_identity(trial):
                    reasons.append(f"{trial.trial_id}: DC-RS state leaked across identities")
                    break
        if trial.arm == "contaminated_filter" and any(
            entry.get("clean_or_contaminated") == "contaminated" for entry in trial.memory_before
        ):
            reasons.append(f"{trial.trial_id}: contaminated-filter DC-RS state retains a corrupted seed")
    return _result(reasons, trials=len(dc_trials), method_calls=sum(len(t.method_calls) for t in dc_trials))


def _nonempty_cheatsheet(raw_response: str) -> bool:
    start = raw_response.find("<cheatsheet>")
    end = raw_response.find("</cheatsheet>", start + len("<cheatsheet>"))
    return start >= 0 and end >= 0 and bool(raw_response[start + len("<cheatsheet>") : end].strip())


def _check_reflexion(trials: list[TrialLog]) -> dict[str, Any]:
    reasons: list[str] = []
    reflexion_trials = [trial for trial in trials if trial.baseline == REFLEXION]
    retry_trials = 0
    for trial in reflexion_trials:
        stages = [call.stage for call in trial.method_calls]
        should_retry = trial.task_name == "game24" and trial.sample_id == "game24_pilot_001"
        if should_retry:
            retry_trials += 1
            if stages != ["reflexion_generate", "reflexion_reflect", "reflexion_generate"]:
                reasons.append(f"{trial.trial_id}: Reflexion retry stages must be generate -> reflect -> generate")
                continue
            reflection = trial.method_calls[1].raw_response.strip()
            retry_text = "\n".join(message["content"] for message in trial.method_calls[2].messages)
            event = trial.memory_write_event or {}
            if not reflection:
                reasons.append(f"{trial.trial_id}: Reflexion retry reflection is empty")
            if reflection not in retry_text:
                reasons.append(f"{trial.trial_id}: Reflexion retry does not consume its reflection")
            if event.get("source_trial_id") != trial.trial_id or event.get("status") != "accepted":
                reasons.append(f"{trial.trial_id}: Reflexion retry does not retain its task/sample identity")
            if not any(reflection in str(entry.get("content", "")) for entry in trial.memory_after[-3:]):
                reasons.append(f"{trial.trial_id}: reflection is absent from retry-visible latest-three memory")
            if trial.raw_response != trial.method_calls[-1].raw_response or not trial.verifier_result.is_correct:
                reasons.append(f"{trial.trial_id}: final correctness is not the retry verifier result")
        elif stages not in (["reflexion_generate"], ["reflexion_generate", "reflexion_reflect"]):
            reasons.append(f"{trial.trial_id}: unexpected Reflexion stage sequence {stages}")
    if retry_trials != 6:
        reasons.append(f"expected six Reflexion retry identities, got {retry_trials}")
    return _result(
        reasons,
        trials=len(reflexion_trials),
        retry_identities=retry_trials,
        method_calls=sum(len(t.method_calls) for t in reflexion_trials),
    )


def _method_metrics(trials: list[TrialLog]) -> dict[str, int]:
    calls = [call for trial in trials for call in trial.method_calls]
    return {
        "method_call_count": len(calls),
        "method_call_error_count": sum(call.error_type is not None for call in calls),
        "prompt_token_total": sum(int(call.token_usage.get("prompt_tokens", 0)) for call in calls),
        "completion_token_total": sum(int(call.token_usage.get("completion_tokens", 0)) for call in calls),
        "total_token_total": sum(int(call.token_usage.get("total_tokens", 0)) for call in calls),
        "latency_ms_total": sum(call.latency_ms or 0 for call in calls),
    }


def _check_accounting(trials: list[TrialLog], aggregate: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    metrics = _method_metrics(trials)
    stages = Counter(call.stage for trial in trials for call in trial.method_calls)
    expected_stages = {
        "dc_rs_synthesize": 54,
        "dc_rs_generate": 54,
        "reflexion_generate": 60,
        "reflexion_reflect": 6,
    }
    if metrics["method_call_count"] != 174:
        reasons.append(f"expected 174 native method calls, got {metrics['method_call_count']}")
    if dict(stages) != expected_stages:
        reasons.append(f"unexpected native method-call stages: {dict(stages)}")
    if aggregate.get("n_trials") != len(trials):
        reasons.append(f"aggregate n_trials {aggregate.get('n_trials')!r} != {len(trials)}")
    groups = aggregate.get("groups")
    if not isinstance(groups, list):
        reasons.append("aggregate groups are missing")
    else:
        for name, actual in metrics.items():
            total = sum(group.get(name, 0) for group in groups)
            if total != actual:
                reasons.append(f"aggregate {name} {total!r} != method-call total {actual}")
    return _result(reasons, **metrics, stage_counts=dict(stages))


def inspect_run(run_dir: Path) -> dict[str, Any]:
    trials = _load_trials(run_dir)
    aggregate = _load_aggregate(run_dir)
    shape = _check_shape(trials)
    dc_rs = _check_dc_rs(trials)
    reflexion = _check_reflexion(trials)
    accounting = _check_accounting(trials, aggregate)
    checks = {"shape": shape, "dc_rs": dc_rs, "reflexion": reflexion, "accounting": accounting}
    report = {
        **{name: result["pass"] for name, result in checks.items()},
        "summary": {
            "trials": len(trials),
            "method_calls": accounting["method_call_count"],
            "dc_rs_calls": dc_rs["method_calls"],
            "reflexion_calls": reflexion["method_calls"],
        },
        "reasons": [reason for result in checks.values() for reason in result["reasons"]],
    }
    report["overall"] = "pass" if all(value["pass"] == "pass" for value in checks.values()) else "fail"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the G0 DC-RS/Reflexion replay fidelity evidence.")
    parser.add_argument("run_dir", type=Path, help="run directory containing trials.jsonl and aggregate.json")
    args = parser.parse_args(argv)
    report = inspect_run(args.run_dir)
    print(json.dumps(report, indent=2))
    return 0 if report["overall"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
