from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from memcontam.logging.schema import CallEvent, FailureEvent, MemoryEvent, TrialLog


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def _jsonl(path: Path, model: Any) -> list[Any]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            rows.append(model.model_validate(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"{path.name}:{line_number}: {exc}") from exc
    return rows


def inspect_run(run_dir: Path) -> dict[str, Any]:
    reasons: list[str] = []
    required = [
        "run.json",
        "resolved_config.json",
        "provider_profile.json",
        "trials.jsonl",
        "calls.jsonl",
        "failures.jsonl",
        "memory_events.jsonl",
    ]
    for filename in required:
        if not (run_dir / filename).is_file():
            reasons.append(f"missing artifact: {filename}")
    if reasons:
        return {"overall": "fail", "reasons": reasons}

    manifest = _json(run_dir / "run.json")
    resolved = _json(run_dir / "resolved_config.json")
    _json(run_dir / "provider_profile.json")
    trials = _jsonl(run_dir / "trials.jsonl", TrialLog)
    calls = _jsonl(run_dir / "calls.jsonl", CallEvent)
    failures = _jsonl(run_dir / "failures.jsonl", FailureEvent)
    memory_events = _jsonl(run_dir / "memory_events.jsonl", MemoryEvent)
    if resolved.get("run", {}).get("fidelity_gate_layer") != "source_contract":
        reasons.append("resolved config is not an F1B source-contract run")
    counts = manifest.get("counts", {})
    expected_counts = {"trials": len(trials), "calls": len(calls), "failures": len(failures), "memory_events": len(memory_events)}
    for name, actual in expected_counts.items():
        if counts.get(name) != actual:
            reasons.append(f"run count {name}={counts.get(name)!r} != {actual}")

    calls_by_id = {call.call_id: call for call in calls}
    failures_by_id = {failure.failure_id: failure for failure in failures}
    events_by_trial = {event.trial_id: event for event in memory_events}
    expected_prompt_hashes = {
        fixture["stage"]: (fixture["messages_sha256"], fixture.get("sample_id"))
        for path in (Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "prompts" / "baseline_fidelity_v2").glob("*.json")
        for fixture in [_json(path)]
    }
    trial_call_ids = [call.call_id for trial in trials for call in trial.method_calls if call.call_id]
    if len(trial_call_ids) != len(set(trial_call_ids)) or set(trial_call_ids) != set(calls_by_id):
        reasons.append("calls.jsonl does not exactly join trial method calls")
    failed_ids = {trial.failure_id for trial in trials if trial.status == "failed" and trial.failure_id}
    if failed_ids != set(failures_by_id):
        reasons.append("failures.jsonl does not exactly join failed trials")
    for trial in trials:
        method_call_ids = {call.call_id for call in trial.method_calls}
        if trial.answer_call_id not in method_call_ids or trial.answer_call_id not in calls_by_id:
            reasons.append(f"{trial.trial_id}: answer-call join failed")
        for call in trial.method_calls:
            event = calls_by_id.get(call.call_id)
            if event is None:
                reasons.append(f"{trial.trial_id}: missing call {call.call_id}")
                continue
            if event.messages != call.messages or event.source_spans != call.source_spans:
                reasons.append(f"{trial.trial_id}: call payload differs from calls.jsonl")
            expected_fixture = expected_prompt_hashes.get(call.stage)
            if expected_fixture is not None and expected_fixture[1] == trial.sample_id:
                expected_hash = expected_fixture[0]
                prompt_bytes = json.dumps(call.messages, sort_keys=True, separators=(",", ":")).replace(
                    trial.run_id, "{{run_id}}"
                )
                if hashlib.sha256(prompt_bytes.encode("utf-8")).hexdigest() != expected_hash:
                    reasons.append(f"{trial.trial_id}: prompt bytes differ from {call.stage} fixture")
            _check_spans(trial, call, reasons)
        if trial.status == "failed":
            failure = failures_by_id.get(trial.failure_id or "")
            triple = (trial.error_type, trial.metadata.get("failure_disposition"), trial.metadata.get("scientific_ineligibility_reason"))
            if failure is None or not all(isinstance(value, str) and value for value in triple):
                reasons.append(f"{trial.trial_id}: missing closed failure triple")
            elif failure.error_type != trial.error_type or failure.disposition != triple[1]:
                reasons.append(f"{trial.trial_id}: failure event does not match trial triple")
        event = events_by_trial.get(trial.trial_id)
        changed = [entry.get("entry_id") for entry in trial.memory_before] != [entry.get("entry_id") for entry in trial.memory_after]
        if changed and event is None:
            reasons.append(f"{trial.trial_id}: state changed without memory event")
        if event is not None and (event.before_entry_ids != [entry.get("entry_id") for entry in trial.memory_before] or event.after_entry_ids != [entry.get("entry_id") for entry in trial.memory_after]):
            reasons.append(f"{trial.trial_id}: memory event state delta mismatch")
        if trial.baseline == "retrieval_rag":
            if len(trial.method_calls) and len(trial.method_calls[0].retrieved_records) != 3:
                reasons.append(f"{trial.trial_id}: RAG did not retrieve top-3")
            if changed or trial.memory_write_event is not None:
                reasons.append(f"{trial.trial_id}: RAG is not read-only")
    changed_trial_ids = {trial.trial_id for trial in trials if trial.memory_before != trial.memory_after}
    if not changed_trial_ids.issubset(events_by_trial) or not set(events_by_trial).issubset(
        {trial.trial_id for trial in trials}
    ):
        reasons.append("memory_events.jsonl does not join changed trial state")
    report = {"overall": "pass" if not reasons else "fail", "trials": len(trials), "calls": len(calls), "failures": len(failures), "memory_events": len(memory_events), "reasons": reasons}
    return report


def _check_spans(trial: TrialLog, call: Any, reasons: list[str]) -> None:
    entry_ids = {
        entry.get("entry_id")
        for entry in [*trial.memory_before, *trial.memory_after, *trial.retrieved_memory]
        if isinstance(entry.get("entry_id"), str)
    }
    for span in call.source_spans:
        try:
            content = call.messages[span.message_index]["content"]
            rendered = content[span.start : span.end]
        except (IndexError, KeyError):
            reasons.append(f"{trial.trial_id}: invalid source span bounds")
            continue
        if hashlib.sha256(rendered.encode("utf-8")).hexdigest() != span.rendered_hash:
            reasons.append(f"{trial.trial_id}: source span hash mismatch")
        derived = span.entry_id.startswith(("dc_rs_synthesized:", "reflexion_failed_actor:"))
        if span.entry_id not in entry_ids and not derived:
            reasons.append(f"{trial.trial_id}: source span references unknown entry {span.entry_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect Baseline-Fidelity-V2 F1B replay artifacts.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = inspect_run(args.run_dir)
    except Exception as exc:
        report = {"overall": "fail", "reasons": [str(exc)]}
    payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["overall"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
