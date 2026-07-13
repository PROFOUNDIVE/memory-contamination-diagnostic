from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from memcontam.logging.schema import TrialLog
from memcontam.memory.corpus import _FORBIDDEN_ANSWER_SUBSTRINGS


EXPECTED_BASELINES = {
    "full_history",
    "reflexion_style",
    "dynamic_cheatsheet_optional",
}
EXPECTED_ARMS = {"clean", "contaminated", "contaminated_filter"}
EXPECTED_TASKS = {"game24", "math_equation_balancer", "word_sorting"}
EXPECTED_MODELS = {"gpt4o", "frontier_reasoning"}
EXPECTED_SAMPLES = {
    "game24": {"game24_pilot_001", "game24_pilot_002", "game24_pilot_003"},
    "math_equation_balancer": {"meb_pilot_001", "meb_pilot_002", "meb_pilot_003"},
    "word_sorting": {"word_sorting_pilot_001", "word_sorting_pilot_002", "word_sorting_pilot_003"},
}
SAMPLE_ORDER = {
    "game24": ["game24_pilot_001", "game24_pilot_002", "game24_pilot_003"],
    "math_equation_balancer": ["meb_pilot_001", "meb_pilot_002", "meb_pilot_003"],
    "word_sorting": ["word_sorting_pilot_001", "word_sorting_pilot_002", "word_sorting_pilot_003"],
}
SOURCE_LABELS = {"injected_corruption", "pilot_warmup_strategy"}


def _load_trials(run_dir: Path) -> list[TrialLog]:
    trials_path = run_dir / "trials.jsonl"
    if not trials_path.exists():
        raise SystemExit(f"trials.jsonl not found: {trials_path}")

    trials: list[TrialLog] = []
    for line_number, line in enumerate(trials_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"line {line_number}: malformed JSON") from exc
        try:
            trials.append(TrialLog.model_validate(raw))
        except Exception as exc:
            raise SystemExit(f"line {line_number}: invalid TrialLog: {exc}") from exc
    return trials


def _identity(trial: TrialLog) -> tuple[str, str, str, str, str]:
    return (trial.run_id, trial.task_name, trial.baseline, trial.arm, trial.backbone)


def _entry_ids(entries: list[dict[str, Any]]) -> set[str]:
    return {entry["entry_id"] for entry in entries if isinstance(entry.get("entry_id"), str)}


def _is_catalog_seed_entry(entry: dict[str, Any]) -> bool:
    memory_type = entry.get("memory_type")
    return memory_type in {"full_history_transcript", "verbal_reflection", "cheatsheet_item"} and entry.get("source_trial_id") is None


def _all_entry_ids(trials: list[TrialLog]) -> set[str]:
    ids: set[str] = set()
    for trial in trials:
        ids.update(_entry_ids(trial.memory_before))
        ids.update(_entry_ids(trial.memory_after))
    return ids


def _check_shape(trials: list[TrialLog]) -> dict[str, Any]:
    reasons: list[str] = []

    if len(trials) != 162:
        reasons.append(f"expected 162 trials, got {len(trials)}")

    baseline_counts: dict[str, int] = {}
    baseline_arm_counts: dict[tuple[str, str], int] = {}
    seen_trial_ids: set[str] = set()
    seen_identities: set[tuple[str, str, str, str, str]] = set()
    tasks: set[str] = set()
    models: set[str] = set()
    arms: set[str] = set()
    baselines: set[str] = set()
    samples_by_task: dict[str, set[str]] = {}

    for trial in trials:
        tasks.add(trial.task_name)
        models.add(trial.backbone)
        arms.add(trial.arm)
        baselines.add(trial.baseline)
        samples_by_task.setdefault(trial.task_name, set()).add(trial.sample_id)

        if trial.trial_id in seen_trial_ids:
            reasons.append(f"duplicate trial_id: {trial.trial_id}")
        seen_trial_ids.add(trial.trial_id)

        baseline_counts[trial.baseline] = baseline_counts.get(trial.baseline, 0) + 1
        baseline_arm_counts[(trial.baseline, trial.arm)] = baseline_arm_counts.get((trial.baseline, trial.arm), 0) + 1

    if baselines != EXPECTED_BASELINES:
        reasons.append(f"unexpected baselines: {baselines ^ EXPECTED_BASELINES}")
    if arms != EXPECTED_ARMS:
        reasons.append(f"unexpected arms: {arms ^ EXPECTED_ARMS}")
    if tasks != EXPECTED_TASKS:
        reasons.append(f"unexpected tasks: {tasks ^ EXPECTED_TASKS}")
    if models != EXPECTED_MODELS:
        reasons.append(f"unexpected models: {models ^ EXPECTED_MODELS}")
    for task, expected in EXPECTED_SAMPLES.items():
        got = samples_by_task.get(task, set())
        if got != expected:
            reasons.append(f"{task}: unexpected samples: {got ^ expected}")

    for baseline in EXPECTED_BASELINES:
        count = baseline_counts.get(baseline, 0)
        if count != 54:
            reasons.append(f"{baseline}: expected 54 trials, got {count}")
    for baseline in EXPECTED_BASELINES:
        for arm in EXPECTED_ARMS:
            count = baseline_arm_counts.get((baseline, arm), 0)
            if count != 18:
                reasons.append(f"{baseline}/{arm}: expected 18 trials, got {count}")

    passed = not reasons
    return {
        "pass": "pass" if passed else "fail",
        "trial_count": len(trials),
        "unique_identities": len(seen_identities),
        "reasons": reasons,
    }


def _check_stages(trials: list[TrialLog]) -> dict[str, Any]:
    reasons: list[str] = []
    stage_counts: dict[str, int] = {}

    for trial in trials:
        stages = [call.stage for call in trial.method_calls]
        for stage in stages:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        if trial.baseline == "full_history":
            expected = ["full_history_generate"]
        elif trial.baseline == "reflexion_style":
            expected = ["reflexion_generate"] if stages == ["reflexion_generate"] else ["reflexion_generate", "reflexion_reflect"]
        else:
            expected = ["dynamic_cheatsheet_generate", "dynamic_cheatsheet_curate"]

        if stages != expected:
            reasons.append(f"{trial.trial_id}: expected stages {expected}, got {stages}")

    expected_counts = {
        "full_history_generate": 54,
        "reflexion_generate": 54,
        "reflexion_reflect": 6,
        "dynamic_cheatsheet_generate": 54,
        "dynamic_cheatsheet_curate": 54,
    }

    for stage, expected in expected_counts.items():
        got = stage_counts.get(stage, 0)
        if got != expected:
            reasons.append(f"stage {stage}: expected {expected} calls, got {got}")

    for stage in stage_counts:
        if stage not in expected_counts:
            reasons.append(f"unexpected stage: {stage}")

    passed = not reasons
    return {
        "pass": "pass" if passed else "fail",
        "stage_counts": stage_counts,
        "reasons": reasons,
    }


def _check_full_history(trials: list[TrialLog]) -> dict[str, Any]:
    fh_trials = [t for t in trials if t.baseline == "full_history"]
    reasons: list[str] = []
    accepted = 0

    for trial in fh_trials:
        event = trial.memory_write_event
        if event is None:
            reasons.append(f"{trial.trial_id}: missing memory_write_event")
            continue
        if event.get("status") != "accepted":
            reasons.append(f"{trial.trial_id}: full_history write status {event.get('status')!r} != accepted")
            continue
        if event.get("type") != "full_history_append":
            reasons.append(f"{trial.trial_id}: full_history write type {event.get('type')!r} != full_history_append")
            continue

        new_entry_id = event.get("new_entry_id")
        if not isinstance(new_entry_id, str):
            reasons.append(f"{trial.trial_id}: full_history write missing new_entry_id")
            continue

        memory_before_ids = _entry_ids(trial.memory_before)
        memory_after_ids = _entry_ids(trial.memory_after)
        if len(memory_after_ids) != len(memory_before_ids) + 1:
            reasons.append(
                f"{trial.trial_id}: memory_after ({len(memory_after_ids)}) != memory_before ({len(memory_before_ids)}) + 1"
            )
            continue
        added = memory_after_ids - memory_before_ids
        if added != {new_entry_id}:
            reasons.append(f"{trial.trial_id}: added entry {added} does not match write event {new_entry_id}")
            continue

        new_entry = next((e for e in trial.memory_after if e.get("entry_id") == new_entry_id), None)
        if new_entry is None:
            reasons.append(f"{trial.trial_id}: new_entry_id {new_entry_id} not found in memory_after")
            continue
        if new_entry.get("memory_type") != "full_history_transcript":
            reasons.append(f"{trial.trial_id}: new entry memory_type {new_entry.get('memory_type')!r}")
            continue

        prompt_text = "\n".join(str(m.get("content", "")) for m in trial.prompt_messages)
        for entry in trial.memory_before:
            content = entry.get("content", "")
            if isinstance(content, str) and content not in prompt_text:
                reasons.append(f"{trial.trial_id}: prompt missing memory_before entry {entry.get('entry_id')}")
                break
        accepted += 1

    passed = not reasons
    return {
        "pass": "pass" if passed else "fail",
        "fh_trials": len(fh_trials),
        "accepted_appends": accepted,
        "reasons": reasons,
    }


def _check_reflexion(trials: list[TrialLog]) -> dict[str, Any]:
    refl_trials = [t for t in trials if t.baseline == "reflexion_style"]
    reasons: list[str] = []
    reflected_trials = 0

    for trial in refl_trials:
        stages = [call.stage for call in trial.method_calls]
        has_reflect = "reflexion_reflect" in stages
        is_correct = trial.verifier_result.is_correct
        should_reflect = trial.sample_id == "game24_pilot_001"

        if is_correct == should_reflect:
            expected = "fail" if should_reflect else "succeed"
            reasons.append(f"{trial.trial_id}: expected Reflexion trial to {expected}")
        if has_reflect != should_reflect:
            expected = "reflexion_reflect" if should_reflect else "no reflexion_reflect"
            reasons.append(f"{trial.trial_id}: expected {expected} call")

        if should_reflect and has_reflect:
            event = trial.memory_write_event
            if event is None or event.get("status") != "accepted":
                reasons.append(f"{trial.trial_id}: failed trial missing accepted reflection append")
                continue
            reflected_trials += 1

        reflection_entries = [
            e for e in trial.memory_before
            if e.get("memory_type") == "verbal_reflection" or str(e.get("content", "")).startswith("Reflection:")
        ]
        prompt_text = "\n".join(str(m.get("content", "")) for m in trial.prompt_messages)
        rendered = len([e for e in reflection_entries if e.get("content", "") in prompt_text or f"Reflection: {str(e.get('content', '')).removeprefix('Reflection:').strip()}" in prompt_text])
        if len(reflection_entries) > 3 and rendered > 3:
            reasons.append(f"{trial.trial_id}: prompt exposes more than 3 reflections")
            continue

    passed = not reasons
    return {
        "pass": "pass" if passed else "fail",
        "refl_trials": len(refl_trials),
        "reflected_trials": reflected_trials,
        "reasons": reasons,
    }


def _check_dynamic_cheatsheet(trials: list[TrialLog]) -> dict[str, Any]:
    dc_trials = [t for t in trials if t.baseline == "dynamic_cheatsheet_optional"]
    reasons: list[str] = []
    preserved = 0
    accepted = 0

    grouped: dict[tuple[str, str, str], list[TrialLog]] = {}
    for trial in dc_trials:
        grouped.setdefault((trial.task_name, trial.arm, trial.backbone), []).append(trial)

    for trial in dc_trials:
        stages = [call.stage for call in trial.method_calls]
        if stages != ["dynamic_cheatsheet_generate", "dynamic_cheatsheet_curate"]:
            continue

        event = trial.memory_write_event
        if event is None:
            reasons.append(f"{trial.trial_id}: missing DC memory_write_event")
            continue
        status = event.get("status")

        if trial.sample_id == "game24_pilot_001":
            if status != "preserved_missing_tag":
                reasons.append(f"{trial.trial_id}: game24_pilot_001 DC status {status!r} != preserved_missing_tag")
                continue
            if trial.memory_after != trial.memory_before:
                reasons.append(f"{trial.trial_id}: game24_pilot_001 DC memory_after changed despite preservation")
                continue
            preserved += 1
        else:
            if status != "accepted":
                reasons.append(f"{trial.trial_id}: DC status {status!r} != accepted")
                continue
            accepted += 1

    for (task, arm, model), group in grouped.items():
        group_sorted = sorted(group, key=lambda t: SAMPLE_ORDER[task].index(t.sample_id))
        for i in range(1, len(group_sorted)):
            prev = group_sorted[i - 1]
            cur = group_sorted[i]
            if cur.memory_write_event and cur.memory_write_event.get("status") == "accepted":
                cur_prompt = "\n".join(str(m.get("content", "")) for m in cur.prompt_messages)
                missing = [
                    e.get("entry_id")
                    for e in prev.memory_after
                    if str(e.get("content", "")) and str(e.get("content", "")) not in cur_prompt
                ]
                if missing:
                    reasons.append(
                        f"{cur.trial_id}: DC generate prompt does not reuse previous cheatsheet entries {missing} from {prev.trial_id}"
                    )

    passed = not reasons
    return {
        "pass": "pass" if passed else "fail",
        "dc_trials": len(dc_trials),
        "preserved_trials": preserved,
        "accepted_trials": accepted,
        "reasons": reasons,
    }


def _check_arms(trials: list[TrialLog]) -> dict[str, Any]:
    reasons: list[str] = []

    for trial in trials:
        exposure = trial.contamination_exposure
        if trial.arm == "clean":
            if exposure.is_exposed or exposure.source_entry_ids:
                reasons.append(f"{trial.trial_id}: clean arm is exposed")
        elif trial.arm == "contaminated":
            if not exposure.is_exposed:
                reasons.append(f"{trial.trial_id}: contaminated arm has no exposure")
            seed_ids = {e["entry_id"] for e in trial.memory_before if e.get("clean_or_contaminated") == "contaminated"}
            if not seed_ids and trial.sample_id.endswith("_001"):
                reasons.append(f"{trial.trial_id}: contaminated arm missing paired seed in memory_before")
        elif trial.arm == "contaminated_filter":
            if trial.filter_decision is None or trial.filter_decision.get("dropped", 0) == 0:
                reasons.append(f"{trial.trial_id}: contaminated_filter missing non-zero drop")
            remaining = [e for e in trial.memory_before if e.get("clean_or_contaminated") == "contaminated"]
            if remaining:
                reasons.append(f"{trial.trial_id}: contaminated_filter still has contaminated memory_before entries")
            if exposure.is_exposed:
                reasons.append(f"{trial.trial_id}: contaminated_filter arm is exposed")

    passed = not reasons
    return {
        "pass": "pass" if passed else "fail",
        "reasons": reasons,
    }


def _check_isolation(trials: list[TrialLog]) -> dict[str, Any]:
    reasons: list[str] = []
    accepted_by_id: dict[str, tuple[str, str, str, str, str]] = {}

    for trial in trials:
        event = trial.memory_write_event
        if event and event.get("status") == "accepted":
            new_entry_id = event.get("new_entry_id")
            if isinstance(new_entry_id, str):
                accepted_by_id[new_entry_id] = _identity(trial)

    for trial in trials:
        identity = _identity(trial)
        visible_ids = _entry_ids(trial.memory_before + trial.memory_after)
        for entry_id, source_identity in accepted_by_id.items():
            if entry_id in visible_ids and identity != source_identity:
                reasons.append(f"{entry_id} from {source_identity} leaked into {identity} ({trial.trial_id})")

    passed = not reasons
    return {
        "pass": "pass" if passed else "fail",
        "accepted_entries": len(accepted_by_id),
        "leakage_count": len(reasons),
        "reasons": reasons,
    }


def _check_no_bot_warmup(trials: list[TrialLog]) -> dict[str, Any]:
    reasons: list[str] = []
    bot_keys = {"warmup", "bot_state", "bot_buffer", "thought_template", "retrieved_records"}

    for trial in trials:
        if trial.baseline not in EXPECTED_BASELINES:
            continue
        metadata = trial.metadata or {}
        for key in metadata:
            if any(bot_key in key.lower() for bot_key in bot_keys):
                reasons.append(f"{trial.trial_id}: native baseline metadata contains bot/retrieval key {key!r}")
        if trial.retrieved_memory:
            reasons.append(f"{trial.trial_id}: native baseline has retrieved_memory entries")
        if trial.retrieved_scores:
            reasons.append(f"{trial.trial_id}: native baseline has retrieved_scores")

    passed = not reasons
    return {
        "pass": "pass" if passed else "fail",
        "reasons": reasons,
    }


def _check_leakage(trials: list[TrialLog]) -> dict[str, Any]:
    reasons: list[str] = []

    for trial in trials:
        for call in trial.method_calls:
            for message in call.messages:
                content = str(message.get("content", ""))
                for sub in SOURCE_LABELS:
                    if sub.lower() in content.lower():
                        reasons.append(f"{trial.trial_id}: method_call message contains source label {sub!r}")

        for message in trial.prompt_messages:
            content = str(message.get("content", ""))
            for sub in SOURCE_LABELS:
                if sub.lower() in content.lower():
                    reasons.append(f"{trial.trial_id}: prompt_message contains source label {sub!r}")

        for entry in trial.memory_before + trial.memory_after:
            content = str(entry.get("content", ""))
            for sub in SOURCE_LABELS:
                if sub.lower() in content.lower():
                    reasons.append(f"{trial.trial_id}: memory entry {entry.get('entry_id')} contains source label {sub!r}")

            if _is_catalog_seed_entry(entry):
                for sub in _FORBIDDEN_ANSWER_SUBSTRINGS:
                    if sub.lower() in content.lower():
                        reasons.append(f"{trial.trial_id}: seed memory entry {entry.get('entry_id')} contains {sub!r}")

        for entry in trial.memory_before + trial.memory_after:
            if "clean_or_contaminated" not in entry:
                reasons.append(f"{trial.trial_id}: memory entry {entry.get('entry_id')} missing clean_or_contaminated")

    passed = not reasons
    return {
        "pass": "pass" if passed else "fail",
        "reasons": reasons,
    }


def _check_logging(trials: list[TrialLog]) -> dict[str, Any]:
    target_trials = [t for t in trials if t.baseline in EXPECTED_BASELINES]
    reasons: list[str] = []
    error_calls = 0
    total_calls = 0

    for trial in target_trials:
        total_calls += len(trial.method_calls)
        for call in trial.method_calls:
            if call.error_type is not None:
                error_calls += 1
                reasons.append(f"{trial.trial_id}: {call.stage} recorded error {call.error_type!r}")

    passed = not reasons
    return {
        "pass": "pass" if passed else "fail",
        "target_trials": len(target_trials),
        "total_calls": total_calls,
        "error_calls": error_calls,
        "reasons": reasons,
    }


def inspect_run(run_dir: Path) -> dict[str, Any]:
    trials = _load_trials(run_dir)
    shape_result = _check_shape(trials)
    stages_result = _check_stages(trials)
    fh_result = _check_full_history(trials)
    refl_result = _check_reflexion(trials)
    dc_result = _check_dynamic_cheatsheet(trials)
    arms_result = _check_arms(trials)
    isolation_result = _check_isolation(trials)
    no_bot_result = _check_no_bot_warmup(trials)
    leakage_result = _check_leakage(trials)
    logging_result = _check_logging(trials)

    checks = [
        shape_result["pass"],
        stages_result["pass"],
        fh_result["pass"],
        refl_result["pass"],
        dc_result["pass"],
        arms_result["pass"],
        isolation_result["pass"],
        no_bot_result["pass"],
        leakage_result["pass"],
        logging_result["pass"],
    ]

    report: dict[str, Any] = {
        "shape": shape_result["pass"],
        "stages": stages_result["pass"],
        "full_history": fh_result["pass"],
        "reflexion": refl_result["pass"],
        "dynamic_cheatsheet": dc_result["pass"],
        "arms": arms_result["pass"],
        "isolation": isolation_result["pass"],
        "no_bot_warmup": no_bot_result["pass"],
        "leakage": leakage_result["pass"],
        "logging": logging_result["pass"],
        "shape_counts": {k: v for k, v in shape_result.items() if k not in {"pass", "reasons"}},
        "stage_counts": stages_result.get("stage_counts", {}),
        "fh_counts": {k: v for k, v in fh_result.items() if k not in {"pass", "reasons"}},
        "refl_counts": {k: v for k, v in refl_result.items() if k not in {"pass", "reasons"}},
        "dc_counts": {k: v for k, v in dc_result.items() if k not in {"pass", "reasons"}},
        "isolation_counts": {k: v for k, v in isolation_result.items() if k not in {"pass", "reasons"}},
        "logging_counts": {k: v for k, v in logging_result.items() if k not in {"pass", "reasons"}},
        "reasons": (
            shape_result.get("reasons", [])
            + stages_result.get("reasons", [])
            + fh_result.get("reasons", [])
            + refl_result.get("reasons", [])
            + dc_result.get("reasons", [])
            + arms_result.get("reasons", [])
            + isolation_result.get("reasons", [])
            + no_bot_result.get("reasons", [])
            + leakage_result.get("reasons", [])
            + logging_result.get("reasons", [])
        ),
    }
    report["overall"] = "pass" if all(c == "pass" for c in checks) else "fail"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a G0 full-history/Reflexion/DC replay run for fidelity.")
    parser.add_argument("run_dir", type=Path, help="path to the run directory containing trials.jsonl")
    args = parser.parse_args(argv)

    report = inspect_run(args.run_dir)
    print(json.dumps(report, indent=2))

    return 0 if report["overall"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
