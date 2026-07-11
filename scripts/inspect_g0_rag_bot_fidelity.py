from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from memcontam.logging.schema import MethodCall, TrialLog


RAG_BASELINE = "retrieval_rag"
BOT_BASELINE = "bot_style"
RAG_GENERATE_STAGE = "rag_generate"
BOT_EXPECTED_STAGES = [
    "bot_problem_distill",
    "bot_instantiate_solve",
    "bot_thought_distill",
    "bot_novelty_decide",
]
RETRIEVAL_RECORD_FIELDS = [
    "document_id",
    "rank",
    "score",
    "text",
    "title_or_type",
    "clean_or_contaminated",
    "source",
    "corpus_hash",
    "embedding_model_id",
    "embedding_revision",
    "embedding_library_version",
]


def _load_trials(run_dir: Path) -> list[TrialLog]:
    trials_path = run_dir / "trials.jsonl"
    if not trials_path.exists():
        raise SystemExit(f"trials.jsonl not found: {trials_path}")

    trials: list[TrialLog] = []
    for line in trials_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        trials.append(TrialLog.model_validate(json.loads(line)))
    return trials


def _identity(trial: TrialLog) -> tuple[str, str, str, str]:
    return (trial.task_name, trial.baseline, trial.arm, trial.backbone)


def _group_trials(trials: list[TrialLog]) -> dict[tuple[str, str, str, str], list[TrialLog]]:
    grouped: dict[tuple[str, str, str, str], list[TrialLog]] = {}
    for trial in trials:
        grouped.setdefault(_identity(trial), []).append(trial)
    return grouped


def _entry_ids(memory: list[dict[str, Any]]) -> set[str]:
    return {entry["entry_id"] for entry in memory if isinstance(entry.get("entry_id"), str)}


def _rag_generate_records(trial: TrialLog) -> list[MethodCall]:
    return [call for call in trial.method_calls if call.stage == RAG_GENERATE_STAGE]


def _check_rag(trials: list[TrialLog]) -> dict[str, Any]:
    rag_trials = [t for t in trials if t.baseline == RAG_BASELINE]
    reasons: list[str] = []
    if not rag_trials:
        reasons.append("no RAG trials found")
        return {"pass": "fail", "rag_trials": 0, "reasons": reasons}

    records: list[Any] = []
    corpus_hashes: set[str] = set()
    model_ids: set[str] = set()
    revisions: set[str] = set()
    library_versions: set[str] = set()
    aligned_trials = 0
    read_only_trials = 0

    for trial in rag_trials:
        generate_calls = _rag_generate_records(trial)
        if not generate_calls:
            reasons.append(f"{trial.trial_id}: missing {RAG_GENERATE_STAGE} method call")
            continue
        for call in generate_calls:
            records.extend(call.retrieved_records)

        if trial.memory_write_event is not None:
            reasons.append(f"{trial.trial_id}: RAG trial has a memory_write_event")
        elif trial.memory_before != trial.memory_after:
            reasons.append(f"{trial.trial_id}: RAG memory_before != memory_after")
        else:
            read_only_trials += 1

        call_records = [
            record for call in generate_calls for record in call.retrieved_records
        ]
        call_ids = {record.document_id for record in call_records}
        memory_ids = _entry_ids(trial.retrieved_memory)
        if call_ids != memory_ids:
            reasons.append(
                f"{trial.trial_id}: retrieved_memory IDs do not match logged retrieval records"
            )
            continue

        prompt_text = "\n".join(
            str(message.get("content", "")) for message in trial.prompt_messages
        )
        missing = [
            record.document_id
            for record in call_records
            if record.document_id not in prompt_text or record.text not in prompt_text
        ]
        if missing:
            reasons.append(
                f"{trial.trial_id}: prompt missing retrieved record IDs/text: {missing}"
            )
            continue
        aligned_trials += 1

    if records:
        corpus_hashes = {record.corpus_hash for record in records}
        model_ids = {record.embedding_model_id for record in records}
        revisions = {record.embedding_revision for record in records}
        library_versions = {record.embedding_library_version for record in records}
        for name, values in [
            ("corpus_hash", corpus_hashes),
            ("embedding_model_id", model_ids),
            ("embedding_revision", revisions),
            ("embedding_library_version", library_versions),
        ]:
            if len(values) > 1:
                reasons.append(f"RAG retrieval records have inconsistent {name}: {sorted(values)}")
        for record in records:
            for field in RETRIEVAL_RECORD_FIELDS:
                if not getattr(record, field, None):
                    reasons.append(
                        f"RAG retrieval record missing field {field!r}: {record.document_id}"
                    )
    else:
        reasons.append("no RAG retrieval records found")

    passed = not reasons
    return {
        "pass": "pass" if passed else "fail",
        "rag_trials": len(rag_trials),
        "aligned_prompt_log_trials": aligned_trials,
        "read_only_trials": read_only_trials,
        "retrieval_record_count": len(records),
        "consistent_corpus_hashes": len(corpus_hashes) if records else 0,
        "consistent_model_ids": len(model_ids) if records else 0,
        "reasons": reasons,
    }


def _check_bot(trials: list[TrialLog]) -> dict[str, Any]:
    bot_trials = [t for t in trials if t.baseline == BOT_BASELINE]
    reasons: list[str] = []
    if not bot_trials:
        reasons.append("no BoT trials found")
        return {"pass": "fail", "bot_trials": 0, "reasons": reasons}

    full_stage_trials = 0
    accepted_events = 0
    rejected_events = 0
    failed_without_accept = 0

    for trial in bot_trials:
        stages = [call.stage for call in trial.method_calls]
        missing = [stage for stage in BOT_EXPECTED_STAGES if stage not in stages]
        if missing:
            reasons.append(f"{trial.trial_id}: missing BoT stages {missing}")
            continue
        positions = [stages.index(stage) for stage in BOT_EXPECTED_STAGES]
        if positions != sorted(positions):
            reasons.append(f"{trial.trial_id}: BoT stages are out of order")
            continue
        full_stage_trials += 1

        event = trial.memory_write_event
        if not event or "status" not in event:
            reasons.append(f"{trial.trial_id}: missing complete write event with status")
            continue
        status = event.get("status")
        if status in {"accepted", "reused"}:
            accepted_events += 1
        elif status == "rejected":
            rejected_events += 1
        else:
            reasons.append(f"{trial.trial_id}: unexpected write event status {status!r}")
            continue

        if not trial.verifier_result.is_correct and status == "accepted":
            reasons.append(f"{trial.trial_id}: accepted buffer update after failed verifier")
        elif not trial.verifier_result.is_correct:
            failed_without_accept += 1

    passed = not reasons
    return {
        "pass": "pass" if passed else "fail",
        "bot_trials": len(bot_trials),
        "full_stage_trials": full_stage_trials,
        "accepted_events": accepted_events,
        "rejected_events": rejected_events,
        "failed_without_accept": failed_without_accept,
        "reasons": reasons,
    }


def _check_persistence(trials: list[TrialLog]) -> dict[str, Any]:
    bot_trials = [t for t in trials if t.baseline == BOT_BASELINE]
    grouped = _group_trials(bot_trials)
    reasons: list[str] = []
    accepted_templates: list[tuple[tuple[str, str, str, str], str, str]] = []
    reused_count = 0
    persistent_identities: set[tuple[str, str, str, str]] = set()

    for identity, group in grouped.items():
        for trial in group:
            event = trial.memory_write_event
            if event and event.get("status") == "accepted":
                new_entry_id = event.get("new_entry_id")
                if isinstance(new_entry_id, str):
                    accepted_templates.append((identity, trial.sample_id, new_entry_id))

    for identity, source_sample, entry_id in accepted_templates:
        group = grouped.get(identity, [])
        seen_source = False
        reused = False
        for trial in group:
            if trial.sample_id == source_sample:
                seen_source = True
                continue
            if seen_source and entry_id in _entry_ids(trial.memory_before + trial.retrieved_memory):
                reused = True
                break
        if reused:
            reused_count += 1
            persistent_identities.add(identity)

    for identity, group in grouped.items():
        identity_accepted = [item for item in accepted_templates if item[0] == identity]
        if not identity_accepted:
            continue
        distinct_samples = {t.sample_id for t in group}
        if len(distinct_samples) > 1 and identity not in persistent_identities:
            reasons.append(
                f"{identity}: accepted templates never reused across {len(distinct_samples)} samples"
            )

    if not persistent_identities:
        reasons.append("no accepted BoT template is reused across samples")

    if not accepted_templates:
        reasons.append("no accepted BoT templates found")

    passed = not reasons
    return {
        "pass": "pass" if passed else "fail",
        "accepted_templates": len(accepted_templates),
        "reused_templates": reused_count,
        "persistent_identities": len(persistent_identities),
        "reasons": reasons,
    }


def _check_isolation(trials: list[TrialLog]) -> dict[str, Any]:
    bot_trials = [t for t in trials if t.baseline == BOT_BASELINE]
    accepted_by_id: dict[str, tuple[str, str, str, str]] = {}
    for trial in bot_trials:
        event = trial.memory_write_event
        if event and event.get("status") == "accepted":
            entry_id = event.get("new_entry_id")
            if isinstance(entry_id, str):
                accepted_by_id[entry_id] = _identity(trial)

    leakage_count = 0
    reasons: list[str] = []
    for trial in trials:
        identity = _identity(trial)
        visible_ids = _entry_ids(trial.memory_before + trial.retrieved_memory)
        for entry_id, source_identity in accepted_by_id.items():
            if entry_id in visible_ids and identity != source_identity:
                leakage_count += 1
                reasons.append(
                    f"{entry_id} from {source_identity} leaked into {identity} ({trial.trial_id})"
                )

    passed = leakage_count == 0
    return {
        "pass": "pass" if passed else "fail",
        "accepted_template_count": len(accepted_by_id),
        "leakage_count": leakage_count,
        "reasons": reasons,
    }


def _check_logging(trials: list[TrialLog]) -> dict[str, Any]:
    target_trials = [t for t in trials if t.baseline in {RAG_BASELINE, BOT_BASELINE}]
    reasons: list[str] = []
    if not target_trials:
        reasons.append("no RAG or BoT trials to log")
        return {"pass": "fail", "target_trials": 0, "reasons": reasons}

    trials_with_calls = 0
    error_calls = 0
    total_calls = 0
    for trial in target_trials:
        if trial.method_calls:
            trials_with_calls += 1
        total_calls += len(trial.method_calls)
        for call in trial.method_calls:
            if call.error_type is not None:
                error_calls += 1
                reasons.append(f"{trial.trial_id}: {call.stage} call recorded error {call.error_type!r}")

    missing_calls = len(target_trials) - trials_with_calls
    if missing_calls:
        reasons.append(f"{missing_calls} RAG/BoT trials have no method_calls")

    passed = not reasons
    return {
        "pass": "pass" if passed else "fail",
        "target_trials": len(target_trials),
        "trials_with_calls": trials_with_calls,
        "total_calls": total_calls,
        "error_calls": error_calls,
        "reasons": reasons,
    }


def inspect_run(run_dir: Path) -> dict[str, Any]:
    trials = _load_trials(run_dir)
    rag_result = _check_rag(trials)
    bot_result = _check_bot(trials)
    persistence_result = _check_persistence(trials)
    isolation_result = _check_isolation(trials)
    logging_result = _check_logging(trials)

    report: dict[str, Any] = {
        "rag": rag_result["pass"],
        "bot": bot_result["pass"],
        "persistence": persistence_result["pass"],
        "isolation": isolation_result["pass"],
        "logging": logging_result["pass"],
        "rag_counts": {k: v for k, v in rag_result.items() if k not in {"pass", "reasons"}},
        "bot_counts": {k: v for k, v in bot_result.items() if k not in {"pass", "reasons"}},
        "persistence_counts": {k: v for k, v in persistence_result.items() if k not in {"pass", "reasons"}},
        "isolation_counts": {k: v for k, v in isolation_result.items() if k not in {"pass", "reasons"}},
        "logging_counts": {k: v for k, v in logging_result.items() if k not in {"pass", "reasons"}},
        "reasons": (
            rag_result.get("reasons", [])
            + bot_result.get("reasons", [])
            + persistence_result.get("reasons", [])
            + isolation_result.get("reasons", [])
            + logging_result.get("reasons", [])
        ),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a G0 RAG/BoT replay run for fidelity.")
    parser.add_argument("run_dir", type=Path, help="path to the run directory containing trials.jsonl")
    args = parser.parse_args(argv)

    report = inspect_run(args.run_dir)
    print(json.dumps(report, indent=2))

    checks = [report["rag"], report["bot"], report["persistence"], report["isolation"], report["logging"]]
    return 0 if all(check == "pass" for check in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
