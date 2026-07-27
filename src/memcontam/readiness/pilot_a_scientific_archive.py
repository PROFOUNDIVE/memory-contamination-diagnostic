from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


STREAMS = (
    "trials.jsonl",
    "calls.jsonl",
    "retrieval_events.jsonl",
    "context_events.jsonl",
    "failures.jsonl",
    "memory_events.jsonl",
    "admission_events.jsonl",
    "intervention_events.jsonl",
    "checkpoint_events.jsonl",
    "eligibility_events.jsonl",
    "seed_status.jsonl",
    "audit/audit_labels.jsonl",
)
SIDECARS = (
    "run.json",
    "resolved_config.json",
    "provider_profile.json",
    "decision_ledger.json",
)
_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer\s|sk-[a-z0-9]{8,}|password|secret)"
)
_SCIENTIFIC_MANIFEST_STATUSES = frozenset({"completed", "blocked", "invalidated", "interrupted"})


class ScientificArchiveError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def write_scientific_archive(run_dir: Path, artifacts: dict[str, Any]) -> None:
    expected = {*STREAMS, *SIDECARS}
    if set(artifacts) != expected:
        raise ScientificArchiveError("SCIENTIFIC_ARCHIVE_ARTIFACT_SET_INVALID")
    (run_dir / "audit").mkdir(parents=True, exist_ok=True)
    for filename in STREAMS:
        _write_jsonl(run_dir / filename, artifacts[filename])
    for filename in SIDECARS:
        _write_json(run_dir / filename, artifacts[filename])
    run_status = artifacts["run.json"].get("status") if isinstance(artifacts["run.json"], dict) else None
    manifest = {
        "status": run_status if run_status in _SCIENTIFIC_MANIFEST_STATUSES else "completed",
        "artifacts": {
            filename: {
                "count": _count(run_dir / filename),
                "sha256": _sha256(run_dir / filename),
            }
            for filename in (*SIDECARS, *STREAMS)
        },
    }
    _write_json(run_dir / "public_artifact_manifest.json", manifest)
    _write_json(
        run_dir / "archive_seal.json",
        {"public_artifact_manifest_sha256": _sha256(run_dir / "public_artifact_manifest.json")},
    )


def validate_scientific_archive(run_dir: Path) -> dict[str, Any]:
    required = (*SIDECARS, *STREAMS, "public_artifact_manifest.json", "archive_seal.json")
    if any(not (run_dir / filename).is_file() for filename in required):
        return _failure(run_dir, "REQUIRED_ARTIFACT_MISSING")
    manifest = _read_json(run_dir / "public_artifact_manifest.json")
    if not isinstance(manifest, dict) or manifest.get("status") not in _SCIENTIFIC_MANIFEST_STATUSES:
        return _failure(run_dir, "ARCHIVE_MANIFEST_INVALID")
    records = manifest.get("artifacts")
    if not isinstance(records, dict) or set(records) != {*SIDECARS, *STREAMS}:
        return _failure(run_dir, "ARCHIVE_MANIFEST_INVALID")
    mismatches = sum(
        not isinstance(record, dict)
        or record.get("sha256") != _sha256(run_dir / filename)
        or record.get("count") != _count(run_dir / filename)
        for filename, record in records.items()
    )
    if mismatches:
        return _failure(run_dir, "ARCHIVE_HASH_MISMATCH", mismatches)
    seal = _read_json(run_dir / "archive_seal.json")
    if not isinstance(seal, dict) or seal.get("public_artifact_manifest_sha256") != _sha256(
        run_dir / "public_artifact_manifest.json"
    ):
        return _failure(run_dir, "ARCHIVE_SEAL_MISMATCH")
    payload = {filename: _read(run_dir / filename) for filename in (*SIDECARS, *STREAMS)}
    if _SECRET_PATTERN.search(json.dumps(payload, sort_keys=True)):
        return _failure(run_dir, "SECRET_LIKE_VALUE_FORBIDDEN")
    run = payload["run.json"]
    provider = payload["provider_profile.json"]
    ledger = payload["decision_ledger.json"]
    trials = payload["trials.jsonl"]
    calls = payload["calls.jsonl"]
    seed_status = payload["seed_status.jsonl"]
    if not all(
        isinstance(value, expected)
        for value, expected in (
            (run, dict),
            (provider, dict),
            (ledger, dict),
            (trials, list),
            (calls, list),
            (seed_status, list),
        )
    ):
        return _failure(run_dir, "ARCHIVE_JSON_INVALID")
    if (
        run.get("scientific_result") is not True
        or run.get("evidence_layer") != "calibration"
        or run.get("run_family") != "pilot_a"
        or provider.get("provider") != "openai_responses"
    ):
        return _failure(run_dir, "SCIENTIFIC_SCOPE_INVALID")
    if run.get("status") != "completed" and not isinstance(run.get("status_reason"), str):
        return _failure(run_dir, "TERMINAL_STATUS_REASON_MISSING")
    trial_ids = {row.get("trial_id") for row in trials if isinstance(row, dict)}
    if None in trial_ids or len(trial_ids) != len(trials):
        return _failure(run_dir, "TRIAL_REFERENCE_INVALID")
    if any(not isinstance(row, dict) or row.get("trial_id") not in trial_ids for row in calls):
        return _failure(run_dir, "CALL_REFERENCE_INVALID")
    event_files = tuple(filename for filename in STREAMS if filename.endswith("events.jsonl"))
    if any(
        not isinstance(row, dict) or row.get("trial_id") not in trial_ids
        for filename in event_files
        for row in payload[filename]
    ):
        return _failure(run_dir, "EVENT_REFERENCE_INVALID")
    cost_total = sum(float(row.get("cost_usd", 0.0)) for row in calls)
    retry_total = sum(int(row.get("retry_count", 0)) for row in calls)
    if (
        cost_total >= float(ledger.get("hard_cost_ceiling_usd", 0.0))
        or len(calls) != ledger.get("live_provider_calls")
        or retry_total != ledger.get("retry_total")
        or abs(cost_total - float(ledger.get("cost_total", -1.0))) > 1e-12
    ):
        return _failure(run_dir, "OPERATIONS_RECONCILIATION_FAILED")
    prefix_trials = sum(
        isinstance(row, dict) and row.get("trial_kind") == "branch_free_prefix" for row in trials
    )
    if (
        ledger.get("prefix") != {"completed_trials": prefix_trials}
        or ledger.get("eligibility") != payload["eligibility_events.jsonl"]
        or ledger.get("joint") != seed_status
        or ledger.get("failure") != payload["failures.jsonl"]
        or not isinstance(ledger.get("provenance"), dict)
        or ledger["provenance"].get("run_id") != run.get("run_id")
        or ledger["provenance"].get("parent_run_id") != run.get("parent_run_id")
        or ledger["provenance"].get("implementation_commit") != run.get("implementation_commit")
        or ledger.get("eligible_seeds")
        != [row.get("seed") for row in seed_status if isinstance(row, dict) and row.get("eligible")]
        or any(
            not isinstance(row, dict) or row.get("fallback_checkpoint_used") is not False
            for row in seed_status
        )
    ):
        return _failure(run_dir, "DECISION_LEDGER_RECONCILIATION_FAILED")
    engineering_failures = [
        row
        for row in payload["failures.jsonl"]
        if isinstance(row, dict) and row.get("failure_kind") == "engineering"
    ]
    if engineering_failures:
        return _failure(run_dir, "ENGINEERING_FAILURE_PRESENT")
    return {
        "cost_total": cost_total,
        "eligible_seeds": ledger["eligible_seeds"],
        "hash_mismatches": 0,
        "live_provider_calls": len(calls),
        "manifest_hash": _sha256(run_dir / "public_artifact_manifest.json"),
        "overall": "pass",
        "reason_code": None,
        "retry_total": retry_total,
        "run_dir": str(run_dir),
        "scientific_result": True,
        "trajectory_seeds": ledger["trajectory_seeds"],
        "unresolved_references": 0,
    }


def _failure(run_dir: Path, code: str, mismatches: int = 0) -> dict[str, Any]:
    return {
        "hash_mismatches": mismatches,
        "overall": "fail",
        "reason_code": code,
        "run_dir": str(run_dir),
        "scientific_result": True,
        "unresolved_references": 1,
    }


def _read(path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return _read_json(path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines()) if path.suffix == ".jsonl" else 1


__all__ = [
    "ScientificArchiveError",
    "validate_scientific_archive",
    "write_scientific_archive",
]
