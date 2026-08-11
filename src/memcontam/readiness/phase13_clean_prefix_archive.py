from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from memcontam.readiness.phase13_clean_prefix import BASELINES, TASKS


def append_trajectory_records(
    task: str,
    seed: int,
    result: Any,
    trials: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    eligibility: list[dict[str, Any]],
) -> None:
    for baseline in BASELINES:
        for position, trial in enumerate(result.trial_results_by_baseline[baseline], start=1):
            outcome = trial.outcome
            trials.append(
                {
                    "task": task,
                    "seed": seed,
                    "baseline": baseline,
                    "position": position,
                    "status": outcome.status,
                    "verifier_result": bool(outcome.verifier_result),
                }
            )
            calls.extend(
                {
                    "task": task,
                    "seed": seed,
                    "baseline": baseline,
                    "position": position,
                    **call.model_dump(mode="json"),
                }
                for call in outcome.method_calls
            )
        checkpoints.extend(
            {
                "task": task,
                "seed": seed,
                "baseline": baseline,
                "checkpoint_id": checkpoint.identity.checkpoint_id,
                "checkpoint_sha256": checkpoint.canonical_sha256,
                "state": checkpoint.state.to_mapping(),
            }
            for checkpoint in result.checkpoints_by_baseline[baseline]
        )
    eligibility.append(
        {
            "task": task,
            "seed": seed,
            "eligible": not result.selection.blocked,
            "joint_eligible_indices": list(
                result.selection.joint_eligibility.joint_eligible_indices
            ),
            "selected_trial_index": result.selection.selected_trial_index,
            "block_reason": result.selection.block_reason,
            "decisions": [
                {
                    "family": decision.baseline_family,
                    "checkpoint_index": decision.checkpoint_index,
                    "eligible": decision.eligible,
                    "reason_codes": list(decision.reason_codes),
                }
                for decision in result.selection.decisions
            ],
        }
    )


def rates(seed_status: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        task_rows = [row for row in seed_status if row["task"] == task]
        eligible = sum(row["status"] == "eligible" for row in task_rows)
        attempted = len(task_rows)
        result[task] = {
            "eligible": eligible,
            "attempted": attempted,
            "rate": f"{eligible}/{attempted}",
        }
    return result


def write_archive(
    run_dir: Path,
    config_bytes: bytes,
    config: dict[str, Any],
    run_id: str,
    records: dict[str, list[dict[str, Any]]],
    frozen_rates: dict[str, dict[str, Any]],
    accounting: dict[str, Any],
    request_bytes: bytes,
    authorization_bytes: bytes,
    *,
    status: str = "completed",
) -> None:
    _write_json(run_dir / "run.json", {"run_id": run_id, "status": status})
    _write_json(run_dir / "resolved_config.json", config)
    _write_json(run_dir / "schedule.json", config["trajectory_seeds"])
    for name, rows in records.items():
        _write_jsonl(run_dir / f"{name}.jsonl", rows)
    _write_json(run_dir / "rates.json", frozen_rates)
    _write_json(run_dir / "accounting.json", accounting)
    (run_dir / "authorized_config.yaml").write_bytes(config_bytes)
    (run_dir / "authorization_request.json").write_bytes(request_bytes)
    (run_dir / "authorization.json").write_bytes(authorization_bytes)
    artifacts = {
        path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(run_dir.iterdir())
    }
    _write_json(run_dir / "artifact_manifest.json", {"status": status, "artifacts": artifacts})
    _write_json(
        run_dir / "archive_seal.json",
        {
            "status": status,
            "artifact_manifest_sha256": _sha256(run_dir / "artifact_manifest.json"),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        },
    )
def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
