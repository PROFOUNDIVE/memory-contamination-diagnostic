from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Final

import yaml


ROOT: Final = Path(__file__).resolve().parents[3]
TASKS: Final = ("game24", "math_equation_balancer", "word_sorting")
BASELINES: Final = ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")
SEEDS: Final = (0, 1, 2, 3)
NOMINAL_CALLS_PER_POSITION: Final = 6
MAXIMUM_CALLS_PER_POSITION: Final = 9


class Phase13CalibrationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def prepare_clean_prefix(config_path: Path, run_id: str, output: Path) -> dict[str, Any]:
    config = load_clean_prefix_config(config_path)
    _validate_run_id(run_id)
    contract = _request_contract(config)
    request = {
        "schema_version": "phase13_clean_prefix_authorization_request_v1",
        "status": "READY_FOR_SEPARATE_PRE_MAIN_CALIBRATION_AUTHORIZATION",
        "scientific_result": False,
        "main_result": False,
        "provider_calls_issued": 0,
        "filter_calls": 0,
        "run_id": run_id,
        "implementation_commit": _git("rev-parse", "HEAD"),
        "tracked_worktree_clean": _execution_worktree_clean(),
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        **contract,
        "execution_command": (
            "python -m memcontam.cli phase13 run-clean-prefix "
            f"--config {config_path} --run-id {run_id} --request {output} "
            "--authorization <authorization.json> "
            "--expected-authorization-sha256 <sha256> --allow-live-calls"
        ),
        "output_manifest_location": str(
            resolve_output_root(config) / run_id / "artifact_manifest.json"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise Phase13CalibrationError("AUTHORIZATION_REQUEST_ALREADY_EXISTS")
    output.write_text(json.dumps(request, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return request


def load_clean_prefix_config(path: Path) -> dict[str, Any]:
    return load_clean_prefix_config_bytes(path.read_bytes())


def load_clean_prefix_config_bytes(raw: bytes) -> dict[str, Any]:
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise Phase13CalibrationError("INVALID_CLEAN_PREFIX_CONFIG")
    if (
        payload.get("config_kind") != "phase13_clean_prefix_calibration_v1"
        or payload.get("baselines") != list(BASELINES)
        or payload.get("suffix_horizon") != 1
        or payload.get("scope")
        != {
            "clean_prefix_only": True,
            "suffix_execution": False,
            "nomem_execution": False,
            "intervention_execution": False,
            "filter_execution": False,
        }
    ):
        raise Phase13CalibrationError("INVALID_CLEAN_PREFIX_CONFIG")
    _input_artifacts(payload)
    _schedule(payload)
    _budget(payload, 44)
    return payload


def resolve_output_root(config: dict[str, Any]) -> Path:
    output = config.get("output")
    if not isinstance(output, dict) or not isinstance(output.get("artifact_root"), str):
        raise Phase13CalibrationError("INVALID_CALIBRATION_OUTPUT_ROOT")
    root = Path(output["artifact_root"])
    return (root if root.is_absolute() else ROOT / root).resolve()


def _request_contract(config: dict[str, Any]) -> dict[str, Any]:
    schedule = _schedule(config)
    return {
        "schedule": schedule,
        "schedule_sha256": _json_hash(schedule),
        "provider_decoding_sha256": _json_hash(
            {"provider": config["provider"], "decoding": config["decoding"]}
        ),
        "budget": _budget(config, schedule["prefix_position_count"]),
        "input_artifacts": _input_artifacts(config),
        "execution_scope": {
            "purpose": "joint_eligibility_route_budget_planning",
            "clean_prefix_only": True,
            "baselines": list(BASELINES),
            "suffix_horizon": 1,
            "nomem_execution": False,
            "intervention_execution": False,
            "filter_execution": False,
        },
    }


def _execution_worktree_clean() -> bool:
    tracked = _git("status", "--porcelain", "--untracked-files=no")
    relevant = _git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "src/memcontam",
        "configs/phase13",
    )
    return not tracked and not relevant


def _schedule(config: dict[str, Any]) -> dict[str, Any]:
    registries = config.get("task_registries")
    schedules = config.get("trajectory_seeds")
    if not isinstance(registries, dict) or not isinstance(schedules, dict):
        raise Phase13CalibrationError("INVALID_CALIBRATION_SCHEDULE")
    rows_by_task: dict[str, tuple[str, ...]] = {}
    frozen: list[dict[str, Any]] = []
    for task in TASKS:
        registry = registries.get(task)
        schedule = schedules.get(task)
        if not isinstance(registry, dict) or not isinstance(schedule, dict):
            raise Phase13CalibrationError("INVALID_CALIBRATION_SCHEDULE")
        rows = tuple(
            json.loads(line)["sample_id"]
            for line in (ROOT / registry["path"]).read_text(encoding="utf-8").splitlines()
            if line
        )
        rows_by_task[task] = rows
        for seed in SEEDS:
            ordered = schedule.get(seed)
            if not isinstance(ordered, list) or tuple(sorted(ordered)) != tuple(sorted(rows)):
                raise Phase13CalibrationError("INVALID_CALIBRATION_SCHEDULE")
            frozen.append({"task": task, "seed": seed, "ordered_task_ids": ordered})
    positions = sum(len(rows_by_task[task]) * len(SEEDS) for task in TASKS)
    return {
        "tasks": list(TASKS),
        "seeds_per_task": len(SEEDS),
        "trajectory_count": len(TASKS) * len(SEEDS),
        "prefix_position_count": positions,
        "trajectories": frozen,
    }


def _budget(config: dict[str, Any], prefix_positions: int) -> dict[str, int]:
    budget = config.get("budget")
    retry = config.get("retry")
    decoding = config.get("decoding")
    if not isinstance(budget, dict) or not isinstance(retry, dict) or not isinstance(decoding, dict):
        raise Phase13CalibrationError("INVALID_CALIBRATION_BUDGET")
    nominal = prefix_positions * NOMINAL_CALLS_PER_POSITION
    maximum = prefix_positions * MAXIMUM_CALLS_PER_POSITION
    attempts = maximum * (1 + retry["retries_after_initial_attempt"])
    input_tokens = attempts * budget["maximum_input_tokens_per_attempt"]
    output_tokens = attempts * decoding["max_output_tokens"]
    microusd = int(
        input_tokens * budget["input_per_1m_tokens"]
        + output_tokens * budget["output_per_1m_tokens"]
    )
    calculated = {
        "nominal_semantic_calls": nominal,
        "maximum_semantic_calls": maximum,
        "maximum_transport_attempts": attempts,
        "maximum_input_tokens": input_tokens,
        "maximum_output_tokens": output_tokens,
        "hard_ceiling_microusd": microusd,
    }
    if any(budget.get(key) != value for key, value in calculated.items()):
        raise Phase13CalibrationError("INVALID_CALIBRATION_BUDGET")
    return calculated


def _input_artifacts(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    task_registries = config.get("task_registries")
    if not isinstance(task_registries, dict):
        raise Phase13CalibrationError("CALIBRATION_INPUT_HASH_MISMATCH")
    values = {
        **{f"task_registry:{task}": task_registries.get(task) for task in TASKS},
        "main_exclusions": config.get("main_exclusions"),
        "clean_corpus": {
            "path": config.get("clean_context", {}).get("corpus_path"),
            "sha256": config.get("clean_context", {}).get("corpus_sha256"),
        },
        "clean_corpus_manifest": {
            "path": config.get("clean_context", {}).get("manifest_path"),
            "sha256": config.get("clean_context", {}).get("manifest_sha256"),
        },
        "readiness_evidence": config.get("readiness_evidence"),
        "embedding_contract": config.get("embedding_contract"),
        "metric_registry": config.get("metric_registry"),
    }
    for identity, record in values.items():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise Phase13CalibrationError("CALIBRATION_INPUT_HASH_MISMATCH")
        path = ROOT / record["path"]
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise Phase13CalibrationError("CALIBRATION_INPUT_HASH_MISMATCH")
        records[identity] = {"path": record["path"], "sha256": record["sha256"]}
    return records


def _validate_run_id(run_id: str) -> None:
    path = Path(run_id)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise Phase13CalibrationError("INVALID_RUN_ID")


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
