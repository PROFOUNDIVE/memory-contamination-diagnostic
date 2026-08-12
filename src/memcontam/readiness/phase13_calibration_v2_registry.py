from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, NamedTuple

from memcontam.main_registry import Task

TASKS: Final[tuple[Task, ...]] = ("game24", "math_equation_balancer", "word_sorting")
ROW_COUNT: Final = 11
SEEDS: Final = tuple(range(10000, 10012))
FORBIDDEN_SELECTOR_FIELDS: Final = frozenset({
    "outcome", "outcomes", "verifier_result", "verifier_results", "eligibility", "eligible",
})


class CalibrationV2Error(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)
SelectionExclusions = NamedTuple(
    "SelectionExclusions", [("pilot", frozenset[str]), ("candidate_control", frozenset[str])]
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationV2Error("REGISTRY_INPUT_MALFORMED") from error
    if not isinstance(value, dict):
        raise CalibrationV2Error("REGISTRY_INPUT_MALFORMED")
    return value


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationV2Error("REGISTRY_INPUT_MALFORMED") from error
    if any(not isinstance(value, dict) for value in values):
        raise CalibrationV2Error("REGISTRY_INPUT_MALFORMED")
    return values


def _signature(row: Mapping[str, object]) -> str:
    signature = row.get("canonical_signature")
    if not isinstance(signature, str) or not signature:
        raise CalibrationV2Error("CANONICAL_SIGNATURE_INVALID")
    return signature


def select_calibration_rows(
    task: str,
    rows: Sequence[Mapping[str, object]],
    exclusions: SelectionExclusions,
) -> tuple[dict[str, object], ...]:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        if FORBIDDEN_SELECTOR_FIELDS.intersection(row):
            raise CalibrationV2Error("SELECTOR_OUTCOME_FIELD_FORBIDDEN")
        signature = _signature(row)
        if signature in exclusions.pilot:
            raise CalibrationV2Error("PILOT_SIGNATURE_OVERLAP")
        if signature in exclusions.candidate_control:
            raise CalibrationV2Error("CANDIDATE_CONTROL_SIGNATURE_OVERLAP")
        if signature in seen:
            raise CalibrationV2Error("CALIBRATION_SIGNATURE_DUPLICATE")
        seen.add(signature)
        if len(selected) < ROW_COUNT:
            payload = dict(row)
            payload["sample_id"] = f"phase13_calibration_v2_{task}_{len(selected) + 1:04d}"
            payload["row_sha256"] = _sha256(_canonical_bytes(payload))
            selected.append(payload)
    if len(selected) != ROW_COUNT:
        raise CalibrationV2Error("TASK_TRAJECTORY_TOO_SHORT")
    return tuple(selected)


def _pilot_signatures(root: Path, task: Task) -> frozenset[str]:
    rows = _load_jsonl(root / f"data/tasks/{task}_pilot.jsonl")
    if task == "game24":
        signatures: set[str] = set()
        for row in rows:
            numbers = row.get("numbers")
            if not isinstance(numbers, list) or not all(isinstance(value, int) for value in numbers):
                raise CalibrationV2Error("PILOT_REGISTRY_INVALID")
            signatures.add(",".join(str(value) for value in sorted(numbers)))
        return frozenset(signatures)
    if task == "math_equation_balancer":
        return frozenset(",".join(re.findall(r"-?\d+", str(row.get("input")))) for row in rows)
    signatures = set()
    for row in rows:
        words = row.get("words")
        if not isinstance(words, list) or not all(isinstance(value, str) for value in words):
            raise CalibrationV2Error("PILOT_REGISTRY_INVALID")
        signatures.add("|".join(sorted(words)))
    return frozenset(signatures)


def _candidate_signatures(root: Path) -> dict[str, str]:
    registry = _load_json(root / "data/phase12/registries/candidate_registry_v1.json")
    triplets = registry.get("triplets")
    if not isinstance(triplets, list) or len(triplets) != len(TASKS):
        raise CalibrationV2Error("CANDIDATE_REGISTRY_INVALID")
    signatures: dict[str, str] = {}
    for triplet in triplets:
        if not isinstance(triplet, dict):
            raise CalibrationV2Error("CANDIDATE_REGISTRY_INVALID")
        task, counterexample = triplet.get("task"), triplet.get("counterexample")
        if task == "game24" and isinstance(counterexample, str):
            values = sorted(int(value) for value in re.findall(r"\d+", counterexample.rsplit("=", 1)[0]))
            signatures[task] = ",".join(str(value) for value in values)
        elif task == "math_equation_balancer" and isinstance(counterexample, str):
            signatures[task] = ",".join(re.findall(r"-?\d+", counterexample))
        elif task == "word_sorting" and isinstance(counterexample, list) and all(isinstance(value, str) for value in counterexample):
            signatures[task] = "|".join(sorted(counterexample))
        else:
            raise CalibrationV2Error("CANDIDATE_REGISTRY_INVALID")
    if set(signatures) != set(TASKS):
        raise CalibrationV2Error("CANDIDATE_REGISTRY_INVALID")
    return signatures


def build_calibration_v2_registry(root: Path, output_root: Path) -> dict[str, object]:
    manifest = _load_json(root / "data/phase13/main/main_registry_manifest_v1.json")
    exclusions_payload = _load_json(root / "data/phase13/main/exclusions_v1.json")
    registries = manifest.get("registries")
    excluded_by_task = exclusions_payload.get("excluded_signatures")
    if not isinstance(registries, dict) or not isinstance(excluded_by_task, dict):
        raise CalibrationV2Error("REGISTRY_INPUT_MALFORMED")
    candidate_path = root / "data/phase12/registries/candidate_registry_v1.json"
    candidate_signatures = _candidate_signatures(root)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks: dict[str, object] = {}
    for task in TASKS:
        source_path = root / f"data/phase13/main/{task}_main_v1.jsonl"
        source_raw = source_path.read_bytes()
        source_ref = registries.get(task)
        task_exclusions = excluded_by_task.get(task)
        if not isinstance(source_ref, dict) or not isinstance(task_exclusions, list):
            raise CalibrationV2Error("REGISTRY_INPUT_MALFORMED")
        expected_hash = source_ref.get("sha256")
        if not isinstance(expected_hash, str) or not all(isinstance(value, str) for value in task_exclusions):
            raise CalibrationV2Error("REGISTRY_INPUT_MALFORMED")
        if _sha256(source_raw) != expected_hash:
            raise CalibrationV2Error("SOURCE_POOL_HASH_MISMATCH")
        source_rows = _load_jsonl(source_path)
        pilot = _pilot_signatures(root, task)
        registered = frozenset(task_exclusions)
        if candidate_signatures[task] not in registered:
            raise CalibrationV2Error("CANDIDATE_CONTROL_SIGNATURE_MISSING")
        selected = select_calibration_rows(
            task, source_rows, SelectionExclusions(pilot, registered - pilot)
        )
        row_raw = b"".join(_canonical_bytes(row) for row in selected)
        (output_root / f"{task}_calibration_v2.jsonl").write_bytes(row_raw)
        sample_ids = tuple(str(row["sample_id"]) for row in selected)
        trajectories = []
        for index, seed in enumerate(SEEDS):
            offset = index % ROW_COUNT
            ordered = sample_ids[offset:] + sample_ids[:offset]
            trajectories.append({
                "seed_id": seed,
                "rotation_offset": offset,
                "ordered_sample_ids": list(ordered),
                "ordered_stream_sha256": _sha256(_canonical_bytes({"ordered_sample_ids": list(ordered)})),
            })
        remainder = source_rows[ROW_COUNT:]
        remainder_signatures = [_signature(row) for row in remainder]
        tasks[task] = {
            "calibration_path": f"{task}_calibration_v2.jsonl",
            "calibration_sha256": _sha256(row_raw),
            "source_main_v1": {"path": str(source_path.relative_to(root)), "sha256": expected_hash},
            "prospective_main_v2": {
                "definition": "source_main_v1_after_calibration_v2_exclusions_in_source_order",
                "source_reference": str(source_path.relative_to(root)),
                "starts_after_calibration_rows": ROW_COUNT,
                "count": len(remainder),
                "ordered_signatures_sha256": _sha256(_canonical_bytes({"signatures": remainder_signatures})),
            },
            "reserved_extension": {
                "count": 0,
                "registry_status": "blocked_pending_future_registry",
                "signatures": [],
            },
            "trajectories": trajectories,
        }
    registry = {
        "schema_version": "phase13_calibration_v2_seed_partition_registry_v1",
        "selection_law": "first_11_source_order_after_registered_signature_exclusions_v1",
        "input_authorities": {
            "main_manifest_sha256": _sha256((root / "data/phase13/main/main_registry_manifest_v1.json").read_bytes()),
            "exclusions_sha256": _sha256((root / "data/phase13/main/exclusions_v1.json").read_bytes()),
            "candidate_registry_sha256": _sha256(candidate_path.read_bytes()),
            "pilot_registry_sha256": {
                task: _sha256((root / f"data/tasks/{task}_pilot.jsonl").read_bytes())
                for task in TASKS
            },
        },
        "evidence_layer_seeds": {
            "historical_calibration_v1": [0, 1, 2, 3],
            "calibration_v2": list(SEEDS),
            "prospective_main_v2": "pending_future_registry",
            "reserved_extension": [],
        },
        "tasks": tasks,
    }
    (output_root / "seed_partition_registry_v1.json").write_text(
        json.dumps(registry, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    validate_calibration_v2_registry(output_root)
    return registry


def validate_calibration_v2_registry(output_root: Path) -> dict[str, object]:
    registry = _load_json(output_root / "seed_partition_registry_v1.json")
    tasks = registry.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(TASKS):
        raise CalibrationV2Error("TASK_PARTITION_INVALID")
    for task in TASKS:
        task_registry = tasks[task]
        rows = _load_jsonl(output_root / f"{task}_calibration_v2.jsonl")
        if len(rows) != ROW_COUNT:
            raise CalibrationV2Error("TASK_TRAJECTORY_TOO_SHORT")
        row_raw = b"".join(_canonical_bytes(row) for row in rows)
        if _sha256(row_raw) != task_registry["calibration_sha256"]:
            raise CalibrationV2Error("STALE_GENERATED_STATE")
        sample_ids = [str(row["sample_id"]) for row in rows]
        signatures = [_signature(row) for row in rows]
        if len(set(signatures)) != ROW_COUNT:
            raise CalibrationV2Error("CALIBRATION_SIGNATURE_DUPLICATE")
        for row in rows:
            row_hash = row.get("row_sha256")
            unhashed = {key: value for key, value in row.items() if key != "row_sha256"}
            if row_hash != _sha256(_canonical_bytes(unhashed)):
                raise CalibrationV2Error("ROW_HASH_MISMATCH")
        remainder = task_registry["prospective_main_v2"]
        if remainder["starts_after_calibration_rows"] != ROW_COUNT:
            raise CalibrationV2Error("MAIN_V2_REFERENCE_INVALID")
        reserved = task_registry["reserved_extension"]
        if reserved != {"count": 0, "registry_status": "blocked_pending_future_registry", "signatures": []}:
            raise CalibrationV2Error("RESERVED_EXTENSION_NOT_EMPTY")
        trajectories = task_registry["trajectories"]
        seeds = [trajectory["seed_id"] for trajectory in trajectories]
        if len(seeds) != len(set(seeds)):
            raise CalibrationV2Error("DUPLICATE_SEED_ID")
        if seeds != list(SEEDS):
            raise CalibrationV2Error("SEED_PARTITION_INVALID")
        for trajectory in trajectories:
            offset = trajectory["rotation_offset"]
            if not isinstance(offset, int) or not 0 <= offset < ROW_COUNT:
                raise CalibrationV2Error("ROTATION_OFFSET_INVALID")
            ordered = trajectory["ordered_sample_ids"]
            if len(ordered) != ROW_COUNT or len(set(ordered)) != ROW_COUNT:
                raise CalibrationV2Error("TASK_TRAJECTORY_TOO_SHORT")
            if ordered != sample_ids[offset:] + sample_ids[:offset]:
                raise CalibrationV2Error("TRAJECTORY_ORDER_INVALID")
            expected_stream_hash = _sha256(_canonical_bytes({"ordered_sample_ids": ordered}))
            if trajectory["ordered_stream_sha256"] != expected_stream_hash:
                raise CalibrationV2Error("ORDERED_STREAM_HASH_MISMATCH")
    return registry
