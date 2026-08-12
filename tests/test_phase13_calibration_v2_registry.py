from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from memcontam.readiness.phase13_calibration_v2_registry import (
    CalibrationV2Error,
    SelectionExclusions,
    build_calibration_v2_registry,
    select_calibration_rows,
    validate_calibration_v2_registry,
)
from memcontam.readiness.phase13_calibration_v2_authority import pilot_signatures

ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = ROOT / "data/phase13/main"
OUTPUT_ROOT = ROOT / "data/phase13/calibration_v2"
TASKS = ("game24", "math_equation_balancer", "word_sorting")
SOURCE_HASHES = {
    "game24": "ae682f138d8035fc1de9382eb8903730d392851def720351a78846df160b615f",
    "math_equation_balancer": "dfa07c8c3ada1b0030a735cca97022f98dfb8da30d8ce86f82013eb51b4a7037",
    "word_sorting": "e7ff0507512af4e71ae027a5226984b175d9b75dca898df79ca88535326c9c54",
}
AUTHORITY_HASHES = {
    "candidate": "7dc76a6816a1d3d641a38db71c9152eccf9fe290b4dfdc9d4fd19f62a73ceef8",
    "game24_pilot": "7a7a39f8697ccd7199e063679de90a373535f912a04ca6c15d1e919f0749e8b5",
    "math_equation_balancer_pilot": "6fa5a5d3be52853f8d9da93a9a9c0ea5399f67c9c08acc64fdbdd4821f68bb41",
    "word_sorting_pilot": "da3f3bb073b55f20690005b25eb7f3a88d1e1ea4ed9a262e7071f902eeda91f8",
}


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _resign_row(output: Path, task: str, mutation: dict[str, object]) -> None:
    row_path = output / f"{task}_calibration_v2.jsonl"
    rows = _read_jsonl(row_path)
    rows[0].update(mutation)
    unhashed = {key: value for key, value in rows[0].items() if key != "row_sha256"}
    rows[0]["row_sha256"] = hashlib.sha256(
        (json.dumps(unhashed, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    raw = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    row_path.write_bytes(raw)
    registry_path = output / "seed_partition_registry_v1.json"
    registry = _read_json(registry_path)
    tasks = registry["tasks"]
    assert isinstance(tasks, dict) and isinstance(tasks[task], dict)
    tasks[task]["calibration_sha256"] = hashlib.sha256(raw).hexdigest()
    registry_path.write_text(json.dumps(registry, sort_keys=True, indent=2) + "\n")


def _copy_authorities(tmp_path: Path) -> Path:
    root = tmp_path / "authority"
    for relative in (
        "data/phase13/main",
        "data/tasks",
        "data/phase12/registries",
    ):
        shutil.copytree(ROOT / relative, root / relative)
    return root


def test_current_main_v1_source_bytes_and_signatures_are_immutable() -> None:
    for task, expected_hash in SOURCE_HASHES.items():
        source = MAIN_ROOT / f"{task}_main_v1.jsonl"
        rows = _read_jsonl(source)

        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash
        assert len({row["canonical_signature"] for row in rows}) == len(rows)
        source_order = [row["source_row"] for row in rows]
        assert all(isinstance(value, int) for value in source_order)
        assert source_order == sorted(value for value in source_order if isinstance(value, int))
    assert hashlib.sha256(
        (ROOT / "data/phase12/registries/candidate_registry_v1.json").read_bytes()
    ).hexdigest() == AUTHORITY_HASHES["candidate"]
    for task in TASKS:
        assert hashlib.sha256(
            (ROOT / f"data/tasks/{task}_pilot.jsonl").read_bytes()
        ).hexdigest() == AUTHORITY_HASHES[f"{task}_pilot"]


def test_build_is_byte_identical_and_freezes_expected_partitions(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_calibration_v2_registry(ROOT, first)
    build_calibration_v2_registry(ROOT, second)

    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    registry = validate_calibration_v2_registry(first)
    tasks = registry["tasks"]
    assert isinstance(tasks, dict)
    for task in TASKS:
        rows = _read_jsonl(first / f"{task}_calibration_v2.jsonl")
        task_registry = tasks[task]
        assert isinstance(task_registry, dict)
        trajectories = task_registry["trajectories"]

        assert len(rows) == 11
        assert len({row["canonical_signature"] for row in rows}) == 11
        source_rows = _read_jsonl(MAIN_ROOT / f"{task}_main_v1.jsonl")
        assert [row["source_row"] for row in rows] == [row["source_row"] for row in source_rows[:11]]
        assert {row["canonical_signature"] for row in rows}.isdisjoint(
            {row["canonical_signature"] for row in source_rows[11:]}
        )
        assert [trajectory["seed_id"] for trajectory in trajectories] == list(range(10000, 10012))
        assert all(len(set(trajectory["ordered_sample_ids"])) == 11 for trajectory in trajectories)
        assert all(set(trajectory["ordered_sample_ids"]) == {row["sample_id"] for row in rows} for trajectory in trajectories)
        assert task_registry["prospective_main_v2"]["count"] >= 11
        assert task_registry["reserved_extension"] == {
            "count": 0,
            "registry_status": "blocked_pending_future_registry",
            "signatures": [],
        }


@pytest.mark.parametrize(
    ("task", "signature", "expected_code"),
    [
        ("game24", "1,3,4,6", "PILOT_SIGNATURE_OVERLAP"),
        ("math_equation_balancer", "1,2,3,7", "CANDIDATE_CONTROL_SIGNATURE_OVERLAP"),
    ],
)
def test_selector_rejects_pilot_or_candidate_control_duplication(
    task: str, signature: str, expected_code: str
) -> None:
    rows = _read_jsonl(MAIN_ROOT / f"{task}_main_v1.jsonl")
    rows[0]["canonical_signature"] = signature
    exclusions = SelectionExclusions(
        pilot=frozenset({"1,3,4,6"}),
        candidate_control=frozenset({"1,2,3,7"}),
    )

    with pytest.raises(CalibrationV2Error, match=expected_code):
        select_calibration_rows(task, rows, exclusions)


def test_selector_rejects_outcome_fields() -> None:
    rows = _read_jsonl(MAIN_ROOT / "game24_main_v1.jsonl")
    rows[0]["outcome"] = "forbidden"

    with pytest.raises(CalibrationV2Error, match="SELECTOR_OUTCOME_FIELD_FORBIDDEN"):
        select_calibration_rows(
            "game24", rows, SelectionExclusions(frozenset(), frozenset())
        )


def test_meb_pilot_signature_includes_target_value() -> None:
    assert pilot_signatures(ROOT, "math_equation_balancer") == {
        "2,5,7",
        "3,6,18",
        "9,4,5",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        {"metadata": {"outcome": "forbidden"}},
        {"metadata": {"verifier_result": "forbidden"}},
        {"metadata": {"eligibility": True}},
        {"metadata": {"success_rate": 1.0}},
    ],
)
def test_validator_rejects_resigned_recursive_semantic_fields(
    tmp_path: Path, mutation: dict[str, object]
) -> None:
    output = tmp_path / "registry"
    build_calibration_v2_registry(ROOT, output)
    _resign_row(output, "game24", mutation)

    with pytest.raises(CalibrationV2Error, match="FORBIDDEN_SEMANTIC_FIELD"):
        validate_calibration_v2_registry(output, ROOT)


@pytest.mark.parametrize(
    ("task", "signature", "expected_code"),
    [
        ("game24", "1,3,4,6", "PILOT_SIGNATURE_OVERLAP"),
        ("math_equation_balancer", "1,2,3,7", "CANDIDATE_CONTROL_SIGNATURE_OVERLAP"),
        ("game24", "1,2,2,9", "PROSPECTIVE_MAIN_V2_SIGNATURE_OVERLAP"),
    ],
)
def test_validator_rejects_resigned_authority_layer_overlap(
    tmp_path: Path, task: str, signature: str, expected_code: str
) -> None:
    output = tmp_path / "registry"
    build_calibration_v2_registry(ROOT, output)
    _resign_row(output, task, {"canonical_signature": signature})

    with pytest.raises(CalibrationV2Error, match=expected_code):
        validate_calibration_v2_registry(output, ROOT)


def test_validator_rejects_resigned_reserved_signature(tmp_path: Path) -> None:
    output = tmp_path / "registry"
    build_calibration_v2_registry(ROOT, output)
    registry_path = output / "seed_partition_registry_v1.json"
    registry = _read_json(registry_path)
    tasks = registry["tasks"]
    assert isinstance(tasks, dict) and isinstance(tasks["game24"], dict)
    tasks["game24"]["reserved_extension"] = {
        "count": 1,
        "registry_status": "blocked_pending_future_registry",
        "signatures": ["2,5,8,11"],
    }
    registry_path.write_text(json.dumps(registry, sort_keys=True, indent=2) + "\n")

    with pytest.raises(CalibrationV2Error, match="RESERVED_EXTENSION_SIGNATURE_OVERLAP"):
        validate_calibration_v2_registry(output, ROOT)


def test_validator_rejects_claimed_authority_hash_without_source_match(tmp_path: Path) -> None:
    output = tmp_path / "registry"
    build_calibration_v2_registry(ROOT, output)
    registry_path = output / "seed_partition_registry_v1.json"
    registry = _read_json(registry_path)
    authorities = registry["input_authorities"]
    assert isinstance(authorities, dict)
    authorities["candidate_registry_sha256"] = "0" * 64
    registry_path.write_text(json.dumps(registry, sort_keys=True, indent=2) + "\n")

    with pytest.raises(CalibrationV2Error, match="INPUT_AUTHORITY_HASH_MISMATCH"):
        validate_calibration_v2_registry(output, ROOT)


def test_validator_rejects_main_source_bytes_drift(tmp_path: Path) -> None:
    authority_root = _copy_authorities(tmp_path)
    output = tmp_path / "registry"
    build_calibration_v2_registry(ROOT, output)
    source = authority_root / "data/phase13/main/game24_main_v1.jsonl"
    source.write_bytes(source.read_bytes() + b"\n")

    with pytest.raises(CalibrationV2Error, match="MAIN_SOURCE_HASH_MISMATCH"):
        validate_calibration_v2_registry(output, authority_root)


def test_validator_rejects_resigned_source_payload_drift(tmp_path: Path) -> None:
    output = tmp_path / "registry"
    build_calibration_v2_registry(ROOT, output)
    _resign_row(output, "game24", {"target": 99})

    with pytest.raises(CalibrationV2Error, match="CALIBRATION_SOURCE_ROW_MISMATCH"):
        validate_calibration_v2_registry(output, ROOT)


@pytest.mark.parametrize(
    "claim",
    [
        {"path": "data/phase13/main/word_sorting_main_v1.jsonl", "sha256": SOURCE_HASHES["game24"]},
        {"path": "data/phase13/main/game24_main_v1.jsonl", "sha256": "0" * 64},
    ],
)
def test_validator_rejects_source_claim_drift(
    tmp_path: Path, claim: dict[str, str]
) -> None:
    output = tmp_path / "registry"
    build_calibration_v2_registry(ROOT, output)
    registry_path = output / "seed_partition_registry_v1.json"
    registry = _read_json(registry_path)
    tasks = registry["tasks"]
    assert isinstance(tasks, dict) and isinstance(tasks["game24"], dict)
    tasks["game24"]["source_main_v1"] = claim
    registry_path.write_text(json.dumps(registry, sort_keys=True, indent=2) + "\n")

    with pytest.raises(CalibrationV2Error, match="SOURCE_MAIN_V1_CLAIM_MISMATCH"):
        validate_calibration_v2_registry(output, ROOT)


@pytest.mark.parametrize("alias", ["outcomeRate", "verifier_outcome", "eligibility_status"])
def test_validator_rejects_resigned_semantic_aliases(tmp_path: Path, alias: str) -> None:
    output = tmp_path / "registry"
    build_calibration_v2_registry(ROOT, output)
    _resign_row(output, "game24", {"metadata": {alias: 1}})

    with pytest.raises(CalibrationV2Error, match="FORBIDDEN_SEMANTIC_FIELD"):
        validate_calibration_v2_registry(output, ROOT)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("short_task", "TASK_TRAJECTORY_TOO_SHORT"),
        ("duplicate_seed", "DUPLICATE_SEED_ID"),
        ("order_wrap", "ROTATION_OFFSET_INVALID"),
    ],
)
def test_registry_mutations_fail_closed(
    tmp_path: Path, mutation: str, expected_code: str
) -> None:
    output = tmp_path / "registry"
    build_calibration_v2_registry(ROOT, output)
    seed_path = output / "seed_partition_registry_v1.json"
    payload = _read_json(seed_path)
    tasks = payload["tasks"]
    assert isinstance(tasks, dict)
    game24 = tasks["game24"]
    assert isinstance(game24, dict)

    if mutation == "short_task":
        game24["trajectories"][0]["ordered_sample_ids"] = game24["trajectories"][0][
            "ordered_sample_ids"
        ][:10]
    elif mutation == "duplicate_seed":
        game24["trajectories"][1]["seed_id"] = game24["trajectories"][0]["seed_id"]
    else:
        game24["trajectories"][0]["rotation_offset"] = 11
    seed_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(CalibrationV2Error, match=expected_code):
        validate_calibration_v2_registry(output)


def test_committed_registry_rebuilds_exactly(tmp_path: Path) -> None:
    rebuilt = tmp_path / "rebuilt"

    build_calibration_v2_registry(ROOT, rebuilt)

    assert {path.name: path.read_bytes() for path in rebuilt.iterdir()} == {
        path.name: path.read_bytes() for path in OUTPUT_ROOT.iterdir()
    }
    validate_calibration_v2_registry(OUTPUT_ROOT)
