from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

import pytest

from memcontam.experiment.phase12.filter_challenge.freeze_a import (
    FreezeAError,
    build_freeze_a,
    validate_freeze_a,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase12" / "filter_v5_bct_calibration.yaml"
SOURCE_UNIVERSE = ROOT / "data" / "phase12" / "filter_v5_bct_v1" / "source_universe_v1.json"
JsonValue: TypeAlias = str | int | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def _json_object(path: Path) -> dict[str, JsonValue]:
    value: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _object(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


def _array(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list)
    return value


def test_freeze_a_builds_exact_disjoint_pool_and_native_construction_artifacts(tmp_path: Path) -> None:
    result = build_freeze_a(CONFIG, SOURCE_UNIVERSE, tmp_path)
    freeze = _json_object(tmp_path / "freeze_a.json")

    assert result == tmp_path / "freeze_a.json"
    probes = _object(freeze["probes"])
    assert tuple(probes) == ("game24", "math_equation_balancer", "word_sorting")
    assert tuple(len(_array(probe_ids)) for probe_ids in probes.values()) == (6, 6, 6)
    assert len(_array(freeze["control_schedule"])) == 72
    assert len(_array(freeze["method_call_schedule"])) == 90
    assert freeze["provider_calls_issued"] == 0
    assert _json_object(tmp_path / "candidate_triplets_v1.json")["render_count"] == 36
    assert _json_object(tmp_path / "checkpoint_manifest_v1.json")["checkpoint_count"] == 12
    assert _json_object(tmp_path / "ordinary_route_false_manifest_v1.json")["realization_count"] == 12


def test_freeze_a_is_byte_repeatable_and_excludes_known_leakage(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_freeze_a(CONFIG, SOURCE_UNIVERSE, first)
    build_freeze_a(CONFIG, SOURCE_UNIVERSE, second)

    assert {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in sorted(first.rglob("*"))
        if path.is_file()
    } == {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in sorted(second.rglob("*"))
        if path.is_file()
    }
    leakage = _object(_json_object(first / "leakage_disjointness_report.json")["excluded_signatures"])
    assert leakage["game24"] == ["3,3,8,8"]
    assert leakage["math_equation_balancer"] == ["1,2,3,7"]
    assert leakage["word_sorting"] == ["ayz|aza"]


def test_freeze_a_validator_rejects_tampered_manifest_hash_and_leaky_probe(tmp_path: Path) -> None:
    build_freeze_a(CONFIG, SOURCE_UNIVERSE, tmp_path)
    manifest = tmp_path / "probe_construction_manifest_v1.json"
    payload = _json_object(manifest)
    probes = _object(payload["probes"])
    game24 = _array(probes["game24"])
    certificate = _object(_object(game24[0])["certificate"])
    certificate["numbers"] = [3, 3, 8, 8]
    manifest.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    with pytest.raises(FreezeAError, match="FREEZE_A_MANIFEST_HASH_MISMATCH"):
        validate_freeze_a(CONFIG, SOURCE_UNIVERSE, tmp_path)
