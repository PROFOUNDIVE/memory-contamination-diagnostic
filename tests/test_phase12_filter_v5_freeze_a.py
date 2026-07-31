from __future__ import annotations

import json
from pathlib import Path

import pytest

from memcontam.experiment.phase12.filter_challenge.freeze_a import (
    FreezeAError,
    build_freeze_a,
    validate_freeze_a,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase12" / "filter_v5_bct_calibration.yaml"
SOURCE_UNIVERSE = ROOT / "data" / "phase12" / "filter_v5_bct_v1" / "source_universe_v1.json"


def test_freeze_a_builds_exact_disjoint_pool_and_native_construction_artifacts(tmp_path: Path) -> None:
    result = build_freeze_a(CONFIG, SOURCE_UNIVERSE, tmp_path)
    freeze = json.loads((tmp_path / "freeze_a.json").read_text(encoding="utf-8"))

    assert result == tmp_path / "freeze_a.json"
    assert tuple(freeze["probes"]) == ("game24", "math_equation_balancer", "word_sorting")
    assert tuple(len(probe_ids) for probe_ids in freeze["probes"].values()) == (6, 6, 6)
    assert len(freeze["control_schedule"]) == 72
    assert len(freeze["method_call_schedule"]) == 90
    assert freeze["provider_calls_issued"] == 0
    assert json.loads((tmp_path / "candidate_triplets_v1.json").read_text(encoding="utf-8"))["render_count"] == 36
    assert json.loads((tmp_path / "checkpoint_manifest_v1.json").read_text(encoding="utf-8"))["checkpoint_count"] == 12
    assert json.loads((tmp_path / "ordinary_route_false_manifest_v1.json").read_text(encoding="utf-8"))["realization_count"] == 12


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
    leakage = json.loads((first / "leakage_disjointness_report.json").read_text(encoding="utf-8"))
    assert leakage["excluded_signatures"]["game24"] == ["3,3,8,8"]
    assert leakage["excluded_signatures"]["math_equation_balancer"] == ["1,2,3,7"]
    assert leakage["excluded_signatures"]["word_sorting"] == ["ayz|aza"]


def test_freeze_a_validator_rejects_tampered_manifest_hash_and_leaky_probe(tmp_path: Path) -> None:
    build_freeze_a(CONFIG, SOURCE_UNIVERSE, tmp_path)
    manifest = tmp_path / "probe_construction_manifest_v1.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["probes"]["game24"][0]["certificate"]["numbers"] = [3, 3, 8, 8]
    manifest.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    with pytest.raises(FreezeAError, match="FREEZE_A_MANIFEST_HASH_MISMATCH"):
        validate_freeze_a(CONFIG, SOURCE_UNIVERSE, tmp_path)
