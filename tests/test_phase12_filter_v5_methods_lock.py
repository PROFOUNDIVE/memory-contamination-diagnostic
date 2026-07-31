from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_phase12_filter_v5_methods_lock.py"
DOCUMENT = ROOT / "docs" / "phase12-filter-v5-bct-methods-lock.md"
CONFIG = ROOT / "configs" / "phase12" / "filter_v5_bct_calibration.yaml"
PLAN = ROOT / ".omo" / "plans" / "phase12-post-filter-v5-calibration-readiness.md"


def _run(document: Path, config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--document",
            str(document),
            "--config",
            str(config),
            "--plan",
            str(PLAN),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_methods_lock_validates_frozen_grid_and_budget() -> None:
    result = _run(DOCUMENT, CONFIG)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "METHODS_LOCK_VALID" in result.stdout


def test_methods_lock_rejects_hard_ceiling_mutation(tmp_path: Path) -> None:
    config = tmp_path / "filter_v5_bct_calibration.yaml"
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["budget"]["shared"]["hard_ceiling_usd"] = 11
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = _run(DOCUMENT, config)

    assert result.returncode != 0
    assert "METHODS_BUDGET_MISMATCH" in result.stdout


def test_methods_lock_rejects_call_capacity_mutation(tmp_path: Path) -> None:
    config = tmp_path / "filter_v5_bct_calibration.yaml"
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["budget"]["bct"]["native_call_multiplier"] = 6
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = _run(DOCUMENT, config)

    assert result.returncode != 0
    assert "METHODS_CALL_RESERVATION_MISMATCH" in result.stdout


def test_methods_lock_rejects_noncontradicted_as_safe(tmp_path: Path) -> None:
    document = tmp_path / "methods.md"
    shutil.copyfile(DOCUMENT, document)
    document.write_text(
        document.read_text(encoding="utf-8").replace(
            "not_contradicted -> active", "not_contradicted -> safe"
        ),
        encoding="utf-8",
    )

    result = _run(document, CONFIG)

    assert result.returncode != 0
    assert "METHODS_DECISION_LAW_MISMATCH" in result.stdout


def test_methods_lock_rejects_missing_no_pooling_law(tmp_path: Path) -> None:
    document = tmp_path / "methods.md"
    shutil.copyfile(DOCUMENT, document)
    document.write_text(
        document.read_text(encoding="utf-8").replace("No pooling across challenge, Main, or code", ""),
        encoding="utf-8",
    )

    result = _run(document, CONFIG)

    assert result.returncode != 0
    assert "METHODS_NO_POOLING_LAW_MISSING" in result.stdout
