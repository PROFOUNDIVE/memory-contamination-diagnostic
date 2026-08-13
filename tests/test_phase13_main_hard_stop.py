from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/phase13/pre_main_calibration_v2.yaml"
ALIASES = ("main", "main-a", "run-main", "run-main-a", "authorize-main", "request-main")


@pytest.mark.parametrize("alias", ALIASES)
def test_main_aliases_are_terminally_forbidden_without_artifacts(alias: str, tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    environment.pop("OPENAI_API_KEY", None)

    result = subprocess.run(
        (sys.executable, "-m", "memcontam.cli", "phase13", alias),
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert result.stderr.rstrip().endswith("MAIN_A_EXECUTION_FORBIDDEN")
    assert tuple(tmp_path.iterdir()) == ()
    assert not (ROOT / "runs/phase13-main-a").exists()


def test_no_report_external_block_emits_main_prohibition() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    environment.pop("OPENAI_API_KEY", None)

    result = subprocess.run(
        (
            sys.executable, "-m", "memcontam.cli", "phase13", "run-calibration-v2",
            "--config", CONFIG,
        ),
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert result.stderr.splitlines() == [
        "CALIBRATION_V2_EXTERNAL_BLOCK",
        "MAIN_A_EXECUTION_FORBIDDEN",
    ]
    assert not (ROOT / "runs/phase13-calibration-v2").exists()
    assert not (ROOT / "runs/phase13-main-a").exists()
