from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from memcontam.experiment.phase12 import cli as phase12_cli


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase12" / "pilot_a_game24_minimal.yaml"


def _module() -> Any:
    assert importlib.util.find_spec("memcontam.readiness.pilot_a_invariants") is not None
    return importlib.import_module("memcontam.readiness.pilot_a_invariants")


def test_run_replay_and_archive_output_are_offline_and_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    run_dir = module.run_replay(CONFIG, "pilot-a", artifact_root=tmp_path)
    output = tmp_path / "archive.json"
    args = argparse.Namespace(replay=None, run_dir=run_dir, output=output)

    report = phase12_cli._validate_archive(args)

    assert report["overall"] == "pass"
    assert report["unresolved_references"] == 0
    assert report["hash_mismatches"] == 0
    assert report["live_provider_calls"] == 0
    assert report["scientific_result"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert output.read_text(encoding="utf-8") == json.dumps(
        report, sort_keys=True, separators=(",", ":")
    ) + "\n"


def test_archive_validation_fails_closed_for_a_missing_required_stream(tmp_path: Path) -> None:
    module = _module()
    run_dir = module.run_replay(CONFIG, "pilot-a", artifact_root=tmp_path)
    mutated = tmp_path / "mutated"
    shutil.copytree(run_dir, mutated)
    (mutated / "context_events.jsonl").unlink()

    report = module.validate_archive(mutated)

    assert report["overall"] == "fail"
    assert report["reason_code"] == "REQUIRED_ARTIFACT_MISSING"


def test_replay_and_inspector_cli_surfaces_write_canonical_reports(tmp_path: Path) -> None:
    env = {**os.environ, "MEMCONTAM_ARTIFACT_ROOT": str(tmp_path)}
    run_id = "pilot-a-cli"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "memcontam.cli",
            "phase12",
            "run-replay",
            "--config",
            str(CONFIG),
            "--run-id",
            run_id,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    run_dir = tmp_path / "runs" / run_id
    invariant_output = tmp_path / "invariants.json"
    archive_output = tmp_path / "archive.json"

    assert run.returncode == 0, run.stderr
    inspect = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_phase12_pilot_a.py",
            "--run-dir",
            str(run_dir),
            "--output",
            str(invariant_output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    archive = subprocess.run(
        [
            sys.executable,
            "-m",
            "memcontam.cli",
            "phase12",
            "validate-archive",
            "--run-dir",
            str(run_dir),
            "--output",
            str(archive_output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert inspect.returncode == 0, inspect.stderr
    assert archive.returncode == 0, archive.stderr
    assert json.loads(invariant_output.read_text(encoding="utf-8"))["overall"] == "pass"
    assert json.loads(archive_output.read_text(encoding="utf-8"))["overall"] == "pass"
