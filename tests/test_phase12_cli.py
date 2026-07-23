from __future__ import annotations

import json
import sys
from pathlib import Path

import memcontam.cli as cli


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "phase12"


def _run_cli(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["memcontam", *args])
    cli.main()


def test_non_scientific_validate_plan_prefix_branch_aggregate_archive_flow(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_path = tmp_path / "readiness.yaml"
    config_path.write_text(
        (FIXTURE_ROOT / "FX-CONFIG-001.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    run_root = tmp_path / "runs"

    _run_cli(monkeypatch, "phase12", "validate", "--config", str(config_path))
    assert "valid phase12 config" in capsys.readouterr().out

    _run_cli(monkeypatch, "phase12", "plan", "--config", str(config_path))
    plan = json.loads(capsys.readouterr().out)
    assert plan["candidate_routes"] == ["3w", "5w"]

    _run_cli(
        monkeypatch,
        "phase12",
        "run-prefix",
        "--replay",
        "FX-BRANCH-001",
        "--fixture-root",
        str(FIXTURE_ROOT),
        "--run-root",
        str(run_root),
        "--run-id",
        "prefix",
    )
    prefix_dir = run_root / "prefix"
    assert (prefix_dir / "run.json").exists()

    _run_cli(
        monkeypatch,
        "phase12",
        "run-branch",
        "--replay",
        "FX-BRANCH-001",
        "--fixture-root",
        str(FIXTURE_ROOT),
        "--run-root",
        str(run_root),
        "--run-id",
        "branch",
    )
    branch_dir = run_root / "branch"
    assert (branch_dir / "public_artifact_manifest.json").exists()
    capsys.readouterr()

    _run_cli(monkeypatch, "phase12", "aggregate", "--run-dir", str(branch_dir))
    assert json.loads(capsys.readouterr().out)["trial_count"] > 0

    _run_cli(monkeypatch, "phase12", "validate-archive", "--run-dir", str(branch_dir))
    assert json.loads(capsys.readouterr().out)["archive_valid"] is True

    _run_cli(
        monkeypatch,
        "phase12",
        "aggregate",
        "--replay",
        "FX-AGG-001",
        "--fixture-root",
        str(FIXTURE_ROOT),
    )
    assert json.loads(capsys.readouterr().out)["clean_minus_contam"] == 0.5

    _run_cli(
        monkeypatch,
        "phase12",
        "validate-archive",
        "--replay",
        "FX-ARCHIVE-001",
        "--fixture-root",
        str(FIXTURE_ROOT),
    )
    assert json.loads(capsys.readouterr().out)["archive_valid"] is True
