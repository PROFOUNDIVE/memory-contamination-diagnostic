from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import memcontam.cli as cli


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "phase12"


def _run_cli(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["memcontam", *args])
    cli.main()


def test_cli_blocks_scientific_and_invalid_contract_requests(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "runs"
    replay_args = ("--replay", "FX-BRANCH-001", "--fixture-root", str(FIXTURE_ROOT))

    for run_family in ("pilot_a", "main_a"):
        with pytest.raises(SystemExit, match="phase12 readiness gate not activated"):
            _run_cli(
                monkeypatch,
                "phase12",
                "run-branch",
                *replay_args,
                "--run-root",
                str(run_root),
                "--run-family",
                run_family,
                "--scientific",
            )
        assert not run_root.exists()

    for flag, value, message in (
        ("--candidate", "unknown", "unsupported phase12 candidate"),
        ("--mode", "python_sandbox", "unsupported phase12 mode"),
    ):
        with pytest.raises(SystemExit, match=message):
            _run_cli(
                monkeypatch,
                "phase12",
                "run-prefix",
                *replay_args,
                "--run-root",
                str(run_root),
                flag,
                value,
            )
        assert not run_root.exists()

    invalid_config = json.loads((FIXTURE_ROOT / "FX-CONFIG-001.json").read_text(encoding="utf-8"))
    invalid_config["tool_mode"] = "python_sandbox"
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps(invalid_config), encoding="utf-8")
    with pytest.raises(SystemExit, match="PRIMARY_TOOL_FORBIDDEN"):
        _run_cli(monkeypatch, "phase12", "plan", "--config", str(config_path))
    assert not run_root.exists()
