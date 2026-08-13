from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, assert_never

import pytest

from memcontam.readiness import phase13_cli
from memcontam.readiness import phase13_calibration_v2
from memcontam.readiness.phase13_calibration_v2 import CalibrationV2ConfigError


CONFIG = Path("configs/phase13/pre_main_calibration_v2.yaml")
HISTORICAL = Path("data/phase13/authority/historical_compatibility_v1.json")


def _command(config: Path, command: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    environment["PYTHONPATH"] = "src"
    return subprocess.run(
        (
            sys.executable,
            "-m",
            "memcontam.cli",
            "phase13",
            command,
            "--config",
            str(config),
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


@pytest.mark.parametrize("command", ["validate-calibration-v2", "prepare-calibration-v2"])
def test_deterministic_commands_never_construct_provider(
    monkeypatch: pytest.MonkeyPatch, command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    constructions = 0

    def forbidden_constructor(*args: object, **kwargs: object) -> None:
        nonlocal constructions
        del args, kwargs
        constructions += 1
        raise AssertionError("provider construction is forbidden")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "memcontam.clients.openai_responses.OpenAIResponsesClient",
        forbidden_constructor,
    )

    phase13_cli.run(argparse.Namespace(phase13_command=command, config=CONFIG))

    output = capsys.readouterr().out
    assert output.rstrip().endswith(
        "AWAITING_CALIBRATION_V2_AUTHORIZATION"
        if command == "prepare-calibration-v2"
        else "DETERMINISTIC_AUTHORITY_SYNC_COMPLETE"
    )
    assert constructions == 0


@pytest.mark.parametrize("mode", ["deleted", "mutated"])
def test_historical_registry_drift_fails_before_provider_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["deleted", "mutated"],
) -> None:
    root = tmp_path / "root"
    for path in (
        Path("data/phase13/calibration_v2"),
        Path("data/phase13/authority"),
        Path("data/phase13/main"),
        Path("data/phase12/registries"),
        Path("data/tasks"),
    ):
        shutil.copytree(path, root / path)
    historical = root / HISTORICAL
    match mode:
        case "deleted":
            historical.unlink()
        case "mutated":
            historical.write_bytes(historical.read_bytes() + b"\n")
        case unreachable:
            assert_never(unreachable)
    monkeypatch.setattr(phase13_calibration_v2, "ROOT", root)

    with pytest.raises(
        CalibrationV2ConfigError,
        match="AUTHORITY_FILE_NOT_REGULAR" if mode == "deleted" else "AUTHORITY_HASH_MISMATCH",
    ):
        phase13_calibration_v2.validate_calibration_v2(CONFIG)


@pytest.mark.parametrize(
    ("command", "terminals"),
    [
        (
            "run-calibration-v2",
            ("CALIBRATION_V2_EXTERNAL_BLOCK", "MAIN_A_EXECUTION_FORBIDDEN"),
        ),
        ("validate-calibration-v2-archive", ("CALIBRATION_V2_EXTERNAL_BLOCK",)),
    ],
)
def test_unimplemented_live_boundaries_are_honest(
    command: str, terminals: tuple[str, ...]
) -> None:
    result = _command(CONFIG, command)

    assert result.returncode != 0
    assert tuple(result.stderr.splitlines()) == terminals


@pytest.mark.parametrize(
    ("old", "new", "error_code"),
    [
        (
            "7c92189f645b74381f3fedf2d3ffbc8c4768a7019eaeb621edb8c22fb4a34970",
            "0" * 64,
            "AUTHORITY_HASH_MISMATCH",
        ),
        ("  H_primary: 5", "  H: 5", "BARE_H_PROHIBITED"),
        (
            "  artifact_root: runs/phase13-calibration-v2",
            "  artifact_root: runs/drift",
            "CONFIG_IDENTITY_MISMATCH",
        ),
    ],
)
def test_validation_rejects_drift_before_provider_construction(
    tmp_path: Path,
    old: str,
    new: str,
    error_code: str,
) -> None:
    payload = CONFIG.read_text(encoding="utf-8").replace(old, new, 1)
    config = tmp_path / "mutated.yaml"
    config.write_text(payload, encoding="utf-8")

    result = _command(config, "validate-calibration-v2")

    assert result.returncode != 0
    assert result.stderr.rstrip().endswith(error_code)
