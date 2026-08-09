from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from memcontam.experiment.phase12.filter_challenge import rootless_local_bootstrap_cli


ROOT = Path(__file__).resolve().parents[1]
PROFILE = "local_rootless_non_authoritative"


def _parse(*arguments: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="phase12_command", required=True)
    rootless_local_bootstrap_cli.add_parser(commands)
    return parser.parse_args(("filter-v5-rootless", "--repo-root", os.fspath(ROOT), *arguments))


def _script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_execution_anchor_exclusive_creates_closed_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repository = tmp_path / "repo"
    (repository / "runs" / "phase12-filter-v5-rootless-qa" / "pre-egress").mkdir(parents=True)
    repository.chmod(0o700)
    arguments = _parse("write-execution-anchor", "--execution-commit", "a" * 40)
    arguments.repo_root = repository

    rootless_local_bootstrap_cli.run(arguments)

    anchor = json.loads((repository / "runs/phase12-filter-v5-rootless-qa/pre-egress/execution-anchor.json").read_bytes())
    status = json.loads(capsys.readouterr().out)
    assert anchor == {"schema_version": "rootless_execution_anchor_v1", "profile": PROFILE, "execution_commit": "a" * 40}
    assert status["command"] == "write-execution-anchor"
    assert status["exit_code"] == 0


@pytest.mark.parametrize(
    "raw",
    (
        b"OPENAI_API_KEY=short\n",
        b"export OPENAI_API_KEY=" + b"a" * 20 + b"\n",
        b"OPENAI_API_KEY='" + b"a" * 20 + b"'\n",
        b"OPENAI_API_KEY=" + b"a" * 20 + b"\nOPENAI_API_KEY=" + b"b" * 20 + b"\n",
        b"OTHER=value\nmalformed\nOPENAI_API_KEY=" + b"a" * 20 + b"\n",
    ),
)
def test_provisioner_rejects_malformed_or_ambiguous_source(tmp_path: Path, raw: bytes) -> None:
    module = _script("provision_phase12_filter_v5_rootless_env.py")
    source = tmp_path / "source.env"
    source.write_bytes(raw)
    source.chmod(0o600)
    checkout = tmp_path / "checkout"
    checkout.mkdir(mode=0o700)

    assert module.provision(source, checkout) == 64
    assert not (checkout / ".env").exists()


def test_provisioner_writes_exact_private_secret_without_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _script("provision_phase12_filter_v5_rootless_env.py")
    source = tmp_path / "source.env"
    secret = "A_b-" * 8
    source.write_text(f"OTHER=value\nOPENAI_API_KEY={secret}\n", encoding="ascii")
    source.chmod(0o600)
    checkout = tmp_path / "checkout"
    checkout.mkdir(mode=0o700)

    assert module.provision(source, checkout) == 0

    target = checkout / ".env"
    assert target.read_bytes() == f"OPENAI_API_KEY={secret}\n".encode()
    assert target.stat().st_mode & 0o777 == 0o600
    assert capsys.readouterr() == ("", "")

    assert module.provision(source, checkout) == 64
    assert target.read_bytes() == f"OPENAI_API_KEY={secret}\n".encode()
