from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from memcontam.experiment.phase12.filter_challenge import rootless_local_bootstrap_cli
from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
    build_stage_binding,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    canonical_json_file,
    canonical_json_value,
)


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


def test_stage_acknowledgement_hash_matches_execution_authority_binding_digest(
    tmp_path: Path,
) -> None:
    # Given: a private state root containing one canonical screening binding.
    state_home = tmp_path / "state"
    root = state_home / "memcontam" / "phase12-filter-v5-rootless-local"
    binding_root = root / "bindings" / "attempt-001"
    key_root = root / "keys"
    binding_root.mkdir(mode=0o700, parents=True)
    key_root.mkdir(mode=0o700)
    state_home.chmod(0o700)
    (root / "plan-bind.md").write_bytes(b"reviewed plan\n")
    (key_root / "ed25519-private.key").write_bytes(bytes(range(32)))
    binding = build_stage_binding(
        attempt_id="attempt-001",
        stage="screening",
        plan_binding_sha256="1" * 64,
        trusted_base_commit="a" * 40,
        execution_commit="b" * 40,
        decoding_authority_sha256="2" * 64,
        rate_card_sha256="3" * 64,
        source_manifest_sha256="4" * 64,
        runtime_manifest_sha256="5" * 64,
        input_manifest_sha256="6" * 64,
        compiler_sha256="7" * 64,
        schedule_sha256="8" * 64,
        registered_slots=90,
        stage_cap_nanousd=2_000_000_000,
        created_at="2026-08-09T12:00:00Z",
    )
    (binding_root / "screening.json").write_bytes(canonical_json_file(binding))
    for path in (root / "plan-bind.md", key_root / "ed25519-private.key", binding_root / "screening.json"):
        path.chmod(0o600)
    arguments = _parse(
        "--state-home",
        os.fspath(state_home),
        "acknowledge-stage",
        "--attempt-id",
        "attempt-001",
        "--set-id",
        "screening",
        "--operator-index",
        "1",
        "--operator-label",
        "operator-1",
        "--stage",
        "screening",
    )

    # When: the administrative CLI signs the stage acknowledgement.
    rootless_local_bootstrap_cli.run(arguments)

    # Then: its binding digest uses the authority builder's no-file-LF representation.
    acknowledgement = json.loads(
        (
            root
            / "acknowledgements/stage/attempt-001/screening/screening/operator-1.json"
        ).read_bytes()
    )
    assert acknowledgement["stage_binding_sha256"] == hashlib.sha256(
        canonical_json_value(binding)
    ).hexdigest()
