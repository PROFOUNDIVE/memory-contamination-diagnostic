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
    JsonValue,
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


def test_bind_bct_builds_from_valid_screening_and_freeze_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a fresh private state with all prerequisite manifests and valid lineage files.
    state_home = tmp_path / "state"
    root = state_home / "memcontam/phase12-filter-v5-rootless-local"
    attempt = "bind-bct-positive"
    manifest_root = root / "manifests" / attempt
    terminal = root / "terminals" / attempt / "screening.json"
    freeze = root / "freeze" / attempt / "freeze_b.json"
    key = root / "keys" / "ed25519-private.key"
    for directory in (manifest_root, terminal.parent, freeze.parent, key.parent):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_home.chmod(0o700)
    key.write_bytes(bytes(range(32)))
    (root / "plan-bind.md").write_bytes(b"reviewed plan\n")
    for name in ("source", "runtime", "input", "compiler", "bct-schedule"):
        (manifest_root / f"{name}.json").write_bytes(b"{}\n")
    terminal.write_bytes(b'{"status":"completed_estimable"}\n')
    freeze.write_bytes(b'{"selected_game24_probe_ids":["a","b"]}\n')
    source: dict[str, JsonValue] = {
        "execution_commit": "b" * 40,
        "signature": "signature",
    }
    captured: dict[str, JsonValue] = {}

    def build(**values: JsonValue) -> dict[str, JsonValue]:
        captured.update(values)
        return {"stage": "bct"}

    monkeypatch.setattr(rootless_local_bootstrap_cli, "validate_rootless_configs", lambda _root: {
        "decoding_authority": "1" * 64,
        "rate_card": "2" * 64,
    })
    monkeypatch.setattr(rootless_local_bootstrap_cli, "_read", lambda _path: source)
    monkeypatch.setattr(rootless_local_bootstrap_cli, "verify_object_signature", lambda *_args: None)
    monkeypatch.setattr(rootless_local_bootstrap_cli, "read_canonical", lambda _path: {
        "execution_commit": "b" * 40,
    })
    monkeypatch.setattr(rootless_local_bootstrap_cli, "build_stage_binding", build)
    monkeypatch.setattr(rootless_local_bootstrap_cli, "_write", lambda _path, _value: "3" * 64)
    monkeypatch.setattr(rootless_local_bootstrap_cli, "_status", lambda *_args: None)
    monkeypatch.setattr(rootless_local_bootstrap_cli, "materialize_bct_schedule", lambda *_args: None)
    arguments = _parse(
        "--state-home",
        os.fspath(state_home),
        "bind-bct",
        "--attempt-id",
        attempt,
    )

    # When: production administration binds BCT after screening and Freeze-B.
    rootless_local_bootstrap_cli._bind(arguments)

    # Then: the positive binding carries both immutable predecessor hashes.
    assert captured["stage"] == "bct"
    assert captured["registered_slots"] == 480
    assert captured["predecessor_terminal_sha256"] == hashlib.sha256(terminal.read_bytes()).hexdigest()
    assert captured["freeze_b_sha256"] == hashlib.sha256(freeze.read_bytes()).hexdigest()


def test_preflight_blocks_missing_frozen_inputs_without_writing_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: no frozen local inputs are supplied to the direct preflight handler.
    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    arguments = _parse("preflight", "--attempt-id", "preflight-missing-inputs")
    arguments.repo_root = repository

    # When: preflight runs before any anchor or provider path is available.
    with pytest.raises(SystemExit) as raised:
        rootless_local_bootstrap_cli.run(arguments)

    # Then: it returns a typed zero-call stop without materializing canonical output.
    status = json.loads(capsys.readouterr().out)
    assert raised.value.code == 65
    assert status == {
        "schema_version": "rootless_cli_status_v1",
        "profile": PROFILE,
        "command": "preflight",
        "outcome": "blocked",
        "next_action": "stop",
        "reason_code": "ROOTLESS_MISSING_EXTERNAL_INPUT",
        "attempt_id": "preflight-missing-inputs",
        "artifact_role": None,
        "artifact_sha256": None,
        "provider_calls_issued": 0,
        "exit_code": 65,
    }
    assert tuple(repository.iterdir()) == ()


def test_preflight_maps_missing_state_home_to_typed_zero_call_stop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)
    arguments = _parse(
        "--state-home",
        os.fspath(tmp_path / "missing-state-home"),
        "preflight",
        "--attempt-id",
        "preflight-missing-state-home",
    )
    arguments.repo_root = repository

    with pytest.raises(SystemExit) as raised:
        rootless_local_bootstrap_cli.run(arguments)

    assert raised.value.code == 65
    assert json.loads(capsys.readouterr().out)["reason_code"] == "ROOTLESS_MISSING_EXTERNAL_INPUT"
    assert tuple(repository.iterdir()) == ()
