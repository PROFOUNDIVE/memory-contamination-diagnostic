from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from memcontam.readiness import phase13_calibration_v2_authorization as authorization
from memcontam.readiness import phase13_cli
from memcontam.readiness.phase13_provider_models import ExecutionTemplateIdentity
from test_phase13_calibration_v2_authorization import CONFIG, NOW, complete_bundle


def _args(request: Path, permit: Path, digest: str, client: object) -> argparse.Namespace:
    return argparse.Namespace(
        phase13_command="run-calibration-v2", config=CONFIG, request=request,
        authorization=permit, expected_authorization_sha256=digest, allow_live_calls=True,
        authorization_now=NOW, authorization_environment={
            "OPENAI_API_KEY": "synthetic", "MEMCONTAM_BGE_CACHE_DIR": "/synthetic",
        }, provider_client=client,
        execution_template=ExecutionTemplateIdentity(task="game24", baseline="fh_bounded", arm_key="Clean"),
    )


@pytest.mark.parametrize("block", ["missing", "stale", "live_only", "credential", "cache", "dirty"])
def test_every_external_block_precedes_factory_and_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, block: str,
) -> None:
    constructions = dispatches = 0
    request, permit, digest, _ = complete_bundle(tmp_path)
    monkeypatch.setattr(authorization, "_tracked_worktree_clean", lambda _root: block != "dirty")
    monkeypatch.setattr(authorization, "_cache_ready", lambda _environment: block != "cache")

    class Client:
        def chat(self, messages, model, config):  # noqa: ANN001, ANN202
            nonlocal dispatches
            del messages, model, config
            dispatches += 1

    def factory(client, root, identity):  # noqa: ANN001, ANN202
        nonlocal constructions
        del client, root, identity
        constructions += 1
        return object()

    monkeypatch.setattr(phase13_cli, "build_calibration_v2_provider", factory)
    args = _args(request, permit, digest, Client())
    if block == "missing":
        permit.unlink()
    elif block == "stale":
        request.write_bytes(request.read_bytes() + b"\n")
    elif block == "live_only":
        args.request = None
        args.authorization = None
        args.expected_authorization_sha256 = None
    elif block == "credential":
        args.authorization_environment.pop("OPENAI_API_KEY")
    with pytest.raises(SystemExit, match="CALIBRATION_V2_EXTERNAL_BLOCK"):
        phase13_cli.run(args)
    assert (constructions, dispatches) == (0, 0)
    assert not (tmp_path / "runs").exists()


def test_valid_authorization_reaches_only_injected_fake_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    constructions = dispatches = 0
    request, permit, digest, _ = complete_bundle(tmp_path)
    monkeypatch.setattr(authorization, "_tracked_worktree_clean", lambda _root: True)
    monkeypatch.setattr(authorization, "_cache_ready", lambda _environment: True)

    class Client:
        def chat(self, messages, model, config):  # noqa: ANN001, ANN202
            nonlocal dispatches
            del messages, model, config
            dispatches += 1

    def factory(client, root, identity):  # noqa: ANN001, ANN202
        nonlocal constructions
        del client, root, identity
        constructions += 1
        return object()

    monkeypatch.setattr(phase13_cli, "build_calibration_v2_provider", factory)
    phase13_cli.run(_args(request, permit, digest, Client()))
    assert capsys.readouterr().out.rstrip() == "CALIBRATION_V2_AUTHORIZED"
    assert (constructions, dispatches) == (1, 0)
    assert not (tmp_path / "runs").exists()
