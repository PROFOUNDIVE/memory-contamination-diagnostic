from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from memcontam.readiness import phase13_calibration_v2_authorization as authorization

from .test_phase13_calibration_v2_authorization import CONFIG, NOW, _freeze, _sha, complete_bundle


def _resign_bundle(
    request: Path, permit: Path, payload: dict[str, object], bindings: dict[str, object],
) -> str:
    request.write_text(json.dumps(bindings, sort_keys=True), encoding="utf-8")
    payload["request_sha256"] = _sha(request)
    payload["bindings"] = bindings
    permit.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return _sha(permit)


def _verify_fails(
    request: Path, permit: Path, digest: str, *, environment: dict[str, str],
    now: datetime | None = NOW,
) -> None:
    with pytest.raises(authorization.CalibrationV2AuthorizationError):
        authorization.verify_calibration_v2_authorization(
            config_path=CONFIG, request_path=request, authorization_path=permit,
            expected_authorization_sha256=digest, allow_live_calls=True,
            environment=environment, now=now,
        )


def test_resigned_arbitrary_freeze_authority_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, permit, _, payload = complete_bundle(tmp_path)
    replacement = tmp_path / "replacement-freeze.json"
    replacement.write_text(json.dumps(_freeze()), encoding="utf-8")
    bindings = payload["bindings"]
    assert isinstance(bindings, dict)
    bindings["freeze"] = {"path": str(replacement), "sha256": _sha(replacement)}
    digest = _resign_bundle(request, permit, payload, bindings)
    monkeypatch.setattr(authorization, "_tracked_worktree_clean", lambda _root: True)
    monkeypatch.setattr(authorization, "_cache_ready", lambda _environment: True)

    _verify_fails(request, permit, digest, environment={"OPENAI_API_KEY": "synthetic"})


@pytest.mark.parametrize("field", ["issued_at", "expires_at"])
def test_naive_authorization_datetime_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    request, permit, _, payload = complete_bundle(tmp_path)
    payload[field] = datetime(2026, 8, 13, 13).isoformat()
    permit.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(authorization, "_tracked_worktree_clean", lambda _root: True)
    monkeypatch.setattr(authorization, "_cache_ready", lambda _environment: True)

    _verify_fails(request, permit, _sha(permit), environment={"OPENAI_API_KEY": "synthetic"})


def test_naive_current_datetime_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, permit, digest, _ = complete_bundle(tmp_path)
    monkeypatch.setattr(authorization, "_tracked_worktree_clean", lambda _root: True)
    monkeypatch.setattr(authorization, "_cache_ready", lambda _environment: True)

    _verify_fails(
        request, permit, digest, environment={"OPENAI_API_KEY": "synthetic"},
        now=datetime(2026, 8, 13, 12),
    )


def test_explicit_empty_environment_does_not_fall_back_to_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, permit, digest, _ = complete_bundle(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "host-secret")
    monkeypatch.setattr(authorization, "_tracked_worktree_clean", lambda _root: True)
    monkeypatch.setattr(authorization, "_cache_ready", lambda _environment: True)

    _verify_fails(request, permit, digest, environment={})


@pytest.mark.parametrize("operation", ["git", "config", "registry", "cache"])
def test_operational_failure_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    request, permit, digest, _ = complete_bundle(tmp_path)
    monkeypatch.setattr(authorization, "_tracked_worktree_clean", lambda _root: True)
    monkeypatch.setattr(authorization, "_cache_ready", lambda _environment: True)
    target = {
        "git": "_git_head", "config": "validate_calibration_v2",
        "registry": "load_execution_registry", "cache": "_cache_ready",
    }[operation]
    monkeypatch.setattr(authorization, target, lambda *_args: (_ for _ in ()).throw(OSError()))

    _verify_fails(request, permit, digest, environment={"OPENAI_API_KEY": "synthetic"})
