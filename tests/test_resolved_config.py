from __future__ import annotations

import hashlib
import json

import pytest

from memcontam.clients.config import ProviderConfig
from memcontam.clients.provider_profile import normalize_provider_profile, provider_profile_id
from memcontam.config.resolution import resolve_run_config
from memcontam.clients.base import LLMResponse
from memcontam.cli import run_config
from memcontam.logging.audit_artifacts import (
    write_provider_profile_atomic,
    write_resolved_config_atomic,
)


def test_resolved_config_records_taxonomy_version_and_redacts_provider_secrets(tmp_path) -> None:
    profile = normalize_provider_profile(
        ProviderConfig(provider="replay"),
        served_models=["replay"],
        model_snapshots={"replay": "v1"},
    )
    config = {
        "run": {"name": "audit", "provider": "replay"},
        "provider_config": {"api_key": "never-persist-this"},
    }

    resolved = resolve_run_config(config, provider_profile=profile)
    profile_path = write_provider_profile_atomic(tmp_path, profile)
    resolved_path = write_resolved_config_atomic(tmp_path, resolved)

    resolved_payload = resolved_path.read_text(encoding="utf-8")
    assert profile_path.name == "provider_profile.json"
    assert resolved["run"]["failure_taxonomy_version"] == "baseline_fidelity_v1"
    assert resolved["run"]["provider_profile_id"] == provider_profile_id(profile)
    assert resolved["run"]["stage"] == "replay"
    assert resolved["run"]["execution_class"] == "offline_contract_replay"
    assert "never-persist-this" not in resolved_payload
    assert json.loads(resolved_payload) == resolved


def test_resolved_config_hash_is_canonical_after_default_resolution() -> None:
    profile = normalize_provider_profile(
        ProviderConfig(provider="replay"),
        served_models=["replay"],
        model_snapshots={"replay": "v1"},
    )
    resolved = resolve_run_config({"run": {"provider": "replay"}}, provider_profile=profile)
    expected = json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode("utf-8")

    assert hashlib.sha256(expected).hexdigest()


def test_invalid_provider_dispatch_and_missing_live_credential_fail_before_run_directory(
    tmp_path, monkeypatch
) -> None:
    invalid = {
        "run": {
            "stage": "replay",
            "execution_class": "live",
            "provider": "openai_compatible",
        },
        "models": ["model"],
        "tasks": [{"name": "game24", "sample_path": str(tmp_path / "unused.jsonl"), "limit": 1}],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "logging": {"output_dir": str(tmp_path / "runs")},
    }
    with pytest.raises(SystemExit, match="unsupported provider configuration"):
        run_config(invalid, "invalid")
    assert not (tmp_path / "runs" / "invalid").exists()

    monkeypatch.delenv("TASK_3B_API_KEY", raising=False)
    live = {
        **invalid,
        "run": {
            "mode": "faithful",
            "stage": "pilot",
            "execution_class": "live",
            "provider": "openai_compatible",
        },
        "provider_config": {"api_key_env": "TASK_3B_API_KEY"},
    }
    with pytest.raises(SystemExit, match="missing API key env var: TASK_3B_API_KEY"):
        run_config(live, "missing-credential")
    assert not (tmp_path / "runs" / "missing-credential").exists()


def test_run_writes_redacted_audit_artifacts_before_the_first_provider_call(tmp_path) -> None:
    class ArtifactCheckingClient:
        def chat(self, messages, model, config):
            run_dir = tmp_path / "runs" / "audited"
            assert (run_dir / "provider_profile.json").exists()
            assert (run_dir / "resolved_config.json").exists()
            return LLMResponse(content="final: 6 / (1 - 3 / 4)", raw={}, token_usage={})

    sample_path = tmp_path / "sample.jsonl"
    sample_path.write_text(
        '{"sample_id":"sample","numbers":[1,3,4,6],"target":24}\n', encoding="utf-8"
    )
    config = {
        "run": {
            "stage": "replay",
            "execution_class": "offline_contract_replay",
            "provider": "replay",
        },
        "models": ["replay"],
        "tasks": [{"name": "game24", "sample_path": str(sample_path), "limit": 1}],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "logging": {"output_dir": str(tmp_path / "runs")},
    }

    run_dir = run_config(config, "audited", _client_override=ArtifactCheckingClient())

    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    trial = json.loads((run_dir / "trials.jsonl").read_text(encoding="utf-8"))
    assert resolved["run"]["failure_taxonomy_version"] == "baseline_fidelity_v1"
    assert (
        trial["metadata"]["config_hash"]
        == hashlib.sha256(
            json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
