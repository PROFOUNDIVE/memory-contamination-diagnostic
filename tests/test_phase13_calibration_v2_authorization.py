from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from memcontam.readiness import phase13_calibration_v2_authorization as authorization


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase13/pre_main_calibration_v2.yaml"
PLAN = Path("/home/hyunwoo/git/memory-contamination-diagnostic/.omo/plans/phase13-canonical-authority-sync-calibration-v2.md")
NOW = datetime(2026, 8, 13, 12, tzinfo=UTC)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze() -> dict[str, object]:
    authority_hashes = {
        "theory": "34f63f37a49e92607c78ced038c4c70b4c9d5e3fa8fc57d6e97de1ee79db59a8",
        "baseline": "c28f0e2b00db6a2731f64933ccc67c5ea5a163d6233c526e6b473e540f988204",
        "protocol": "06d23e29dff6c607bc2035c5641fbb696fb5c09dd86f2ce190a99c6baa57eefc",
        "experiment_design": "6b8ab4e414c86dbcb4afc9c2781b13f9312e8ba2834d20473d261f264e6e1acf",
    }
    payload: dict[str, object] = {
        "schema_version": "phase13-authority-freeze-v1",
        "closure_id": "phase13-calibration-v2-prospective",
        "authorities": [
            {
                "kind": "authority",
                "authority_role": role,
                "artifact": {"kind": "artifact", "artifact_id": f"phase13-{role}", "path": f"external/{role}.md", "sha256": digest},
            }
            for role, digest in authority_hashes.items()
        ],
        "parameter_classifications": [
            {"kind": "scientific_design", "class_code": "A", "H_primary": 5, "primary_analysis_window_id": "accuracy-h5-primary"},
            {"kind": "execution", "class_code": "B", "H_run": 10},
            {"kind": "inference", "class_code": "C", "estimator_id": "paired-seed-risk-difference-v1"},
            {"kind": "planning", "class_code": "D", "calibration_seed_count_per_task": 12},
            {"kind": "reproducibility", "class_code": "E", "bootstrap_replicates": 20000, "bootstrap_rng_seed": 13, "serialization_version": "canonical-json-v1"},
        ],
        "registries": [
            {
                "kind": "registry",
                "registry_kind": kind,
                "registry_id": f"phase13-{kind}-registry-v1",
                "artifact": {"kind": "artifact", "artifact_id": f"phase13-{kind}-registry-v1", "path": f"fixture/{kind}.json", "sha256": hashlib.sha256(kind.encode()).hexdigest()},
            }
            for kind in ("calibration_v2", "execution", "analysis")
        ],
    }
    payload["closure_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def complete_bundle(tmp_path: Path) -> tuple[Path, Path, str, dict[str, object]]:
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(_freeze()), encoding="utf-8")
    files = {
        "config": CONFIG,
        "partition": ROOT / "data/phase13/calibration_v2/seed_partition_registry_v1.json",
        "execution": ROOT / "data/phase13/authority/execution_registry_v1.json",
        "analysis": ROOT / "data/phase13/authority/analysis_registry_v1.json",
        "structural": ROOT / "data/phase13/authority/structural_checkpoint_registry_v1.json",
    }
    implementation = {
        role: ROOT / path
        for role, path in {
            "cli": "src/memcontam/readiness/phase13_cli.py",
            "provider_runtime": "src/memcontam/readiness/phase13_provider_runtime.py",
            "trajectory_runtime": "src/memcontam/readiness/phase13_calibration_v2_runtime.py",
        }.items()
    }
    bindings: dict[str, object] = {
        "schema_version": "phase13_calibration_v2_request_v1",
        "run_id": "authorized-synthetic",
        "freeze": {"path": str(freeze_path), "sha256": _sha(freeze_path)},
        "config": {"path": str(CONFIG), "sha256": _sha(CONFIG)},
        "registries": {role: {"path": str(path), "sha256": _sha(path)} for role, path in files.items() if role != "config"},
        "stream_registry_id": "phase13-calibration-v2-rotations-v1",
        "suffix_registry_id": "phase13-calibration-v2-suffix-v1",
        "analysis_window_registry_id": "phase13-analysis-window-registry-v1",
        "primary_analysis_window_id": "accuracy-h5-primary",
        "identities": {
            "provider_id": "openai-responses-v1", "model_snapshot_id": "gpt-4o-2024-11-20",
            "decoding_contract_id": "phase13-decoding-zero-v1", "prompt_contract_id": "baseline-fidelity-v2-prompts",
            "tool_contract_id": "text-only-equal-availability-v1", "session_contract_id": "paired-isolated-session-v1",
            "failure_contract_id": "baseline-fidelity-v2-failure-taxonomy", "resource_contract_id": "phase13-resource-envelope-v1",
        },
        "owners": {"prefix": "phase13-clean-prefix-owner-v1", "execution": "phase13-h10-execution-owner-v1", "offline": "phase13-offline-compute-owner-v1"},
        "capacity": {
            "maximum_semantic_calls": 14327, "maximum_transport_attempts": 57308,
            "maximum_input_tokens": 234733568, "maximum_output_tokens": 117366784,
            "maximum_cost_microusd": 15000000, "per_request_timeout_seconds": 120,
            "maximum_latency_milliseconds": 7200000, "maximum_storage_bytes": 1000000000,
            "maximum_wall_clock_seconds": 10800, "provider_requests_per_minute": 500,
            "provider_concurrency": 8,
        },
        "output_root": str(ROOT / "runs/phase13-calibration-v2/authorized-synthetic"),
        "plan": {"path": str(PLAN), "sha256": _sha(PLAN)},
        "implementation_commit": authorization._git_head(ROOT),
        "implementation": {role: {"path": str(path), "sha256": _sha(path)} for role, path in implementation.items()},
        "credential_env_name": "OPENAI_API_KEY", "cache_env_name": "MEMCONTAM_BGE_CACHE_DIR",
        "runtime_python": sys.version.split()[0], "tracked_worktree_clean": True,
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(bindings, sort_keys=True), encoding="utf-8")
    issued = NOW - timedelta(minutes=1)
    payload = {
        "schema_version": "phase13_calibration_v2_authorization_v1",
        "authorization_id": "separate-operator-authorization-v1",
        "issued_at": issued.isoformat(), "expires_at": (NOW + timedelta(hours=1)).isoformat(),
        "request_sha256": _sha(request_path), "bindings": bindings,
    }
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return request_path, authorization_path, _sha(authorization_path), payload


def _verify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> authorization.VerifiedCalibrationV2Authorization:
    request, permit, digest, _ = complete_bundle(tmp_path)
    monkeypatch.setattr(authorization, "_tracked_worktree_clean", lambda _root: True)
    monkeypatch.setattr(authorization, "_cache_ready", lambda _environment: True)
    return authorization.verify_calibration_v2_authorization(
        config_path=CONFIG, request_path=request, authorization_path=permit,
        expected_authorization_sha256=digest, allow_live_calls=True,
        environment={"OPENAI_API_KEY": "synthetic", "MEMCONTAM_BGE_CACHE_DIR": "/synthetic"}, now=NOW,
    )


def test_complete_separate_authorization_is_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verified = _verify(tmp_path, monkeypatch)
    assert verified.authorization.authorization_id == "separate-operator-authorization-v1"
    assert verified.request.capacity.maximum_semantic_calls == 14327


@pytest.mark.parametrize(
    "mutation",
    [
        "request", "config", "freeze", "partition", "execution", "analysis", "structural",
        "stream", "suffix", "window", "model", "decoding", "prompt", "tool", "session",
        "failure", "resource", "owner", "capacity", "output", "plan", "implementation",
        "credential_name", "cache_name", "runtime", "commit", "tracked", "expired", "extra",
    ],
)
def test_every_stale_or_mismatched_binding_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    request, permit, digest, payload = complete_bundle(tmp_path)
    monkeypatch.setattr(authorization, "_tracked_worktree_clean", lambda _root: True)
    monkeypatch.setattr(authorization, "_cache_ready", lambda _environment: True)
    if mutation == "request":
        request.write_bytes(request.read_bytes() + b"\n")
    elif mutation == "expired":
        payload["expires_at"] = (NOW - timedelta(seconds=1)).isoformat()
        permit.write_text(json.dumps(payload), encoding="utf-8")
        digest = _sha(permit)
    elif mutation == "extra":
        payload["unexpected"] = True
        permit.write_text(json.dumps(payload), encoding="utf-8")
        digest = _sha(permit)
    else:
        bindings = payload["bindings"]
        assert isinstance(bindings, dict)
        targets = {
            "config": ("config", "sha256"), "freeze": ("freeze", "sha256"),
            "partition": ("registries", "partition"), "execution": ("registries", "execution"),
            "analysis": ("registries", "analysis"), "structural": ("registries", "structural"),
            "stream": ("stream_registry_id",), "suffix": ("suffix_registry_id",),
            "window": ("primary_analysis_window_id",), "model": ("identities", "model_snapshot_id"),
            "decoding": ("identities", "decoding_contract_id"), "prompt": ("identities", "prompt_contract_id"),
            "tool": ("identities", "tool_contract_id"), "session": ("identities", "session_contract_id"),
            "failure": ("identities", "failure_contract_id"), "resource": ("identities", "resource_contract_id"),
            "owner": ("owners", "execution"), "capacity": ("capacity", "maximum_cost_microusd"),
            "output": ("output_root",), "plan": ("plan", "sha256"),
            "implementation": ("implementation", "cli"), "credential_name": ("credential_env_name",),
            "cache_name": ("cache_env_name",), "runtime": ("runtime_python",),
            "commit": ("implementation_commit",), "tracked": ("tracked_worktree_clean",),
        }
        target = targets[mutation]
        parent = bindings
        for key in target[:-1]:
            value = parent[key]
            assert isinstance(value, dict)
            parent = value
        key = target[-1]
        current = parent[key]
        parent[key] = False if mutation == "tracked" else ({"path": "/drift", "sha256": "0" * 64} if isinstance(current, dict) else "0" * 64)
        request.write_text(json.dumps(bindings, sort_keys=True), encoding="utf-8")
        payload["request_sha256"] = _sha(request)
        permit.write_text(json.dumps(payload), encoding="utf-8")
        digest = _sha(permit)
    with pytest.raises(authorization.CalibrationV2AuthorizationError, match="CALIBRATION_V2_EXTERNAL_BLOCK"):
        authorization.verify_calibration_v2_authorization(
            config_path=CONFIG, request_path=request, authorization_path=permit,
            expected_authorization_sha256=digest, allow_live_calls=True,
            environment={"OPENAI_API_KEY": "synthetic", "MEMCONTAM_BGE_CACHE_DIR": "/synthetic"}, now=NOW,
        )


@pytest.mark.parametrize("missing", ["authorization", "credential", "cache", "live_flag"])
def test_missing_prerequisite_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str,
) -> None:
    request, permit, digest, _ = complete_bundle(tmp_path)
    monkeypatch.setattr(authorization, "_tracked_worktree_clean", lambda _root: True)
    monkeypatch.setattr(authorization, "_cache_ready", lambda _environment: missing != "cache")
    environment = {"OPENAI_API_KEY": "synthetic", "MEMCONTAM_BGE_CACHE_DIR": "/synthetic"}
    if missing == "authorization":
        permit.unlink()
    if missing == "credential":
        environment.pop("OPENAI_API_KEY")
    with pytest.raises(authorization.CalibrationV2AuthorizationError, match="CALIBRATION_V2_EXTERNAL_BLOCK"):
        authorization.verify_calibration_v2_authorization(
            config_path=CONFIG, request_path=request, authorization_path=permit,
            expected_authorization_sha256=digest, allow_live_calls=missing != "live_flag",
            environment=environment, now=NOW,
        )


@pytest.mark.parametrize("kind", ["request", "authorization", "freeze"])
def test_trusted_inputs_reject_symlinks_and_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    request, permit, digest, payload = complete_bundle(tmp_path)
    monkeypatch.setattr(authorization, "_tracked_worktree_clean", lambda _root: True)
    monkeypatch.setattr(authorization, "_cache_ready", lambda _environment: True)
    if kind == "freeze":
        bindings = payload["bindings"]
        assert isinstance(bindings, dict)
        freeze = bindings["freeze"]
        assert isinstance(freeze, dict)
        target = Path(str(freeze["path"]))
    else:
        target = {"request": request, "authorization": permit}[kind]
    original = tmp_path / f"{kind}.original"
    target.rename(original)
    target.symlink_to(original)
    with pytest.raises(authorization.CalibrationV2AuthorizationError, match="CALIBRATION_V2_EXTERNAL_BLOCK"):
        authorization.verify_calibration_v2_authorization(
            config_path=CONFIG, request_path=request, authorization_path=permit,
            expected_authorization_sha256=digest, allow_live_calls=True,
            environment={"OPENAI_API_KEY": "synthetic", "MEMCONTAM_BGE_CACHE_DIR": "/synthetic"}, now=NOW,
        )
