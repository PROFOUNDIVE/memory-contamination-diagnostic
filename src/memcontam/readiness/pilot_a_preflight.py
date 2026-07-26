from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]

from memcontam.memory.embeddings import BgeM3EmbeddingProvider


AUTHORITY_FILES = (
    ("AGENTS.md", "authority"),
    ("Phase 12 — THEORETICAL ARTIFACT.md", "Theory"),
    ("Phase 12-Compatible Baseline Memory and Lightweight Filter Design revised-v3.md", "Baseline"),
    (
        "Phase 12-Compatible Contamination Construction Intervention Timing and Sensitivity Protocol.md",
        "Contamination Protocol",
    ),
    ("Phase 12-Compatible Pilot Main and Exploratory Experiment Design.md", "Experiment Design"),
    ("Phase 11 — PROVENANCE CITATION-REVISION.md", "reference"),
)
AUTHORITY_PRIORITY = ("Theory", "Baseline", "Contamination Protocol", "Experiment Design")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_EXPECTED_MODEL_ID = "gpt-4o-2024-11-20"
_EVIDENCE_PATH = Path(".sisyphus/evidence/pilot-a-unblock/t0-preflight.json")
_REQUIRED_CONFIG_FIELDS = {
    "config_kind",
    "base_audit_commit",
    "cost",
    "decoding",
    "task_family",
    "live_calls",
    "provider",
    "retry",
    "tool_mode",
    "evidence_layers",
}
_REQUIRED_PROVIDER_FIELDS = {
    "provider",
    "endpoint",
    "model_family",
    "model_id",
    "service_tier",
    "store",
}


class PreflightError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PilotAPreflightConfig:
    base_audit_commit: str
    model_id: str


def run_preflight(config_path: Path, *, repo_root: Path | None = None) -> dict[str, str]:
    config = load_preflight_config(config_path)
    root = (repo_root or Path.cwd()).resolve()
    implementation_commit = _implementation_commit(root)
    environment = _preflight_environment(root)
    artifact_root = _required_directory(environment, "MEMCONTAM_ARTIFACT_ROOT", "missing_artifact_root")
    _require_access(artifact_root, read=True, write=True, code="missing_artifact_root")
    cache_root = _required_directory(environment, "MEMCONTAM_BGE_CACHE_DIR", "missing_bge_cache")
    _require_access(cache_root, read=True, write=False, code="missing_bge_cache")
    _require_bge_revision(cache_root)
    authority_root = _required_directory(
        environment,
        "MEMCONTAM_THEORETICAL_ARTIFACT_ROOT",
        "missing_theoretical_artifact_root",
    )
    _require_access(authority_root, read=True, write=False, code="missing_theoretical_artifact_root")
    _require_game24_assets(root)
    authorities = _authority_manifest(authority_root)
    evidence_path = root / _EVIDENCE_PATH
    previous = _load_previous_manifest(evidence_path)
    _verify_authority_freeze(previous, authorities)
    plan_base_commit = _plan_base_commit(previous, implementation_commit)
    manifest = {
        "schema_version": "pilot_a_preflight_v1",
        "authority_priority": list(AUTHORITY_PRIORITY),
        "authority_files": authorities,
        "commits": {
            "base_audit_commit": config.base_audit_commit,
            "plan_base_commit": plan_base_commit,
            "implementation_commit": implementation_commit,
        },
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    return {"status": "pass", "evidence_path": str(evidence_path)}


def load_preflight_config(path: Path) -> PilotAPreflightConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise PreflightError("invalid_preflight_config") from error
    if not isinstance(payload, dict) or set(payload) != _REQUIRED_CONFIG_FIELDS:
        raise PreflightError("invalid_preflight_config")
    provider = payload.get("provider")
    if not isinstance(provider, dict) or set(provider) != _REQUIRED_PROVIDER_FIELDS:
        raise PreflightError("invalid_preflight_config")
    if payload.get("config_kind") != "phase12_pilot_a_preflight_v1":
        raise PreflightError("invalid_preflight_config")
    base_audit_commit = payload.get("base_audit_commit")
    if not isinstance(base_audit_commit, str) or not _COMMIT_RE.fullmatch(base_audit_commit):
        raise PreflightError("invalid_preflight_config")
    if payload.get("task_family") != "game24" or payload.get("tool_mode") != "text_only":
        raise PreflightError("invalid_preflight_config")
    if payload.get("decoding") != {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": 2048,
        "requested_seed": 0,
    } or payload.get("retry") != {
        "retries_after_initial_attempt": 3,
        "backoff_seconds": [1, 2, 4],
    }:
        raise PreflightError("invalid_preflight_config")
    if payload.get("cost") != {
        "currency": "USD",
        "warning": 3.0,
        "hard_ceiling": 5.0,
        "input_per_1m_tokens": 2.5,
        "cached_input_per_1m_tokens": 1.25,
        "output_per_1m_tokens": 10.0,
    } or payload.get("live_calls") != {"enabled": True}:
        raise PreflightError("invalid_preflight_config")
    evidence_layers = payload.get("evidence_layers")
    if not isinstance(evidence_layers, list) or set(evidence_layers) & {"main", "extension"}:
        raise PreflightError("main_extension_data_forbidden")
    if evidence_layers != ["build", "calibration"]:
        raise PreflightError("invalid_preflight_config")
    if provider != {
        "provider": "openai",
        "endpoint": "responses",
        "model_family": "gpt-4o",
        "model_id": _EXPECTED_MODEL_ID,
        "service_tier": "default",
        "store": False,
    }:
        if provider.get("model_id") != _EXPECTED_MODEL_ID:
            raise PreflightError("invalid_model_identity")
        raise PreflightError("invalid_preflight_config")
    return PilotAPreflightConfig(base_audit_commit, _EXPECTED_MODEL_ID)


def _implementation_commit(repo_root: Path) -> str:
    result = _git(repo_root, "rev-parse", "HEAD")
    commit = result.stdout.strip()
    if result.returncode != 0 or not _COMMIT_RE.fullmatch(commit):
        raise PreflightError("git_commit_unavailable")
    return commit


def _preflight_environment(repo_root: Path) -> dict[str, str]:
    env_path = repo_root / ".env"
    if not env_path.is_file():
        raise PreflightError("env_missing")
    ignored = _git(repo_root, "check-ignore", "--quiet", ".env")
    if ignored.returncode != 0:
        raise PreflightError("env_not_ignored")
    tracked = _git(repo_root, "ls-files", "--error-unmatch", ".env")
    if tracked.returncode == 0:
        raise PreflightError("env_tracked")
    if tracked.returncode != 1:
        raise PreflightError("git_commit_unavailable")
    values = _dotenv_values(env_path)
    environment = {**values, **os.environ}
    if not environment.get("OPENAI_API_KEY"):
        raise PreflightError("missing_api_key")
    return environment


def _dotenv_values(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise PreflightError("env_missing") from error
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _required_directory(environment: Mapping[str, str], name: str, code: str) -> Path:
    value = environment.get(name)
    if not value:
        raise PreflightError(code)
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise PreflightError(code)
    return path.resolve()


def _require_access(path: Path, *, read: bool, write: bool, code: str) -> None:
    required = os.X_OK | (os.R_OK if read else 0) | (os.W_OK if write else 0)
    try:
        next(path.iterdir(), None)
    except OSError as error:
        raise PreflightError(code) from error
    if not os.access(path, required):
        raise PreflightError(code)


def _require_bge_revision(cache_root: Path) -> None:
    snapshot = (
        cache_root
        / f"models--{BgeM3EmbeddingProvider.MODEL_ID.replace('/', '--')}"
        / "snapshots"
        / BgeM3EmbeddingProvider.REVISION
    )
    if not snapshot.is_dir() or not os.access(snapshot, os.R_OK | os.X_OK):
        raise PreflightError("missing_bge_cache")


def _require_game24_assets(repo_root: Path) -> None:
    required = (
        (repo_root / "src/memcontam/tasks/game24.py", "missing_game24_task"),
        (repo_root / "src/memcontam/verifiers/game24.py", "missing_game24_verifier"),
        (
            repo_root / "data/phase12/registries/candidate_registry_v1.json",
            "missing_candidate_registry",
        ),
    )
    for path, code in required:
        if not path.is_file() or not os.access(path, os.R_OK):
            raise PreflightError(code)


def _authority_manifest(authority_root: Path) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    for filename, priority in AUTHORITY_FILES:
        path = authority_root / filename
        if not path.is_file():
            raise PreflightError("missing_authority_file")
        try:
            digest = _sha256(path)
        except OSError as error:
            raise PreflightError("missing_authority_file") from error
        manifest.append({"path": str(path.resolve()), "sha256": digest, "priority": priority})
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_previous_manifest(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreflightError("preflight_manifest_invalid") from error
    if not isinstance(payload, dict):
        raise PreflightError("preflight_manifest_invalid")
    return payload


def _verify_authority_freeze(
    previous: Mapping[str, Any] | None, authorities: list[dict[str, str]]
) -> None:
    if previous is None:
        return
    frozen = previous.get("authority_files")
    if not isinstance(frozen, list):
        raise PreflightError("preflight_manifest_invalid")
    frozen_hashes = {
        item.get("path"): item.get("sha256")
        for item in frozen
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and isinstance(item.get("sha256"), str)
    }
    current_hashes = {item["path"]: item["sha256"] for item in authorities}
    if frozen_hashes != current_hashes:
        raise PreflightError("authority_hash_changed")


def _plan_base_commit(previous: Mapping[str, Any] | None, implementation_commit: str) -> str:
    if previous is None:
        return implementation_commit
    commits = previous.get("commits")
    plan_base_commit = commits.get("plan_base_commit") if isinstance(commits, dict) else None
    if not isinstance(plan_base_commit, str) or not _COMMIT_RE.fullmatch(plan_base_commit):
        raise PreflightError("preflight_manifest_invalid")
    return plan_base_commit


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise PreflightError("git_commit_unavailable") from error
