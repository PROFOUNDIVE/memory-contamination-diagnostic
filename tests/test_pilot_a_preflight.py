from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import memcontam.cli as cli


AUTHORITY_FILENAMES = (
    "AGENTS.md",
    "Phase 12 — THEORETICAL ARTIFACT.md",
    "Phase 12-Compatible Baseline Memory and Lightweight Filter Design revised-v3.md",
    "Phase 12-Compatible Contamination Construction Intervention Timing and Sensitivity Protocol.md",
    "Phase 12-Compatible Pilot Main and Exploratory Experiment Design.md",
    "Phase 11 — PROVENANCE CITATION-REVISION.md",
)


def _git(repo_root: Path, *args: str) -> str:
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Preflight Test",
        "GIT_AUTHOR_EMAIL": "preflight@example.invalid",
        "GIT_COMMITTER_NAME": "Preflight Test",
        "GIT_COMMITTER_EMAIL": "preflight@example.invalid",
    }
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed.stdout.strip()


def _write_config(path: Path, base_audit_commit: str, *, model_id: str = "gpt-4o-2024-11-20") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config_kind": "phase12_pilot_a_preflight_v1",
                "base_audit_commit": base_audit_commit,
                "task_family": "game24",
                "provider": {
                    "provider": "openai",
                    "endpoint": "responses",
                    "model_family": "gpt-4o",
                    "model_id": model_id,
                    "service_tier": "default",
                    "store": False,
                },
                "decoding": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_output_tokens": 2048,
                    "requested_seed": 0,
                },
                "retry": {"retries_after_initial_attempt": 3, "backoff_seconds": [1, 2, 4]},
                "cost": {
                    "currency": "USD",
                    "warning": 3.0,
                    "hard_ceiling": 5.0,
                    "input_per_1m_tokens": 2.5,
                    "cached_input_per_1m_tokens": 1.25,
                    "output_per_1m_tokens": 10.0,
                },
                "live_calls": {"enabled": True},
                "tool_mode": "text_only",
                "evidence_layers": ["build", "calibration"],
            }
        ),
        encoding="utf-8",
    )


def _prepare_checkout(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo_root = tmp_path / "checkout"
    authority_root = tmp_path / "authority root"
    artifact_root = tmp_path / "artifacts"
    cache_root = artifact_root / "huggingface" / "hub"

    (repo_root / "src" / "memcontam" / "tasks").mkdir(parents=True)
    (repo_root / "src" / "memcontam" / "verifiers").mkdir(parents=True)
    (repo_root / "data" / "phase12" / "registries").mkdir(parents=True)
    (repo_root / ".gitignore").write_text(".env\n", encoding="utf-8")
    (repo_root / "src" / "memcontam" / "tasks" / "game24.py").write_text(
        "fixture\n", encoding="utf-8"
    )
    (repo_root / "src" / "memcontam" / "verifiers" / "game24.py").write_text(
        "fixture\n", encoding="utf-8"
    )
    (repo_root / "data" / "phase12" / "registries" / "candidate_registry_v1.json").write_text(
        "{}\n", encoding="utf-8"
    )
    _git(repo_root, "init")
    _git(repo_root, "add", ".gitignore", "src", "data")
    _git(repo_root, "commit", "-m", "fixture")
    commit = _git(repo_root, "rev-parse", "HEAD")

    authority_root.mkdir()
    for filename in AUTHORITY_FILENAMES:
        (authority_root / filename).write_text(f"fixture: {filename}\n", encoding="utf-8")
    snapshot = cache_root / "models--BAAI--bge-m3" / "snapshots" / "5617a9f61b028005a4858fdac845db406aefb181"
    snapshot.mkdir(parents=True)
    (repo_root / ".env").write_text(
        "\n".join(
            (
                "OPENAI_API_KEY=test-key",
                f"MEMCONTAM_ARTIFACT_ROOT={artifact_root}",
                f"MEMCONTAM_BGE_CACHE_DIR={cache_root}",
                f'MEMCONTAM_THEORETICAL_ARTIFACT_ROOT="{authority_root}"',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = repo_root / "configs" / "phase12" / "pilot_a_game24_minimal.yaml"
    _write_config(config_path, commit)
    return repo_root, authority_root, config_path


def _run_cli(monkeypatch: pytest.MonkeyPatch, repo_root: Path, config_path: Path) -> None:
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(
        sys,
        "argv",
        ["memcontam", "phase12", "preflight", "--config", str(config_path)],
    )
    cli.main()


def test_preflight_writes_redacted_authority_freeze_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root, authority_root, config_path = _prepare_checkout(tmp_path)

    _run_cli(monkeypatch, repo_root, config_path)

    result = json.loads(capsys.readouterr().out)
    manifest_path = repo_root / ".sisyphus" / "evidence" / "pilot-a-unblock" / "t0-preflight.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result == {"evidence_path": str(manifest_path), "status": "pass"}
    assert manifest["authority_priority"] == [
        "Theory",
        "Baseline",
        "Contamination Protocol",
        "Experiment Design",
    ]
    assert [Path(item["path"]).name for item in manifest["authority_files"]] == list(
        AUTHORITY_FILENAMES
    )
    assert all(set(item) == {"path", "priority", "sha256"} for item in manifest["authority_files"])
    assert all(item["sha256"] for item in manifest["authority_files"])
    assert manifest["authority_files"][-1]["priority"] == "reference"
    assert manifest["commits"]["base_audit_commit"] == _git(repo_root, "rev-parse", "HEAD")
    assert manifest["commits"]["plan_base_commit"] == _git(repo_root, "rev-parse", "HEAD")
    assert manifest["commits"]["implementation_commit"] == _git(repo_root, "rev-parse", "HEAD")
    assert str(authority_root) in manifest_path.read_text(encoding="utf-8")
    assert "fixture: " not in manifest_path.read_text(encoding="utf-8")


def test_preflight_rejects_changed_frozen_authority_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root, authority_root, config_path = _prepare_checkout(tmp_path)
    _run_cli(monkeypatch, repo_root, config_path)
    capsys.readouterr()
    (authority_root / AUTHORITY_FILENAMES[1]).write_text("changed\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="authority_hash_changed"):
        _run_cli(monkeypatch, repo_root, config_path)


def test_preflight_requires_canonical_experiment_design_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, authority_root, config_path = _prepare_checkout(tmp_path)
    canonical = authority_root / AUTHORITY_FILENAMES[4]
    canonical.rename(authority_root / "Phase 12-Compatible Pilot Main and Exploratory Experiment Design(3).md")

    with pytest.raises(SystemExit, match="missing_authority_file"):
        _run_cli(monkeypatch, repo_root, config_path)


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("remove_api_key", "missing_api_key"),
        ("remove_cache", "missing_bge_cache"),
        ("remove_authority_root", "missing_theoretical_artifact_root"),
        ("wrong_model", "invalid_model_identity"),
    ],
)
def test_preflight_returns_stable_reason_codes_for_missing_prerequisites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, reason_code: str
) -> None:
    repo_root, _, config_path = _prepare_checkout(tmp_path)
    env_path = repo_root / ".env"
    if mutation == "remove_api_key":
        env_path.write_text(
            "\n".join(line for line in env_path.read_text(encoding="utf-8").splitlines() if "API_KEY" not in line)
            + "\n",
            encoding="utf-8",
        )
    elif mutation == "remove_cache":
        env_path.write_text(
            env_path.read_text(encoding="utf-8").replace("MEMCONTAM_BGE_CACHE_DIR=", "MEMCONTAM_BGE_CACHE_DIR=/missing/"),
            encoding="utf-8",
        )
    elif mutation == "remove_authority_root":
        env_path.write_text(
            env_path.read_text(encoding="utf-8").replace(
                "MEMCONTAM_THEORETICAL_ARTIFACT_ROOT=", "MEMCONTAM_THEORETICAL_ARTIFACT_ROOT=/missing/"
            ),
            encoding="utf-8",
        )
    else:
        _write_config(config_path, _git(repo_root, "rev-parse", "HEAD"), model_id="gpt-4o")

    with pytest.raises(SystemExit, match=reason_code):
        _run_cli(monkeypatch, repo_root, config_path)


def test_preflight_rejects_main_or_extension_evidence_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root, _, config_path = _prepare_checkout(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["evidence_layers"] = ["build", "main"]
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="main_extension_data_forbidden"):
        _run_cli(monkeypatch, repo_root, config_path)
