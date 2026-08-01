from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from memcontam.clients.config import ProviderConfig
from memcontam.clients.provider_profile import normalize_provider_profile
from memcontam.config.resolution import resolve_run_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "pilot_multitask_replay.yaml"
RESET = ROOT / "scripts" / "reset_phase12_filter_v5_replay_gate.py"
VERIFY = ROOT / "scripts" / "verify_phase12_filter_v5_replay_gate.py"


def test_replay_gate_helpers_reject_unowned_or_missing_runs(tmp_path: Path) -> None:
    # Given: a missing named scratch run and a sibling that must not be touched.
    run_root = tmp_path / "runs" / "phase12-filter-v5-bct-replay-gate"
    sibling = tmp_path / "runs" / "other-run"
    sibling.mkdir(parents=True)

    # When: reset and verification use the planned run identifier.
    reset = subprocess.run(
        [sys.executable, str(RESET), "--run-root", str(run_root), "--expected-run-id", run_root.name],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    verify = subprocess.run(
        [
            sys.executable,
            str(VERIFY),
            "--run-root",
            str(run_root),
            "--expected-run-id",
            run_root.name,
            "--expected-config",
            "configs/pilot_multitask_replay.yaml",
            "--expected-rows",
            "90",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: reset is safe and verification rejects the absent replay evidence without sibling deletion.
    assert reset.returncode == 0, reset.stdout + reset.stderr
    assert verify.returncode != 0
    assert sibling.exists()


def test_replay_gate_rejects_a_copied_run_for_a_different_config(tmp_path: Path) -> None:
    # Given: a syntactically complete replay run whose config identity is wrong.
    run_root = tmp_path / "phase12-filter-v5-bct-replay-gate"
    run_root.mkdir()
    (run_root / "run.json").write_text(
        json.dumps({"run_metadata": {"run_id": run_root.name}}), encoding="utf-8"
    )
    (run_root / "resolved_config.json").write_text(
        json.dumps({"config_sha256": "0" * 64}), encoding="utf-8"
    )
    (run_root / "trials.jsonl").write_text("{}\n" * 90, encoding="utf-8")

    # When: the gate receives the approved replay config.
    result = subprocess.run(
        [sys.executable, str(VERIFY), "--run-root", str(run_root), "--expected-run-id", run_root.name,
         "--expected-config", "configs/pilot_multitask_replay.yaml", "--expected-rows", "90"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )

    # Then: a copied run cannot claim the approved config identity.
    assert result.returncode != 0
    assert result.stdout == "REPLAY_GATE_EVIDENCE_INVALID\n"


def test_replay_gate_accepts_the_runner_resolved_config_digest(tmp_path: Path) -> None:
    from memcontam.cli import load_config

    run_root = tmp_path / "phase12-filter-v5-bct-replay-gate"
    run_root.mkdir()
    config = load_config(CONFIG)
    provider = ProviderConfig.from_run_config(config)
    resolved = resolve_run_config(
        config,
        provider_profile=normalize_provider_profile(
            provider,
            served_models=config["models"],
            model_snapshots=config.get("run", {}).get("model_snapshots", {}),
        ),
    )
    config_sha256 = hashlib.sha256(
        json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (run_root / "run.json").write_text(
        json.dumps({"run_metadata": {"run_id": run_root.name, "config_hash": config_sha256}}),
        encoding="utf-8",
    )
    (run_root / "resolved_config.json").write_text(
        json.dumps(resolved, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    (run_root / "trials.jsonl").write_text("{}\n" * 90, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(VERIFY), "--run-root", str(run_root), "--expected-run-id", run_root.name,
         "--expected-config", str(CONFIG), "--expected-rows", "90"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "REPLAY_GATE_VALID\n"
