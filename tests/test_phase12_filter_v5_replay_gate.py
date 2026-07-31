from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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
