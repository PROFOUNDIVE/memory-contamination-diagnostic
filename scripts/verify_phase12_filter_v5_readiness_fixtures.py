from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "docs/evidence/phase12-filter-v5-bct-v1"
PLAN = ROOT / ".omo/plans/phase12-post-filter-v5-calibration-readiness.md"
EXPECTED = {
    "awaiting-screening": "AWAITING_SCREENING_AUTHORIZATION",
    "invalid-calibration": "FILTER_V5_PILOT_B_BLOCKED_BY_INVALID_CALIBRATION_EVIDENCE",
    "not-estimable": "FILTER_V5_PILOT_B_NOT_ESTIMABLE",
    "awaiting-bct": "AWAITING_BCT_AUTHORIZATION",
    "invalid-bct": "FILTER_V5_PILOT_B_BLOCKED_BY_INVALID_BCT_EVIDENCE",
    "ready": "READY_FOR_SEPARATE_FILTER_V5_PILOT_B_AUTHORIZATION",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--code-prespec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name, expected in EXPECTED.items():
        output = args.output_root / name
        bundle = output / "bundle"
        shutil.copytree(BUNDLE, bundle)
        (bundle / "pilot_b_readiness_report.json").unlink(missing_ok=True)
        stage = output / "stage-result.json"
        command = [
            sys.executable, "-m", "memcontam.cli", "phase12", "filter-v5", "pilot-b-readiness",
            "--bundle", str(bundle), "--code-prespec", str(args.code_prespec),
            "--fixture", str(args.fixture_root / f"{name}.json"), "--stage-result", str(stage),
        ]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode != 0 or json.loads(result.stdout)["terminal_status"] != expected:
            return 2
        build = subprocess.run(
            [sys.executable, "scripts/build_phase12_filter_v5_bct_evidence.py", "--report", "pilot-b-readiness", "--bundle", str(bundle), "--plan", str(PLAN), "--artifact-root", "runs/phase12-filter-v5-bct-live-v1", "--stage-result", str(stage), "--code-prespec", str(args.code_prespec)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        if build.returncode != 0 or len(tuple(bundle.glob("*_report.json"))) != 9:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
