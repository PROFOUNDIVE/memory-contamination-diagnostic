from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.bct_archive import validate_evidence_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", choices=("authority-methods", "freeze-a", "screening", "freeze-b", "bct", "readiness"), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    arguments = parser.parse_args()
    report = validate_evidence_bundle(arguments.bundle, hashlib.sha256(arguments.plan.read_bytes()).hexdigest(), arguments.through)
    print("APPROVE" if report.valid else report.reason_code)
    return 0 if report.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
