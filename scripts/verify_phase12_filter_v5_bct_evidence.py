from __future__ import annotations

import argparse
from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.bct_archive import validate_evidence_bundle
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    EvidenceBuildError,
    approval_descriptor_path,
    approved_plan_sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", choices=("authority-methods", "freeze-a", "screening", "freeze-b", "bct", "readiness"), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        digest = approved_plan_sha256(arguments.plan, approval_descriptor_path(arguments.plan))
    except EvidenceBuildError as error:
        print(error.code)
        return 2
    report = validate_evidence_bundle(arguments.bundle, digest, arguments.through)
    print("APPROVE" if report.valid else report.reason_code)
    return 0 if report.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
