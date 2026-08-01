from __future__ import annotations

import argparse
from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    ArchiveValidation,
    validate_evidence_bundle,
)
from memcontam.experiment.phase12.filter_challenge.bct_waiting_evidence import (
    validate_waiting_bct_reports,
)
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
    try:
        report = validate_evidence_bundle(arguments.bundle, digest, arguments.through)
    except ValueError as error:
        print(error)
        return 2
    if report.valid and arguments.through in {"bct", "readiness"}:
        waiting_valid = validate_waiting_bct_reports(
            arguments.bundle, digest, arguments.artifact_root
        )
        report = ArchiveValidation(
            waiting_valid, None if waiting_valid else "EVIDENCE_BCT_WAITING_INVALID"
        )
    print("APPROVE" if report.valid else report.reason_code)
    return 0 if report.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
