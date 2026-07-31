from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.bct_archive import build_evidence_report


REPORTS = (
    "authority-transition", "methods-lock", "freeze-a", "screening", "freeze-b-search-config",
    "bct-execution", "archive-validation", "claim-scope", "pilot-b-readiness",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", choices=REPORTS)
    parser.add_argument("--report-set", choices=("authority-methods", "bct", "terminal-fill"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--stage-result", type=Path)
    arguments = parser.parse_args()
    names = (arguments.report,) if arguments.report else {
        "authority-methods": REPORTS[:2], "bct": REPORTS[5:8], "terminal-fill": REPORTS[4:8],
    }.get(arguments.report_set, ())
    if not names:
        parser.error("one report selector is required")
    if arguments.stage_result is None and set(names) - {"authority-transition", "methods-lock"}:
        parser.error("--stage-result is required for stage-bound reports")
    digest = hashlib.sha256(arguments.plan.read_bytes()).hexdigest()
    for name in names:
        path = arguments.bundle / f"{name.replace('-', '_')}_report.json"
        if not path.exists():
            build_evidence_report(arguments.bundle, name, arguments.stage_result, digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
