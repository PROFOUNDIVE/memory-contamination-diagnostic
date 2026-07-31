from __future__ import annotations

import argparse
import hashlib
import json
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
    parser.add_argument("--freeze-a", type=Path)
    parser.add_argument("--authorization-request", type=Path)
    arguments = parser.parse_args()
    names = (arguments.report,) if arguments.report else {
        "authority-methods": REPORTS[:2], "bct": REPORTS[5:8], "terminal-fill": REPORTS[4:8],
    }.get(arguments.report_set, ())
    if not names:
        parser.error("one report selector is required")
    if arguments.stage_result is None and set(names) - {"authority-transition", "methods-lock", "freeze-a"}:
        parser.error("--stage-result is required for stage-bound reports")
    digest = hashlib.sha256(arguments.plan.read_bytes()).hexdigest()
    for name in names:
        path = arguments.bundle / f"{name.replace('-', '_')}_report.json"
        if not path.exists():
            build_evidence_report(arguments.bundle, name, arguments.stage_result, digest)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["input_digests"] = {
                "freeze_a": None if arguments.freeze_a is None else hashlib.sha256(arguments.freeze_a.read_bytes()).hexdigest(),
                "authorization_request": None if arguments.authorization_request is None else hashlib.sha256(arguments.authorization_request.read_bytes()).hexdigest(),
            }
            path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
