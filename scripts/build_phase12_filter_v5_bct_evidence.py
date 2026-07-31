from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    build_evidence_report,
    validate_evidence_bundle,
)
from memcontam.experiment.phase12.filter_challenge.bct_waiting_evidence import (
    BCT_REPORT_IDS,
    waiting_bct_report_fields,
)
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    EvidenceBuildError,
    approval_descriptor_path,
    approved_plan_sha256,
)
from memcontam.experiment.phase12.filter_challenge.registry_calibration import CalibrationStageResult


REPORTS = (
    "authority-transition", "methods-lock", "freeze-a", "screening", "freeze-b-search-config",
    "bct-execution", "archive-validation", "claim-scope", "pilot-b-readiness",
)
WAITING_SCREENING_TERMINAL = "AWAITING_SCREENING_AUTHORIZATION"


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
    parser.add_argument("--reseal-existing", action="store_true")
    arguments = parser.parse_args()
    names = (arguments.report,) if arguments.report else {
        "authority-methods": REPORTS[:2], "bct": REPORTS[5:8], "terminal-fill": REPORTS[4:8],
    }.get(arguments.report_set, ())
    if not names:
        parser.error("one report selector is required")
    if arguments.stage_result is None and set(names) - {"authority-transition", "methods-lock", "freeze-a"}:
        parser.error("--stage-result is required for stage-bound reports")
    try:
        digest = approved_plan_sha256(arguments.plan, approval_descriptor_path(arguments.plan))
    except EvidenceBuildError as error:
        print(error.code)
        return 2
    if "freeze-b-search-config" in names:
        if arguments.stage_result is None or arguments.artifact_root.exists():
            print("EVIDENCE_FREEZE_B_WAITING_INVALID")
            return 2
        upstream = validate_evidence_bundle(arguments.bundle, digest, "screening")
        stage = CalibrationStageResult.model_validate_json(
            arguments.stage_result.read_text(encoding="utf-8")
        )
        if (
            not upstream.valid
            or stage.stage != "screening"
            or stage.disposition != "blocked_before_stage"
            or stage.terminal_status != WAITING_SCREENING_TERMINAL
            or stage.provider_calls_issued != 0
        ):
            print("EVIDENCE_FREEZE_B_WAITING_INVALID")
            return 2
    bct_fields = None
    if set(names) & set(BCT_REPORT_IDS):
        if arguments.stage_result is None or arguments.artifact_root.exists():
            print("EVIDENCE_BCT_WAITING_INVALID")
            return 2
        bct_fields = waiting_bct_report_fields(arguments.bundle, digest, arguments.stage_result)
        if bct_fields is None:
            print("EVIDENCE_BCT_WAITING_INVALID")
            return 2
    for name in names:
        path = arguments.bundle / f"{name.replace('-', '_')}_report.json"
        if path.exists() and not arguments.reseal_existing:
            continue
        if path.exists():
            _validate_resealable(path, name)
        with tempfile.TemporaryDirectory(dir=arguments.bundle.parent) as temporary:
            temporary_bundle = Path(temporary) / "bundle"
            replacement = build_evidence_report(
                temporary_bundle,
                name,
                arguments.stage_result,
                digest,
            )
            payload = json.loads(replacement.read_text(encoding="utf-8"))
        if name in BCT_REPORT_IDS:
            assert bct_fields is not None
            payload |= bct_fields
        else:
            payload["input_digests"] = {
                "freeze_a": _sha256(arguments.freeze_a),
                "authorization_request": _sha256(arguments.authorization_request),
            }
        if name == "freeze-b-search-config":
            payload["terminal_status"] = WAITING_SCREENING_TERMINAL
            payload["input_digests"] |= {
                "freeze_b": None,
                "search_config": None,
                "bct_authorization_request": None,
            }
            payload["upstream_report_sha256"] = {
                report_id: _sha256(arguments.bundle / f"{report_id.replace('-', '_')}_report.json")
                for report_id in REPORTS[:4]
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    return 0


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_resealable(path: Path, report_id: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "phase12_fv5_evidence_report_v1"
        or payload.get("report_id") != report_id
        or payload.get("provider_calls_issued") != 0
    ):
        raise ValueError("EVIDENCE_REPORT_RESEAL_REFUSED")


if __name__ == "__main__":
    raise SystemExit(main())
