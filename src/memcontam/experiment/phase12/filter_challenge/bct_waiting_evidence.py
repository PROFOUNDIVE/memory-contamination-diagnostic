from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

from memcontam.experiment.phase12.filter_challenge.bct_archive import validate_evidence_bundle
from memcontam.experiment.phase12.filter_challenge.registry_calibration import CalibrationStageResult


BCT_REPORT_IDS: Final = ("bct-execution", "archive-validation", "claim-scope")
WAITING_SCREENING_TERMINAL: Final = "AWAITING_SCREENING_AUTHORIZATION"


def waiting_screening_stage(bundle: Path, plan_digest: str) -> CalibrationStageResult | None:
    if not validate_evidence_bundle(bundle, plan_digest, "freeze-b").valid:
        return None
    payload = _payload(bundle / "freeze_b_search_config_report.json")
    stage_path = payload.get("stage_result_path")
    if not isinstance(stage_path, str):
        return None
    try:
        stage = CalibrationStageResult.model_validate_json(Path(stage_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None
    if (
        stage.stage == "screening"
        and stage.disposition == "blocked_before_stage"
        and stage.terminal_status == WAITING_SCREENING_TERMINAL
        and stage.provider_calls_issued == 0
    ):
        return stage
    return None


def waiting_bct_report_fields(bundle: Path, plan_digest: str, stage_result: Path) -> dict[str, object] | None:
    screening = waiting_screening_stage(bundle, plan_digest)
    if screening is None:
        return None
    try:
        stage = CalibrationStageResult.model_validate_json(stage_result.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None
    if (
        stage.stage != "bct"
        or stage.disposition != "blocked_before_stage"
        or stage.terminal_status != screening.terminal_status
        or stage.reason_code != screening.reason_code
        or stage.provider_calls_issued != 0
    ):
        return None
    inputs = _payload(bundle / "freeze_b_search_config_report.json").get("input_digests")
    if not isinstance(inputs, dict):
        return None
    return {
        "input_digests": inputs,
        "terminal_status": screening.terminal_status,
        "upstream_report_sha256": {
            report_id: _sha256(bundle / f"{report_id.replace('-', '_')}_report.json")
            for report_id in (
                "authority-transition",
                "methods-lock",
                "freeze-a",
                "screening",
                "freeze-b-search-config",
            )
        },
    }


def validate_waiting_bct_reports(bundle: Path, plan_digest: str, artifact_root: Path) -> bool:
    if artifact_root.exists():
        return False
    for report_id in BCT_REPORT_IDS:
        path = bundle / f"{report_id.replace('-', '_')}_report.json"
        payload = _payload(path)
        stage_path = payload.get("stage_result_path")
        fields = waiting_bct_report_fields(
            bundle, plan_digest, Path(stage_path) if isinstance(stage_path, str) else path
        )
        if (
            fields is None
            or payload.get("schema_version") != "phase12_fv5_evidence_report_v1"
            or payload.get("report_id") != report_id
            or payload.get("approved_plan_sha256") != plan_digest
            or payload.get("stage_disposition") != "blocked_before_stage"
            or payload.get("provider_calls_issued") != 0
            or any(payload.get(key) != value for key, value in fields.items())
        ):
            return False
    return True


def _payload(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
