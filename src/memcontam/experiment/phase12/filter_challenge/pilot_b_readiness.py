from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from memcontam.experiment.phase12.filter_challenge.bct_archive import validate_evidence_bundle
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    EvidenceBuildError,
    read_regular_nofollow,
    sha256_regular_nofollow,
)
from memcontam.experiment.phase12.filter_challenge.code_prespec import (
    CodePrespecError,
    validate_code_prespec,
)
from memcontam.experiment.phase12.filter_challenge.registry_calibration import CalibrationStageResult
from memcontam.experiment.phase12.filter_challenge.rootless_local_firewall import (
    ROOTLESS_PROFILE_FORBIDDEN,
    has_forbidden_rootless_profile,
)


Terminal = Literal[
    "AWAITING_SCREENING_AUTHORIZATION",
    "FILTER_V5_PILOT_B_BLOCKED_BY_INVALID_CALIBRATION_EVIDENCE",
    "FILTER_V5_PILOT_B_NOT_ESTIMABLE",
    "AWAITING_BCT_AUTHORIZATION",
    "FILTER_V5_PILOT_B_BLOCKED_BY_INVALID_BCT_EVIDENCE",
    "READY_FOR_SEPARATE_FILTER_V5_PILOT_B_AUTHORIZATION",
]


@dataclass(frozen=True, slots=True)
class ReadinessEvidence:
    screening: Literal["missing", "invalid", "valid"]
    common_strict_probes: int = 0
    freeze_b: Literal["missing", "invalid", "valid"] = "missing"
    bct_authorization: Literal["missing", "invalid", "valid"] = "missing"
    bct_archive: Literal["missing", "invalid", "completed"] = "missing"
    completed_bct_families: tuple[str, ...] = ()
    behavioral_false_negative: bool = False


def derive_terminal(evidence: ReadinessEvidence) -> CalibrationStageResult:
    match evidence.screening:
        case "missing":
            return CalibrationStageResult.waiting("pilot_b_readiness", "AWAITING_SCREENING_AUTHORIZATION")
        case "invalid":
            return _result("blocked_before_stage", "FILTER_V5_PILOT_B_BLOCKED_BY_INVALID_CALIBRATION_EVIDENCE", "SCREENING_CALIBRATION_INVALID")
        case "valid":
            return _after_screening(evidence)


def readiness_from_bundle(bundle: Path, plan_digest: str) -> CalibrationStageResult:
    for report_name in (
        "authority_transition",
        "methods_lock",
        "freeze_a",
        "screening",
        "freeze_b_search_config",
        "bct_execution",
    ):
        try:
            raw_report = read_regular_nofollow(
                bundle / f"{report_name}_report.json", "EVIDENCE_READINESS_INVALID"
            )
        except (EvidenceBuildError, OSError):
            continue
        if has_forbidden_rootless_profile(raw_report):
            raise ValueError(ROOTLESS_PROFILE_FORBIDDEN)
        try:
            report = json.loads(raw_report)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        stage_path = report.get("stage_result_path") if isinstance(report, dict) else None
        if not isinstance(stage_path, str):
            continue
        try:
            raw_stage = read_regular_nofollow(Path(stage_path), "EVIDENCE_READINESS_INVALID")
        except (EvidenceBuildError, OSError):
            continue
        if has_forbidden_rootless_profile(raw_stage):
            raise ValueError(ROOTLESS_PROFILE_FORBIDDEN)
    if not validate_evidence_bundle(bundle, plan_digest, "screening").valid:
        return _result(
            "blocked_before_stage",
            "FILTER_V5_PILOT_B_BLOCKED_BY_INVALID_CALIBRATION_EVIDENCE",
            "PARTIAL_REPORT_CHAIN_INVALID",
        )
    screening = _report(bundle, "screening")
    stage = _stage(screening)
    if stage is None or stage.terminal_status == "AWAITING_SCREENING_AUTHORIZATION":
        return derive_terminal(ReadinessEvidence(screening="missing"))
    if stage.disposition != "completed":
        return derive_terminal(ReadinessEvidence(screening="invalid"))
    freeze_b = _report(bundle, "freeze-b-search-config")
    common_strict_probes = _integer(freeze_b.get("common_strict_probes"), 2)
    freeze_b_status = _status(freeze_b.get("freeze_b_status"), "missing")
    authorization = _status(freeze_b.get("bct_authorization_status"), "missing")
    bct = _report(bundle, "bct-execution")
    return derive_terminal(
        ReadinessEvidence(
            screening="valid",
            common_strict_probes=common_strict_probes,
            freeze_b=freeze_b_status,
            bct_authorization=authorization,
            bct_archive=_archive_status(bct.get("bct_archive_status")),
            completed_bct_families=_families(bct.get("completed_bct_families")),
            behavioral_false_negative=bct.get("behavioral_false_negative") is True,
        )
    )


def readiness_from_fixture(path: Path) -> CalibrationStageResult:
    try:
        raw = path.read_bytes()
        if has_forbidden_rootless_profile(raw):
            raise ValueError(ROOTLESS_PROFILE_FORBIDDEN)
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError
        return derive_terminal(ReadinessEvidence(**payload))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        if str(error) == ROOTLESS_PROFILE_FORBIDDEN:
            raise
        raise ValueError("READINESS_FIXTURE_INVALID") from error


def validate_readiness_report(bundle: Path) -> None:
    payload = _report(bundle, "pilot-b-readiness")
    stage_path = payload.get("stage_result_path")
    code_path = payload.get("code_prespec_path")
    if not isinstance(stage_path, str) or not isinstance(code_path, str):
        raise ValueError("EVIDENCE_READINESS_INVALID")
    try:
        stage = CalibrationStageResult.model_validate_json(
            read_regular_nofollow(Path(stage_path), "EVIDENCE_READINESS_INVALID")
        )
        expected = {
            report_id: _sha(bundle / f"{report_id.replace('-', '_')}_report.json")
            for report_id in (
                "authority-transition", "methods-lock", "freeze-a", "screening", "freeze-b-search-config",
                "bct-execution", "archive-validation", "claim-scope",
            )
        }
        validate_code_prespec(Path(code_path), Path.cwd())
    except (CodePrespecError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("EVIDENCE_READINESS_INVALID") from error
    if (
        payload.get("prior_report_sha256") != expected
        or payload.get("code_prespec_sha256") != _sha(Path(code_path))
        or payload.get("schema_version") != "phase12_fv5_pilot_b_readiness_report_v1"
        or payload.get("common_envelope") != "phase12_fv5_evidence_report_v1"
        or payload.get("stage_disposition") != stage.disposition
        or payload.get("terminal_status") != stage.terminal_status
        or stage.stage != "pilot_b_readiness"
        or stage.provider_calls_issued != 0
        or payload.get("provider_calls_issued") != 0
    ):
        raise ValueError("EVIDENCE_READINESS_INVALID")


def _after_screening(evidence: ReadinessEvidence) -> CalibrationStageResult:
    if evidence.common_strict_probes < 2:
        return _result("skipped_structural", "FILTER_V5_PILOT_B_NOT_ESTIMABLE", "COMMON_STRICT_PROBES_INSUFFICIENT")
    match evidence.freeze_b:
        case "missing" | "invalid":
            return _result("blocked_before_stage", "FILTER_V5_PILOT_B_BLOCKED_BY_INVALID_CALIBRATION_EVIDENCE", "FREEZE_B_OR_PREVIEW_INVALID")
        case "valid":
            return _after_freeze_b(evidence)


def _after_freeze_b(evidence: ReadinessEvidence) -> CalibrationStageResult:
    match evidence.bct_authorization:
        case "missing":
            return CalibrationStageResult.waiting("pilot_b_readiness", "AWAITING_BCT_AUTHORIZATION")
        case "invalid":
            return _result("blocked_before_stage", "FILTER_V5_PILOT_B_BLOCKED_BY_INVALID_CALIBRATION_EVIDENCE", "BCT_AUTHORIZATION_INVALID")
        case "valid":
            return _after_bct_authorization(evidence)


def _after_bct_authorization(evidence: ReadinessEvidence) -> CalibrationStageResult:
    if evidence.bct_archive != "completed" or set(evidence.completed_bct_families) != {
        "BCT-FV5-01", "BCT-FV5-02", "BCT-FV5-03", "BCT-FV5-04"
    }:
        return _result("invalidated", "FILTER_V5_PILOT_B_BLOCKED_BY_INVALID_BCT_EVIDENCE", "BCT_ARCHIVE_INCOMPLETE_OR_INVALID")
    return _result("completed", "READY_FOR_SEPARATE_FILTER_V5_PILOT_B_AUTHORIZATION", "ALL_BCT_FAMILIES_COMPLETED")


def _result(
    disposition: Literal["completed", "blocked_before_stage", "skipped_structural", "invalidated"],
    terminal_status: Terminal,
    reason_code: str,
) -> CalibrationStageResult:
    return CalibrationStageResult(
        stage="pilot_b_readiness",
        disposition=disposition,
        terminal_status=terminal_status,
        reason_code=reason_code,
        provider_calls_issued=0,
    )


def _report(bundle: Path, report_id: str) -> dict[str, object]:
    try:
        value = json.loads(
            read_regular_nofollow(bundle / f"{report_id.replace('-', '_')}_report.json", "EVIDENCE_READINESS_INVALID")
        )
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _stage(payload: dict[str, object]) -> CalibrationStageResult | None:
    path = payload.get("stage_result_path")
    if not isinstance(path, str):
        return None
    try:
        return CalibrationStageResult.model_validate_json(
            read_regular_nofollow(Path(path), "EVIDENCE_READINESS_INVALID")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


def _integer(value: object, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _status(value: object, default: Literal["missing"]) -> Literal["missing", "invalid", "valid"]:
    match value:
        case "missing":
            return "missing"
        case "invalid":
            return "invalid"
        case "valid":
            return "valid"
        case _:
            return default


def _archive_status(value: object) -> Literal["missing", "invalid", "completed"]:
    match value:
        case "missing":
            return "missing"
        case "invalid":
            return "invalid"
        case "completed":
            return "completed"
        case _:
            return "missing"


def _families(value: object) -> tuple[str, ...]:
    return tuple(item for item in value if isinstance(item, str)) if isinstance(value, list) else ()


def _sha(path: Path) -> str:
    return sha256_regular_nofollow(path, "EVIDENCE_READINESS_INVALID")
