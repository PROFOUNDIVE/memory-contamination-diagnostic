from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from memcontam.experiment.phase12.filter_challenge.bct_archive_models import (
    ArchiveValidation,
    LedgerError,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive_evidence_inputs import (
    validate_current_readiness_inputs,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive_storage import _hash, _sha256_path
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    EvidenceBuildError,
    read_regular_nofollow,
    sha256_regular_nofollow,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.experiment.phase12.filter_challenge.registry_calibration import CalibrationStageResult


_REPORT_IDS: Final = (
    "authority-transition",
    "methods-lock",
    "freeze-a",
    "screening",
    "freeze-b-search-config",
    "bct-execution",
    "archive-validation",
    "claim-scope",
    "pilot-b-readiness",
)
_REPORT_SCHEMAS: Final = {
    report_id: f"phase12_fv5_{report_id.replace('-', '_')}_report_v1" for report_id in _REPORT_IDS
}


def build_evidence_report(
    bundle: Path, report_id: str, stage_result: Path | None, plan_digest: str
) -> Path:
    if report_id not in _REPORT_SCHEMAS:
        raise LedgerError("EVIDENCE_REPORT_CONTRACT_INVALID")
    stage = None
    if stage_result is not None:
        stage = CalibrationStageResult.model_validate_json(
            read_regular_nofollow(stage_result, "EVIDENCE_STAGE_DIGEST_MISMATCH")
        )
    bundle.mkdir(parents=True, exist_ok=True)
    path = bundle / f"{report_id.replace('-', '_')}_report.json"
    if path.exists():
        raise LedgerError("EVIDENCE_REPORT_EXISTS")
    payload: dict[str, object] = {
        "schema_version": _REPORT_SCHEMAS[report_id],
        "common_envelope": "phase12_fv5_evidence_report_v1",
        "report_id": report_id,
        "producer_argv": "scripts/build_phase12_filter_v5_bct_evidence.py",
        "producer_version": "phase12-filter-v5-bct-v1",
        "producer_code_commit": "6b415fbf3f27103d7d25726f8ce6447f9830a8e3",
        "approved_plan_sha256": plan_digest,
        "stage_result_sha256": None
        if stage_result is None
        else sha256_regular_nofollow(stage_result, "EVIDENCE_STAGE_DIGEST_MISMATCH"),
        "stage_result_path": None if stage_result is None else str(stage_result),
        "stage_disposition": "completed" if stage is None else stage.disposition,
        "provider_calls_issued": 0 if stage is None else stage.provider_calls_issued,
    }
    if report_id == "freeze-a":
        payload["input_digests"] = {
            "source_universe": sha256_regular_nofollow(
                Path.cwd() / "data/phase12/filter_v5_bct_v1/source_universe_v1.json",
                "EVIDENCE_SOURCE_UNIVERSE_INVALID",
            )
        }
    payload["output_seal"] = _hash(payload)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return path


def validate_evidence_bundle(
    bundle: Path, plan_digest: str, through: str = "screening"
) -> ArchiveValidation:
    try:
        required = {
            "authority-methods": ("authority_transition", "methods_lock"),
            "freeze-a": ("authority_transition", "methods_lock", "freeze_a"),
            "screening": ("authority_transition", "methods_lock", "freeze_a", "screening"),
            "freeze-b": (
                "authority_transition",
                "methods_lock",
                "freeze_a",
                "screening",
                "freeze_b_search_config",
            ),
            "bct": (
                "authority_transition",
                "methods_lock",
                "freeze_a",
                "screening",
                "freeze_b_search_config",
                "bct_execution",
                "archive_validation",
                "claim_scope",
            ),
            "readiness": (
                "authority_transition",
                "methods_lock",
                "freeze_a",
                "screening",
                "freeze_b_search_config",
                "bct_execution",
                "archive_validation",
                "claim_scope",
                "pilot_b_readiness",
            ),
        }.get(through)
        if required is None:
            raise LedgerError("EVIDENCE_REPORT_MISSING")
        reports: dict[str, dict[str, JsonValue]] = {}
        for name in required:
            path = bundle / f"{name}_report.json"
            payload = json.loads(read_regular_nofollow(path, "EVIDENCE_REPORT_INVALID"))
            if not isinstance(payload, dict):
                raise LedgerError("EVIDENCE_REPORT_INVALID")
            report_id = name.replace("_", "-")
            stage_path = payload.get("stage_result_path")
            if (
                payload.get("schema_version") != _REPORT_SCHEMAS[report_id]
                or payload.get("common_envelope") != "phase12_fv5_evidence_report_v1"
                or payload.get("report_id") != report_id
                or payload.get("approved_plan_sha256") != plan_digest
                or payload.get("provider_calls_issued") != 0
                or payload.get("producer_argv") != "scripts/build_phase12_filter_v5_bct_evidence.py"
                or payload.get("producer_version") != "phase12-filter-v5-bct-v1"
                or not isinstance(payload.get("producer_code_commit"), str)
                or payload.get("all_passed") is not None
                or payload.get("output_seal")
                != _hash({key: value for key, value in payload.items() if key != "output_seal"})
            ):
                raise LedgerError("EVIDENCE_REPORT_CONTRACT_INVALID")
            if report_id == "freeze-a":
                inputs = payload.get("input_digests")
                source = Path.cwd() / "data/phase12/filter_v5_bct_v1/source_universe_v1.json"
                if not isinstance(inputs, dict) or inputs.get("source_universe") != sha256_regular_nofollow(
                    source, "EVIDENCE_SOURCE_UNIVERSE_INVALID"
                ):
                    raise LedgerError("EVIDENCE_SOURCE_UNIVERSE_INVALID")
            reports[name] = payload
            if stage_path is not None and (
                not isinstance(stage_path, str)
                or payload.get("stage_result_sha256")
                != sha256_regular_nofollow(Path(stage_path), "EVIDENCE_STAGE_DIGEST_MISMATCH")
            ):
                raise LedgerError("EVIDENCE_STAGE_DIGEST_MISMATCH")
            if stage_path is not None:
                stage = CalibrationStageResult.model_validate_json(
                    read_regular_nofollow(Path(stage_path), "EVIDENCE_STAGE_DIGEST_MISMATCH")
                )
                if payload.get("stage_disposition") != stage.disposition:
                    raise LedgerError("EVIDENCE_REPORT_CONTRACT_INVALID")
        if "freeze_b_search_config" in required:
            _validate_freeze_b_waiting_report(bundle, plan_digest)
        if "pilot_b_readiness" in required:
            validate_current_readiness_inputs(reports)
            from memcontam.experiment.phase12.filter_challenge.pilot_b_readiness import (
                validate_readiness_report,
            )

            validate_readiness_report(bundle)
    except (EvidenceBuildError, LedgerError, OSError, UnicodeError, json.JSONDecodeError) as error:
        return ArchiveValidation(False, error.code if isinstance(error, LedgerError) else "EVIDENCE_REPORT_INVALID")
    return ArchiveValidation(True)


def _validate_freeze_b_waiting_report(bundle: Path, plan_digest: str) -> None:
    payload = json.loads(
        read_regular_nofollow(bundle / "freeze_b_search_config_report.json", "EVIDENCE_REPORT_INVALID")
    )
    stage_path = payload.get("stage_result_path")
    if not isinstance(stage_path, str):
        raise LedgerError("EVIDENCE_FREEZE_B_WAITING_INVALID")
    stage = CalibrationStageResult.model_validate_json(
        read_regular_nofollow(Path(stage_path), "EVIDENCE_STAGE_DIGEST_MISMATCH")
    )
    upstream = {
        report_id: _sha256_path(bundle / f"{report_id.replace('-', '_')}_report.json")
        for report_id in ("authority-transition", "methods-lock", "freeze-a", "screening")
    }
    inputs = payload.get("input_digests")
    if (
        payload.get("approved_plan_sha256") != plan_digest
        or payload.get("stage_disposition") != "blocked_before_stage"
        or payload.get("terminal_status") != "AWAITING_SCREENING_AUTHORIZATION"
        or payload.get("provider_calls_issued") != 0
        or stage.stage != "screening"
        or stage.disposition != "blocked_before_stage"
        or stage.terminal_status != "AWAITING_SCREENING_AUTHORIZATION"
        or stage.provider_calls_issued != 0
        or payload.get("upstream_report_sha256") != upstream
        or not isinstance(inputs, dict)
        or inputs.get("freeze_b") is not None
        or inputs.get("search_config") is not None
        or inputs.get("bct_authorization_request") is not None
    ):
        raise LedgerError("EVIDENCE_FREEZE_B_WAITING_INVALID")
