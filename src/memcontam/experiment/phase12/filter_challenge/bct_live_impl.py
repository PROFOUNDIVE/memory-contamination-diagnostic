from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Final, Literal, Mapping, TypeVar

from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    BudgetLedger,
    append_archive_record,
    validate_live_archive,
)
from memcontam.experiment.phase12.filter_challenge.bct_live_cli import (
    add_calibration_parsers,
    run_calibration_command as _dispatch_calibration_command,
)
from memcontam.experiment.phase12.filter_challenge.bct_live_preview import build_cost_preview
from memcontam.experiment.phase12.filter_challenge.bct_live_authorization import (
    CalibrationAuthorizationError,
    load_authorization as _load_authorization,
    validate_runtime_authorization,
)
from memcontam.experiment.phase12.filter_challenge.bct_waiting_evidence import (
    waiting_screening_stage,
)
from memcontam.experiment.phase12.filter_challenge.code_prespec import CodePrespecError, validate_code_prespec
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    approval_descriptor_path,
    approved_plan_sha256,
)
from memcontam.experiment.phase12.filter_challenge.pilot_b_readiness import (
    readiness_from_bundle,
    readiness_from_fixture,
)
from memcontam.experiment.phase12.filter_challenge.registry_calibration import (
    BCTAuthorizationV1,
    CalibrationConfigError,
    CalibrationAuthorization,
    CalibrationStageResult,
    ScreeningAuthorizationV1,
    require_artifact_root,
    validate_calibration_config,
)


REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[5]
_Authorization = TypeVar("_Authorization", bound=CalibrationAuthorization)
__all__ = (
    "CalibrationAuthorizationError",
    "_run_cli_stage",
    "_validate_config",
    "add_calibration_parsers",
    "load_authorization",
    "run_calibration_command",
    "run_screen_controls",
)


def run_calibration_command(args: argparse.Namespace) -> None:
    def validate() -> None:
        _validate_config(args.config)
        _print({"valid": True, "provider_calls_issued": 0})

    def validate_archive() -> None:
        payload = asdict(validate_live_archive(args.archive))
        args.output.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        _print(payload)

    try:
        _dispatch_calibration_command(args, {
            "validate-calibration-config": validate,
            "screening-cost-preview": lambda: _write_cost_preview(args, "screening"),
            "bct-cost-preview": lambda: _write_cost_preview(args, "bct"),
            "screen-controls": lambda: _print(_run_cli_stage(args, "screening").model_dump(mode="json")),
            "bct-run": lambda: _print(_run_cli_stage(args, "bct").model_dump(mode="json")),
            "validate-bct-archive": validate_archive,
            "pilot-b-readiness": lambda: _run_pilot_b_readiness(args),
        })
    except ValueError as error:
        raise CalibrationAuthorizationError(str(error)) from error


def _run_pilot_b_readiness(args: argparse.Namespace) -> None:
    try:
        validate_code_prespec(args.code_prespec, REPOSITORY_ROOT)
        plan = REPOSITORY_ROOT / ".omo/plans/phase12-post-filter-v5-calibration-readiness.md"
        result = readiness_from_fixture(args.fixture) if args.fixture is not None else readiness_from_bundle(
            args.bundle, approved_plan_sha256(plan, approval_descriptor_path(plan))
        )
    except CodePrespecError as error:
        result = _blocked("bct", error.code).model_copy(update={"stage": "pilot_b_readiness"})
    result.write_atomic(args.stage_result)
    _print(result.model_dump(mode="json"))


def run_screen_controls(
    *,
    artifact_root: Path,
    run_id: str,
    stage_result: Path,
    authorization: Path | None,
    expected_authorization_sha256: str | None,
    client_factory: Callable[[], None],
) -> CalibrationStageResult:
    return _run_stage(
        "screening", artifact_root, run_id, stage_result, authorization, expected_authorization_sha256,
        ScreeningAuthorizationV1, client_factory,
    )


def _run_cli_stage(args: argparse.Namespace, stage: Literal["screening", "bct"]) -> CalibrationStageResult:
    if stage == "bct":
        require_artifact_root(args.artifact_root)
        if args.artifact_root.exists():
            result = _blocked("bct", "LIVE_ARTIFACT_ROOT_EXISTS")
            result.write_atomic(args.stage_result)
            return result
        plan = REPOSITORY_ROOT / ".omo/plans/phase12-post-filter-v5-calibration-readiness.md"
        raw_screening = waiting_screening_stage(
            REPOSITORY_ROOT / "docs/evidence/phase12-filter-v5-bct-v1",
            approved_plan_sha256(plan, approval_descriptor_path(plan)),
        )
        if raw_screening is not None:
            result = CalibrationStageResult.waiting("bct", raw_screening.terminal_status)
            result.write_atomic(args.stage_result)
            return result

    def factory() -> None:
        _build_live_factory(args.config)

    model = ScreeningAuthorizationV1 if stage == "screening" else BCTAuthorizationV1
    freeze = args.freeze_a if stage == "screening" else args.freeze_b
    return _run_stage(stage, args.artifact_root, args.run_id, args.stage_result, args.authorization, args.expected_authorization_sha256, model, factory, args.authorization_request, args.config, freeze)


def _run_stage(
    stage: Literal["screening", "bct"],
    artifact_root: Path,
    run_id: str,
    stage_result: Path,
    authorization: Path | None,
    expected_digest: str | None,
    model: type[_Authorization],
    client_factory: Callable[[], None],
    authorization_request: Path | None = None,
    config: Path | None = None,
    freeze: Path | None = None,
) -> CalibrationStageResult:
    require_artifact_root(artifact_root)
    if authorization is None and expected_digest is None:
        status = "AWAITING_SCREENING_AUTHORIZATION" if stage == "screening" else "AWAITING_BCT_AUTHORIZATION"
        result = CalibrationStageResult.waiting(stage, status)
    elif authorization is None or expected_digest is None:
        result = _blocked(stage, "AUTHORIZATION_INPUT_PAIR_REQUIRED")
    else:
        try:
            approved = load_authorization(authorization, expected_digest, model)
            if authorization_request is None:
                raise CalibrationAuthorizationError("AUTHORIZATION_REQUEST_MISMATCH")
            if config is None or freeze is None:
                raise CalibrationAuthorizationError("AUTHORIZATION_TRUSTED_INPUT_INVALID")
            validate_runtime_authorization(approved, run_id, authorization_request, config, freeze, artifact_root, REPOSITORY_ROOT, stage)
        except CalibrationAuthorizationError as error:
            result = _blocked(stage, error.code)
        else:
            artifact_root.mkdir(parents=True, exist_ok=True)
            ledger = BudgetLedger(artifact_root / "budget-ledger.jsonl")
            reservation = ledger.reserve_process(stage, run_id)
            try:
                client_factory()
            except TimeoutError:
                ledger.invalidate_timeout(reservation)
                result = CalibrationStageResult(
                    stage=stage,
                    disposition="invalidated",
                    terminal_status="FILTER_V5_PILOT_B_BLOCKED_BY_INVALID_BCT_EVIDENCE",
                    reason_code="CALIBRATION_DEADLINE_EXCEEDED",
                    provider_calls_issued=0,
                )
            else:
                archive_payload: dict[str, object] = {"run_id": run_id, "status": "planned"}
                append_archive_record(artifact_root / run_id, "public", archive_payload)
                append_archive_record(artifact_root / run_id, "audit", {**archive_payload, "stage": stage})
                result = CalibrationStageResult(
                    stage=stage,
                    disposition="completed",
                    terminal_status="CALIBRATION_EXECUTION_READY",
                    reason_code="AUTHORIZATION_VALIDATED",
                    provider_calls_issued=0,
                )
    result.write_atomic(stage_result)
    return result


def _blocked(stage: Literal["screening", "bct"], reason: str) -> CalibrationStageResult:
    return CalibrationStageResult(
        stage=stage,
        disposition="blocked_before_stage",
        terminal_status="FILTER_V5_PILOT_B_BLOCKED_BY_INVALID_CALIBRATION_EVIDENCE",
        reason_code=reason,
        provider_calls_issued=0,
    )


def _write_cost_preview(args: argparse.Namespace, stage: Literal["screening", "bct"]) -> None:
    require_artifact_root(args.ledger.parent)
    payload = build_cost_preview(args, stage, _validate_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    _print(payload)


def _validate_config(path: Path) -> None:
    try:
        validate_calibration_config(path, REPOSITORY_ROOT)
    except CalibrationConfigError as error:
        raise CalibrationAuthorizationError(error.code) from error


load_authorization = _load_authorization


def _build_live_factory(config: Path) -> None:
    _validate_config(config)


def _print(payload: Mapping[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
