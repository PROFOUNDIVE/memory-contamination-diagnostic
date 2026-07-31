from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import stat
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Literal, Mapping, TypeVar

from pydantic import ValidationError

from memcontam.experiment.phase12.filter_challenge.bct_archive import (
    BudgetLedger,
    append_archive_record,
    validate_live_archive,
)
from memcontam.experiment.phase12.filter_challenge.registry_calibration import (
    ARTIFACT_ROOT,
    BCTAuthorizationV1,
    CalibrationAuthorization,
    CalibrationStageResult,
    LEDGER_ID,
    ScreeningAuthorizationV1,
    require_artifact_root,
)


class CalibrationAuthorizationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


_Authorization = TypeVar("_Authorization", bound=CalibrationAuthorization)


def add_calibration_parsers(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    validate = commands.add_parser("validate-calibration-config")
    validate.add_argument("--config", type=Path, required=True)
    for name, freeze in (("screening-cost-preview", "--freeze-a"), ("bct-cost-preview", "--freeze-b")):
        preview = commands.add_parser(name)
        preview.add_argument("--config", type=Path, required=True)
        preview.add_argument(freeze, type=Path, required=True)
        preview.add_argument("--ledger", type=Path, required=True)
        preview.add_argument("--output", type=Path, required=True)
    for name, authorization in (("screen-controls", "screening"), ("bct-run", "bct")):
        run = commands.add_parser(name)
        run.add_argument("--config", type=Path, required=True)
        run.add_argument("--freeze-a" if authorization == "screening" else "--freeze-b", type=Path, required=True)
        run.add_argument("--authorization-request", type=Path, required=True)
        run.add_argument("--authorization", type=Path)
        run.add_argument("--expected-authorization-sha256")
        run.add_argument("--artifact-root", type=Path, required=True)
        run.add_argument("--run-id", required=True)
        run.add_argument("--stage-result", type=Path, required=True)
    archive = commands.add_parser("validate-bct-archive")
    archive.add_argument("--config", type=Path, required=True)
    archive.add_argument("--freeze-b", type=Path, required=True)
    archive.add_argument("--archive", type=Path, required=True)
    archive.add_argument("--output", type=Path, required=True)
    readiness = commands.add_parser("pilot-b-readiness")
    readiness.add_argument("--bundle", type=Path, required=True)
    readiness.add_argument("--stage-result", type=Path, required=True)


def run_calibration_command(args: argparse.Namespace) -> None:
    command = args.filter_v5_command
    match command:
        case "validate-calibration-config":
            _validate_config(args.config)
            _print({"valid": True, "provider_calls_issued": 0})
        case "screening-cost-preview" | "bct-cost-preview":
            _cost_preview(args, "screening" if command == "screening-cost-preview" else "bct")
        case "screen-controls":
            _print(_run_cli_stage(args, "screening").model_dump(mode="json"))
        case "bct-run":
            _print(_run_cli_stage(args, "bct").model_dump(mode="json"))
        case "validate-bct-archive":
            report = validate_live_archive(args.archive)
            payload = asdict(report)
            args.output.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            _print(payload)
        case "pilot-b-readiness":
            result = CalibrationStageResult.waiting("pilot_b_readiness", "AWAITING_SCREENING_AUTHORIZATION")
            result.write_atomic(args.stage_result)
            _print(result.model_dump(mode="json"))
        case _:
            raise CalibrationAuthorizationError("CALIBRATION_COMMAND_UNKNOWN")


def load_authorization(path: Path, expected_digest: str, model: type[_Authorization]) -> _Authorization:
    if not hmac.compare_digest(expected_digest.encode("ascii"), hashlib.sha256(_read_nofollow(path)).hexdigest().encode("ascii")):
        raise CalibrationAuthorizationError("AUTHORIZATION_DIGEST_MISMATCH")
    try:
        return model.model_validate_json(_read_nofollow(path))
    except (ValidationError, UnicodeDecodeError) as error:
        raise CalibrationAuthorizationError("AUTHORIZATION_INVALID") from error


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
    def factory() -> None:
        _build_live_factory(args.config)

    model = ScreeningAuthorizationV1 if stage == "screening" else BCTAuthorizationV1
    return _run_stage(stage, args.artifact_root, args.run_id, args.stage_result, args.authorization, args.expected_authorization_sha256, model, factory, args.authorization_request)


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
            _validate_runtime_authorization(approved, run_id)
            if authorization_request is None or approved.request_sha256 != hashlib.sha256(authorization_request.read_bytes()).hexdigest():
                raise CalibrationAuthorizationError("AUTHORIZATION_REQUEST_MISMATCH")
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


def _validate_runtime_authorization(authorization: CalibrationAuthorization, run_id: str) -> None:
    if authorization.run_id != run_id:
        raise CalibrationAuthorizationError("AUTHORIZATION_RUN_ID_MISMATCH")
    if authorization.expires_at <= datetime.now(UTC):
        raise CalibrationAuthorizationError("AUTHORIZATION_EXPIRED")
    if authorization.ledger_id != LEDGER_ID:
        raise CalibrationAuthorizationError("AUTHORIZATION_LEDGER_MISMATCH")


def _cost_preview(args: argparse.Namespace, stage: Literal["screening", "bct"]) -> None:
    _validate_config(args.config)
    require_artifact_root(args.ledger.parent)
    calls, wall_seconds, hard_ceiling = (90, 3600, 2) if stage == "screening" else (480, 7200, 8)
    freeze_path = args.freeze_a if stage == "screening" else args.freeze_b
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (AttributeError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationAuthorizationError("CALIBRATION_FREEZE_INVALID") from error
    if not isinstance(freeze, dict):
        raise CalibrationAuthorizationError("CALIBRATION_FREEZE_INVALID")
    schedule = freeze.get("method_call_schedule")
    if stage == "screening" and (not isinstance(schedule, list) or len(schedule) != calls):
        raise CalibrationAuthorizationError("CALL_SCHEDULE_MISMATCH")
    payload = {"schema_version": "phase12_fv5_authorization_request_v1", "stage": stage, "maximum_calls": calls, "maximum_input_tokens": 368640 if stage == "screening" else 1966080, "maximum_output_tokens": 57600 if stage == "screening" else 307200, "wall_seconds": wall_seconds, "hard_ceiling_usd": hard_ceiling, "ledger_id": LEDGER_ID, "artifact_root": str(ARTIFACT_ROOT), "freeze_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(), "schedule_sha256": hashlib.sha256(json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(), "approved_plan_sha256": freeze.get("approved_plan_sha256"), "provider_calls_issued": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    _print(payload)


def _validate_config(path: Path) -> None:
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise CalibrationAuthorizationError("CALIBRATION_CONFIG_INVALID") from error
    if not payload.startswith("schema_version: phase12_fv5_bct_calibration_methods_v1\n"):
        raise CalibrationAuthorizationError("CALIBRATION_CONFIG_INVALID")


def _read_nofollow(path: Path) -> bytes:
    target = path.absolute()
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        parts = target.parts[1:]
        for part in parts[:-1]:
            next_descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        file_descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
        try:
            if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
                raise CalibrationAuthorizationError("AUTHORIZATION_INVALID")
            return os.read(file_descriptor, os.fstat(file_descriptor).st_size)
        finally:
            os.close(file_descriptor)
    except OSError as error:
        raise CalibrationAuthorizationError("AUTHORIZATION_INVALID") from error
    finally:
        os.close(descriptor)


def _build_live_factory(config: Path) -> None:
    _validate_config(config)


def _print(payload: Mapping[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
