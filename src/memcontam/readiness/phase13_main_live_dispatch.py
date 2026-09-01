from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from memcontam.logging.schema import MethodCall
from memcontam.readiness.phase13_main_live_evidence import (
    DispatchEvidenceInput,
    MainEvidenceValidationError,
    MainReconciliationEvidence,
    MainUnitEvidence,
    validate_dispatch_evidence,
)
from memcontam.readiness.phase13_main_runner_models import (
    DispatchCompleted,
    ExecutionUnit,
    InFlightContext,
    InFlightEvidence,
)


class MainLiveDispatchError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MainUnitDispatchOutput(_FrozenModel):
    evidence: JsonValue
    provider_calls: tuple[MethodCall, ...]
    realized_cost_krw: int = Field(ge=0)


MainUnitBackend = Callable[[ExecutionUnit], MainUnitDispatchOutput]


class DurableMainDispatch:
    def __init__(self, root: Path, backend: MainUnitBackend) -> None:
        self._root = root
        self._backend = backend

    def __call__(self, unit: ExecutionUnit) -> DispatchCompleted:
        output = self._backend(unit)
        return persist_unit_dispatch(self._root, unit, output)


class MainTelemetrySummary(_FrozenModel):
    schema_version: Literal["phase13_main_telemetry_summary_v1"]
    unit_count: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    transport_attempt_count: int = Field(ge=0)
    latency_ms: int = Field(ge=0)
    token_usage: dict[str, int]
    provider_cost_usd: str
    realized_cost_krw: int = Field(ge=0)


def persist_unit_dispatch(
    root: Path,
    unit: ExecutionUnit,
    output: MainUnitDispatchOutput,
) -> DispatchCompleted:
    try:
        joined_evidence, reconciled_cost_krw = validate_dispatch_evidence(
            unit,
            DispatchEvidenceInput(
                output.evidence,
                output.provider_calls,
                output.realized_cost_krw,
            ),
        )
    except MainEvidenceValidationError as error:
        raise MainLiveDispatchError(error.code) from error
    record = MainUnitEvidence(
        schema_version="phase13_main_unit_evidence_v1",
        sequence=unit.sequence,
        unit_id=unit.unit_id,
        kind=unit.kind,
        seed=unit.seed,
        task=unit.task,
        memory_baseline=unit.memory_baseline,
        arm=unit.arm,
        evidence=joined_evidence,
        provider_calls=output.provider_calls,
        realized_cost_krw=reconciled_cost_krw,
    )
    raw = json.dumps(
        record.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    directory = root / "units"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{unit.sequence:06d}-{unit.unit_id}.json"
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as evidence_file:
            evidence_file.write(raw)
            evidence_file.flush()
            os.fsync(evidence_file.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise MainLiveDispatchError("MAIN_UNIT_EVIDENCE_ALREADY_EXISTS") from error
        _fsync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return DispatchCompleted(hashlib.sha256(raw).hexdigest(), reconciled_cost_krw)


def persist_reconciliation_evidence(
    root: Path,
    disposition: Literal["NO_PROVIDER_REQUEST", "TERMINAL_FAILURE"],
    context: InFlightContext,
    *,
    realized_cost_krw: int = 0,
    failure_code: str | None = None,
) -> InFlightEvidence:
    record = MainReconciliationEvidence(
        schema_version="phase13_main_reconciliation_evidence_v1",
        disposition=disposition,
        unit_id=context.unit_id,
        intent_event_hash=context.intent_event_hash,
        package_sha256=context.package_sha256,
        authorization_sha256=context.authorization_sha256,
        realized_cost_krw=realized_cost_krw,
        failure_code=failure_code,
    )
    raw = json.dumps(
        record.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    directory = root / "reconciliation"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{context.intent_event_hash}.json"
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as evidence_file:
            evidence_file.write(raw)
            evidence_file.flush()
            os.fsync(evidence_file.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise MainLiveDispatchError("MAIN_RECONCILIATION_EVIDENCE_ALREADY_EXISTS") from error
        _fsync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(raw).hexdigest()
    if disposition == "NO_PROVIDER_REQUEST":
        return InFlightEvidence.no_provider_request(context, digest)
    if failure_code is None:
        raise MainLiveDispatchError("MAIN_RECONCILIATION_EVIDENCE_INVALID")
    return InFlightEvidence.terminal_failure(
        context,
        digest,
        failure_code,
        realized_cost_krw,
    )


def summarize_telemetry(root: Path) -> MainTelemetrySummary:
    records = tuple(_load_record(path) for path in sorted((root / "units").glob("*.json")))
    calls = tuple(call for record in records for call in record.provider_calls)
    token_usage: dict[str, int] = {}
    for call in calls:
        for name, value in call.token_usage.items():
            token_usage[name] = token_usage.get(name, 0) + value
    cost = sum(
        (Decimal(str(call.provider_cost_usd)) for call in calls if call.provider_cost_usd is not None),
        start=Decimal(0),
    )
    return MainTelemetrySummary(
        schema_version="phase13_main_telemetry_summary_v1",
        unit_count=len(records),
        provider_call_count=len(calls),
        transport_attempt_count=sum(call.transport_attempts for call in calls),
        latency_ms=sum(call.latency_ms or 0 for call in calls),
        token_usage=dict(sorted(token_usage.items())),
        provider_cost_usd=_decimal_text(cost),
        realized_cost_krw=sum(record.realized_cost_krw for record in records),
    )


def load_live_environment(path: Path, *, required_keys: tuple[str, ...]) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise MainLiveDispatchError("MAIN_LIVE_ENV_INVALID") from error
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            values[key] = value
    environment = {**os.environ, **values}
    if any(not environment.get(key) for key in required_keys):
        raise MainLiveDispatchError("MAIN_LIVE_CREDENTIAL_MISSING")
    return environment


def _load_record(path: Path) -> MainUnitEvidence:
    try:
        return MainUnitEvidence.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as error:
        raise MainLiveDispatchError("MAIN_UNIT_EVIDENCE_INVALID") from error


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "DurableMainDispatch",
    "MainLiveDispatchError",
    "MainTelemetrySummary",
    "MainUnitDispatchOutput",
    "load_live_environment",
    "persist_unit_dispatch",
    "persist_reconciliation_evidence",
    "summarize_telemetry",
]
