from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, assert_never

from pydantic import TypeAdapter, ValidationError

from memcontam.experiment.phase13_ordinary_runtime import (
    OrdinaryArm,
    OrdinaryBaseline,
    ProspectiveBaseline,
)
from memcontam.memory.checkpoint_v3 import (
    CheckpointError,
    NativeState,
    Phase12Checkpoint,
    deserialize_checkpoint,
    serialize_checkpoint,
)
from memcontam.readiness.phase13_main_live_dispatch import MainUnitDispatchOutput
from memcontam.readiness.phase13_main_live_evidence import MainUnitEvidence, PrefixUnitEvidence
from memcontam.readiness.phase13_main_production import ProductionObject


class MainProductionBackendError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class PrefixRuntimeOutput:
    checkpoint: Phase12Checkpoint
    dispatch: MainUnitDispatchOutput


@dataclass(frozen=True, slots=True)
class OrdinaryRuntimeRequest:
    unit: ProductionObject
    baseline: ProspectiveBaseline
    arm: OrdinaryArm
    scientific_arm: str
    prefix_unit_id: str | None
    checkpoint: Phase12Checkpoint | None


PrefixRuntime = Callable[[ProductionObject], PrefixRuntimeOutput]
OrdinaryRuntime = Callable[[OrdinaryRuntimeRequest], MainUnitDispatchOutput]
CompletedEvidenceSha256 = Callable[[str], str]
_MEMORY_BASELINE_ADAPTER: Final = TypeAdapter(OrdinaryBaseline)
_ORDINARY_ARM_ADAPTER: Final = TypeAdapter(OrdinaryArm)


class MainProductionBackend:
    def __init__(
        self,
        root: Path,
        execute_prefix: PrefixRuntime,
        execute_ordinary: OrdinaryRuntime,
        completed_evidence_sha256: CompletedEvidenceSha256 | None = None,
    ) -> None:
        self._root = root
        self._execute_prefix = execute_prefix
        self._execute_ordinary = execute_ordinary
        self._completed_evidence_sha256 = completed_evidence_sha256

    def __call__(self, unit: ProductionObject) -> MainUnitDispatchOutput:
        match unit.kind:
            case "CLEAN_PREFIX":
                return self._prefix(unit)
            case "MEMORY_BEARING":
                return self._memory(unit)
            case "NO_MEMORY_SINGLETON":
                return self._nomem(unit)
            case unreachable:
                assert_never(unreachable)

    def _prefix(self, unit: ProductionObject) -> MainUnitDispatchOutput:
        baseline = unit.memory_baseline
        if baseline is None or unit.arm != "NOT_APPLICABLE" or unit.prefix_unit_id is not None:
            raise MainProductionBackendError("MAIN_PRODUCTION_UNIT_INVALID")
        output = self._execute_prefix(unit)
        try:
            state = deserialize_checkpoint(output.checkpoint)
        except CheckpointError as error:
            raise MainProductionBackendError("MAIN_PREFIX_CHECKPOINT_INVALID") from error
        if state.baseline != baseline:
            raise MainProductionBackendError("MAIN_PREFIX_CHECKPOINT_INVALID")
        return MainUnitDispatchOutput(
            evidence={
                "evidence_kind": "CLEAN_PREFIX",
                "prefix_unit_id": unit.unit_id,
                "checkpoint": {
                    "schema_version": "phase13_main_prefix_checkpoint_v1",
                    "baseline": state.baseline,
                    "checkpoint_id": output.checkpoint.identity.checkpoint_id,
                    "checkpoint_identity_sha256": output.checkpoint.identity.sha256,
                    "canonical_sha256": output.checkpoint.canonical_sha256,
                    "canonical_state_utf8": output.checkpoint.canonical_bytes.decode("utf-8"),
                },
                "runtime_evidence": output.dispatch.evidence,
            },
            provider_calls=output.dispatch.provider_calls,
            realized_cost_krw=output.dispatch.realized_cost_krw,
        )

    def _memory(self, unit: ProductionObject) -> MainUnitDispatchOutput:
        if unit.memory_baseline is None or unit.prefix_unit_id is None:
            raise MainProductionBackendError("MAIN_PRODUCTION_UNIT_INVALID")
        checkpoint = self._load_checkpoint(unit)
        output = self._execute_ordinary(
            OrdinaryRuntimeRequest(
                unit=unit,
                baseline=_memory_baseline(unit.memory_baseline),
                arm=_ordinary_arm(unit.arm),
                scientific_arm=unit.arm,
                prefix_unit_id=unit.prefix_unit_id,
                checkpoint=checkpoint,
            )
        )
        return MainUnitDispatchOutput(
            evidence={
                "evidence_kind": "MEMORY_BEARING",
                "prefix_unit_id": unit.prefix_unit_id,
                "consumed_checkpoint_id": checkpoint.identity.checkpoint_id,
                "consumed_checkpoint_identity_sha256": checkpoint.identity.sha256,
                "consumed_checkpoint_canonical_sha256": checkpoint.canonical_sha256,
                "runtime_evidence": output.evidence,
            },
            provider_calls=output.provider_calls,
            realized_cost_krw=output.realized_cost_krw,
        )

    def _nomem(self, unit: ProductionObject) -> MainUnitDispatchOutput:
        if (
            unit.memory_baseline is not None
            or unit.arm != "NOT_APPLICABLE"
            or unit.prefix_unit_id is not None
        ):
            raise MainProductionBackendError("MAIN_PRODUCTION_UNIT_INVALID")
        output = self._execute_ordinary(
            OrdinaryRuntimeRequest(unit, "nomem", "clean", unit.arm, None, None)
        )
        return MainUnitDispatchOutput(
            evidence={
                "evidence_kind": "NO_MEMORY_SINGLETON",
                "internal_baseline": "nomem",
                "internal_arm": "clean",
                "scientific_arm": "NOT_APPLICABLE",
                "runtime_evidence": output.evidence,
            },
            provider_calls=output.provider_calls,
            realized_cost_krw=output.realized_cost_krw,
        )

    def _load_checkpoint(self, unit: ProductionObject) -> Phase12Checkpoint:
        assert unit.prefix_unit_id is not None
        try:
            if self._completed_evidence_sha256 is None:
                raise OSError
            paths = tuple((self._root / "units").glob(f"*-{unit.prefix_unit_id}.json"))
            if len(paths) != 1:
                raise OSError
            raw = paths[0].read_bytes()
            if hashlib.sha256(raw).hexdigest() != self._completed_evidence_sha256(
                unit.prefix_unit_id
            ):
                raise OSError
            unit_record = MainUnitEvidence.model_validate_json(raw)
            evidence = unit_record.evidence
            if not isinstance(evidence, PrefixUnitEvidence):
                raise TypeError
            record = evidence.checkpoint
            state_payload = json.loads(record.canonical_state_utf8)
            if not isinstance(state_payload, dict):
                raise TypeError
            checkpoint = serialize_checkpoint(NativeState.from_mapping(state_payload))
        except (OSError, ValidationError, json.JSONDecodeError, KeyError, TypeError, CheckpointError) as error:
            raise MainProductionBackendError("MAIN_PREFIX_CHECKPOINT_INVALID") from error
        if (
            unit_record.unit_id != unit.prefix_unit_id
            or unit_record.kind != "CLEAN_PREFIX"
            or unit_record.seed != unit.seed
            or unit_record.task != unit.task
            or unit_record.memory_baseline != unit.memory_baseline
            or unit_record.arm != "NOT_APPLICABLE"
            or evidence.prefix_unit_id != unit.prefix_unit_id
            or record.baseline != unit.memory_baseline
            or record.checkpoint_id != checkpoint.identity.checkpoint_id
            or record.checkpoint_identity_sha256 != checkpoint.identity.sha256
            or record.canonical_sha256 != checkpoint.canonical_sha256
        ):
            raise MainProductionBackendError("MAIN_PREFIX_CHECKPOINT_INVALID")
        return checkpoint


def _memory_baseline(value: str) -> ProspectiveBaseline:
    try:
        return _MEMORY_BASELINE_ADAPTER.validate_python(value, strict=True)
    except ValidationError as error:
        raise MainProductionBackendError("MAIN_PRODUCTION_UNIT_INVALID") from error


def _ordinary_arm(value: str) -> OrdinaryArm:
    try:
        return _ORDINARY_ARM_ADAPTER.validate_python(value, strict=True)
    except ValidationError as error:
        raise MainProductionBackendError("MAIN_PRODUCTION_UNIT_INVALID") from error


__all__ = [
    "MainProductionBackend",
    "MainProductionBackendError",
    "OrdinaryRuntimeRequest",
    "PrefixRuntimeOutput",
]
