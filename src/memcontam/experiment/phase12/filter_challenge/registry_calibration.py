from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Final, Literal, TypeAlias

from pydantic import Field, model_validator

from memcontam.experiment.phase12.filter_challenge.registry_common import StrictRegistry
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    EvidenceBuildError,
    approval_descriptor_path,
    approved_plan_sha256,
)


ARTIFACT_ROOT: Final = Path(
    "/home/hyunwoo/git/memory-contamination-diagnostic-filter-v5/runs/phase12-filter-v5-bct-live-v1"
)
LEDGER_ID: Final = "filter-v5-bct-budget-v1"
Task: TypeAlias = Literal["game24", "math_equation_balancer", "word_sorting"]
Baseline: TypeAlias = Literal["full_history", "rag_frozen", "bot_style", "reflexion_style"]
CandidateClass: TypeAlias = Literal["certified_false", "correct", "irrelevant", "ordinary_false"]
TASKS: Final[tuple[Task, ...]] = ("game24", "math_equation_balancer", "word_sorting")
BASELINES: Final[tuple[Baseline, ...]] = ("full_history", "rag_frozen", "bot_style", "reflexion_style")
CANDIDATE_CLASSES: Final[tuple[CandidateClass, ...]] = ("certified_false", "correct", "irrelevant", "ordinary_false")
CONTROL_SIDE: Final[tuple[Literal["control"], ...]] = ("control",)
_CALIBRATION_SCHEMA: Final = "schema_version: phase12_fv5_bct_calibration_methods_v1\n"
_PLAN_DIGEST: Final = re.compile(r"^approved_plan_sha256: ([0-9a-f]{64})$", re.MULTILINE)


class CalibrationRegistryError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CalibrationConfigError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def validate_calibration_config(path: Path, repository_root: Path) -> None:
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise CalibrationConfigError("CALIBRATION_CONFIG_INVALID") from error
    if not payload.startswith(_CALIBRATION_SCHEMA):
        raise CalibrationConfigError("CALIBRATION_CONFIG_INVALID")
    matches = _PLAN_DIGEST.findall(payload)
    if len(matches) != 1:
        raise CalibrationConfigError("CALIBRATION_CONFIG_INVALID")
    plan = repository_root / ".omo" / "plans" / "phase12-post-filter-v5-calibration-readiness.md"
    try:
        approved = approved_plan_sha256(plan, approval_descriptor_path(plan))
    except EvidenceBuildError as error:
        raise CalibrationConfigError("CALIBRATION_CONFIG_PLAN_MISMATCH") from error
    if matches[0] != approved:
        raise CalibrationConfigError("CALIBRATION_CONFIG_PLAN_MISMATCH")


class CalibrationAuthorization(StrictRegistry):
    authorization_id: str
    issued_at: datetime
    expires_at: datetime
    run_id: str
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    artifact_root: str
    ledger_id: Literal["filter-v5-bct-budget-v1"]
    model_id: Literal["gpt-4o-2024-11-20"]
    approved_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authority_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    freeze_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: Literal["openai_responses"]
    decoding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    maximum_calls: int = Field(ge=1, le=570)
    maximum_input_tokens: int = Field(ge=1, le=2_334_720)
    maximum_output_tokens: int = Field(ge=1, le=364_800)
    hard_ceiling_microusd: int = Field(ge=1, le=10_000_000)
    maximum_wall_seconds: int = Field(ge=1, le=10_800)

    @model_validator(mode="after")
    def _validate_expiry(self) -> CalibrationAuthorization:
        if self.expires_at <= self.issued_at:
            raise CalibrationRegistryError("AUTHORIZATION_EXPIRY_INVALID")
        if self.artifact_root != str(ARTIFACT_ROOT):
            raise CalibrationRegistryError("ARTIFACT_ROOT_INVALID")
        return self


class ScreeningAuthorizationV1(CalibrationAuthorization):
    schema_version: Literal["phase12_fv5_screening_authorization_v1"] = (
        "phase12_fv5_screening_authorization_v1"
    )


class BCTAuthorizationV1(CalibrationAuthorization):
    schema_version: Literal["phase12_fv5_bct_authorization_v1"] = "phase12_fv5_bct_authorization_v1"
    screening_terminal_seal: str = Field(pattern=r"^[0-9a-f]{64}$")
    ledger_head: str = Field(pattern=r"^[0-9a-f]{64}$")


class CalibrationStageResult(StrictRegistry):
    schema_version: Literal["phase12_fv5_calibration_stage_result_v1"] = (
        "phase12_fv5_calibration_stage_result_v1"
    )
    stage: Literal["screening", "bct", "pilot_b_readiness"]
    disposition: Literal["completed", "blocked_before_stage", "skipped_structural", "invalidated"]
    terminal_status: str
    reason_code: str
    provider_calls_issued: int = Field(ge=0)

    @classmethod
    def waiting(cls, stage: Literal["screening", "bct", "pilot_b_readiness"], status: str) -> CalibrationStageResult:
        return cls(
            stage=stage,
            disposition="blocked_before_stage",
            terminal_status=status,
            reason_code=status,
            provider_calls_issued=0,
        )

    def write_atomic(self, path: Path) -> None:
        _write_atomic(path, self.model_dump(mode="json"))


class ScheduledCall(StrictRegistry):
    call_id: str
    task: Task
    baseline: Baseline
    probe_id: str
    side: Literal["control", "challenge"]
    candidate_class: CandidateClass | None
    replicate: int | None
    native_stage: Literal["answer", "bot_problem_distill", "bot_instantiate_solve"]


def screening_schedule(probes: dict[str, tuple[str, ...]]) -> tuple[ScheduledCall, ...]:
    _validate_probes(probes, 6)
    return tuple(
        _scheduled("screen", task, baseline, probe, side, None, None, stage, ordinal)
        for task in TASKS
        for baseline in BASELINES
        for probe in probes[task]
        for side in CONTROL_SIDE
        for ordinal, stage in enumerate(_stages(baseline), start=1)
    )


def bct_schedule(probes: dict[str, tuple[str, ...]]) -> tuple[ScheduledCall, ...]:
    _validate_probes(probes, 2)
    return tuple(
        _scheduled("bct", task, baseline, probe, side, candidate, replicate, stage, ordinal)
        for task in TASKS
        for baseline in BASELINES
        for probe in probes[task]
        for candidate in CANDIDATE_CLASSES
        for replicate in (1, 2)
        for side in (("control", "challenge") if replicate == 1 else ("challenge", "control"))
        for ordinal, stage in enumerate(_stages(baseline), start=1)
    )


def stable_json_hash(payload: StrictRegistry) -> str:
    canonical = json.dumps(payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def require_artifact_root(path: Path) -> Path:
    if path.resolve(strict=False) != ARTIFACT_ROOT:
        raise CalibrationRegistryError("ARTIFACT_ROOT_INVALID")
    return ARTIFACT_ROOT


def _scheduled(
    kind: str,
    task: Task,
    baseline: Baseline,
    probe: str,
    side: str,
    candidate: CandidateClass | None,
    replicate: int | None,
    stage: Literal["answer", "bot_problem_distill", "bot_instantiate_solve"],
    ordinal: int,
) -> ScheduledCall:
    if side == "control":
        normalized_side: Literal["control", "challenge"] = "control"
    elif side == "challenge":
        normalized_side = "challenge"
    else:
        raise CalibrationRegistryError("CALIBRATION_PROBE_SCHEDULE_INVALID")
    candidate_suffix = "" if candidate is None else f"-{candidate}-r{replicate}"
    return ScheduledCall(
        call_id=f"fv5-{kind}-{task}-{baseline}-{probe}{candidate_suffix}-{side}-{stage}-call{ordinal}",
        task=task,
        baseline=baseline,
        probe_id=probe,
        side=normalized_side,
        candidate_class=candidate,
        replicate=replicate,
        native_stage=stage,
    )


def _stages(baseline: Baseline) -> tuple[Literal["answer", "bot_problem_distill", "bot_instantiate_solve"], ...]:
    return ("bot_problem_distill", "bot_instantiate_solve") if baseline == "bot_style" else ("answer",)


def _validate_probes(probes: dict[str, tuple[str, ...]], expected_count: int) -> None:
    if tuple(probes) != TASKS or any(len(probes[task]) != expected_count for task in TASKS):
        raise CalibrationRegistryError("CALIBRATION_PROBE_SCHEDULE_INVALID")
    if any(len(set(probes[task])) != expected_count for task in TASKS):
        raise CalibrationRegistryError("CALIBRATION_PROBE_SCHEDULE_INVALID")


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
