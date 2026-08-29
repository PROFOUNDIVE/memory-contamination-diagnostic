from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue

from .phase13_authority_files import read_regular_nofollow
from .phase13_readiness0_f1c import F1CRuntimeError, verify_f1c_runtime
from .phase13_readiness0_f1c_report import F1CReportError, validate_f1c_registry
from .phase13_readiness0_package import (
    Readiness0PackageError,
    validate_implementation_manifest,
    validate_window_proof,
)
from .phase13_readiness0_evidence_validate import EvidenceValidationError, validate_pass_evidence
from .phase13_readiness0_evidence_store import EvidenceStore, EvidenceStoreInputs
from memcontam.readiness.phase13_readiness0_live_models import (
    ArtifactBinding,
    CaseEvidence,
    CaseExecutor,
    EvidenceClosureReport,
    EvidenceManifest,
    F1CRegistry,
    F1CRuntimeMetadata as F1CRuntimeMetadata,
    LiveAuthorization,
    LiveRequest,
    PilotResult,
    Readiness0Case,
    VerifiedReadiness0,
)


class Readiness0LiveError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


READINESS0_CASES: Final = (
    Readiness0Case(
        "nomem_mmlu_engineering_seed0_suffix1", "mmlu_pro_engineering", "nomem",
        ("no_memory_generate",),
    ),
    Readiness0Case(
        "nomem_mmlu_physics_seed0_suffix1", "mmlu_pro_physics", "nomem",
        ("no_memory_generate",),
    ),
    Readiness0Case(
        "fh_bounded_game24_clean_seed0_suffix1", "game24", "fh_bounded",
        ("full_history_generate",),
    ),
    Readiness0Case(
        "rag_frozen_game24_clean_seed0_suffix1", "game24", "rag_frozen",
        ("rag_generate",),
    ),
    Readiness0Case(
        "bot_style_game24_clean_seed0_suffix1", "game24", "bot_style",
        ("bot_problem_distill", "bot_instantiate_solve", "bot_thought_distill"),
    ),
    Readiness0Case(
        "reflexion_game24_clean_seed0_suffix1", "game24", "reflexion_style",
        ("reflexion_generate", "reflexion_reflect", "reflexion_generate"),
    ),
    Readiness0Case(
        "dc_rs_game24_clean_seed0_suffix1", "game24", "dc_rs",
        ("dc_rs_synthesize", "dc_rs_generate"),
    ),
)


def verify_preflight(
    *,
    request_path: Path,
    authorization_path: Path,
    expected_authorization_sha256: str,
    f1c_registry_path: Path,
    repository_root: Path,
    core_root: Path,
    cache_root: Path,
    output_dir: Path,
    allow_live_calls: bool,
) -> VerifiedReadiness0:
    if not allow_live_calls:
        raise Readiness0LiveError("READINESS0_LIVE_CALLS_NOT_ALLOWED")
    request_raw = _read(request_path, "READINESS0_REQUEST_INVALID")
    authorization_raw = _read(authorization_path, "READINESS0_AUTHORIZATION_INVALID")
    f1c_raw = _read(f1c_registry_path, "READINESS0_F1C_INVALID")
    authorization_sha256 = _sha256(authorization_raw)
    if authorization_sha256 != expected_authorization_sha256:
        raise Readiness0LiveError("READINESS0_AUTHORIZATION_HASH_MISMATCH")
    try:
        request = LiveRequest.model_validate_json(request_raw)
        authorization = LiveAuthorization.model_validate_json(authorization_raw)
        f1c = F1CRegistry.model_validate_json(f1c_raw)
    except ValidationError as error:
        raise Readiness0LiveError("READINESS0_PRELIVE_ARTIFACT_INVALID") from error
    request_sha256 = _sha256(request_raw)
    f1c_sha256 = _sha256(f1c_raw)
    if authorization.request_sha256 != request_sha256:
        raise Readiness0LiveError("READINESS0_AUTHORIZATION_REQUEST_MISMATCH")
    if request.request_hash != _canonical_hash(
        request.model_dump(mode="json", exclude={"request_hash"})
    ):
        raise Readiness0LiveError("READINESS0_REQUEST_HASH_MISMATCH")
    if request.case_ids != tuple(case.case_id for case in READINESS0_CASES):
        raise Readiness0LiveError("READINESS0_CASE_MATRIX_MISMATCH")
    if request.f1c_registry.sha256 != f1c_sha256:
        raise Readiness0LiveError("READINESS0_F1C_BINDING_MISMATCH")
    try:
        _require_bindings(repository_root, request, f1c)
    except (F1CReportError, Readiness0PackageError) as error:
        raise Readiness0LiveError(error.code) from error
    if not core_root.is_dir() or not cache_root.is_dir():
        raise Readiness0LiveError("READINESS0_LOCAL_RESOURCE_MISSING")
    configured_cache = os.environ.get(f1c.cache_environment_variable)
    if configured_cache is None or Path(configured_cache).resolve() != cache_root.resolve():
        raise Readiness0LiveError("READINESS0_F1C_CACHE_BINDING_MISMATCH")
    if output_dir.exists():
        raise Readiness0LiveError("READINESS0_OUTPUT_ALREADY_EXISTS")
    if not os.environ.get("OPENAI_API_KEY"):
        raise Readiness0LiveError("READINESS0_CREDENTIAL_MISSING")
    try:
        f1c_runtime = verify_f1c_runtime(f1c, cache_root)
    except F1CRuntimeError as error:
        raise Readiness0LiveError(error.code) from error
    if f1c_runtime.runtime_hash != f1c.runtime_hash:
        raise Readiness0LiveError("READINESS0_F1C_RUNTIME_IDENTITY_MISMATCH")
    return VerifiedReadiness0(
        request, authorization, f1c, request_sha256, authorization_sha256, f1c_sha256,
        output_dir, f1c_runtime,
    )


def run_readiness0_live(
    *,
    request_path: Path,
    authorization_path: Path,
    expected_authorization_sha256: str,
    f1c_registry_path: Path,
    repository_root: Path,
    core_root: Path,
    cache_root: Path,
    output_dir: Path,
    allow_live_calls: bool,
    executor: CaseExecutor | None = None,
) -> PilotResult:
    verified = verify_preflight(
        request_path=request_path,
        authorization_path=authorization_path,
        expected_authorization_sha256=expected_authorization_sha256,
        f1c_registry_path=f1c_registry_path,
        repository_root=repository_root,
        core_root=core_root,
        cache_root=cache_root,
        output_dir=output_dir,
        allow_live_calls=allow_live_calls,
    )
    active_executor = executor
    if active_executor is None:
        from memcontam.readiness.phase13_readiness0_live_runtime import ProductionCaseExecutor

        active_executor = ProductionCaseExecutor(repository_root, core_root, cache_root)
    return execute_verified_pilot(verified, executor=active_executor)


def execute_verified_pilot(
    verified: VerifiedReadiness0, *, executor: CaseExecutor
) -> PilotResult:
    evidence: list[CaseEvidence] = []
    provider_calls = 0
    store = EvidenceStore(
        EvidenceStoreInputs(
            verified.output_dir,
            verified.request_sha256,
            verified.authorization_sha256,
            verified.f1c_sha256,
            verified.f1c_runtime,
        )
    )
    try:
        for case in READINESS0_CASES:
            result = executor(case)
            expected_stages = (
                case.stages
                if result.status == "succeeded"
                else case.stages[: len(result.stages)]
            )
            if result.case_id != case.case_id or not result.stages or result.stages != expected_stages:
                raise Readiness0LiveError("READINESS0_STAGE_EVIDENCE_MISMATCH")
            if result.provider_calls != len(result.stages) or len(result.calls) != len(result.stages):
                raise Readiness0LiveError("READINESS0_CALL_COUNT_MISMATCH")
            if (
                result.status == "succeeded"
                and case.baseline == "reflexion_style"
                and result.routing_verifier_results != (False, True)
            ):
                raise Readiness0LiveError("READINESS0_REFLEXION_ROUTE_MISMATCH")
            evidence.append(result)
            provider_calls += result.provider_calls
            if provider_calls > verified.request.maximum_provider_calls:
                raise Readiness0LiveError("READINESS0_CALL_CEILING_EXCEEDED")
            store.append(result, "FAILED" if result.status == "failed" else "PARTIAL")
            if result.status == "failed":
                return PilotResult("FAILED", provider_calls, _sha256(
                    (verified.output_dir / "evidence_manifest.json").read_bytes()
                ))
    except (OSError, RuntimeError, ValueError) as error:
        store.fail(type(error).__name__)
        raise
    return PilotResult("PASS", provider_calls, store.pass_())


def validate_evidence_closure(root: Path, expected_manifest_sha256: str) -> EvidenceClosureReport:
    raw = _read(root / "evidence_manifest.json", "READINESS0_EVIDENCE_INVALID")
    if _sha256(raw) != expected_manifest_sha256:
        raise Readiness0LiveError("READINESS0_EVIDENCE_MANIFEST_HASH_MISMATCH")
    try:
        manifest = EvidenceManifest.model_validate_json(raw)
        rows = tuple(
            CaseEvidence.model_validate_json(line)
            for line in (root / manifest.cases.path).read_text(encoding="utf-8").splitlines()
        )
    except (OSError, ValidationError) as error:
        raise Readiness0LiveError("READINESS0_EVIDENCE_INVALID") from error
    if (
        manifest.manifest_hash
        != _canonical_hash(manifest.model_dump(mode="json", exclude={"manifest_hash"}))
        or _sha256((root / manifest.cases.path).read_bytes()) != manifest.cases.sha256
        or manifest.case_count != len(rows)
        or manifest.provider_call_count != sum(row.provider_calls for row in rows)
        or tuple(row.case_id for row in rows) != tuple(case.case_id for case in READINESS0_CASES[:len(rows)])
    ):
        raise Readiness0LiveError("READINESS0_EVIDENCE_CLOSURE_MISMATCH")
    try:
        validate_pass_evidence(manifest, rows, READINESS0_CASES)
    except EvidenceValidationError as error:
        raise Readiness0LiveError(error.code) from error
    return EvidenceClosureReport(
        manifest.case_count, manifest.provider_call_count,
        manifest.scientific_result, manifest.main_result,
    )


def _require_bindings(root: Path, request: LiveRequest, f1c: F1CRegistry) -> None:
    for binding in (
        request.implementation_manifest, request.window_proof, request.f1c_registry,
        request.core_manifest, request.legacy_rag_manifest,
        request.checkpoint_registry, request.observability_packet,
    ):
        path = Path(binding.path)
        if path.is_absolute() or ".." in path.parts or _sha256(_read(root / path, "READINESS0_BINDING_MISSING")) != binding.sha256:
            raise Readiness0LiveError("READINESS0_ARTIFACT_BINDING_MISMATCH")
    if f1c.legacy_rag_manifest != request.legacy_rag_manifest:
        raise Readiness0LiveError("READINESS0_F1C_BINDING_MISMATCH")
    report_raw = _read(root / f1c.report.path, "READINESS0_F1C_REPORT_INVALID")
    if _sha256(report_raw) != f1c.report.sha256:
        raise Readiness0LiveError("READINESS0_F1C_BINDING_MISMATCH")
    validate_f1c_registry(
        _read(root / request.f1c_registry.path, "READINESS0_F1C_INVALID"),
        report_raw,
        root,
    )
    validate_implementation_manifest(
        _read(
            root / request.implementation_manifest.path,
            "READINESS0_IMPLEMENTATION_MANIFEST_INVALID",
        ),
        root,
    )
    validate_window_proof(
        _read(root / request.window_proof.path, "READINESS0_WINDOW_PROOF_INVALID"),
        root / "data/phase13/main/mr_p4",
        root,
    )


def _read(path: Path, code: str) -> bytes:
    try:
        return read_regular_nofollow(path)
    except OSError as error:
        raise Readiness0LiveError(code) from error


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_hash(payload: dict[str, JsonValue]) -> str:
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


__all__ = [
    "READINESS0_CASES", "ArtifactBinding", "CaseEvidence", "F1CRuntimeMetadata", "PilotResult",
    "Readiness0Case",
    "Readiness0LiveError", "VerifiedReadiness0", "execute_verified_pilot",
    "run_readiness0_live", "validate_evidence_closure", "verify_preflight",
]
