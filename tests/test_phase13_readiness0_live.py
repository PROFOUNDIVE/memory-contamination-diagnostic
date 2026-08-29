from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import pytest

import memcontam.readiness.phase13_readiness0_live as live
import memcontam.readiness.phase13_readiness0_live_runtime as live_runtime
from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.experiment.phase12.runtime_registry import RuntimeTrialResult
from memcontam.logging.schema import MethodCall, PromptSourceSpan
from memcontam.readiness.phase13_cli import add_parser, run
from memcontam.readiness.phase13_cost_policy import load_cost_policy_bundle
from memcontam.readiness.phase13_production_runtime_models import ProductionOrdinaryRunIdentity
from memcontam.readiness.phase13_readiness0_evidence_models import (
    ProviderCallEvidence,
    RuntimeJoinEvidence,
)
from memcontam.readiness.phase13_readiness0_evidence_validate import (
    EvidenceValidationError,
    validate_pass_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "data/phase13/main/mr_p4"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified(output_dir: Path) -> live.VerifiedReadiness0:
    binding = live.ArtifactBinding(path="test", sha256="0" * 64)
    request = live.LiveRequest(
        schema_version="phase13_readiness0_live_request_v1", status="PRE_LIVE_AUTHORIZED",
        scientific_result=False, main_result=False, measured_main_a_trajectory_count=0,
        case_ids=tuple(case.case_id for case in live.READINESS0_CASES), maximum_provider_calls=12,
        f1c_registry=binding, core_manifest=binding, legacy_rag_manifest=binding,
        checkpoint_registry=binding, observability_packet=binding,
        implementation_manifest=binding, window_proof=binding,
        credentials_source="CURRENT_PROCESS_ENVIRONMENT_ONLY", request_hash="0" * 64,
    )
    authorization = live.LiveAuthorization(
        schema_version="phase13_readiness0_live_authorization_v1",
        scope="MINIMUM_PRODUCTION_FACING_READINESS0_LIVE_PILOT", request_sha256="0" * 64,
        allow_live_calls=True, maximum_provider_calls=12,
        authorizes_provider_backed_scientific_calibration=False, authorizes_mr_p5=False,
        authorizes_mr_p6=False, authorizes_main_a=False,
        answer_correctness_acceptance_criterion=False,
    )
    f1c = live.F1CRegistry(
        schema_version="phase13_readiness0_f1c_registry_v1", status="PASS",
        cache_environment_variable="MEMCONTAM_BGE_CACHE_DIR", local_files_only=True,
        model_id="BAAI/bge-m3", revision="5617a9f61b028005a4858fdac845db406aefb181",
        normalize_embeddings=True, vector_dimension=1024, report=binding,
        runtime_hash="0" * 64, legacy_rag_manifest=binding,
        ready_legacy_cells=("game24", "math_equation_balancer", "word_sorting"),
        f1c_hash="0" * 64,
    )
    f1c_runtime = live.F1CRuntimeMetadata(
        provider_identity="BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181",
        vector_dimension=1024, normalize_embeddings=True, python="test",
        sentence_transformers="test", torch="test", device="cpu", dtype="float32",
        local_files_only=True, network_attempts=0, runtime_hash="0" * 64,
    )
    return live.VerifiedReadiness0(
        request, authorization, f1c, "0" * 64, "0" * 64, "0" * 64, output_dir,
        f1c_runtime,
    )


def _case_evidence(
    case: live.Readiness0Case,
    status: Literal["succeeded", "failed"] = "succeeded",
) -> live.CaseEvidence:
    answer_stage = {
        "nomem": "no_memory_generate",
        "fh_bounded": "full_history_generate",
        "rag_frozen": "rag_generate",
        "bot_style": "bot_instantiate_solve",
        "reflexion_style": "reflexion_generate",
        "dc_rs": "dc_rs_generate",
    }[case.baseline]
    bundle = load_cost_policy_bundle(ROOT)
    stages = {stage.semantic_stage_id: stage for stage in bundle.registry.stages}
    calls = tuple(
        ProviderCallEvidence(
            call_id=f"{case.case_id}:call:{index}", stage=stage, raw_response="answer",
            transport_attempts=1, token_usage={"total_tokens": 2}, latency_ms=1,
            provider_cost_usd=0.000001, provider_response_id=f"resp-{index}",
            provider_usage={"input_tokens": 1, "output_tokens": 1},
            provider_service_tier="default", requested_model="gpt-5.6-luna",
            returned_model="gpt-5.6-luna", response_status="completed",
            reasoning_mode="standard", reasoning_effort="none",
            reasoning_context="current_turn", previous_response_id=None,
            store=False, tools=(), maximum_input_tokens=stages[stage].maximum_input_tokens,
            maximum_output_tokens=stages[stage].maximum_output_tokens,
            execution_envelope_id="CORE_EXECUTION_ENVELOPE_REGISTRY_V2",
            execution_envelope_sha256="4c48fca92d1d70105d2eb34b5b86984c732c03e3600cb00965501ecabd2d1769",
            failure_contract_id="CORE_TRANSPORT_ATTEMPT_CONTRACT_V2",
            failure_contract_sha256="1ee66fcb795f97d483c2ef976133ee61dbd5108c9dae851c2c2786ff496d788f",
            terminal_failure_contract_id="CORE_TERMINAL_TECHNICAL_MISSINGNESS_V1",
            terminal_failure_contract_sha256="9bbcdd9dd1686af034f7c0d2114ac86d5837a07de0cc6ba8fef7940bbc822b75",
            raw_usage={"input_tokens": 1, "output_tokens": 1},
            normalized_usage={"total_tokens": 2}, authoritative_provider_cost_usd=None,
            derived_cost_usd=0.000001, cost_source="DERIVED_FROM_PROVIDER_USAGE",
            rate_card_sha256="50975b67dce4c59ba9267c3234a873076137ded5078aa3e8b5c9a2fad4ff3e06",
            source_spans=(),
        )
        for index, stage in enumerate(case.stages, start=1)
    )
    reflexion = case.baseline == "reflexion_style"
    source_span_join = tuple(
        {"call_id": call.call_id, "source_spans": ()} for call in calls
    )
    return live.CaseEvidence(
        case_id=case.case_id,
        status=status,
        stages=case.stages,
        provider_calls=len(calls),
        calls=calls,
        answer_call_id=next(call.call_id for call in reversed(calls) if call.stage == answer_stage),
        runtime=RuntimeJoinEvidence(
            task=case.task, baseline=case.baseline, sample_id="sample-1", suffix_position=1,
            sample_order=1, trajectory_seed=0, concrete_seed_id="0",
            execution_template_id=f"readiness0|{case.task}|{case.baseline}|clean",
            ordered_sample_ids_sha256="1" * 64,
            checkpoint_registry_sha256=_sha256(
                ARTIFACT_ROOT / "main_a_common_checkpoint_registry_v1.json"
            ),
            registration_packet_sha256=_sha256(
                ROOT / "data/phase13/observability/registration_packet_v1.json"
            ),
            retrieval_query_sha256=("6" * 64 if case.baseline == "rag_frozen" else None),
            retrieval_candidates_sha256=(
                hashlib.sha256(
                    json.dumps(
                        {"entry_ids": ("entry-1",), "scores": (0.9,)},
                        sort_keys=True, separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                if case.baseline == "rag_frozen" else None
            ),
            retrieval_source_span_sha256=hashlib.sha256(
                json.dumps(source_span_join, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            retrieval_event_id=("retrieval-1" if case.baseline == "rag_frozen" else None),
            retrieved_entry_ids=(("entry-1",) if case.baseline == "rag_frozen" else ()),
            retrieved_scores=((0.9,) if case.baseline == "rag_frozen" else ()),
            memory_before_sha256="4" * 64,
            memory_after_sha256="5" * 64,
            capacity_law_id=(
                "luna_common_visible_memory_capacity_v1"
                if case.baseline in {"fh_bounded", "dc_rs"}
                else None
            ),
            capacity_tokens=8192 if case.baseline in {"fh_bounded", "dc_rs"} else None,
            capacity_artifact_sha256=bundle.proof.common_capacity_sha256,
            task_order_sha256=_sha256(ARTIFACT_ROOT / "task_seed_orders_v1.json"),
            analysis_window_id="core_prefix_50",
            analysis_window_registry_sha256=_sha256(
                ARTIFACT_ROOT / "readiness0_window_proof_v1.json"
            ),
            text_only=True, tool_execution_count=0,
        ),
        reflexion_route_policy_id=(
            "readiness0_reflexion_fail_then_pass_v1" if reflexion else None
        ),
        routing_verifier_results=(False, True) if reflexion else (),
        actual_verifier_results=(False, False) if reflexion else (False,),
        scientific_result=False,
        main_result=False,
    )


def test_readiness0_live_matrix_is_exactly_seven_cases_and_twelve_stages() -> None:
    assert [(case.case_id, case.stages) for case in live.READINESS0_CASES] == [
        ("nomem_mmlu_engineering_seed0_suffix1", ("no_memory_generate",)),
        ("nomem_mmlu_physics_seed0_suffix1", ("no_memory_generate",)),
        ("fh_bounded_game24_clean_seed0_suffix1", ("full_history_generate",)),
        ("rag_frozen_game24_clean_seed0_suffix1", ("rag_generate",)),
        (
            "bot_style_game24_clean_seed0_suffix1",
            ("bot_problem_distill", "bot_instantiate_solve", "bot_thought_distill"),
        ),
        (
            "reflexion_game24_clean_seed0_suffix1",
            ("reflexion_generate", "reflexion_reflect", "reflexion_generate"),
        ),
        (
            "dc_rs_game24_clean_seed0_suffix1",
            ("dc_rs_synthesize", "dc_rs_generate"),
        ),
    ]
    assert sum(len(case.stages) for case in live.READINESS0_CASES) == 12
    assert all(case.suffix_position == 1 for case in live.READINESS0_CASES)


def test_case_evidence_schema_requires_operational_and_runtime_joins() -> None:
    assert {
        "calls",
        "answer_call_id",
        "runtime",
        "reflexion_route_policy_id",
        "routing_verifier_results",
        "actual_verifier_results",
    } <= live.CaseEvidence.model_fields.keys()


def test_case_evidence_extracts_answer_call_telemetry_and_runtime_joins() -> None:
    span = PromptSourceSpan(
        message_index=0, start=0, end=4, rendered_hash="sha256:test", entry_id="entry-1",
        source_ids=["entry-1"], parent_ids=[], lineage_id="lineage-1", version="v1",
        origin="full_history", clean_or_contaminated="clean",
    )
    call = MethodCall(
        call_id="trial:call:1", stage="full_history_generate", raw_response="1+1+1+1",
        model="gpt-5.6-luna", latency_ms=23,
        token_usage={"prompt_tokens": 7, "completion_tokens": 11}, transport_attempts=1,
        provider_cost_usd=0.25, provider_response_id="resp_complete",
        provider_usage={"input_tokens": 7, "output_tokens": 11},
        provider_service_tier="default", provider_returned_model="gpt-5.6-luna",
        provider_response_status="completed",
        provider_request_contract={
            "model": "gpt-5.6-luna", "input_sha256": "4" * 64,
            "temperature": 0.0, "top_p": 1,
            "reasoning": {"mode": "standard", "effort": "none", "context": "current_turn"},
            "previous_response_id": None, "service_tier": "default", "store": False,
            "tools": [], "max_output_tokens": 2048,
        },
        provider_authority_contract={
            "maximum_input_tokens": 8192, "maximum_output_tokens": 2048,
            "execution_envelope_id": "CORE_EXECUTION_ENVELOPE_REGISTRY_V2",
            "execution_envelope_sha256": "4c48fca92d1d70105d2eb34b5b86984c732c03e3600cb00965501ecabd2d1769",
            "failure_contract_id": "CORE_TRANSPORT_ATTEMPT_CONTRACT_V2",
            "failure_contract_sha256": "1ee66fcb795f97d483c2ef976133ee61dbd5108c9dae851c2c2786ff496d788f",
            "terminal_failure_contract_id": "CORE_TERMINAL_TECHNICAL_MISSINGNESS_V1",
            "terminal_failure_contract_sha256": "9bbcdd9dd1686af034f7c0d2114ac86d5837a07de0cc6ba8fef7940bbc822b75",
            "rate_card_sha256": "50975b67dce4c59ba9267c3234a873076137ded5078aa3e8b5c9a2fad4ff3e06",
        },
        derived_cost_usd=0.25, provider_cost_source="DERIVED_FROM_PROVIDER_USAGE",
        source_spans=[span],
    )
    trial = RuntimeTrialResult(
        BaselineExecutionOutcome(
            status="succeeded", final_response="1+1+1+1", parsed_answer="1+1+1+1",
            verifier_result=False, answer_call_id="trial:call:1", method_calls=(call,),
            memory_before=({"entry_id": "before"},),
            memory_after=({"entry_id": "after"},),
        ),
        state=(),
    )
    identity = ProductionOrdinaryRunIdentity(
        execution_template_id="readiness0|game24|fh_bounded|clean", trajectory_seed=0,
        concrete_seed_id="0", ordered_sample_ids_sha256="1" * 64,
        registration_packet_sha256="2" * 64, scientific_result=False,
        checkpoint_registry_sha256="3" * 64,
    )

    evidence = live_runtime.build_case_evidence(
        live_runtime.CaseEvidenceInput(
            live.READINESS0_CASES[2], trial, identity, "sample-1", (), (False,), ROOT
        )
    )

    assert evidence.answer_call_id == "trial:call:1"
    assert evidence.calls[0].provider_response_id == "resp_complete"
    assert evidence.calls[0].provider_usage == {"input_tokens": 7, "output_tokens": 11}
    assert evidence.calls[0].source_spans[0].entry_id == "entry-1"
    assert evidence.runtime.sample_id == "sample-1"
    assert evidence.runtime.capacity_tokens == 8192
    assert evidence.runtime.text_only is True
    assert evidence.actual_verifier_results == (False,)

    failed_call = call.model_copy(
        update={
            "raw_response": None,
            "error_type": "APIStatusError",
            "failure_code": "provider_unavailable",
            "provider_cost_usd": None,
            "provider_response_id": None,
            "provider_usage": None,
            "provider_returned_model": None,
            "provider_response_status": None,
            "derived_cost_usd": None,
            "provider_cost_source": None,
        }
    )
    failed = live_runtime.build_case_evidence(
        live_runtime.CaseEvidenceInput(
            live.READINESS0_CASES[2],
            RuntimeTrialResult(
                BaselineExecutionOutcome(
                    status="failed",
                        answer_call_id="trial:call:1",
                        error_type="ProviderCallFailure",
                        failure_disposition="provider_call_failed",
                        scientific_ineligibility_reason="provider_call_failed",
                        method_calls=(failed_call,),
                    memory_before=({"entry_id": "before"},),
                    memory_after=({"entry_id": "before"},),
                ),
                state=(),
            ),
            identity,
            "sample-1",
            (),
            (),
            ROOT,
        )
    )

    assert failed.answer_call_id is None
    assert failed.calls[0].error_type == "APIStatusError"
    assert failed.calls[0].cost_source is None


def test_preflight_rejects_stale_authorization_before_cache_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-dispatched")
    monkeypatch.setenv("MEMCONTAM_BGE_CACHE_DIR", str(tmp_path))
    with pytest.raises(live.Readiness0LiveError, match="READINESS0_AUTHORIZATION_REQUEST_MISMATCH"):
        live.verify_preflight(
            request_path=ARTIFACT_ROOT / "readiness0_live_request_v1.json",
            authorization_path=ARTIFACT_ROOT / "readiness0_live_authorization_v1.json",
            expected_authorization_sha256=_sha256(
                ARTIFACT_ROOT / "readiness0_live_authorization_v1.json"
            ),
            f1c_registry_path=ARTIFACT_ROOT / "readiness0_f1c_registry_v1.json",
            repository_root=ROOT,
            core_root=ROOT / "data/phase13/core",
            cache_root=tmp_path,
            output_dir=tmp_path / "output",
            allow_live_calls=True,
        )


def test_preflight_records_real_local_bge_runtime_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_value = os.environ.get("MEMCONTAM_BGE_CACHE_DIR")
    if cache_value is None:
        pytest.skip("MEMCONTAM_BGE_CACHE_DIR is unavailable")
    cache_root = Path(cache_value)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-dispatched")
    authorization_path = tmp_path / "authorization.json"
    authorization = json.loads(
        (ARTIFACT_ROOT / "readiness0_live_authorization_v1.json").read_text(encoding="utf-8")
    )
    authorization["request_sha256"] = _sha256(
        ARTIFACT_ROOT / "readiness0_live_request_v1.json"
    )
    authorization_path.write_text(json.dumps(authorization, indent=2) + "\n", encoding="utf-8")

    verified = live.verify_preflight(
        request_path=ARTIFACT_ROOT / "readiness0_live_request_v1.json",
        authorization_path=authorization_path,
        expected_authorization_sha256=_sha256(authorization_path),
        f1c_registry_path=ARTIFACT_ROOT / "readiness0_f1c_registry_v1.json",
        repository_root=ROOT,
        core_root=ROOT / "data/phase13/core",
        cache_root=cache_root,
        output_dir=tmp_path / "output",
        allow_live_calls=True,
    )

    assert verified.f1c_runtime.provider_identity == (
        "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181"
    )
    assert verified.f1c_runtime.vector_dimension == 1024
    assert verified.f1c_runtime.normalize_embeddings is True
    assert verified.f1c_runtime.local_files_only is True


def test_preflight_hash_failure_dispatches_zero_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def executor(case: live.Readiness0Case) -> live.CaseEvidence:
        del case
        nonlocal calls
        calls += 1
        raise AssertionError("executor must not run")

    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-dispatched")
    monkeypatch.setenv("MEMCONTAM_BGE_CACHE_DIR", str(tmp_path))
    with pytest.raises(live.Readiness0LiveError, match="READINESS0_AUTHORIZATION_HASH_MISMATCH"):
        live.run_readiness0_live(
            request_path=ARTIFACT_ROOT / "readiness0_live_request_v1.json",
            authorization_path=ARTIFACT_ROOT / "readiness0_live_authorization_v1.json",
            expected_authorization_sha256="0" * 64,
            f1c_registry_path=ARTIFACT_ROOT / "readiness0_f1c_registry_v1.json",
            repository_root=ROOT,
            core_root=ROOT / "data/phase13/core",
            cache_root=tmp_path,
            output_dir=tmp_path / "output",
            allow_live_calls=True,
            executor=executor,
        )
    assert calls == 0
    assert not (tmp_path / "output").exists()


def test_stale_authorization_precedes_missing_credential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MEMCONTAM_BGE_CACHE_DIR", str(tmp_path))
    with pytest.raises(live.Readiness0LiveError, match="READINESS0_AUTHORIZATION_REQUEST_MISMATCH"):
        live.run_readiness0_live(
            request_path=ARTIFACT_ROOT / "readiness0_live_request_v1.json",
            authorization_path=ARTIFACT_ROOT / "readiness0_live_authorization_v1.json",
            expected_authorization_sha256=_sha256(
                ARTIFACT_ROOT / "readiness0_live_authorization_v1.json"
            ),
            f1c_registry_path=ARTIFACT_ROOT / "readiness0_f1c_registry_v1.json",
            repository_root=ROOT,
            core_root=ROOT / "data/phase13/core",
            cache_root=tmp_path,
            output_dir=tmp_path / "output",
            allow_live_calls=True,
            executor=lambda case: pytest.fail(f"provider seam reached: {case.case_id}"),
        )


def test_live_orchestrator_stops_on_first_failure_without_retry(tmp_path: Path) -> None:
    attempted: list[str] = []

    def executor(case: live.Readiness0Case) -> live.CaseEvidence:
        attempted.append(case.case_id)
        return _case_evidence(case, "failed" if len(attempted) == 2 else "succeeded")

    result = live.execute_verified_pilot(
        _verified(tmp_path), executor=executor
    )

    assert result.status == "FAILED"
    assert attempted == [case.case_id for case in live.READINESS0_CASES[:2]]
    assert result.provider_calls_issued == 2

    with pytest.raises(live.Readiness0LiveError, match="READINESS0_EVIDENCE_NOT_PASS"):
        live.validate_evidence_closure(tmp_path, result.manifest_sha256)


def test_unexpected_executor_failure_seals_prior_prefix(tmp_path: Path) -> None:
    attempted = 0

    def executor(case: live.Readiness0Case) -> live.CaseEvidence:
        nonlocal attempted
        attempted += 1
        if attempted == 2:
            raise RuntimeError("scripted provider failure")
        return _case_evidence(case)

    with pytest.raises(RuntimeError, match="scripted provider failure"):
        live.execute_verified_pilot(_verified(tmp_path), executor=executor)

    manifest = json.loads((tmp_path / "evidence_manifest.json").read_text())
    rows = (tmp_path / "cases.jsonl").read_text().splitlines()
    assert manifest["status"] == "FAILED"
    assert manifest["case_count"] == 1
    assert len(rows) == 1


@pytest.mark.parametrize("target_index", [4, 5, 6])
def test_first_stage_failure_is_accepted_as_exact_prefix(
    tmp_path: Path,
    target_index: int,
) -> None:
    def executor(case: live.Readiness0Case) -> live.CaseEvidence:
        row = _case_evidence(case, "failed" if case is live.READINESS0_CASES[target_index] else "succeeded")
        if case is not live.READINESS0_CASES[target_index]:
            return row
        return row.model_copy(
            update={
                "stages": row.stages[:1],
                "calls": row.calls[:1],
                "provider_calls": 1,
                "answer_call_id": None,
            }
        )

    result = live.execute_verified_pilot(_verified(tmp_path), executor=executor)

    assert result.status == "FAILED"
    rows = (tmp_path / "cases.jsonl").read_text().splitlines()
    terminal = live.CaseEvidence.model_validate_json(rows[-1])
    assert terminal.case_id == live.READINESS0_CASES[target_index].case_id
    assert terminal.stages == live.READINESS0_CASES[target_index].stages[:1]
    assert terminal.answer_call_id is None


def test_successful_fake_run_closes_non_scientific_evidence(tmp_path: Path) -> None:
    def executor(case: live.Readiness0Case) -> live.CaseEvidence:
        return _case_evidence(case)

    result = live.execute_verified_pilot(
        _verified(tmp_path), executor=executor
    )
    report = live.validate_evidence_closure(tmp_path, result.manifest_sha256)

    assert result.status == "PASS"
    assert result.provider_calls_issued == 12
    assert report.case_count == 7
    assert report.provider_call_count == 12
    assert report.scientific_result is False
    assert report.main_result is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("requested_model", "wrong-model"),
        ("returned_model", "wrong-model"),
        ("response_status", "incomplete"),
        ("reasoning_mode", "wrong"),
        ("reasoning_effort", "low"),
        ("reasoning_context", "previous_turn"),
        ("previous_response_id", "resp-prior"),
        ("store", True),
        ("tools", ("python",)),
        ("maximum_input_tokens", 0),
        ("maximum_output_tokens", 0),
        ("execution_envelope_id", "wrong"),
        ("execution_envelope_sha256", "0" * 64),
        ("failure_contract_id", "wrong"),
        ("failure_contract_sha256", "0" * 64),
        ("terminal_failure_contract_id", "wrong"),
        ("terminal_failure_contract_sha256", "0" * 64),
        ("raw_usage", None),
        ("normalized_usage", {}),
        ("derived_cost_usd", None),
        ("cost_source", "AUTHORITATIVE_PROVIDER"),
        ("rate_card_sha256", "0" * 64),
    ],
)
def test_pass_evidence_rejects_provider_contract_tamper(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    result = live.execute_verified_pilot(
        _verified(tmp_path), executor=lambda case: _case_evidence(case)
    )
    manifest = live.EvidenceManifest.model_validate_json(
        (tmp_path / "evidence_manifest.json").read_bytes()
    )
    rows = tuple(
        live.CaseEvidence.model_validate_json(line)
        for line in (tmp_path / "cases.jsonl").read_text().splitlines()
    )
    changed_call = rows[0].calls[0].model_copy(update={field: value})
    changed_row = rows[0].model_copy(update={"calls": (changed_call,)})

    with pytest.raises(EvidenceValidationError):
        validate_pass_evidence(manifest, (changed_row, *rows[1:]), live.READINESS0_CASES)
    assert result.status == "PASS"


@pytest.mark.parametrize(
    ("row_index", "field", "value"),
    [
        (0, "ordered_sample_ids_sha256", "0" * 64),
        (0, "checkpoint_registry_sha256", "0" * 64),
        (0, "registration_packet_sha256", "0" * 64),
        (3, "retrieval_query_sha256", "0" * 64),
        (3, "retrieval_candidates_sha256", "0" * 64),
        (3, "retrieval_source_span_sha256", "0" * 64),
        (2, "capacity_artifact_sha256", "0" * 64),
        (0, "task_order_sha256", "0" * 64),
        (0, "analysis_window_id", "wrong"),
        (0, "analysis_window_registry_sha256", "0" * 64),
    ],
)
def test_pass_evidence_rejects_runtime_join_tamper(
    tmp_path: Path,
    row_index: int,
    field: str,
    value: object,
) -> None:
    live.execute_verified_pilot(_verified(tmp_path), executor=lambda case: _case_evidence(case))
    manifest = live.EvidenceManifest.model_validate_json(
        (tmp_path / "evidence_manifest.json").read_bytes()
    )
    rows = list(
        live.CaseEvidence.model_validate_json(line)
        for line in (tmp_path / "cases.jsonl").read_text().splitlines()
    )
    rows[row_index] = rows[row_index].model_copy(
        update={"runtime": rows[row_index].runtime.model_copy(update={field: value})}
    )

    with pytest.raises(EvidenceValidationError):
        validate_pass_evidence(manifest, tuple(rows), live.READINESS0_CASES)


def test_pass_evidence_rejects_wrong_answer_stage_join(tmp_path: Path) -> None:
    live.execute_verified_pilot(_verified(tmp_path), executor=lambda case: _case_evidence(case))
    manifest = live.EvidenceManifest.model_validate_json(
        (tmp_path / "evidence_manifest.json").read_bytes()
    )
    rows = list(
        live.CaseEvidence.model_validate_json(line)
        for line in (tmp_path / "cases.jsonl").read_text().splitlines()
    )
    bot = rows[4]
    rows[4] = bot.model_copy(update={"answer_call_id": bot.calls[0].call_id})

    with pytest.raises(EvidenceValidationError, match="READINESS0_ANSWER_CALL_JOIN_MISMATCH"):
        validate_pass_evidence(manifest, tuple(rows), live.READINESS0_CASES)


def test_cli_registers_and_dispatches_readiness0_live_with_fake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(
        [
            "phase13", "run-readiness0-live",
            "--request", "request.json", "--authorization", "authorization.json",
            "--expected-authorization-sha256", "a" * 64,
            "--f1c-registry", "f1c.json", "--output", str(tmp_path),
            "--repository-root", ".", "--core-root", "core", "--cache-root", "cache",
            "--allow-live-calls",
        ]
    )
    monkeypatch.setattr(
        live,
        "run_readiness0_live",
        lambda **_kwargs: live.PilotResult("PASS", 12, "b" * 64),
    )

    run(args)

    assert json.loads(capsys.readouterr().out) == {
        "manifest_sha256": "b" * 64,
        "provider_calls_issued": 12,
        "status": "PASS",
    }
