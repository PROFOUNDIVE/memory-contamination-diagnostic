from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from memcontam.experiment.phase12.runtime_registry import RuntimeTrialResult
from memcontam.logging.schema import MethodCall
from memcontam.logging.schema_v3 import RetrievalEvent
from memcontam.readiness.phase13_production_runtime_models import ProductionOrdinaryRunIdentity
from memcontam.readiness.phase13_cost_policy import load_cost_policy_bundle
from memcontam.readiness.phase13_readiness0_evidence_models import (
    ProviderAuthorityContract,
    ProviderCallEvidence,
    ProviderRequestContract,
    RuntimeJoinEvidence,
)
from memcontam.readiness.phase13_readiness0_live_models import CaseEvidence, Readiness0Case


class CaseEvidenceBuildError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class CaseEvidenceInput:
    case: Readiness0Case
    trial: RuntimeTrialResult
    identity: ProductionOrdinaryRunIdentity
    sample_id: str
    routing_verifier_results: tuple[bool, ...]
    actual_verifier_results: tuple[bool, ...]
    repository_root: Path


def build_case_evidence(value: CaseEvidenceInput) -> CaseEvidence:
    outcome = value.trial.outcome
    method_calls = tuple(
        MethodCall.model_validate(call, from_attributes=True) for call in outcome.method_calls
    )
    calls = tuple(_provider_evidence(call) for call in method_calls)
    if outcome.status == "succeeded" and (
        outcome.answer_call_id is None
        or not any(call.call_id == outcome.answer_call_id for call in calls)
    ):
        raise CaseEvidenceBuildError("READINESS0_ANSWER_CALL_JOIN_MISSING")
    retrieval = (
        None
        if value.trial.retrieval_event is None
        else RetrievalEvent.model_validate(value.trial.retrieval_event, from_attributes=True)
    )
    checkpoint_hash = value.identity.checkpoint_registry_sha256
    if checkpoint_hash is None:
        raise CaseEvidenceBuildError("READINESS0_CHECKPOINT_JOIN_MISSING")
    if value.identity.trajectory_seed != 0 or value.identity.concrete_seed_id != "0":
        raise CaseEvidenceBuildError("READINESS0_SEED_IDENTITY_INVALID")
    capacity_applies = value.case.baseline in {"fh_bounded", "dc_rs"}
    bundle = load_cost_policy_bundle(value.repository_root)
    source_span_payload = tuple(
        {
            "call_id": call.call_id,
            "source_spans": tuple(
                span.model_dump(mode="json") for span in call.source_spans
            ),
        }
        for call in calls
    )
    retrieval_candidates = None
    if retrieval is not None:
        retrieval_candidates = _canonical_hash(
            {
                "entry_ids": tuple(retrieval.retrieved_entry_ids),
                "scores": tuple(retrieval.retrieved_scores),
            }
        )
    return CaseEvidence(
        case_id=value.case.case_id,
        status=outcome.status,
        stages=tuple(call.stage for call in calls),
        provider_calls=len(calls),
        calls=calls,
        answer_call_id=(
            outcome.answer_call_id
            if any(
                call.call_id == outcome.answer_call_id and call.raw_response is not None
                for call in calls
            )
            else None
        ),
        runtime=RuntimeJoinEvidence(
            task=value.case.task,
            baseline=value.case.baseline,
            sample_id=value.sample_id,
            suffix_position=1,
            sample_order=1,
            trajectory_seed=value.identity.trajectory_seed,
            concrete_seed_id=value.identity.concrete_seed_id,
            execution_template_id=value.identity.execution_template_id,
            ordered_sample_ids_sha256=value.identity.ordered_sample_ids_sha256,
            checkpoint_registry_sha256=checkpoint_hash,
            registration_packet_sha256=value.identity.registration_packet_sha256,
            retrieval_query_sha256=None if retrieval is None else retrieval.query_hash,
            retrieval_candidates_sha256=retrieval_candidates,
            retrieval_source_span_sha256=_canonical_hash(source_span_payload),
            retrieval_event_id=None if retrieval is None else retrieval.event_id,
            retrieved_entry_ids=() if retrieval is None else tuple(retrieval.retrieved_entry_ids),
            retrieved_scores=() if retrieval is None else tuple(retrieval.retrieved_scores),
            memory_before_sha256=_canonical_hash(outcome.memory_before),
            memory_after_sha256=_canonical_hash(outcome.memory_after),
            capacity_law_id=(
                "luna_common_visible_memory_capacity_v1" if capacity_applies else None
            ),
            capacity_tokens=8192 if capacity_applies else None,
            capacity_artifact_sha256=bundle.proof.common_capacity_sha256,
            task_order_sha256=_file_hash(
                value.repository_root
                / "data/phase13/main/mr_p4/task_seed_orders_v1.json"
            ),
            analysis_window_id=value.identity.analysis_window_id,
            analysis_window_registry_sha256=_file_hash(
                value.repository_root
                / "data/phase13/main/mr_p4/readiness0_window_proof_v1.json"
            ),
            text_only=True,
            tool_execution_count=0,
        ),
        reflexion_route_policy_id=(
            "readiness0_reflexion_fail_then_pass_v1"
            if value.case.baseline == "reflexion_style"
            else None
        ),
        routing_verifier_results=value.routing_verifier_results,
        actual_verifier_results=value.actual_verifier_results,
        scientific_result=False,
        main_result=False,
    )


def _provider_evidence(call: MethodCall) -> ProviderCallEvidence:
    if call.call_id is None:
        raise CaseEvidenceBuildError("READINESS0_CALL_ID_MISSING")
    try:
        request = ProviderRequestContract.model_validate(call.provider_request_contract)
        authority = ProviderAuthorityContract.model_validate(call.provider_authority_contract)
    except ValidationError as error:
        raise CaseEvidenceBuildError("READINESS0_PROVIDER_CONTRACT_MISSING") from error
    if call.provider_cost_source == "AUTHORITATIVE_PROVIDER":
        cost_source = "AUTHORITATIVE_PROVIDER"
    elif call.provider_cost_source == "DERIVED_FROM_PROVIDER_USAGE":
        cost_source = "DERIVED_FROM_PROVIDER_USAGE"
    elif call.error_type is not None:
        cost_source = None
    else:
        raise CaseEvidenceBuildError("READINESS0_PROVIDER_COST_SOURCE_MISSING")
    return ProviderCallEvidence(
        call_id=call.call_id,
        stage=call.stage,
        raw_response=call.raw_response,
        transport_attempts=call.transport_attempts,
        token_usage=call.token_usage,
        latency_ms=call.latency_ms,
        provider_cost_usd=call.provider_cost_usd,
        provider_response_id=call.provider_response_id,
        provider_usage=call.provider_usage,
        provider_service_tier=call.provider_service_tier,
        requested_model=request.model,
        returned_model=call.provider_returned_model,
        response_status=call.provider_response_status,
        failure_code=call.failure_code,
        error_type=call.error_type,
        provider_status=call.provider_status,
        provider_incomplete_reason=call.provider_incomplete_reason,
        reasoning_mode=request.reasoning.mode,
        reasoning_effort=request.reasoning.effort,
        reasoning_context=request.reasoning.context,
        previous_response_id=request.previous_response_id,
        store=request.store,
        tools=(),
        maximum_input_tokens=authority.maximum_input_tokens,
        maximum_output_tokens=authority.maximum_output_tokens,
        execution_envelope_id=authority.execution_envelope_id,
        execution_envelope_sha256=authority.execution_envelope_sha256,
        failure_contract_id=authority.failure_contract_id,
        failure_contract_sha256=authority.failure_contract_sha256,
        terminal_failure_contract_id=authority.terminal_failure_contract_id,
        terminal_failure_contract_sha256=authority.terminal_failure_contract_sha256,
        raw_usage=call.provider_usage,
        normalized_usage=call.token_usage,
        authoritative_provider_cost_usd=call.authoritative_provider_cost_usd,
        derived_cost_usd=call.derived_cost_usd,
        cost_source=cost_source,
        rate_card_sha256=authority.rate_card_sha256,
        source_spans=tuple(call.source_spans),
    )


def _canonical_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["CaseEvidenceBuildError", "CaseEvidenceInput", "build_case_evidence"]
