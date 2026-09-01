from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from memcontam.readiness.phase13_authority_files import read_regular_nofollow
from memcontam.readiness.phase13_cost_policy import load_cost_policy_bundle
from memcontam.readiness.phase13_main_checkpoint import CommonCheckpointRegistry
from memcontam.readiness.phase13_readiness0_f1c_models import F1CReport
from memcontam.readiness.phase13_readiness0_live_models import (
    CaseEvidence,
    EvidenceManifest,
    Readiness0Case,
)


class EvidenceValidationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


_ROOT = Path(__file__).parents[3]
_ENVELOPE_HASH = "4c48fca92d1d70105d2eb34b5b86984c732c03e3600cb00965501ecabd2d1769"
_FAILURE_HASH = "1ee66fcb795f97d483c2ef976133ee61dbd5108c9dae851c2c2786ff496d788f"
_TERMINAL_HASH = "9bbcdd9dd1686af034f7c0d2114ac86d5837a07de0cc6ba8fef7940bbc822b75"
_RATE_CARD_HASH = "50975b67dce4c59ba9267c3234a873076137ded5078aa3e8b5c9a2fad4ff3e06"
_HISTORICAL_CAPACITY_HASH = "102da3f554294c0e802c6894cbcf03074704f465e9595ef27134b3897a56ad31"
_ANSWER_STAGE = {
    "nomem": "no_memory_generate",
    "fh_bounded": "full_history_generate",
    "rag_frozen": "rag_generate",
    "bot_style": "bot_instantiate_solve",
    "reflexion_style": "reflexion_generate",
    "dc_rs": "dc_rs_generate",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(read_regular_nofollow(path)).hexdigest()


def _canonical_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def validate_pass_evidence(
    manifest: EvidenceManifest,
    rows: Sequence[CaseEvidence],
    cases: Sequence[Readiness0Case],
) -> None:
    bundle = load_cost_policy_bundle(_ROOT)
    stages = {stage.semantic_stage_id: stage for stage in bundle.registry.stages}
    checkpoint = CommonCheckpointRegistry.model_validate_json(
        read_regular_nofollow(
            _ROOT / "data/phase13/main/mr_p4/main_a_common_checkpoint_registry_v1.json"
        )
    )
    f1c = F1CReport.model_validate_json(
        read_regular_nofollow(_ROOT / "data/phase13/main/mr_p4/readiness0_f1c_report_v1.json")
    )
    checkpoint_hash = _sha256(
        _ROOT / "data/phase13/main/mr_p4/main_a_common_checkpoint_registry_v1.json"
    )
    registration_hash = _sha256(
        _ROOT / "data/phase13/observability/registration_packet_v1.json"
    )
    task_order_hash = _sha256(
        _ROOT / "data/phase13/main/mr_p4/task_seed_orders_v1.json"
    )
    window_hash = _sha256(
        _ROOT / "data/phase13/main/mr_p4/readiness0_window_proof_v1.json"
    )
    if manifest.status != "PASS":
        raise EvidenceValidationError("READINESS0_EVIDENCE_NOT_PASS")
    if (
        manifest.case_count != 7
        or manifest.provider_call_count != 12
        or manifest.scientific_result
        or manifest.main_result
        or manifest.measured_main_a_trajectory_count != 0
        or manifest.terminal_case_id is not None
        or manifest.terminal_stage is not None
        or manifest.failure_code is not None
        or len(rows) != len(cases)
    ):
        raise EvidenceValidationError("READINESS0_EVIDENCE_CLOSURE_MISMATCH")
    capacity_hashes = {row.runtime.capacity_artifact_sha256 for row in rows}
    if len(capacity_hashes) != 1 or not capacity_hashes <= {
        _HISTORICAL_CAPACITY_HASH,
        bundle.proof.common_capacity_sha256,
    }:
        raise EvidenceValidationError("READINESS0_CAPACITY_JOIN_MISMATCH")
    call_ids: set[str] = set()
    for row, case in zip(rows, cases, strict=True):
        seed = checkpoint.tasks[case.task].seeds[0]
        if (
            row.status != "succeeded"
            or row.case_id != case.case_id
            or row.stages != case.stages
            or row.provider_calls != len(case.stages)
            or len(row.calls) != len(case.stages)
            or tuple(call.stage for call in row.calls) != case.stages
            or row.answer_call_id is None
            or row.answer_call_id not in {call.call_id for call in row.calls}
            or row.runtime.task != case.task
            or row.runtime.baseline != case.baseline
            or row.runtime.sample_id != seed.suffix_sample_ids[0]
            or row.runtime.execution_template_id
            != f"readiness0|{case.task}|{case.baseline}|clean"
            or row.runtime.ordered_sample_ids_sha256 != seed.suffix_sample_ids_sha256
            or row.runtime.suffix_position != 1
            or row.runtime.sample_order != 1
            or not row.runtime.text_only
            or row.runtime.tool_execution_count != 0
            or not row.actual_verifier_results
        ):
            raise EvidenceValidationError("READINESS0_EVIDENCE_CLOSURE_MISMATCH")
        answer_calls = tuple(call for call in row.calls if call.call_id == row.answer_call_id)
        if len(answer_calls) != 1 or answer_calls[0].stage != _ANSWER_STAGE[case.baseline]:
            raise EvidenceValidationError("READINESS0_ANSWER_CALL_JOIN_MISMATCH")
        if any(
            call.transport_attempts != 1
            or call.latency_ms is None
            or call.provider_cost_usd is None
            or call.provider_response_id is None
            or call.provider_usage is None
            or call.provider_service_tier is None
            for call in row.calls
        ):
            raise EvidenceValidationError("READINESS0_PROVIDER_EVIDENCE_INCOMPLETE")
        for call in row.calls:
            stage = stages.get(call.stage)
            if stage is None or (
                call.requested_model != "gpt-5.6-luna"
                or call.returned_model != "gpt-5.6-luna"
                or call.response_status != "completed"
                or call.reasoning_mode != "standard"
                or call.reasoning_effort != "none"
                or call.reasoning_context != "current_turn"
                or call.previous_response_id is not None
                or call.store
                or call.tools
                or call.maximum_input_tokens != stage.maximum_input_tokens
                or call.maximum_output_tokens != stage.maximum_output_tokens
                or call.execution_envelope_id != "CORE_EXECUTION_ENVELOPE_REGISTRY_V2"
                or call.execution_envelope_sha256 != _ENVELOPE_HASH
                or call.failure_contract_id != "CORE_TRANSPORT_ATTEMPT_CONTRACT_V2"
                or call.failure_contract_sha256 != _FAILURE_HASH
                or call.terminal_failure_contract_id
                != "CORE_TERMINAL_TECHNICAL_MISSINGNESS_V1"
                or call.terminal_failure_contract_sha256 != _TERMINAL_HASH
                or call.raw_usage != call.provider_usage
                or call.normalized_usage != call.token_usage
                or call.derived_cost_usd is None
                or call.cost_source != "DERIVED_FROM_PROVIDER_USAGE"
                or call.authoritative_provider_cost_usd is not None
                or call.provider_cost_usd != call.derived_cost_usd
                or call.rate_card_sha256 != _RATE_CARD_HASH
            ):
                raise EvidenceValidationError("READINESS0_PROVIDER_CONTRACT_MISMATCH")
        row_call_ids = {call.call_id for call in row.calls}
        if len(row_call_ids) != len(row.calls) or call_ids.intersection(row_call_ids):
            raise EvidenceValidationError("READINESS0_CALL_ID_DUPLICATE")
        call_ids.update(row_call_ids)
        if case.baseline == "reflexion_style":
            if (
                row.reflexion_route_policy_id
                != "readiness0_reflexion_fail_then_pass_v1"
                or row.routing_verifier_results != (False, True)
                or len(row.actual_verifier_results) != 2
            ):
                raise EvidenceValidationError("READINESS0_REFLEXION_EVIDENCE_INCOMPLETE")
        elif row.reflexion_route_policy_id is not None or row.routing_verifier_results:
            raise EvidenceValidationError("READINESS0_REFLEXION_EVIDENCE_UNEXPECTED")
        capacity_applies = case.baseline in {"fh_bounded", "dc_rs"}
        runtime = row.runtime
        if (
            capacity_applies != (runtime.capacity_tokens == 8192)
            or runtime.checkpoint_registry_sha256 != checkpoint_hash
            or runtime.registration_packet_sha256 != registration_hash
            or runtime.ordered_sample_ids_sha256 == "0" * 64
            or runtime.task_order_sha256 != task_order_hash
            or runtime.analysis_window_id != "core_prefix_50"
            or runtime.analysis_window_registry_sha256 != window_hash
        ):
            raise EvidenceValidationError("READINESS0_CAPACITY_JOIN_MISMATCH")
        source_spans = tuple(
            {
                "call_id": call.call_id,
                "source_spans": tuple(
                    span.model_dump(mode="json") for span in call.source_spans
                ),
            }
            for call in row.calls
        )
        if runtime.retrieval_source_span_sha256 != _canonical_hash(source_spans):
            raise EvidenceValidationError("READINESS0_RETRIEVAL_JOIN_MISMATCH")
        if case.baseline == "rag_frozen":
            matching_rows = tuple(
                item
                for item in f1c.rows
                if item.task == case.task
                and item.baseline == case.baseline
                and item.arm == "clean"
            )
            if len(matching_rows) != 1:
                raise EvidenceValidationError("READINESS0_RETRIEVAL_JOIN_MISSING")
            expected = matching_rows[0]
            scores = dict(zip(expected.candidate_ids, expected.scores, strict=True))
            expected_scores = tuple(scores[entry_id] for entry_id in expected.selected_ids)
            candidates = {
                "entry_ids": runtime.retrieved_entry_ids,
                "scores": runtime.retrieved_scores,
            }
            answer_span_ids = tuple(span.entry_id for span in answer_calls[0].source_spans)
            if (
                runtime.retrieval_event_id is None
                or runtime.retrieval_query_sha256 != expected.query_sha256
                or runtime.retrieved_entry_ids != expected.selected_ids
                or runtime.retrieved_scores != expected_scores
                or runtime.retrieval_candidates_sha256 != _canonical_hash(candidates)
                or answer_span_ids != expected.selected_ids
            ):
                raise EvidenceValidationError("READINESS0_RETRIEVAL_JOIN_MISSING")


__all__ = ["EvidenceValidationError", "validate_pass_evidence"]
