from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import pytest

from memcontam.clients import openai_responses
from memcontam.experiment import phase13_ordinary_runtime as ordinary_runtime
from memcontam.memory.embeddings import BgeM3EmbeddingProvider, FakeEmbeddingProvider
from memcontam.readiness import phase13_readiness0_live_runtime as live_runtime
from memcontam.readiness.phase13_readiness0_live import (
    READINESS0_CASES,
    execute_verified_pilot,
)
from memcontam.readiness.phase13_readiness0_live_runtime import ProductionCaseExecutor
from memcontam.readiness.phase13_readiness0_live_models import (
    F1CRegistry,
    F1CRuntimeMetadata,
    LiveAuthorization,
    LiveRequest,
    VerifiedReadiness0,
)


ROOT = Path(__file__).resolve().parents[1]


class _ContractFakeEmbeddingProvider(FakeEmbeddingProvider, BgeM3EmbeddingProvider):
    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model_id": "BAAI/bge-m3",
            "revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "embedding_library_version": "5.6.0",
            "vector_dimension": 1024,
            "normalize_embeddings": True,
        }


class _Usage:
    def model_dump(self) -> dict[str, int | dict[str, int]]:
        return {
            "input_tokens": 7,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens": 11,
            "total_tokens": 18,
        }


class _Response:
    model = "gpt-5.6-luna"
    usage = _Usage()
    service_tier = "default"
    status = "completed"
    incomplete_details: Any = None

    def __init__(self, response_id: str, output_text: str) -> None:
        self.id = response_id
        self.output_text = output_text


class _ScriptedResponses:
    def __init__(self, outputs: tuple[str, ...], fail_at: int | None = None) -> None:
        self._outputs = list(outputs)
        self._fail_at = fail_at
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        output = self._outputs.pop(0)
        response = _Response(f"resp-{len(self.calls)}", output)
        if len(self.calls) == self._fail_at:
            response.status = "incomplete"
            response.incomplete_details = type("Incomplete", (), {"reason": "max_output_tokens"})()
        return response


class _ScriptedOpenAI:
    outputs: tuple[str, ...] = ()
    fail_at: int | None = None
    instance: _ScriptedOpenAI

    def __init__(self, **_kwargs: Any) -> None:
        self.responses = _ScriptedResponses(self.outputs, self.fail_at)
        type(self).instance = self


@pytest.fixture(autouse=True)
def _stable_frozen_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)


def _outputs() -> tuple[str, ...]:
    return (
        "final: A",
        "final: A",
        "final: 1+1+1+1",
        "final: 1+1+1+1",
        json.dumps(
            {
                "key_information": "four numbers",
                "restrictions": "use every number once",
                "distilled_task": "construct an arithmetic expression",
            }
        ),
        json.dumps(
            {
                "selected_structure": "procedure-based",
                "solution_trace": "combine the four numbers",
                "final_answer": "final: 1+1+1+1",
            }
        ),
        json.dumps(
            {
                "description": "Game24 procedure",
                "template": "combine every number once and verify the target",
                "category": "procedure-based",
                "explicitly_used_memory_ids": [],
            }
        ),
        "final: 1+1+1+1",
        json.dumps(
            {
                "mode": "corrective",
                "failure_class": "incorrect_answer",
                "reflection_text": "try a different arithmetic combination",
                "explicitly_used_memory_ids": [],
            }
        ),
        "final: 1+1+1+1",
        "<cheatsheet>combine all four numbers and verify the result</cheatsheet>",
        "final: 1+1+1+1",
    )


def test_scripted_responses_drive_all_production_readiness0_cases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _ScriptedOpenAI.outputs = _outputs()
    _ScriptedOpenAI.fail_at = None
    monkeypatch.setattr(openai_responses, "OpenAI", _ScriptedOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "scripted-no-network")
    executor = ProductionCaseExecutor(
        ROOT,
        ROOT / "data/phase13/core",
        tmp_path,
        embedding_provider=_ContractFakeEmbeddingProvider(1024),
    )

    rows = tuple(executor(case) for case in READINESS0_CASES)

    assert tuple(row.case_id for row in rows) == tuple(case.case_id for case in READINESS0_CASES)
    assert tuple(row.stages for row in rows) == tuple(case.stages for case in READINESS0_CASES)
    assert sum(row.provider_calls for row in rows) == 12
    assert rows[5].routing_verifier_results == (False, True)
    assert rows[5].actual_verifier_results == (False, False)
    requests = _ScriptedOpenAI.instance.responses.calls
    assert [request["max_output_tokens"] for request in requests] == [
        512, 512, 512, 512, 384, 512, 384, 512, 384, 512, 8192, 512,
    ]
    assert all(
        request["reasoning"]
        == {"mode": "standard", "effort": "none", "context": "current_turn"}
        and request["tools"] == []
        and request["store"] is False
        and "previous_response_id" not in request
        for request in requests
    )
    assert all(
        call.provider_response_id is not None
        and call.provider_usage is not None
        and call.provider_cost_usd is not None
        and call.transport_attempts == 1
        for row in rows
        for call in row.calls
    )
    stage_inputs = {
        "no_memory_generate": 1160,
        "full_history_generate": 9330,
        "rag_generate": 290,
        "bot_problem_distill": 1177,
        "bot_instantiate_solve": 1949,
        "bot_thought_distill": 2545,
        "reflexion_generate": 2282,
        "reflexion_reflect": 3349,
        "dc_rs_synthesize": 13521,
        "dc_rs_generate": 9212,
    }
    assert all(
        call.requested_model == "gpt-5.6-luna"
        and call.returned_model == "gpt-5.6-luna"
        and call.response_status == "completed"
        and call.reasoning_mode == "standard"
        and call.reasoning_effort == "none"
        and call.reasoning_context == "current_turn"
        and call.previous_response_id is None
        and call.store is False
        and call.tools == ()
        and call.maximum_input_tokens == stage_inputs[call.stage]
        and call.execution_envelope_sha256
        == "4c48fca92d1d70105d2eb34b5b86984c732c03e3600cb00965501ecabd2d1769"
        and call.failure_contract_sha256
        == "1ee66fcb795f97d483c2ef976133ee61dbd5108c9dae851c2c2786ff496d788f"
        and call.terminal_failure_contract_sha256
        == "9bbcdd9dd1686af034f7c0d2114ac86d5837a07de0cc6ba8fef7940bbc822b75"
        and call.rate_card_sha256
        == "50975b67dce4c59ba9267c3234a873076137ded5078aa3e8b5c9a2fad4ff3e06"
        and call.raw_usage == call.provider_usage
        and call.normalized_usage == call.token_usage
        and call.authoritative_provider_cost_usd is None
        and call.derived_cost_usd == call.provider_cost_usd
        and call.cost_source == "DERIVED_FROM_PROVIDER_USAGE"
        for row in rows
        for call in row.calls
    )
    capacity_hash = hashlib.sha256((ROOT / "data/phase13/common_capacity_v1.json").read_bytes()).hexdigest()
    order_hash = hashlib.sha256(
        (ROOT / "data/phase13/main/mr_p4/task_seed_orders_v1.json").read_bytes()
    ).hexdigest()
    window_hash = hashlib.sha256(
        (ROOT / "data/phase13/main/mr_p4/readiness0_window_proof_v1.json").read_bytes()
    ).hexdigest()
    assert all(
        row.runtime.capacity_artifact_sha256 == capacity_hash
        and row.runtime.task_order_sha256 == order_hash
        and row.runtime.analysis_window_id == "core_prefix_50"
        and row.runtime.analysis_window_registry_sha256 == window_hash
        and row.runtime.retrieval_source_span_sha256 != "0" * 64
        for row in rows
    )
    rag_runtime = rows[3].runtime
    assert rag_runtime.retrieval_query_sha256 is not None
    assert rag_runtime.retrieval_candidates_sha256 is not None


@pytest.mark.parametrize(
    ("target_index", "request_count"),
    ((4, 5), (5, 8), (6, 11)),
)
def test_first_multistage_provider_failure_is_durably_sealed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target_index: int,
    request_count: int,
) -> None:
    _ScriptedOpenAI.outputs = _outputs()
    _ScriptedOpenAI.fail_at = request_count
    monkeypatch.setattr(openai_responses, "OpenAI", _ScriptedOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "scripted-no-network")
    executor = ProductionCaseExecutor(
        ROOT,
        ROOT / "data/phase13/core",
        tmp_path,
        embedding_provider=_ContractFakeEmbeddingProvider(1024),
    )

    result = execute_verified_pilot(_verified(tmp_path / "evidence"), executor=executor)

    manifest = json.loads((tmp_path / "evidence/evidence_manifest.json").read_bytes())
    rows = [json.loads(line) for line in (tmp_path / "evidence/cases.jsonl").read_text().splitlines()]
    terminal = rows[-1]
    call = terminal["calls"][0]
    assert result.status == "FAILED"
    assert len(_ScriptedOpenAI.instance.responses.calls) == request_count
    assert len(rows) == target_index + 1
    assert manifest["status"] == "FAILED"
    assert manifest["case_count"] == target_index + 1
    assert manifest["provider_call_count"] == request_count
    assert manifest["terminal_case_id"] == READINESS0_CASES[target_index].case_id
    assert manifest["terminal_stage"] == READINESS0_CASES[target_index].stages[0]
    assert terminal["status"] == "failed"
    assert terminal["stages"] == [READINESS0_CASES[target_index].stages[0]]
    assert terminal["provider_calls"] == 1
    assert terminal["answer_call_id"] is None
    assert call["transport_attempts"] == 1
    assert call["failure_code"] == "LUNA_PROVIDER_INCOMPLETE"
    assert call["provider_status"] == "incomplete"
    assert call["provider_incomplete_reason"] == "max_output_tokens"


def _verified(output_dir: Path) -> VerifiedReadiness0:
    artifact_root = ROOT / "data/phase13/main/mr_p4"
    report = json.loads((artifact_root / "readiness0_f1c_report_v1.json").read_bytes())
    return VerifiedReadiness0(
        request=LiveRequest.model_validate_json(
            (artifact_root / "readiness0_live_request_v1.json").read_bytes()
        ),
        authorization=LiveAuthorization.model_validate_json(
            (artifact_root / "readiness0_live_authorization_v1.json").read_bytes()
        ),
        f1c=F1CRegistry.model_validate_json(
            (artifact_root / "readiness0_f1c_registry_v1.json").read_bytes()
        ),
        request_sha256="1" * 64,
        authorization_sha256="2" * 64,
        f1c_sha256="3" * 64,
        output_dir=output_dir,
        f1c_runtime=F1CRuntimeMetadata.model_validate(report["runtime"]),
    )


def test_production_executor_blocks_thirteenth_request_before_client(
) -> None:
    delegate = _ScriptedResponses(("final: A",) * 13)
    budgeted = live_runtime.BudgetedResponses(delegate, maximum_calls=12)

    for _index in range(12):
        budgeted.create(model="gpt-5.6-luna")
    with pytest.raises(
        live_runtime.ProviderCallBudgetError,
        match="READINESS0_PROVIDER_CALL_CEILING_EXCEEDED",
    ):
        budgeted.create(model="gpt-5.6-luna")

    assert len(delegate.calls) == 12
