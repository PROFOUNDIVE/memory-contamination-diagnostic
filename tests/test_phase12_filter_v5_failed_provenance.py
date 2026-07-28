from __future__ import annotations

import json
from pathlib import Path

from memcontam.baselines.bot_runtime import BotRuntime
from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.baselines.full_history import FullHistoryState
from memcontam.baselines.full_history_adapter import FullHistoryAdapter
from memcontam.baselines.reflexion_adapter import ReflexionAdapter, ReflexionState
from memcontam.baselines.retrieval_rag_phase12 import (
    RagFrozenPhase12Adapter,
    RagFrozenStateV3,
    RagFrozenTrialContextV3,
)
from memcontam.clients.base import LLMResponse
from memcontam.clients.recording import MethodCallRecorder
from memcontam.experiment.phase12.filter_challenge.contracts import AnswerCallRelation
from memcontam.experiment.phase12.filter_challenge.provenance import AnswerCallProvenanceObserver
from memcontam.memory.bot_buffer import BotBufferIdentity
from memcontam.rag.branch_index import BGE_M3_PRIMARY_IDENTITY, BranchIndex
from memcontam.rag.phase12_corpus import BranchCorpus, Document
from memcontam.tasks.base import TaskInstance
from memcontam.tools import SubprocessTestDouble, load_tool_runtime_contract


class _FailureClient:
    def __init__(self, successful_stages: dict[str, LLMResponse]) -> None:
        self._successful_stages = successful_stages
        self.provider_calls_issued = 0

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        del messages, model
        stage = config["method_stage"]
        assert isinstance(stage, str)
        if stage in self._successful_stages:
            return self._successful_stages[stage]
        raise ConnectionError(stage)


class _ToolContinuationFailureClient(_FailureClient):
    def __init__(self, initial_tool_response: LLMResponse, distillation: LLMResponse) -> None:
        super().__init__({"bot_problem_distill": distillation, "bot_instantiate_solve": initial_tool_response})
        self._tool_calls = 0

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        if config["method_stage"] == "bot_instantiate_solve":
            self._tool_calls += 1
            if self._tool_calls == 2:
                raise ConnectionError("bot_instantiate_solve")
        return super().chat(messages, model, config)


class _Observer(AnswerCallProvenanceObserver):
    def __init__(self) -> None:
        super().__init__()
        self.relations: dict[str, AnswerCallRelation] = {}

    def finalize(self, answer_call_id: str) -> AnswerCallRelation:
        relation = super().finalize(answer_call_id)
        self.relations[answer_call_id] = relation
        return relation


class _Embedder:
    def encode_query(self, text: str) -> list[float]:
        assert text
        return [1.0, 0.0]


class _BotEmbedder:
    def encode_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]

    def encode_document(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="failed-provenance",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6], "target": 24},
    )


def _noisy_records(monkeypatch) -> None:
    original = MethodCallRecorder.get_records

    def reordered(self):
        records = original(self)
        return [] if len(records) == 1 else [records[0], *records[1:], records[0]]

    monkeypatch.setattr(MethodCallRecorder, "get_records", reordered)


def _assert_missing(outcome: BaselineExecutionOutcome, observer: _Observer, call_id: str) -> None:
    assert outcome.answer_call_id == call_id
    assert tuple(observer.relations) == (call_id,)
    assert observer.relations[call_id].answer_call_provenance_status == "missing"


def test_full_history_provider_failure_uses_its_generated_call_id(monkeypatch) -> None:
    # Given: a provider failure and a recorder whose record list cannot identify it.
    observer = _Observer()
    _noisy_records(monkeypatch)

    # When: full-history invokes its native answer surface.
    outcome = FullHistoryAdapter().execute(
        _task(),
        FullHistoryState(),
        client=_FailureClient({}),
        model="replay",
        config={"_logging_answer_call_provenance_observer": observer},
    )

    # Then: the generated failed-call ID owns one unresolved relation.
    _assert_missing(outcome, observer, "unknown:game24:failed-provenance:full_history:clean:replay:call:1")


def test_rag_provider_failure_uses_its_generated_call_id(monkeypatch) -> None:
    # Given: a frozen RAG answer request and a provider failure after list noise.
    observer = _Observer()
    _noisy_records(monkeypatch)
    document = Document("source", "Use fractions.")
    corpus = BranchCorpus("clean", (document,), (document.document_id,), "failed-corpus")
    index = BranchIndex(
        "clean", (document,), {"production_identity": BGE_M3_PRIMARY_IDENTITY},
        {document.document_id: (1.0, 0.0)}, "failed-index", _Embedder()
    )

    # When: RAG executes its native answer call.
    outcome = RagFrozenPhase12Adapter().execute(
        RagFrozenTrialContextV3(
            task=_task(), client=_FailureClient({}), model="replay", run_id="failed",
            trial_id="failed:rag", condition_id="rag", branch="clean", rag_mode="frozen",
            provenance_observer=observer,
        ),
        RagFrozenStateV3("clean", corpus, index),
    ).outcome

    # Then: no record-list position controls the unresolved answer relation.
    _assert_missing(outcome, observer, "failed:rag:call:1")


def test_reflexion_provider_failure_uses_its_generated_call_id(monkeypatch) -> None:
    # Given: an actor provider failure and unusable recorder-list ordering.
    observer = _Observer()
    _noisy_records(monkeypatch)

    # When: Reflexion invokes its actor answer surface.
    outcome = ReflexionAdapter().execute(
        _task(), ReflexionState(), client=_FailureClient({}), model="replay",
        config={"_logging_answer_call_provenance_observer": observer},
    )

    # Then: exactly the generated failed actor ID is finalized.
    _assert_missing(outcome, observer, "unknown:game24:failed-provenance:reflexion_style:clean:replay:call:1")


def test_bot_text_provider_failure_uses_its_generated_call_id(monkeypatch) -> None:
    # Given: successful auxiliary distillation, then a failed text answer and trailing noise.
    observer = _Observer()
    _noisy_records(monkeypatch)
    distillation = LLMResponse(
        content=json.dumps({"key_information": "numbers", "restrictions": "all", "distilled_task": "solve"}),
        raw={}, token_usage={}, latency_ms=0,
    )

    # When: BoT reaches the native text answer call.
    outcome = BotRuntime().run(
        identity=BotBufferIdentity("failed", "game24", "bot_style", "clean", "replay"),
        task=_task(), buffer_snapshot=[], client=_FailureClient({"bot_problem_distill": distillation}),
        model="replay", config={"embedding_provider": _BotEmbedder(), "_logging_answer_call_provenance_observer": observer},
    )

    # Then: the failed second call, not auxiliary record order, is finalized.
    _assert_missing(outcome, observer, "failed:game24:failed-provenance:bot_style:clean:replay:call:2")


def test_bot_tool_initial_provider_failure_uses_its_generated_call_id(monkeypatch) -> None:
    # Given: tool mode reaches a provider failure before an initial tool action exists.
    observer = _Observer()
    _noisy_records(monkeypatch)
    policy = load_tool_runtime_contract(
        Path(__file__).resolve().parents[1] / "containers" / "python-sandbox" / "image.lock.json",
        scientific=False,
    )
    distillation = LLMResponse(
        content=json.dumps({"key_information": "numbers", "restrictions": "all", "distilled_task": "solve"}),
        raw={}, token_usage={}, latency_ms=0,
    )

    # When: BoT issues its initial tool answer request.
    outcome = BotRuntime().run(
        identity=BotBufferIdentity("failed", "game24", "bot_style", "clean", "replay"),
        task=_task(), buffer_snapshot=[], client=_FailureClient({"bot_problem_distill": distillation}), model="replay",
        config={"embedding_provider": _BotEmbedder(), "tool_mode": "python_sandbox", "tool_executor": SubprocessTestDouble(), "tool_runtime_contract": policy, "_logging_answer_call_provenance_observer": observer},
    )

    # Then: the failed answer call is unresolved and classified as a provider failure.
    _assert_missing(outcome, observer, "failed:game24:failed-provenance:bot_style:clean:replay:call:2")
    assert outcome.failure_disposition == "provider_call_failed"


def test_bot_tool_continuation_failure_uses_its_generated_call_id(monkeypatch) -> None:
    # Given: a valid initial tool action followed by a failed continuation request.
    observer = _Observer()
    _noisy_records(monkeypatch)
    distillation = LLMResponse(
        content=json.dumps({"key_information": "numbers", "restrictions": "all", "distilled_task": "solve"}),
        raw={}, token_usage={}, latency_ms=0,
    )
    action = LLMResponse(
        content=json.dumps({"action": "execute_python", "code": "print(24)"}),
        raw={}, token_usage={}, latency_ms=0,
    )
    policy = load_tool_runtime_contract(
        Path(__file__).resolve().parents[1] / "containers" / "python-sandbox" / "image.lock.json",
        scientific=False,
    )

    # When: the tool loop requests its continuation answer.
    outcome = BotRuntime().run(
        identity=BotBufferIdentity("failed", "game24", "bot_style", "clean", "replay"),
        task=_task(), buffer_snapshot=[], client=_ToolContinuationFailureClient(action, distillation),
        model="replay",
        config={"embedding_provider": _BotEmbedder(), "tool_mode": "python_sandbox", "tool_executor": SubprocessTestDouble(), "tool_runtime_contract": policy, "_logging_answer_call_provenance_observer": observer},
    )

    # Then: no response is fabricated and the failed continuation alone is unresolved.
    _assert_missing(outcome, observer, "failed:game24:failed-provenance:bot_style:clean:replay:call:3")
    assert outcome.final_response is None
    assert outcome.failure_disposition == "provider_call_failed"
