from __future__ import annotations

import json

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
from memcontam.clients.recording import MethodCallRecorder, RecordedResponse
from memcontam.experiment.phase12.filter_challenge.contracts import AnswerCallRelation
from memcontam.experiment.phase12.filter_challenge.provenance import AnswerCallProvenanceObserver
from memcontam.rag.branch_index import BGE_M3_PRIMARY_IDENTITY, BranchIndex
from memcontam.rag.phase12_corpus import BranchCorpus, Document
from memcontam.tasks.base import TaskInstance


class _ScriptedClient:
    def __init__(self, responses_by_stage: dict[str, list[LLMResponse]]) -> None:
        self._responses_by_stage = responses_by_stage
        self.issued: list[LLMResponse] = []
        self.provider_calls_issued = 0

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        del messages, model
        stage = config["method_stage"]
        assert isinstance(stage, str)
        response = self._responses_by_stage[stage].pop(0)
        self.issued.append(response)
        return response


class _Observer(AnswerCallProvenanceObserver):
    def __init__(self) -> None:
        super().__init__()
        self.recorded: dict[str, RecordedResponse] = {}
        self.relations: dict[str, AnswerCallRelation] = {}

    def record_answer(self, recorded: RecordedResponse) -> None:
        self.recorded[recorded.call_id] = recorded
        super().record_answer(recorded)

    def finalize(self, answer_call_id: str) -> AnswerCallRelation:
        relation = super().finalize(answer_call_id)
        self.relations[answer_call_id] = relation
        return relation


class _Embedder:
    def encode_query(self, text: str) -> list[float]:
        assert text
        return [1.0, 0.0]


def _response(content: str) -> LLMResponse:
    return LLMResponse(content=content, raw={"private": content}, token_usage={}, latency_ms=0)


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="provenance-sample",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6], "target": 24},
    )


def _assert_explicit(
    outcome: BaselineExecutionOutcome, observer: _Observer, response: LLMResponse
) -> None:
    assert outcome.answer_call_id is not None
    relation = observer.relations[outcome.answer_call_id]
    assert relation.answer_call_provenance_status == "explicit_matched"
    assert relation.answer_call_id == relation.parsed_response_source_call_id == outcome.answer_call_id
    assert observer.recorded[outcome.answer_call_id].response is response


def _reject_legacy_answer_chat(monkeypatch, stages: set[str]) -> None:
    legacy_chat = MethodCallRecorder.chat

    def rejected_chat(self, messages, model, config):
        if config.get("method_stage") in stages:
            raise AssertionError("answer path used legacy chat")
        return legacy_chat(self, messages, model, config)

    monkeypatch.setattr(MethodCallRecorder, "chat", rejected_chat)


def test_full_history_and_rag_frozen_bind_the_returned_response_object(monkeypatch) -> None:
    # Given: actual full-history and frozen-RAG surfaces with distinct scripted answer objects.
    full_response, rag_response = _response("final: 24"), _response("final: 24")
    full_client = _ScriptedClient({"full_history_generate": [full_response]})
    rag_client = _ScriptedClient({"rag_generate": [rag_response]})
    full_observer, rag_observer = _Observer(), _Observer()
    _reject_legacy_answer_chat(monkeypatch, {"full_history_generate", "rag_generate"})
    document = Document("source-1", "Use exact fractions.")
    corpus = BranchCorpus("clean", (document,), (document.document_id,), "provenance-corpus")
    index = BranchIndex(
        "clean",
        (document,),
        {"production_identity": BGE_M3_PRIMARY_IDENTITY},
        {document.document_id: (1.0, 0.0)},
        "provenance-index",
        _Embedder(),
    )

    # When: each native execution surface produces its final answer.
    full_outcome = FullHistoryAdapter().execute(
        _task(),
        FullHistoryState(),
        client=full_client,
        model="replay",
        config={"_logging_answer_call_provenance_observer": full_observer},
        verifier=lambda _answer, _task: True,
    )
    rag_outcome = RagFrozenPhase12Adapter().execute(
        RagFrozenTrialContextV3(
            task=_task(),
            client=rag_client,
            model="replay",
            run_id="provenance",
            trial_id="provenance:rag",
            condition_id="rag_frozen",
            branch="clean",
            rag_mode="frozen",
            verifier=lambda _answer, _task: True,
            provenance_observer=rag_observer,
        ),
        RagFrozenStateV3("clean", corpus, index),
    ).outcome

    # Then: each outcome ID denotes its explicitly finalized, exact raw response object.
    _assert_explicit(full_outcome, full_observer, full_response)
    _assert_explicit(rag_outcome, rag_observer, rag_response)
    assert full_client.provider_calls_issued == rag_client.provider_calls_issued == 0


def test_reflexion_excludes_reflection_calls_from_answer_provenance(monkeypatch) -> None:
    # Given: a failed actor, an auxiliary reflection, and a successful retry.
    first, reflection, final = _response("final: wrong"), _response(
        json.dumps(
            {
                "mode": "corrective",
                "failure_class": "incorrect_answer",
                "reflection_text": "Use fractional arithmetic.",
                "explicitly_used_memory_ids": [],
            }
        )
    ), _response("final: 24")
    client = _ScriptedClient(
        {"reflexion_generate": [first, final], "reflexion_reflect": [reflection]}
    )
    observer = _Observer()
    _reject_legacy_answer_chat(monkeypatch, {"reflexion_generate"})

    # When: the real actor/reflection/retry loop executes.
    outcome = ReflexionAdapter().execute(
        _task(),
        ReflexionState(),
        client=client,
        model="replay",
        config={"max_attempts": 2, "_logging_answer_call_provenance_observer": observer},
        verifier=lambda answer, _task: answer == "24",
    )

    # Then: both actor answers are explicit while the reflection response has no relation.
    _assert_explicit(outcome, observer, final)
    assert len(observer.relations) == 2
    observed_responses = tuple(recorded.response for recorded in observer.recorded.values())
    assert any(response is first for response in observed_responses)
    assert any(response is final for response in observed_responses)
    assert all(response is not reflection for response in observed_responses)
    assert client.provider_calls_issued == 0


def test_full_history_parse_failure_finalizes_one_unresolved_relation() -> None:
    # Given: a returned but unparsable answer response.
    response = _response("not a final answer")
    observer = _Observer()

    # When: the full-history surface handles its parse failure.
    outcome = FullHistoryAdapter().execute(
        _task(),
        FullHistoryState(),
        client=_ScriptedClient({"full_history_generate": [response]}),
        model="replay",
        config={"_logging_answer_call_provenance_observer": observer},
    )

    # Then: its returned answer call has exactly one finalized unresolved relation.
    assert outcome.answer_call_id is not None
    assert tuple(observer.relations) == (outcome.answer_call_id,)
    assert observer.relations[outcome.answer_call_id].answer_call_provenance_status == "missing"
    assert observer.recorded[outcome.answer_call_id].response is response


def test_full_history_verifier_failure_finalizes_one_unresolved_relation() -> None:
    # Given: a parseable returned answer and a verifier-contract failure.
    response = _response("final: 24")
    observer = _Observer()

    # When: the full-history surface invokes the failing verifier.
    outcome = FullHistoryAdapter().execute(
        _task(),
        FullHistoryState(),
        client=_ScriptedClient({"full_history_generate": [response]}),
        model="replay",
        config={"_logging_answer_call_provenance_observer": observer},
        verifier=lambda _answer, _task: (_ for _ in ()).throw(RuntimeError("verifier failed")),
    )

    # Then: the returned answer call has exactly one finalized unresolved relation.
    assert outcome.answer_call_id is not None
    assert tuple(observer.relations) == (outcome.answer_call_id,)
    assert observer.relations[outcome.answer_call_id].answer_call_provenance_status == "missing"
    assert observer.recorded[outcome.answer_call_id].response is response
