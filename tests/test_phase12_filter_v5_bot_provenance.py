from __future__ import annotations

import json
from pathlib import Path

from memcontam.baselines.bot_runtime import BotRuntime
from memcontam.clients.base import LLMResponse
from memcontam.clients.recording import MethodCallRecorder, RecordedResponse
from memcontam.experiment.phase12.filter_challenge.contracts import AnswerCallRelation
from memcontam.experiment.phase12.filter_challenge.provenance import AnswerCallProvenanceObserver
from memcontam.memory.bot_buffer import BotBufferIdentity
from memcontam.tasks.game24 import build_instance
from memcontam.tools import SubprocessTestDouble, load_tool_runtime_contract


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


class _EmbeddingProvider:
    def encode_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]

    def encode_document(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


def _response(content: str) -> LLMResponse:
    return LLMResponse(content=content, raw={"private": content}, token_usage={}, latency_ms=0)


def _task():
    return build_instance({"sample_id": "bot-provenance", "numbers": [1, 3, 4, 6], "target": 24})


def _identity() -> BotBufferIdentity:
    return BotBufferIdentity("provenance", "game24", "bot_style", "clean", "replay")


def _problem() -> str:
    return json.dumps(
        {
            "key_information": "numbers = [1, 3, 4, 6], target = 24",
            "restrictions": "Use every number exactly once.",
            "distilled_task": "Construct an expression equal to 24.",
        }
    )


def _solve() -> str:
    return json.dumps(
        {
            "selected_structure": "programming-based",
            "solution_trace": "Use the result.",
            "final_answer": "final: 24",
        }
    )


def _thought() -> str:
    return json.dumps(
        {
            "description": "Validate the result.",
            "template": "Validate then answer.",
            "category": "programming-based",
            "explicitly_used_memory_ids": [],
        }
    )


def _run(client: _ScriptedClient, observer: _Observer, config: dict, verifier) -> object:
    return BotRuntime().run(
        identity=_identity(),
        task=_task(),
        buffer_snapshot=[],
        client=client,
        model="replay",
        config={"embedding_provider": _EmbeddingProvider(), **config, "_logging_answer_call_provenance_observer": observer},
        verifier=verifier,
    )


def _assert_explicit(outcome: object, observer: _Observer, response: LLMResponse) -> None:
    answer_call_id = outcome.answer_call_id
    assert answer_call_id is not None
    relation = observer.relations[answer_call_id]
    assert relation.answer_call_provenance_status == "explicit_matched"
    assert relation.answer_call_id == relation.parsed_response_source_call_id == answer_call_id
    assert observer.recorded[answer_call_id].response is response


def _reject_legacy_answer_chat(monkeypatch) -> None:
    legacy_chat = MethodCallRecorder.chat

    def rejected_chat(self, messages, model, config):
        if config.get("method_stage") == "bot_instantiate_solve":
            raise AssertionError("answer path used legacy chat")
        return legacy_chat(self, messages, model, config)

    monkeypatch.setattr(MethodCallRecorder, "chat", rejected_chat)


def test_bot_text_answer_is_the_only_provenance_relation(monkeypatch) -> None:
    # Given: native distillation, text solving, and thought distillation responses.
    distilled, answer, thought = _response(_problem()), _response(_solve()), _response(_thought())
    client = _ScriptedClient(
        {
            "bot_problem_distill": [distilled],
            "bot_instantiate_solve": [answer],
            "bot_thought_distill": [thought],
        }
    )
    observer = _Observer()
    _reject_legacy_answer_chat(monkeypatch)

    # When: the text-only BoT surface completes.
    outcome = _run(client, observer, {"tool_mode": "text_only"}, lambda _answer: True)

    # Then: auxiliary calls do not become answer relations.
    _assert_explicit(outcome, observer, answer)
    assert len(observer.relations) == 1
    assert all(recorded.response is not distilled and recorded.response is not thought for recorded in observer.recorded.values())
    assert client.provider_calls_issued == 0


def test_bot_tool_continuation_carries_its_exact_final_response(monkeypatch) -> None:
    # Given: an auxiliary tool action followed by a distinct final continuation response.
    action = _response(json.dumps({"action": "execute_python", "code": "print(24)"}))
    final = _response(json.dumps({"action": "final", "answer": _solve()}))
    client = _ScriptedClient(
        {
            "bot_problem_distill": [_response(_problem())],
            "bot_instantiate_solve": [action, final],
            "bot_thought_distill": [_response(_thought())],
        }
    )
    observer = _Observer()
    _reject_legacy_answer_chat(monkeypatch)
    policy = load_tool_runtime_contract(
        Path(__file__).resolve().parents[1] / "containers" / "python-sandbox" / "image.lock.json",
        scientific=False,
    )

    # When: the native tool-augmented BoT surface reaches its final continuation.
    outcome = _run(
        client,
        observer,
        {"tool_mode": "python_sandbox", "tool_executor": SubprocessTestDouble(), "tool_runtime_contract": policy},
        lambda _answer: True,
    )

    # Then: only the final continuation is finalized as the answer.
    _assert_explicit(outcome, observer, final)
    assert outcome.answer_call_id == outcome.method_calls[2].call_id
    assert len(observer.relations) == 1
    assert all(recorded.response is not action for recorded in observer.recorded.values())
    assert client.provider_calls_issued == 0


def test_bot_parse_failure_finalizes_one_unresolved_relation() -> None:
    # Given: a returned solve response that cannot be parsed.
    answer = _response("not bot JSON")
    client = _ScriptedClient({"bot_problem_distill": [_response(_problem())], "bot_instantiate_solve": [answer]})
    observer = _Observer()

    # When: the native text surface handles its parse failure.
    outcome = _run(client, observer, {"tool_mode": "text_only"}, lambda _answer: True)

    # Then: its returned answer response has one unresolved finalized relation.
    assert outcome.answer_call_id is not None
    assert tuple(observer.relations) == (outcome.answer_call_id,)
    assert observer.relations[outcome.answer_call_id].answer_call_provenance_status == "missing"
    assert observer.recorded[outcome.answer_call_id].response is answer


def test_bot_verifier_failure_finalizes_one_unresolved_relation() -> None:
    # Given: a parseable returned answer response and a failing verifier.
    answer = _response(_solve())
    client = _ScriptedClient(
        {
            "bot_problem_distill": [_response(_problem())],
            "bot_instantiate_solve": [answer],
            "bot_thought_distill": [_response(_thought())],
        }
    )
    observer = _Observer()

    # When: the native text surface invokes the failing verifier.
    outcome = _run(
        client,
        observer,
        {"tool_mode": "text_only"},
        lambda _answer: (_ for _ in ()).throw(RuntimeError("verifier failed")),
    )

    # Then: its returned answer response has one unresolved finalized relation.
    assert outcome.answer_call_id is not None
    assert tuple(observer.relations) == (outcome.answer_call_id,)
    assert observer.relations[outcome.answer_call_id].answer_call_provenance_status == "missing"
    assert observer.recorded[outcome.answer_call_id].response is answer


def test_bot_malformed_final_continuation_finalizes_one_unresolved_relation() -> None:
    # Given: a returned terminal continuation whose final action is malformed.
    action = _response(json.dumps({"action": "execute_python", "code": "print(24)"}))
    malformed = _response(json.dumps({"action": "final", "answer": ""}))
    observer = _Observer()
    policy = load_tool_runtime_contract(
        Path(__file__).resolve().parents[1] / "containers" / "python-sandbox" / "image.lock.json",
        scientific=False,
    )

    # When: the native tool loop rejects the malformed terminal continuation.
    outcome = _run(
        _ScriptedClient(
            {
                "bot_problem_distill": [_response(_problem())],
                "bot_instantiate_solve": [action, malformed],
            }
        ),
        observer,
        {"tool_mode": "python_sandbox", "tool_executor": SubprocessTestDouble(), "tool_runtime_contract": policy},
        lambda _answer: True,
    )

    # Then: the returned terminal response is finalized once as unresolved.
    assert outcome.answer_call_id is not None
    assert tuple(observer.relations) == (outcome.answer_call_id,)
    assert observer.relations[outcome.answer_call_id].answer_call_provenance_status == "missing"
    assert observer.recorded[outcome.answer_call_id].response is malformed
