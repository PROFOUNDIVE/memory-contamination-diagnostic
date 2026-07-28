from __future__ import annotations

import json
from pathlib import Path

from memcontam.baselines.bot_runtime import BotRuntime
from memcontam.clients.base import LLMResponse
from memcontam.experiment.phase12.filter_challenge.adapters.bot_style import (
    BoTChallengeExecution,
    BoTStyleChallengeAdapter,
)
from memcontam.experiment.phase12.filter_challenge.contracts import (
    AnswerCallRelation,
    ChallengeCandidate,
)
from memcontam.experiment.phase12.filter_challenge.provenance import (
    AnswerCallProvenanceObserver,
)
from memcontam.memory.bot_buffer import BotBufferIdentity
from memcontam.memory.cards_v3 import canonical_content_hash
from memcontam.memory.checkpoint_v3 import (
    NATIVE_ENTRY_V1,
    NativeEntry,
    NativeState,
    Phase12Checkpoint,
    serialize_checkpoint,
)
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance
from memcontam.tools import SubprocessTestDouble, load_tool_runtime_contract


_PROBLEM = json.dumps(
    {
        "key_information": "numbers = [1, 3, 4, 6], target = 24",
        "restrictions": "Use every number exactly once.",
        "distilled_task": "Construct an expression equal to 24.",
    }
)
_RETRIEVED_SOLUTION = json.dumps(
    {
        "selected_structure": "retrieved-template",
        "solution_trace": "Use the selected template.",
        "final_answer": "final: 24",
    }
)
_THOUGHT = json.dumps(
    {
        "description": "Build useful values.",
        "template": "Build values before combining them.",
        "category": "procedure-based",
        "explicitly_used_memory_ids": [],
    }
)


class _TieEmbeddingProvider:
    def encode_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]

    def encode_document(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


class _AdmittingEmbeddingProvider:
    def encode_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if text.startswith("{") else [0.0, 1.0]

    def encode_document(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


class _ScriptedClient:
    def __init__(self, responses: dict[str, list[str]]) -> None:
        self._responses = responses
        self.stages: list[str] = []
        self.provider_calls_issued = 0

    def chat(
        self, messages: list[dict[str, str]], model: str, config: dict[str, object]
    ) -> LLMResponse:
        del messages, model
        stage = config["method_stage"]
        assert isinstance(stage, str)
        self.stages.append(stage)
        return LLMResponse(content=self._responses[stage].pop(0), raw={}, token_usage={}, latency_ms=0)


class _Observer(AnswerCallProvenanceObserver):
    def __init__(self) -> None:
        super().__init__()
        self.relations: dict[str, AnswerCallRelation] = {}

    def finalize(self, answer_call_id: str) -> AnswerCallRelation:
        relation = super().finalize(answer_call_id)
        self.relations[answer_call_id] = relation
        return relation


def _task() -> TaskInstance:
    return TaskInstance(
        sample_id="bot-adapter",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6], "target": 24},
    )


def _identity() -> BotBufferIdentity:
    return BotBufferIdentity("filter-v5", "game24", "bot_style", "clean", "replay")


def _template(entry_id: str) -> NativeEntry:
    content = f"Use template {entry_id}."
    return NativeEntry(
        entry_id=entry_id,
        semantic_kind="thought_template",
        schema_version=NATIVE_ENTRY_V1,
        native_component="buffer",
        content=content,
        content_hash=canonical_content_hash(content),
    )


def _checkpoint(*, capacity: int, entries: tuple[NativeEntry, ...]) -> Phase12Checkpoint:
    return serialize_checkpoint(
        NativeState(
            "bot_style",
            entries,
            {"active_capacity": capacity, "clean_competitor_ids": [], "templates": [entry.entry_id for entry in entries]},
        )
    )


def _candidate(checkpoint: Phase12Checkpoint) -> ChallengeCandidate:
    return ChallengeCandidate(
        candidate_entry_id="candidate-template",
        candidate_native_content="Apply the candidate template.",
        candidate_native_kind="thought_template",
        baseline_family="bot_style",
        rag_mode="not_applicable",
        source_checkpoint_id=checkpoint.identity.checkpoint_id,
        source_active_state_hash=checkpoint.canonical_sha256,
        routability={"routability": "challenge_routable_v1", "challenge_suite_key": "synthetic"},
    )


def _execution(
    checkpoint: Phase12Checkpoint,
    client: _ScriptedClient,
    observer: _Observer,
    *,
    tool_mode: str = "text_only",
) -> BoTChallengeExecution:
    config: dict[str, object] = {
        "embedding_provider": _TieEmbeddingProvider(),
        "tool_mode": tool_mode,
        "_logging_answer_call_provenance_observer": observer,
    }
    if tool_mode == "python_sandbox":
        config["tool_executor"] = SubprocessTestDouble()
        config["tool_runtime_contract"] = load_tool_runtime_contract(
            Path(__file__).resolve().parents[1] / "containers" / "python-sandbox" / "image.lock.json",
            scientific=False,
        )
    return BoTChallengeExecution(
        checkpoint=checkpoint,
        task=_task(),
        client=client,
        model="replay",
        identity=_identity(),
        config=config,
        verifier=lambda _answer: True,
    )


def test_bot_runtime_default_keeps_distillation_and_writeback_enabled() -> None:
    # Given: an ordinary BoT run with a native template and all three replay responses.
    template = MemoryEntry(
        entry_id="template-a",
        content="Use template a.",
        memory_type="thought_template",
        metadata={"description": "Use template a.", "category": "procedure-based"},
    )
    client = _ScriptedClient(
        {
            "bot_problem_distill": [_PROBLEM],
            "bot_instantiate_solve": [_RETRIEVED_SOLUTION],
            "bot_thought_distill": [_THOUGHT],
        }
    )

    # When: no update setting is provided to the native runtime.
    outcome = BotRuntime().run(
        identity=_identity(),
        task=_task(),
        buffer_snapshot=[template],
        client=client,
        model="replay",
        config={"embedding_provider": _AdmittingEmbeddingProvider()},
        verifier=lambda _answer: True,
    )

    # Then: the legacy writeback sequence and append remain intact.
    assert client.stages == ["bot_problem_distill", "bot_instantiate_solve", "bot_thought_distill"]
    assert len(outcome.memory_after) == len(outcome.memory_before) + 1
    assert outcome.memory_write_event is not None


def test_bot_challenge_selects_the_native_candidate_without_writeback() -> None:
    # Given: a frozen two-template buffer with space for one lexically winning tied candidate.
    checkpoint = _checkpoint(capacity=3, entries=(_template("template-z"), _template("template-y")))
    source_bytes, source_hash = checkpoint.canonical_bytes, checkpoint.canonical_sha256
    observer = _Observer()
    client = _ScriptedClient(
        {"bot_problem_distill": [_PROBLEM], "bot_instantiate_solve": [_RETRIEVED_SOLUTION]}
    )

    # When: the adapter performs a provisional challenge execution.
    result = BoTStyleChallengeAdapter().execute(_execution(checkpoint, client, observer), _candidate(checkpoint))

    # Then: native insertion and tie-breaking select the candidate, but writeback remains absent.
    assert result.provisional_template_ids == ("template-z", "template-y", "candidate-template")
    assert result.selected_template_id == "candidate-template"
    assert result.nonselected_template_ids == ("template-z", "template-y")
    assert result.displaced_template_ids == ()
    assert result.final_context_source_ids == ("candidate-template",)
    assert result.candidate_final_context_inclusion
    assert [call.stage for call in result.outcome.method_calls] == [
        "bot_problem_distill",
        "bot_instantiate_solve",
    ]
    assert result.outcome.memory_after == result.outcome.memory_before
    assert result.outcome.memory_write_event is None
    assert checkpoint.canonical_bytes == source_bytes
    assert checkpoint.canonical_sha256 == source_hash
    assert observer.relations[result.outcome.answer_call_id].answer_call_provenance_status == "explicit_matched"
    assert client.provider_calls_issued == 0


def test_bot_challenge_preserves_capacity_when_candidate_cannot_compete() -> None:
    # Given: a full native buffer where the candidate has no admission slot.
    checkpoint = _checkpoint(capacity=2, entries=(_template("template-y"), _template("template-z")))
    observer = _Observer()
    client = _ScriptedClient(
        {"bot_problem_distill": [_PROBLEM], "bot_instantiate_solve": [_RETRIEVED_SOLUTION]}
    )
    control_client = _ScriptedClient(
        {"bot_problem_distill": [_PROBLEM], "bot_instantiate_solve": [_RETRIEVED_SOLUTION]}
    )

    # When: the candidate is provisionally evaluated against the unchanged capacity rule.
    result = BoTStyleChallengeAdapter().execute(_execution(checkpoint, client, observer), _candidate(checkpoint))
    control = BoTStyleChallengeAdapter().execute_control(
        _execution(checkpoint, control_client, _Observer())
    )

    # Then: the selected native template, rather than mere candidate storage, determines exposure.
    assert result.provisional_template_ids == ("template-y", "template-z")
    assert result.selected_template_id == "template-y"
    assert result.nonselected_template_ids == ("template-z",)
    assert result.displaced_template_ids == ()
    assert result.final_context_source_ids == ("template-y",)
    assert not result.candidate_final_context_inclusion
    assert control.provisional_checkpoint is checkpoint
    assert control.final_context_source_ids == ("template-y",)
    assert [call.stage for call in result.outcome.method_calls] == [
        "bot_problem_distill",
        "bot_instantiate_solve",
    ]


def test_bot_challenge_keeps_final_tool_response_as_the_provenance_answer() -> None:
    # Given: a candidate-only provisional buffer and a native tool continuation.
    checkpoint = _checkpoint(capacity=1, entries=())
    observer = _Observer()
    client = _ScriptedClient(
        {
            "bot_problem_distill": [_PROBLEM],
            "bot_instantiate_solve": [
                json.dumps({"action": "execute_python", "code": "print(24)"}),
                json.dumps({"action": "final", "answer": _RETRIEVED_SOLUTION}),
            ],
        }
    )

    # When: read-only challenge execution follows the native tool path.
    result = BoTStyleChallengeAdapter().execute(
        _execution(checkpoint, client, observer, tool_mode="python_sandbox"), _candidate(checkpoint)
    )

    # Then: only the terminal continuation is the explicit answer and no thought template is distilled.
    assert result.outcome.answer_call_id == result.outcome.method_calls[-1].call_id
    assert observer.relations[result.outcome.answer_call_id].answer_call_provenance_status == "explicit_matched"
    assert [call.stage for call in result.outcome.method_calls] == [
        "bot_problem_distill",
        "bot_instantiate_solve",
        "bot_instantiate_solve",
    ]
    assert result.outcome.memory_write_event is None
    assert client.provider_calls_issued == 0
