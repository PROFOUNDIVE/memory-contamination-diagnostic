from __future__ import annotations

import importlib
import inspect
import json

import pytest

from memcontam.baselines.bot_style import BotStylePolicy
from memcontam.clients.base import LLMResponse
from memcontam.clients.replay import ReplayClient
from memcontam.memory.bot_buffer import BotBufferIdentity
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance


def _problem():
    bot_read = importlib.import_module("memcontam.baselines.bot_read")
    return bot_read.DistilledProblem(
        key_information="numbers = [1, 2, 3, 4], target = 24",
        restrictions="Use every number exactly once.",
        distilled_task="Construct an expression equal to 24.",
    )


def _entry(*, metadata: dict[str, str] | None = None) -> MemoryEntry:
    return MemoryEntry(
        entry_id="template-1",
        content="template body must not be embedded for retrieval",
        memory_type="thought_template",
        clean_or_contaminated="clean",
        metadata=metadata
        or {
            "description": "Combine factor pairs before the final arithmetic step.",
            "category": "procedure-based",
        },
    )


class _SimilarityProvider:
    def __init__(self, similarity: float) -> None:
        self.similarity = similarity
        self.queries: list[str] = []
        self.documents: list[str] = []

    def encode_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]

    def encode_document(self, text: str) -> list[float]:
        self.documents.append(text)
        return [self.similarity, (1 - self.similarity**2) ** 0.5]


def _solve_output(selected_structure: str) -> str:
    return json.dumps(
        {
            "selected_structure": selected_structure,
            "solution_trace": "Build two pairs, then combine them.",
            "final_answer": "final: 24",
        }
    )


def _thought_output() -> str:
    return json.dumps(
        {
            "description": "Build useful pairs before combining them.",
            "template": "Create pairs, then combine the intermediate values.",
            "category": "procedure-based",
            "explicitly_used_memory_ids": [],
        }
    )


def test_retrieval_decision_matches_at_threshold_and_embeds_only_description() -> None:
    bot_read = importlib.import_module("memcontam.baselines.bot_read")
    provider = _SimilarityProvider(0.7)

    result = bot_read.retrieve_top_template(_problem(), [_entry()], provider)

    assert result.decision == "matched"
    assert result.matched_entry is not None
    assert result.matched_entry.entry_id == "template-1"
    assert result.top_similarity == pytest.approx(0.7)
    assert result.threshold == pytest.approx(0.7)
    assert provider.queries == [bot_read.build_distilled_query(_problem())]
    assert provider.documents == ["Combine factor pairs before the final arithmetic step."]


def test_retrieval_decision_returns_miss_below_threshold() -> None:
    bot_read = importlib.import_module("memcontam.baselines.bot_read")

    result = bot_read.retrieve_top_template(_problem(), [_entry()], _SimilarityProvider(0.699))

    assert result.decision == "miss"
    assert result.matched_entry is None
    assert result.top_similarity == pytest.approx(0.699)


def test_retrieval_decision_returns_empty_buffer_without_embedding() -> None:
    bot_read = importlib.import_module("memcontam.baselines.bot_read")

    result = bot_read.retrieve_top_template(_problem(), [], _SimilarityProvider(1.0))

    assert result.decision == "empty_buffer"
    assert result.matched_entry is None
    assert result.top_similarity is None


@pytest.mark.parametrize(
    "metadata",
    [{"category": "procedure-based"}, {"description": "only"}, {"description": "only", "category": ""}],
)
def test_retrieval_rejects_templates_without_explicit_description_and_category(
    metadata: dict[str, str],
) -> None:
    bot_read = importlib.import_module("memcontam.baselines.bot_read")

    with pytest.raises(ValueError, match="description and category"):
        bot_read.retrieve_top_template(_problem(), [_entry(metadata=metadata)], _SimilarityProvider(1.0))


def test_miss_prompt_renders_all_coarse_structures_and_requires_a_selection() -> None:
    bot_read = importlib.import_module("memcontam.baselines.bot_read")
    bot_solve = importlib.import_module("memcontam.baselines.bot_solve")
    decision = bot_read.retrieve_top_template(_problem(), [], _SimilarityProvider(1.0))

    prompt, source_spans = bot_solve.render_bot_solve_prompt(
        TaskInstance(sample_id="sample", task_name="game24", input={"numbers": [1, 2, 3, 4]}),
        _problem(),
        decision,
    )

    assert "prompt-based" in prompt
    assert "procedure-based" in prompt
    assert "programming-based" in prompt
    assert "selected_structure, solution_trace, final_answer" in prompt
    assert source_spans == []


def test_matched_retrieval_rejects_a_coarse_fallback_selection() -> None:
    runtime = importlib.import_module("memcontam.baselines.bot_runtime").BotRuntime()
    task = TaskInstance(sample_id="sample", task_name="game24", input={"numbers": [1, 2, 3, 4]})
    client = ReplayClient(
        responses_by_sample={
            "sample": {
                "bot_problem_distill": _problem().model_dump_json(),
                "bot_instantiate_solve": _solve_output("procedure-based"),
            }
        }
    )

    outcome = runtime.run(
        identity=BotBufferIdentity("run", "game24", "bot_style", "clean", "replay"),
        task=task,
        buffer_snapshot=[_entry()],
        client=client,
        model="replay",
        config={"embedding_provider": _SimilarityProvider(1.0)},
    )

    assert outcome.failure_disposition == "bot_invalid_solve_result"
    assert [call.stage for call in outcome.method_calls] == [
        "bot_problem_distill",
        "bot_instantiate_solve",
    ]


def test_runtime_requires_an_explicit_embedding_provider() -> None:
    runtime = importlib.import_module("memcontam.baselines.bot_runtime").BotRuntime()

    with pytest.raises(ValueError, match="embedding_provider"):
        runtime.run(
            identity=BotBufferIdentity("run", "game24", "bot_style", "clean", "replay"),
            task=TaskInstance(sample_id="sample", task_name="game24", input={}),
            buffer_snapshot=[],
            client=ReplayClient(responses_by_sample={}),
            model="replay",
            config={},
        )


def test_empty_buffer_logs_selected_coarse_fallback_with_three_semantic_calls() -> None:
    runtime = importlib.import_module("memcontam.baselines.bot_runtime").BotRuntime()
    task = TaskInstance(sample_id="sample", task_name="game24", input={"numbers": [1, 2, 3, 4]})
    client = ReplayClient(
        responses_by_sample={
            "sample": {
                "bot_problem_distill": _problem().model_dump_json(),
                "bot_instantiate_solve": _solve_output("procedure-based"),
                "bot_thought_distill": _thought_output(),
            }
        }
    )

    outcome = runtime.run(
        identity=BotBufferIdentity("run", "game24", "bot_style", "clean", "replay"),
        task=task,
        buffer_snapshot=[],
        client=client,
        model="replay",
        config={"embedding_provider": _SimilarityProvider(1.0)},
        verifier=lambda _answer: True,
    )

    assert outcome.metadata["retrieval_decision"] == {
        "decision": "empty_buffer",
        "matched_entry_id": None,
        "top_similarity": None,
        "threshold": 0.7,
    }
    assert outcome.metadata["selected_structure"] == "procedure-based"
    assert [call.stage for call in outcome.method_calls] == [
        "bot_problem_distill",
        "bot_instantiate_solve",
        "bot_thought_distill",
    ]


def test_miss_logs_similarity_without_retrieved_memory_or_score() -> None:
    runtime = importlib.import_module("memcontam.baselines.bot_runtime").BotRuntime()
    task = TaskInstance(sample_id="sample", task_name="game24", input={"numbers": [1, 2, 3, 4]})
    client = ReplayClient(
        responses_by_sample={
            "sample": {
                "bot_problem_distill": _problem().model_dump_json(),
                "bot_instantiate_solve": _solve_output("procedure-based"),
                "bot_thought_distill": _thought_output(),
            }
        }
    )

    outcome = runtime.run(
        identity=BotBufferIdentity("run", "game24", "bot_style", "clean", "replay"),
        task=task,
        buffer_snapshot=[_entry()],
        client=client,
        model="replay",
        config={"embedding_provider": _SimilarityProvider(0.699)},
        verifier=lambda _answer: True,
    )

    assert outcome.metadata["retrieval_decision"]["top_similarity"] == pytest.approx(0.699)
    assert outcome.retrieved_memory == ()
    assert outcome.retrieved_scores == ()


def test_legacy_solve_compatibility_response_selects_a_coarse_structure() -> None:
    cli = importlib.import_module("memcontam.cli")

    class LegacySolveClient:
        def chat(self, *_args: object, **_kwargs: object) -> LLMResponse:
            return LLMResponse(content="final: 24", raw={}, token_usage={})

    response = cli._ReplayBotSolveCompatibilityClient(LegacySolveClient()).chat(
        [], "replay", {"method_stage": "bot_instantiate_solve"}
    )
    matched_response = cli._ReplayBotSolveCompatibilityClient(LegacySolveClient()).chat(
        [],
        "replay",
        {"method_stage": "bot_instantiate_solve", "_bot_retrieval_decision": "matched"},
    )

    assert json.loads(response.content)["selected_structure"] == "procedure-based"
    assert json.loads(matched_response.content)["selected_structure"] == "retrieved-template"


def test_build_prompt_requires_an_explicit_provider() -> None:
    assert "embedding_provider" in inspect.signature(BotStylePolicy().build_prompt).parameters
