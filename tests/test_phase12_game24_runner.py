from __future__ import annotations

import json

from memcontam.baselines.bot_phase12 import BoTStateV3
from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.baselines.reflexion_phase12 import ReflexionStateV3
from memcontam.baselines.retrieval_rag_phase12 import RagFrozenStateV3
from memcontam.clients.base import LLMResponse
from memcontam.experiment.phase12.game24_runner import (
    Game24RuntimeContext,
    RuntimeIdentities,
    RuntimeWriterCallbacks,
    run_clean_game24_trial,
)
from memcontam.memory.checkpoint_v3 import NativeEntry
from memcontam.memory.stores import MemoryEntry
from memcontam.rag.branch_index import build_branch_indices
from memcontam.rag.phase12_corpus import CleanCorpus, build_branch_corpora
from memcontam.tasks.game24 import build_instance
from memcontam.verifiers.game24 import verify_expression


ANSWER = "final: 6 / (1 - 3 / 4)"


class _Client:
    def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
        del messages, model
        stage = config["method_stage"]
        responses = {
            "no_memory_generate": ANSWER,
            "full_history_generate": ANSWER,
            "rag_generate": ANSWER,
            "bot_problem_distill": json.dumps(
                {
                    "key_information": "numbers = [1, 3, 4, 6], target = 24",
                    "restrictions": "Use every number exactly once.",
                    "distilled_task": "Construct an expression equal to 24.",
                }
            ),
            "bot_instantiate_solve": json.dumps(
                {
                    "selected_structure": "retrieved-template",
                    "solution_trace": "Use rational intermediate values.",
                    "final_answer": ANSWER,
                }
            ),
            "bot_thought_distill": json.dumps(
                {
                    "description": "Use rational intermediate values.",
                    "template": "Build exact fractions before combining values.",
                    "category": "procedure-based",
                    "explicitly_used_memory_ids": [],
                }
            ),
            "reflexion_generate": ANSWER,
        }
        return LLMResponse(content=responses[stage], raw={}, token_usage={}, latency_ms=0)


class _Embedder:
    embedding_contract = {
        "dimension": 2,
        "normalized": True,
        "production_identity": "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181",
        "provider": "test",
    }

    def encode_document(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]

    def encode_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


def _rag_state(embedder: _Embedder) -> RagFrozenStateV3:
    corpora = build_branch_corpora(
        CleanCorpus.from_documents(
            [
                {"id": "clean-a", "text": "Use rational intermediate values."},
                {"id": "clean-b", "text": "Check arithmetic exactly."},
            ],
            corpus_id="game24-clean",
        ),
        {
            "false": {"id": "false", "text": "Use only integer intermediates."},
            "correct": {"id": "correct", "text": "Fractions are valid."},
            "irrelevant": {"id": "irrelevant", "text": "Sort alphabetically."},
        },
    )
    indices = build_branch_indices(corpora, embedder, filter_policy=None)
    return RagFrozenStateV3("clean", corpora.branches["clean"], indices.branches["clean"])


def test_one_mocked_clean_game24_trial_uses_each_native_baseline_flow() -> None:
    embedder = _Embedder()
    bot_entries: list[MemoryEntry | NativeEntry] = [
        MemoryEntry(
            entry_id="bot-a",
            content="Use rational intermediate values.",
            memory_type="thought_template",
            metadata={"description": "Use rational intermediate values.", "category": "procedure-based"},
        ),
        MemoryEntry(
            entry_id="bot-b",
            content="Check arithmetic exactly.",
            memory_type="thought_template",
            metadata={"description": "Check arithmetic exactly.", "category": "procedure-based"},
        ),
    ]
    outcomes = []
    context = Game24RuntimeContext(
        task=build_instance({"sample_id": "game24-1", "numbers": [1, 3, 4, 6]}),
        client=_Client(),
        model="test-model",
        verifier=lambda answer, task: verify_expression(
            answer, task.input["numbers"], task.verifier_spec["target"]
        ),
        decoding={"temperature": 0, "max_tokens": 128},
        branch="clean",
        identities=RuntimeIdentities("run-1", "trial-1", 1),
        writer_callbacks=RuntimeWriterCallbacks(on_outcome=outcomes.append),
        embedding_provider=embedder,
        baseline_configs={"fh_bounded": {"context_window_tokens": 10_000}},
        initial_states={
            "fh_bounded": FullHistoryStateV3(records=[]),
            "rag_frozen": _rag_state(embedder),
            "bot_style": BoTStateV3(
                entries=bot_entries,
                clean_competitor_ids=("bot-a", "bot-b"),
                active_capacity=3,
            ),
            "reflexion_style": ReflexionStateV3(reflections=[], active_capacity=3),
        },
    )

    results = run_clean_game24_trial(context)

    assert tuple(results) == (
        "nomem",
        "fh_bounded",
        "rag_frozen",
        "bot_style",
        "reflexion_style",
    )
    assert [result.outcome.status for result in results.values()] == ["succeeded"] * 5
    assert {
        baseline: [call.stage for call in result.outcome.method_calls]
        for baseline, result in results.items()
    } == {
        "nomem": ["no_memory_generate"],
        "fh_bounded": ["full_history_generate"],
        "rag_frozen": ["rag_generate"],
        "bot_style": [
            "bot_problem_distill",
            "bot_instantiate_solve",
            "bot_thought_distill",
        ],
        "reflexion_style": ["reflexion_generate"],
    }
    assert len(outcomes) == 5
