from __future__ import annotations

import inspect

from memcontam.baselines.bot_phase12 import BoTStateV3
from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.baselines.reflexion_phase12 import ReflexionStateV3
from memcontam.baselines.retrieval_rag_phase12 import RagFrozenStateV3
from memcontam.experiment.phase12.game24_runner import (
    Game24RuntimeContext,
    RuntimeIdentities,
)
from memcontam.experiment.phase12.runtime_registry import LIVE_BASELINE_REGISTRY
from memcontam.memory.checkpoint_v3 import NativeEntry
from memcontam.memory.stores import MemoryEntry
from memcontam.rag.branch_index import build_branch_indices
from memcontam.rag.phase12_corpus import CleanCorpus, build_branch_corpora
from memcontam.tasks.game24 import build_instance


class _Client:
    def chat(self, messages, model, config):  # noqa: ANN001, ANN201
        del messages, model, config
        raise AssertionError("state tests do not call the client")


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


def _context() -> Game24RuntimeContext:
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
    return Game24RuntimeContext(
        task=build_instance({"sample_id": "game24-1", "numbers": [1, 3, 4, 6]}),
        client=_Client(),
        model="test-model",
        verifier=lambda _answer, _task: True,
        decoding={"temperature": 0},
        branch="clean",
        identities=RuntimeIdentities("run-1", "trial-1", 1),
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


def test_live_registry_is_narrow_and_round_trips_native_states() -> None:
    context = _context()

    assert tuple(LIVE_BASELINE_REGISTRY) == (
        "nomem",
        "fh_bounded",
        "rag_frozen",
        "bot_style",
        "reflexion_style",
    )
    for entry in LIVE_BASELINE_REGISTRY.values():
        assert all(
            callable(getattr(entry, attribute))
            for attribute in (
                "initial_state",
                "execute_trial",
                "serialize_state",
                "restore_state",
                "maturity_view",
            )
        )
        state = entry.initial_state(context)
        restored = entry.restore_state(entry.serialize_state(state), context)
        assert entry.serialize_state(restored) == entry.serialize_state(state)

    nomem = LIVE_BASELINE_REGISTRY["nomem"]
    assert nomem.initial_state(context) is nomem.restore_state(
        nomem.serialize_state(nomem.initial_state(context)), context
    )


def test_live_modules_do_not_reference_replay_policies() -> None:
    import memcontam.experiment.phase12.game24_runner as runner
    import memcontam.experiment.phase12.runtime_registry as registry

    for module in (registry, runner):
        source = inspect.getsource(module)
        assert "_ReplayPrefixPolicy" not in source
        assert "_ReplaySuffixPolicy" not in source
