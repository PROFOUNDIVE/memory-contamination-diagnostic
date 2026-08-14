from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import assert_never, cast

import pytest

from memcontam.baselines.bot_phase12 import BoTStateV3
from memcontam.baselines import dynamic_cheatsheet_phase12 as dc
from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.baselines.reflexion_phase12 import ReflexionStateV3
from memcontam.baselines.retrieval_rag_phase12 import RagFrozenStateV3
from memcontam.contamination.phase12.models import CandidateVariant, canonical_content_hash
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.live_branch import Arm, build_live_reduced_main_branches
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, serialize_checkpoint
from memcontam.memory.stores import MemoryEntry
from memcontam.rag.branch_index import build_branch_indices
from memcontam.rag.phase12_corpus import CleanCorpus, build_branch_corpora
from memcontam.tasks.base import TaskInstance

REGISTRY_PATH = Path("data/phase12/registries/candidate_registry_v1.json")
LiveState = FullHistoryStateV3 | RagFrozenStateV3 | BoTStateV3 | ReflexionStateV3 | dc.DcRsStateV3


class _Client:
    def chat(self, messages, model, config):  # noqa: ANN001, ANN201
        del messages, model, config
        raise AssertionError("native-rendering QA does not call a provider")


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


def _clean_state(baseline: str, embedder: _Embedder) -> LiveState:
    if baseline == "fh_bounded":
        return FullHistoryStateV3(records=[])
    if baseline == "rag_frozen":
        corpora = build_branch_corpora(
            CleanCorpus.from_documents(
                [
                    {"id": "clean-rule", "text": "Follow the registered task procedure."},
                    {"id": "clean-check", "text": "Verify the final answer."},
                ],
                corpus_id="phase13-main-clean",
            ),
            {
                "false": {"id": "unused-false", "text": "unused"},
                "correct": {"id": "unused-correct", "text": "unused"},
                "irrelevant": {"id": "unused-irrelevant", "text": "unused"},
            },
        )
        indices = build_branch_indices(corpora, embedder, filter_policy=None)
        return RagFrozenStateV3("clean", corpora.branches["clean"], indices.branches["clean"])
    if baseline == "bot_style":
        entries: list[MemoryEntry | NativeEntry] = [
            MemoryEntry(entry_id="bot-a", content="Follow the task rules.", memory_type="thought_template"),
            MemoryEntry(entry_id="bot-b", content="Check the answer.", memory_type="thought_template"),
        ]
        return BoTStateV3(entries=entries, clean_competitor_ids=("bot-a", "bot-b"))
    if baseline == "dc_rs":
        return dc.DcRsStateV3(archive=[])
    return ReflexionStateV3(reflections=[])


TASKS = (
    TaskInstance(
        sample_id="game24-main-1",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6]},
        verifier_spec={"target": 24},
    ),
    TaskInstance(
        sample_id="meb-main-1",
        task_name="math_equation_balancer",
        input={"input": "14 ? 16 ? 20 ? 15 ? 23 = 25"},
        verifier_spec={"target": "14 - 16 / 20 * 15 + 23 = 25", "target_value": 25},
    ),
    TaskInstance(
        sample_id="words-main-1",
        task_name="word_sorting",
        input={"words": ["syndrome", "therefrom"]},
        verifier_spec={"sorted_words": ["syndrome", "therefrom"]},
    ),
)


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize(
    "baseline",
    tuple(
        baseline
        for baseline in PHASE13_CORE_BASELINE_REGISTRY
        if baseline not in {"nomem", "dc_rs"}
    ),
)
def test_reduced_main_materializes_native_roots_through_live_state_surface(
    task: TaskInstance, baseline: str
) -> None:
    embedder = _Embedder()
    clean_state = _clean_state(baseline, embedder)
    context = Game24RuntimeContext(
        task=task,
        client=_Client(),
        model="test-model",
        verifier=lambda _answer, _task: False,
        decoding={"temperature": 0.0},
        branch="clean",
        identities=RuntimeIdentities("run-1", f"{task.sample_id}-{baseline}", 1),
        embedding_provider=embedder,
        baseline_configs={"fh_bounded": {"context_window_tokens": 10_000}},
        initial_states={baseline: clean_state},
    )
    runtime = PHASE13_CORE_BASELINE_REGISTRY[baseline]
    prefix = serialize_checkpoint(cast(NativeState, runtime.serialize_state(clean_state)))
    registry = load_candidate_registry(REGISTRY_PATH)
    triplet = next(item for item in registry.triplets if item.task == task.task_name)
    branches = build_live_reduced_main_branches(
        prefix=prefix,
        context=context,
        candidate_registry=registry,
        registry=PHASE13_CORE_BASELINE_REGISTRY,
    )

    assert len({id(branch.state) for branch in branches.arms.values()}) == 4
    candidates: dict[Arm, CandidateVariant] = {
        "correct": triplet.correct_twin,
        "irrelevant": triplet.irrelevant_control,
        "contam": triplet.false_candidate,
    }
    for arm, candidate in candidates.items():
        branch = branches.arms[arm]
        state = cast(LiveState, branch.state)
        match state:
            case FullHistoryStateV3(records=records):
                consumed_content = records[-1].content
            case RagFrozenStateV3(corpus=corpus):
                assert corpus is not None
                consumed_content = next(
                    document.text
                    for document in corpus.active_documents
                    if document.document_id == candidate.candidate_id
                )
            case BoTStateV3(entries=entries):
                root = entries[-1]
                assert isinstance(root, NativeEntry)
                consumed_content = root.content
            case ReflexionStateV3(reflections=reflections):
                root = reflections[-1]
                assert isinstance(root, NativeEntry)
                consumed_content = root.content
            case dc.DcRsStateV3(archive=archive):
                consumed_content = dc._archive_native(archive[-1]).content
            case unreachable:
                assert_never(unreachable)
        checkpoint_root = branch.checkpoint.state.entries[-1]
        assert isinstance(checkpoint_root, NativeEntry)
        assert consumed_content == checkpoint_root.content
        assert checkpoint_root.content_hash == canonical_content_hash(consumed_content)
        assert branch.injected_root_id == candidate.candidate_id
        branch_context = replace(context, branch=arm, initial_states={baseline: branch.state})
        restored = runtime.restore_state(runtime.serialize_state(branch.state), branch_context)
        assert runtime.serialize_state(restored) == runtime.serialize_state(branch.state)
