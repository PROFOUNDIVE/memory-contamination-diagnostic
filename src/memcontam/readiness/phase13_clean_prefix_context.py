from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, Literal

from memcontam.baselines.bot_phase12 import BoTStateV3
from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.baselines.reflexion_phase12 import ReflexionStateV3
from memcontam.baselines.retrieval_rag_phase12 import RagFrozenStateV3
from memcontam.clients.base import LLMClient
from memcontam.experiment.phase12.contracts import BaselineConditionSpec, MemoryArmExecutionKey
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.memory.checkpoint_v3 import NativeEntry
from memcontam.memory.stores import MemoryEntry
from memcontam.rag.branch_index import EmbeddingProvider, build_branch_indices
from memcontam.rag.phase12_corpus import CleanCorpus, build_branch_corpora
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.game24 import build_instance as build_game24
from memcontam.tasks.math_equation_balancer import build_instance as build_meb
from memcontam.tasks.word_sorting import build_instance as build_word_sorting
from memcontam.verifiers.game24 import verify_expression
from memcontam.verifiers.math_equation_balancer import verify_rhs_completion_answer
from memcontam.verifiers.word_sorting import verify_words

from memcontam.readiness.phase13_clean_prefix import ROOT, TASKS


Builder = Callable[[dict[str, Any]], TaskInstance]
Verifier = Callable[[str, TaskInstance], bool]


def load_instances(config: dict[str, Any]) -> dict[str, dict[str, TaskInstance]]:
    builders: dict[str, Builder] = {
        "game24": build_game24,
        "math_equation_balancer": build_meb,
        "word_sorting": build_word_sorting,
    }
    result: dict[str, dict[str, TaskInstance]] = {}
    for task in TASKS:
        path = ROOT / config["task_registries"][task]["path"]
        rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        result[task] = {row["sample_id"]: builders[task](row) for row in rows}
    return result


def load_corpus_rows(config: dict[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    path = ROOT / config["clean_context"]["corpus_path"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {
        task: [{"id": row["entry_id"], "text": row["content"]} for row in rows if row["task"] == task]
        for task in TASKS
    }


def build_contexts(
    config: dict[str, Any],
    run_id: str,
    task: str,
    seed: int,
    instances: dict[str, TaskInstance],
    documents: list[Mapping[str, Any]],
    client: LLMClient,
    embedder: EmbeddingProvider,
) -> tuple[Game24RuntimeContext, ...]:
    schedule = config["trajectory_seeds"][task][seed]
    corpus = CleanCorpus.from_documents(documents, corpus_id=f"phase13-clean-prefix-{task}-v1")
    branches = build_branch_corpora(
        corpus,
        {
            "false": {"id": f"unused-{task}-false", "text": "unused false branch"},
            "correct": {"id": f"unused-{task}-correct", "text": "unused correct branch"},
            "irrelevant": {"id": f"unused-{task}-irrelevant", "text": "unused irrelevant branch"},
        },
    )
    indices = build_branch_indices(branches, embedder, filter_policy=None)
    bot_entries: list[MemoryEntry | NativeEntry] = [
        MemoryEntry(
            entry_id=f"bot-{document['id']}",
            content=document["text"],
            memory_type="thought_template",
            metadata={"description": document["text"], "category": "procedure-based"},
        )
        for document in documents[:2]
    ]
    initial_states = {
        "fh_bounded": FullHistoryStateV3(records=[]),
        "rag_frozen": RagFrozenStateV3(
            "clean", branches.branches["clean"], indices.branches["clean"]
        ),
        "bot_style": BoTStateV3(
            entries=bot_entries,
            clean_competitor_ids=tuple(entry.entry_id for entry in bot_entries),
            active_capacity=config["baseline_config"]["bot_style"]["active_capacity"],
        ),
        "reflexion_style": ReflexionStateV3(
            reflections=[],
            active_capacity=config["baseline_config"]["reflexion_style"]["active_capacity"],
        ),
    }
    verifier = _verifier(task)
    return tuple(
        Game24RuntimeContext(
            task=instances[sample_id],
            client=client,
            model=config["provider"]["model_id"],
            verifier=verifier,
            decoding=config["decoding"],
            branch="clean",
            identities=RuntimeIdentities(
                run_id,
                f"{run_id}:{task}:seed:{seed}:trial:{index}",
                index,
            ),
            embedding_provider=embedder,
            baseline_configs=config["baseline_config"],
            initial_states=initial_states,
        )
        for index, sample_id in enumerate(schedule, start=1)
    )


def conditions() -> dict[str, BaselineConditionSpec]:
    families: dict[str, Literal["full_history", "rag", "bot", "reflexion"]] = {
        "fh_bounded": "full_history",
        "rag_frozen": "rag",
        "bot_style": "bot",
        "reflexion_style": "reflexion",
    }
    return {
        baseline: BaselineConditionSpec(
            condition_id=f"{baseline}-clean",
            baseline_family=family,
            fidelity_label="bounded",
            rag_mode="frozen" if family == "rag" else "not_applicable",
            fh_mode="bounded",
            execution_key_example=MemoryArmExecutionKey(kind="memory_arm", arm="clean"),
        )
        for baseline, family in families.items()
    }


def _verifier(task: str) -> Verifier:
    if task == "game24":
        return lambda answer, seen: verify_expression(
            answer, seen.input["numbers"], seen.verifier_spec["target"]
        ).is_correct
    if task == "math_equation_balancer":
        return lambda answer, seen: verify_rhs_completion_answer(
            answer, seen.verifier_spec
        ).is_correct
    return lambda answer, seen: verify_words(
        answer.split(), seen.verifier_spec["sorted_words"]
    ).is_correct
