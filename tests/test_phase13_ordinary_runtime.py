from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memcontam.baselines.bot_phase12 import BoTStateV3
from memcontam.baselines.dynamic_cheatsheet_phase12 import DcRsStateV3
from memcontam.clients.base import LLMResponse
from memcontam.experiment.phase13_ordinary_runtime import (
    ORDINARY_BASELINES,
    ORDINARY_TASKS,
    OrdinaryBaseline,
    OrdinaryTask,
    ProspectiveOrdinaryError,
    ProspectiveOrdinaryRun,
    execute_prospective_ordinary,
)
from memcontam.readiness import phase13_core_datasets as core_datasets
from memcontam.readiness.phase13_core_bundle import (
    CoreSources,
    CoreTask,
    SelectionProvenance,
    SourceArtifact,
    write_bundle,
)
from memcontam.readiness.phase13_core_datasets import (
    GPQA_REVISION,
    GPQA_SOURCE_SHA256,
    MMLU_PRO_REVISION,
    MMLU_PRO_TEST_SHA256,
    SELECTION_PATH,
)
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance


class _Client:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        del messages, model
        self.calls += 1
        stage = config.get("method_stage")
        if stage == "dc_rs_synthesize":
            content = "<cheatsheet>ordinary strategy</cheatsheet>"
        elif stage == "bot_problem_distill":
            content = json.dumps(
                {
                    "key_information": "multiple-choice input",
                    "restrictions": "choose one option",
                    "distilled_task": "select the answer",
                }
            )
        elif stage == "bot_instantiate_solve":
            content = json.dumps(
                {
                    "selected_structure": "retrieved-template",
                    "solution_trace": "apply the retrieved procedure",
                    "final_answer": "final: B",
                }
            )
        elif stage == "bot_thought_distill":
            content = json.dumps(
                {
                    "description": "multiple-choice procedure",
                    "template": "compare each option and select the supported answer",
                    "category": "procedure-based",
                    "explicitly_used_memory_ids": [],
                }
            )
        else:
            content = "final: B"
        return LLMResponse(content=content, raw={"replay": True}, token_usage={}, latency_ms=0)


class _EmbeddingProvider:
    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model_id": "BAAI/bge-m3",
            "revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "embedding_library_version": "test",
            "vector_dimension": 1024,
            "normalize_embeddings": True,
        }

    def encode_document(self, _text: str) -> list[float]:
        return [1.0, 0.0]

    def encode_query(self, _text: str) -> list[float]:
        return [1.0, 0.0]


def _core_row(task: str, sample_id: str, index: int) -> TaskInstance:
    return TaskInstance(
        sample_id=sample_id,
        task_name=task,
        input={"question": f"question {index}", "options": ["A", "B", "C", "D"]},
        verifier_spec={"answer_index": 1, "answer_label": "B"},
        metadata={"upstream_question_id": index},
    )


def _archive_source_trial_id(state: DcRsStateV3) -> str:
    entry = state.archive[0]
    assert isinstance(entry, MemoryEntry)
    assert entry.source_trial_id is not None
    return entry.source_trial_id


def _bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "core"
    sources = CoreSources(
        mmlu_pro=SourceArtifact(
            repo="TIGER-Lab/MMLU-Pro",
            revision=MMLU_PRO_REVISION,
            path="data/test-00000-of-00001.parquet",
            sha256=MMLU_PRO_TEST_SHA256,
        ),
        gpqa=SourceArtifact(
            repo="Idavidrein/gpqa",
            revision=GPQA_REVISION,
            path="gpqa_diamond.csv",
            sha256=GPQA_SOURCE_SHA256,
        ),
    )
    selection = SelectionProvenance(
        resource="memcontam.readiness/data/mmlu_pro_dc_selection_v1.json",
        sha256=hashlib.sha256(SELECTION_PATH.read_bytes()).hexdigest(),
        upstream_revision=MMLU_PRO_REVISION,
        dynamic_cheatsheet_revision="5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9",
    )
    write_bundle(
        root,
        {
            "mmlu_pro_engineering": tuple(
                _core_row("mmlu_pro_engineering", f"engineering:{index}", index)
                for index in range(250)
            ),
            "mmlu_pro_physics": tuple(
                _core_row("mmlu_pro_physics", f"physics:{index}", index)
                for index in range(250)
            ),
            "gpqa_diamond": tuple(
                _core_row("gpqa_diamond", f"gpqa:{index}", index)
                for index in range(198)
            ),
        },
        sources=sources,
        selection=selection,
    )
    manifest = json.loads((root / "manifest.json").read_text())
    monkeypatch.setattr(
        core_datasets,
        "CANONICAL_CORE_ARTIFACT_SHA256",
        {task: artifact["sha256"] for task, artifact in manifest["artifacts"].items()},
    )
    return root


def test_ordinary_surface_declares_all_six_tasks_and_five_memory_baselines() -> None:
    assert ORDINARY_TASKS == (
        "game24",
        "math_equation_balancer",
        "word_sorting",
        "mmlu_pro_engineering",
        "mmlu_pro_physics",
        "gpqa_diamond",
    )
    assert ORDINARY_BASELINES == (
        "fh_bounded",
        "rag_frozen",
        "bot_style",
        "reflexion_style",
        "dc_rs",
    )


def test_core_ordinary_execution_consumes_seeded_bundle_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path, monkeypatch)
    client = _Client()
    run = ProspectiveOrdinaryRun(
        task_name="mmlu_pro_engineering",
        baseline="fh_bounded",
        run_id="ordinary-core",
        model="replay",
        client=client,
        verifier=lambda answer, task: answer == task.verifier_spec["answer_label"],
        decoding={"temperature": 0.0},
        core_bundle=bundle,
        trajectory_seed=17,
    )

    first = execute_prospective_ordinary(run)
    second = execute_prospective_ordinary(run)

    assert first.sample_ids == second.sample_ids
    assert set(first.sample_ids) == {f"engineering:{index}" for index in range(250)}
    assert first.sample_ids != tuple(sorted(first.sample_ids))
    assert all(trial.outcome.status == "succeeded" for trial in first.trials)


def test_core_rag_remains_explicitly_unavailable_without_scientific_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client()
    run = ProspectiveOrdinaryRun(
        task_name="mmlu_pro_physics",
        baseline="rag_frozen",
        run_id="ordinary-rag",
        model="replay",
        client=client,
        verifier=lambda _answer, _task: False,
        decoding={"temperature": 0.0},
        core_bundle=_bundle(tmp_path, monkeypatch),
        trajectory_seed=17,
    )

    with pytest.raises(
        ProspectiveOrdinaryError,
        match="NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN",
    ):
        execute_prospective_ordinary(run)
    assert client.calls == 0


@pytest.mark.parametrize("baseline", ("fh_bounded", "bot_style", "reflexion_style"))
def test_existing_memory_runtimes_execute_core_ordinary_trajectory(
    baseline: OrdinaryBaseline,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_states = (
        {
            "bot_style": BoTStateV3(
                entries=[
                    MemoryEntry(
                        entry_id=f"clean-template-{index}",
                        content=f"clean procedure {index}",
                        memory_type="thought_template",
                        metadata={
                            "description": f"clean procedure {index}",
                            "category": "procedure-based",
                        },
                    )
                    for index in range(2)
                ],
                clean_competitor_ids=("clean-template-0", "clean-template-1"),
            )
        }
        if baseline == "bot_style"
        else {}
    )
    result = execute_prospective_ordinary(
        ProspectiveOrdinaryRun(
            task_name="gpqa_diamond",
            baseline=baseline,
            run_id=f"ordinary-{baseline}",
            model="replay",
            client=_Client(),
            verifier=lambda answer, task: answer == task.verifier_spec["answer_label"],
            decoding={"temperature": 0.0},
            core_bundle=_bundle(tmp_path, monkeypatch),
            trajectory_seed=29,
            embedding_provider=_EmbeddingProvider(),
            initial_states=initial_states,
        )
    )

    assert len(result.trials) == 198
    failures = [
        trial.outcome.failure_disposition
        for trial in result.trials
        if trial.outcome.status != "succeeded"
    ]
    assert failures == []


@pytest.mark.parametrize(
    "task_name",
    ("mmlu_pro_engineering", "mmlu_pro_physics", "gpqa_diamond"),
)
def test_dc_rs_ordinary_execution_accepts_each_core_task_stream(
    task_name: CoreTask,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = execute_prospective_ordinary(
        ProspectiveOrdinaryRun(
            task_name=task_name,
            baseline="dc_rs",
            run_id=f"ordinary-dc-{task_name}",
            model="replay",
            client=_Client(),
            verifier=lambda answer, task: answer == task.verifier_spec["answer_label"],
            decoding={"temperature": 0.0},
            core_bundle=_bundle(tmp_path, monkeypatch),
            trajectory_seed=31,
            embedding_provider=_EmbeddingProvider(),
            baseline_configs={
                "dc_rs": {
                    "embedding_mode": "test_double",
                    "serialized_cheatsheet_budget_bytes": 1024,
                    "tool_mode": "text_only",
                }
            },
            initial_states={"dc_rs": DcRsStateV3(archive=[])},
        )
    )

    assert result.sample_ids
    assert all(trial.outcome.status == "succeeded" for trial in result.trials)


@pytest.mark.parametrize(
    ("task_name", "task"),
    (
        (
            "game24",
            TaskInstance(
                sample_id="game24:1",
                task_name="game24",
                input={"numbers": [1, 3, 4, 6], "target": 24},
                verifier_spec={"target": 24},
            ),
        ),
        (
            "math_equation_balancer",
            TaskInstance(
                sample_id="meb:1",
                task_name="math_equation_balancer",
                input={"input": "1 + 1 = 2"},
                verifier_spec={"target": "1 + 1 = 2"},
            ),
        ),
        (
            "word_sorting",
            TaskInstance(
                sample_id="words:1",
                task_name="word_sorting",
                input={"words": ["a", "b"]},
                verifier_spec={"sorted_words": ["a", "b"]},
            ),
        ),
    ),
)
def test_dc_rs_ordinary_execution_accepts_original_task_native_inputs(
    task_name: OrdinaryTask,
    task: TaskInstance,
) -> None:
    result = execute_prospective_ordinary(
        ProspectiveOrdinaryRun(
            task_name=task_name,
            baseline="dc_rs",
            run_id=f"ordinary-{task.task_name}",
            model="replay",
            client=_Client(),
            verifier=lambda _answer, _task: True,
            decoding={"temperature": 0.0},
            tasks=(task,),
            embedding_provider=_EmbeddingProvider(),
            baseline_configs={
                "dc_rs": {
                    "embedding_mode": "test_double",
                    "serialized_cheatsheet_budget_bytes": 1024,
                    "tool_mode": "text_only",
                }
            },
            initial_states={"dc_rs": DcRsStateV3(archive=[])},
        )
    )

    assert result.trials[0].outcome.status == "succeeded"
    assert isinstance(result.trials[0].state, DcRsStateV3)
    assert _archive_source_trial_id(result.trials[0].state).startswith(
        f"ordinary-{task.task_name}:trial:1:"
    )
