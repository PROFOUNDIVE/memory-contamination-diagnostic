from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from memcontam.baselines.bot_phase12 import BoTStateV3
from memcontam.baselines.dynamic_cheatsheet_phase12 import DcRsStateV3
from memcontam.clients.base import LLMResponse
from memcontam.contamination.phase12.registry import load_current_candidate_registry
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.live_branch import Arm, build_live_reduced_main_branches
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.logging.schema_v3 import ContextEvent, RetrievalEvent
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, serialize_checkpoint
from memcontam.readiness.phase13_main_execution_models import MainExecutionFreeze
from memcontam.readiness.phase13_main_live_runtime import ProductionMainRuntime
from memcontam.readiness.phase13_main_live_runtime_support import core_task_name
from memcontam.readiness.phase13_main_new_mcq_runtime import build_new_mcq_live_branches

from .test_phase13_native_rendering import _clean_state, _Embedder
from .test_phase13_readiness0_production_dry_run import _ContractFakeEmbeddingProvider

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = MainExecutionFreeze.model_validate_json(
    (ROOT / "data/phase13/main/mr_p5/execution_package_v1.json").read_bytes()
)
ARMS = tuple(dict.fromkeys(
    arm for sequence in PACKAGE.arm_order.sequences for arm in sequence.arms
))
ROUTES = tuple(
    (task, baseline, arm)
    for task, baseline in PACKAGE.active_cells.included_task_baseline_pairs
    for arm in ARMS
) + tuple((task, "nomem", "clean") for task in PACKAGE.dispatch.task_order)


class NativeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        del model
        stage = config["method_stage"]
        self.calls.append((stage, messages))
        content = {
            "bot_problem_distill": '{"key_information":"task","restrictions":"follow rules","distilled_task":"solve"}',
            "bot_thought_distill": '{"description":"task","template":"solve","category":"procedure-based","explicitly_used_memory_ids":[]}',
            "reflexion_reflect": '{"mode":"corrective","failure_class":"incorrect_answer","reflection_text":"Check answer","explicitly_used_memory_ids":[]}',
            "dc_rs_synthesize": "<cheatsheet>Check answer</cheatsheet>",
        }.get(stage, "final: 0")
        if stage == "bot_instantiate_solve":
            content = json.dumps({"selected_structure": "retrieved-template" if config.get("source_spans")
                                  else "procedure-based", "solution_trace": "solve", "final_answer": "final: 0"})
        return LLMResponse(content, {"replay": True, "attempts": 1},
                           {"prompt_tokens": 1, "completion_tokens": 1}, 0)


@pytest.mark.parametrize(("task_name", "baseline", "arm"), ROUTES)
def test_current_package_route_executes_native_consumer(
    task_name: str, baseline: str, arm: Arm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memcontam.readiness import phase13_main_live_runtime

    monkeypatch.setattr(phase13_main_live_runtime, "OpenAIResponsesClient", lambda *_args, **_kwargs: pytest.fail(
        "provider construction forbidden in native-consumption matrix"))
    client = NativeTransport()
    runtime = ProductionMainRuntime(ROOT, tmp_path / "cache", client=client)
    task = runtime._tasks(task_name, 0, prefix=False)[0]
    embedder = _ContractFakeEmbeddingProvider(vector_dimension=1024) if baseline == "dc_rs" else _Embedder()
    clean_state = (
        DcRsStateV3(archive=[], allow_unparented_strategies=True)
        if baseline == "dc_rs" else BoTStateV3(entries=[])
        if baseline == "bot_style" else _clean_state(baseline, _Embedder())
        if baseline != "nomem" else None
    )
    def native_verifier(_answer: str, _task) -> bool:
        return baseline != "reflexion_style" or sum(
            stage == "reflexion_generate" for stage, _ in client.calls
        ) > 1

    context = Game24RuntimeContext(
        task=task, client=client, model="replay", verifier=native_verifier,
        decoding={"temperature": 0.0}, branch="clean",
        identities=RuntimeIdentities("h3-native", f"h3-native:trial:1:{arm}:{task.sample_id}", 1,
                                    f"{baseline}:{arm}"),
        embedding_provider=embedder,
        baseline_configs={"fh_bounded": {"context_window_tokens": 10000},
                          "dc_rs": {"embedding_mode": "test_double", "serialized_cheatsheet_budget_tokens": 8192}},
        initial_states={} if clean_state is None else {baseline: clean_state},
    )
    entry = PHASE13_CORE_BASELINE_REGISTRY[baseline]
    if baseline == "nomem":
        state = entry.initial_state(context)
        root = None
    else:
        snapshot = entry.serialize_state(clean_state)
        assert isinstance(snapshot, NativeState)
        prefix = serialize_checkpoint(snapshot)
        if task_name in runtime._new_mcq_registry.tasks:
            branches = build_new_mcq_live_branches(
                prefix=prefix, context=context, task=core_task_name(task_name),
                registry=runtime._new_mcq_registry, runtime_registry=PHASE13_CORE_BASELINE_REGISTRY,
            )
        else:
            branches = build_live_reduced_main_branches(
                prefix=prefix, context=context,
                candidate_registry=load_current_candidate_registry(
                    ROOT / "data/phase12/registries/candidate_registry_v2.json"),
                registry=PHASE13_CORE_BASELINE_REGISTRY, renderers=runtime._renderers,
            )
        assert len({branches.arms[name].injected_root_id for name in ("correct", "irrelevant", "contam")}) == 3
        branch = branches.arms[arm]
        root = next((item for item in branch.checkpoint.state.entries
                     if isinstance(item, NativeEntry) and item.entry_id == branch.injected_root_id), None)
        context = replace(context, branch=arm, initial_states={baseline: branch.state},
                          expected_intervention=root)
        state = entry.restore_state(entry.serialize_state(branch.state), context)

    result = entry.execute_trial(context, state)

    assert result.outcome.method_calls
    assert result.outcome.status == "succeeded", (result.outcome.failure_disposition, result.outcome.metadata)
    assert tuple(stage for stage, _ in client.calls) == tuple(
        call.stage for call in result.outcome.method_calls
    )
    assert task.sample_id == runtime._checkpoint_registry.tasks[task_name].seeds[0].suffix_sample_ids[0]
    if baseline in {"rag_frozen", "bot_style"}:
        assert isinstance(result.retrieval_event, RetrievalEvent)
        assert isinstance(result.context_event, ContextEvent)
    if baseline in {"fh_bounded", "reflexion_style", "dc_rs"}:
        assert result.write_envelopes
    if root is not None:
        payload = json.loads(root.content) if baseline == "dc_rs" else None
        needle = payload["raw_output"] if payload is not None else root.content
        assert any(needle in message["content"] for _, messages in client.calls for message in messages), (
            task_name, baseline, arm, client.calls
        )
        if baseline in {"rag_frozen", "bot_style"}:
            assert isinstance(result.retrieval_event, RetrievalEvent)
            assert isinstance(result.context_event, ContextEvent)
            assert root.entry_id in result.retrieval_event.retrieved_entry_ids
            assert root.entry_id in result.context_event.final_entry_ids
        restored_snapshot = entry.serialize_state(state)
        assert isinstance(restored_snapshot, NativeState)
        assert root.entry_id in {item.entry_id for item in restored_snapshot.entries
                                 if isinstance(item, NativeEntry)}


def test_matrix_domain_matches_active_package_without_full_shadow() -> None:
    assert len(PACKAGE.active_cells.included_task_baseline_pairs) == 23
    assert len(ROUTES) == len(set(ROUTES)) == 97
    assert set(ARMS) == {"clean", "correct", "irrelevant", "contam"}
    assert all((task, "rag_frozen") not in PACKAGE.active_cells.included_task_baseline_pairs
               for task in ("mmlu_pro_engineering", "mmlu_pro_physics"))
    assert {(task, baseline, arm) for task, baseline, arm in ROUTES if baseline == "nomem"} == {
        (task, "nomem", "clean") for task in PACKAGE.dispatch.task_order
    }
