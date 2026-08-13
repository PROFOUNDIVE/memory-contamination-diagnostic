from __future__ import annotations

from dataclasses import replace
import argparse
from pathlib import Path

import pytest

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.clients.base import LLMResponse
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.live_branch import (
    LiveArmBranch,
    LiveBranchEvent,
    LiveThreeArmBranches,
)
from memcontam.experiment.phase12.runtime_registry import RuntimeEntry, RuntimeTrialResult
from memcontam.memory.checkpoint_v3 import (
    NativeEntry,
    NativeState,
    append_native_entry,
    serialize_checkpoint,
)
from memcontam.contamination.phase12.controls import construct_correct_control, construct_irrelevant_control
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.contamination.phase12.renderers import RendererRegistry
from memcontam.readiness.phase13_calibration_v2 import load_calibration_v2_config
from memcontam.readiness.phase13_calibration_v2_runtime import (
    AuthorizedTrajectoryExecution,
    CalibrationV2RuntimeError,
    InvalidatedTrajectory,
    TrajectoryRequest,
    VerifiedRuntimeAuthorization,
    execute_calibration_trajectory,
    verify_runtime_context,
)
from memcontam.readiness.phase13_provider_models import ExecutionTemplateIdentity
from memcontam.readiness.phase13_provider_runtime import Phase13V2ProviderRuntime
from memcontam.tasks.base import TaskInstance
from memcontam.readiness import phase13_cli

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")
ARMS = ("clean", "correct", "irrelevant", "contam")


class _Provider:
    def __init__(self) -> None:
        self.configs: list[dict[str, object]] = []

    def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
        del messages, model
        self.configs.append(dict(config))
        return LLMResponse("final: 24", {"attempts": 1}, {"prompt_tokens": 1, "completion_tokens": 1}, 1)


def _runtime(
    provider: _Provider,
    baseline: str,
    arm: str,
) -> Phase13V2ProviderRuntime:
    names = {"clean": "Clean", "correct": "Correct", "irrelevant": "Irrelevant", "contam": "Contam"}
    return Phase13V2ProviderRuntime.from_provider(
        provider,
        ROOT,
        ExecutionTemplateIdentity.model_validate(
            {"task": "game24", "baseline": baseline, "arm_key": names[arm]}
        ),
    )


def _fixture(
    *,
    leak: bool = False,
    rewind: bool = False,
    rag_write: bool = False,
    replace_state: bool = False,
):
    provider = _Provider()
    clients = {(baseline, arm): _runtime(provider, baseline, arm) for baseline in BASELINES for arm in ARMS}
    clients[("nomem", "clean")] = Phase13V2ProviderRuntime.from_provider(
        provider,
        ROOT,
        ExecutionTemplateIdentity(task="game24", baseline="nomem", arm_key="star_NoMem"),
    )
    checkpoints = {
        "fh_bounded": serialize_checkpoint(NativeState("fh_bounded", (), {"checkpoint_index": 1, "records": [{"id": "trial-1"}], "first_eviction_trial_id": None})),
        "rag_frozen": serialize_checkpoint(NativeState("rag_frozen", (), {"branch": "clean", "checkpoint_index": 1, "corpus_id": "corpus-v1", "index_id": "index-v1", "read_only": True})),
        "bot_style": serialize_checkpoint(NativeState("bot_style", (), {"templates": [], "checkpoint_index": 1, "clean_competitor_ids": [], "active_capacity": 8})),
        "reflexion_style": serialize_checkpoint(NativeState("reflexion_style", (), {"checkpoint_index": 1, "reflections": [], "active_capacity": 8})),
    }
    candidate_registry = load_candidate_registry(
        ROOT / "data/phase12/registries/candidate_registry_v1.json"
    )
    triplet = next(row for row in candidate_registry.triplets if row.task == "game24")
    renderers = RendererRegistry.native()
    branch_checkpoints = {
        (baseline, arm): (
            checkpoints[baseline]
            if arm == "clean"
            else append_native_entry(
                checkpoints[baseline],
                construct_correct_control(baseline, triplet, checkpoints[baseline])
                if arm == "correct"
                else construct_irrelevant_control(baseline, triplet, checkpoints[baseline])
                if arm == "irrelevant"
                else renderers.render_false(baseline, triplet, checkpoints[baseline]),
            )
        )
        for baseline in BASELINES
        for arm in ARMS
    }
    roots: dict[tuple[str, str], NativeEntry] = {}
    for key, checkpoint in branch_checkpoints.items():
        if key[1] == "clean":
            continue
        root = checkpoint.state.entries[-1]
        assert isinstance(root, NativeEntry)
        roots[key] = root
    branches = {
        baseline: LiveThreeArmBranches(
            baseline,
            "gpt-4o-2024-11-20",
            {"temperature": 0.0, **({"future_horizon": 10} if leak else {})},
            {
                arm: LiveArmBranch(
                    arm,
                    checkpoints[baseline].identity.checkpoint_id,
                    branch_checkpoints[(baseline, arm)].identity.checkpoint_id,
                    branch_checkpoints[(baseline, arm)],
                    {"step": 0}, 0 if arm == "clean" else 1,
                    None
                    if arm == "clean"
                    else roots[(baseline, arm)].entry_id,
                )
                for arm in ARMS
            },
            tuple(
                LiveBranchEvent(
                    "branch_constructed",
                    arm,
                    checkpoints[baseline].identity.checkpoint_id,
                    branch_checkpoints[(baseline, arm)].identity.checkpoint_id,
                    None
                    if arm == "clean"
                    else roots[(baseline, arm)].entry_id,
                    None,
                    None,
                )
                for arm in ARMS
            )
            + tuple(
                LiveBranchEvent(
                    "intervention_applied",
                    arm,
                    checkpoints[baseline].identity.checkpoint_id,
                    branch_checkpoints[(baseline, arm)].identity.checkpoint_id,
                    roots[(baseline, arm)].entry_id,
                    triplet.triplet_id,
                    roots[(baseline, arm)].render_id,
                )
                for arm in ARMS
                if arm != "clean"
            ),
        )
        for baseline in BASELINES
    }
    tasks = tuple(
        TaskInstance(
            sample_id=f"phase13_calibration_v2_game24_{index:04d}",
            task_name="game24",
            input={"numbers": [1, 3, 4, 6]},
            verifier_spec={"target": 24},
        )
        for index in range(2, 12)
    )
    contexts = tuple(
        Game24RuntimeContext(
            task,
            clients[("fh_bounded", "clean")],
            "gpt-4o-2024-11-20",
            lambda _answer, _task: True,
            {"temperature": 0.0, **({"future_horizon": 10} if leak else {})},
            "clean",
            RuntimeIdentities("run-1", f"trial-{index}", index),
        )
        for index, task in enumerate(tasks, start=2)
    )

    def execute(context, state):  # noqa: ANN001, ANN202
        calls = 2 if context.identities.condition_id in {"bot_style", "reflexion_style"} else 1
        for _ in range(calls):
            context.client.chat(
                [{"role": "user", "content": "solve"}],
                context.model,
                {**context.decoding, **context.baseline_configs.get(context.identities.condition_id, {})},
            )
        prior = state["step"]
        state["step"] = prior - 1 if rewind and prior == 4 else prior + 1
        writes = ("forbidden",) if rag_write and context.identities.condition_id == "rag_frozen" else ()
        return RuntimeTrialResult(
            outcome=BaselineExecutionOutcome("succeeded"),
            state={"step": state["step"]} if replace_state else state,
            write_envelopes=writes,
        )

    def serialize(state):  # noqa: ANN001, ANN202
        return NativeState("fh_bounded", (), {"step": state["step"]})

    registry = {
        baseline: RuntimeEntry(lambda _context: {"step": 0}, execute, serialize, lambda value, _context: value, lambda _state, _context: None)
        for baseline in BASELINES
    }
    singleton = {"nomem": True}

    def nomem_execute(context, state):  # noqa: ANN001, ANN202
        context.client.chat([{"role": "user", "content": "solve"}], context.model, dict(context.decoding))
        return RuntimeTrialResult(BaselineExecutionOutcome("succeeded"), state)

    registry["nomem"] = RuntimeEntry(lambda _context: singleton, nomem_execute, lambda value: value, lambda value, _context: value, lambda _state, _context: None)
    verified = verify_runtime_context(
        ROOT,
        load_calibration_v2_config(ROOT / "configs/phase13/pre_main_calibration_v2.yaml"),
        authorization=VerifiedRuntimeAuthorization(
            "verified-test-authorization",
            "phase13-pre-main-calibration-v2",
            "acb769e1e1adbc3eb69e4302322c8eac81829dc836611519caea2ba960900c38",
            "82960a8f65d316c53bcf55da3e215f0c4b62781643c21155307b40aa9adf4eee",
            "phase13-h10-execution-owner-v1",
        ),
    )
    return provider, registry, TrajectoryRequest(
        verified=verified,
        stream_id="game24-seed-10000",
        task="game24",
        seed_id=10000,
        source_ordered_stream_sha256="9f74ad462d286796e671544745f55cae323eb48aed22e151638ed99345230bb8",
        session_id="session-10000",
        branches_by_baseline=branches,
        contexts=contexts,
        providers=clients,
        registry=registry,
    )


def test_executes_one_causal_h10_source_with_owned_calls_and_nomem_singleton() -> None:
    provider, _, request = _fixture()

    result = execute_calibration_trajectory(request)

    assert result.status == "completed"
    assert len(result.events) == 4 * 4 * 10
    assert {event.event_time for event in result.events} == set(range(10))
    assert all(
        left.state_after_sha256 == right.state_before_sha256
        for left, right in zip(result.events, result.events[1:])
        if (left.baseline, left.arm) == (right.baseline, right.arm)
    )
    assert result.nomem_underlying_execution_count == 1
    assert len(provider.configs) == 250
    assert {event.source_checkpoint_id for event in result.events if event.baseline == "fh_bounded"} == {
        "checkpoint-3de74961a1870cb9"
    }
    assert tuple(event.suffix_id for event in result.events[:10]) == tuple(
        f"phase13_calibration_v2_game24_{index:04d}" for index in range(2, 12)
    )
    assert not any(set(config) & {"horizon", "future_horizon", "analysis_window", "task"} for config in provider.configs)


@pytest.mark.parametrize(
    ("mutation", "code"),
    (({"leak": True}, "PROVIDER_CONFIG_LEAKAGE"), ({"rewind": True}, "STATE_REWIND"), ({"rag_write": True}, "RAG_WRITE_FORBIDDEN")),
)
def test_violation_seals_invalidated_partial_trajectory(mutation: dict[str, bool], code: str) -> None:
    _, _, request = _fixture(**mutation)

    result = execute_calibration_trajectory(request)

    assert isinstance(result, InvalidatedTrajectory)
    assert result.status == "invalidated"
    assert result.failure_code == code
    assert result.sealed


def test_rejects_filter_branch_before_dispatch() -> None:
    _, _, request = _fixture()
    bad_branches = dict(request.branches_by_baseline)
    branch = bad_branches["fh_bounded"]
    arms = dict(branch.arms)
    arms["filter"] = replace(branch.arms["contam"], arm="filter")
    object.__setattr__(branch, "arms", arms)

    with pytest.raises(CalibrationV2RuntimeError, match="FILTER_BRANCH_FORBIDDEN"):
        execute_calibration_trajectory(replace(request, branches_by_baseline=bad_branches))


def test_rejects_raw_provider_and_future_suffix_task_before_dispatch() -> None:
    provider, _, request = _fixture()
    raw: dict[tuple[str, str], object] = dict(request.providers)
    raw[("fh_bounded", "clean")] = provider
    wrong_task = request.contexts[0].task.model_copy(update={"sample_id": "future-sample"})

    with pytest.raises(CalibrationV2RuntimeError, match="OWNED_PROVIDER_REQUIRED"):
        execute_calibration_trajectory(replace(request, providers=raw))
    with pytest.raises(CalibrationV2RuntimeError, match="SUFFIX_TASK_DRIFT"):
        execute_calibration_trajectory(
            replace(request, contexts=(replace(request.contexts[0], task=wrong_task), *request.contexts[1:]))
        )
    assert provider.configs == []


def test_rejects_future_field_added_by_baseline_config_before_provider_dispatch() -> None:
    provider, _, request = _fixture()
    contexts = tuple(
        replace(context, baseline_configs={"fh_bounded": {"future_horizon": 10}})
        for context in request.contexts
    )

    result = execute_calibration_trajectory(replace(request, contexts=contexts))

    assert isinstance(result, InvalidatedTrajectory)
    assert result.failure_code == "PROVIDER_CONFIG_LEAKAGE"
    assert not any("future_horizon" in config for config in provider.configs)


def test_owned_runtime_exposes_exact_dispatched_payloads() -> None:
    _, _, request = _fixture()

    result = execute_calibration_trajectory(request)

    assert result.status == "completed"
    runtime = request.providers[("fh_bounded", "clean")]
    assert len(runtime.dispatched_payloads) == 10
    assert all(payload.config["execution_template_id"] == "game24-fh_bounded-clean" for payload in runtime.dispatched_payloads)
    assert {payload.session_id for payload in runtime.dispatched_payloads} == {"session-10000"}
    assert all("seed" not in payload.config and "task" not in payload.config for payload in runtime.dispatched_payloads)


def test_rejects_swapped_branch_checkpoint_source_and_root() -> None:
    provider, _, request = _fixture()
    branches = dict(request.branches_by_baseline)
    panel = branches["fh_bounded"]
    arms = dict(panel.arms)
    arms["correct"] = replace(
        arms["correct"],
        checkpoint=arms["contam"].checkpoint,
        source_identity="swapped-source",
        injected_root_id="swapped-root",
    )
    object.__setattr__(panel, "arms", arms)

    with pytest.raises(CalibrationV2RuntimeError, match="BRANCH_LINEAGE_MISMATCH"):
        execute_calibration_trajectory(replace(request, branches_by_baseline=branches))
    assert provider.configs == []


def test_rejects_swapped_intervention_identity() -> None:
    provider, _, request = _fixture()
    panel = request.branches_by_baseline["fh_bounded"]
    events = list(panel.events)
    event = next(index for index, row in enumerate(events) if row.arm == "correct" and row.kind == "intervention_applied")
    events[event] = replace(events[event], candidate_triplet_id="swapped-triplet")
    object.__setattr__(panel, "events", tuple(events))

    with pytest.raises(CalibrationV2RuntimeError, match="INTERVENTION_LINEAGE_MISMATCH"):
        execute_calibration_trajectory(request)
    assert provider.configs == []


def test_rejects_unrelated_state_replacement_and_seals_accounting_closure() -> None:
    _, _, request = _fixture(replace_state=True)

    result = execute_calibration_trajectory(request)

    assert isinstance(result, InvalidatedTrajectory)
    assert result.failure_code == "STATE_IDENTITY_DRIFT"
    assert result.accounting_closure.status == "closed_partial"
    assert result.accounting_closure.settled_semantic_calls == 1
    assert result.accounting_closure.expected_semantic_calls == 250
    assert result.accounting_closure.templates[0].calls[0].attempts[0].raw_evidence


def test_reconciles_exact_template_call_counts_and_sessions() -> None:
    _, _, request = _fixture()

    result = execute_calibration_trajectory(request)

    assert result.status == "completed"
    assert result.accounting_closure.status == "closed_complete"
    assert result.accounting_closure.settled_semantic_calls == 250
    assert result.accounting_closure.expected_semantic_calls == 250
    assert {row.session_id for row in result.accounting_closure.templates} == {"session-10000"}


def test_rejects_nominal_template_call_shortfall() -> None:
    _, registry, request = _fixture()
    original = registry["bot_style"]

    def one_call(context, state):  # noqa: ANN001, ANN202
        context.client.chat([{"role": "user", "content": "solve"}], context.model, dict(context.decoding))
        state["step"] += 1
        return RuntimeTrialResult(BaselineExecutionOutcome("succeeded"), state)

    registry["bot_style"] = replace(original, execute_trial=one_call)

    result = execute_calibration_trajectory(replace(request, registry=registry))

    assert isinstance(result, InvalidatedTrajectory)
    assert result.failure_code == "PROVIDER_CALL_LEDGER_MISMATCH"
    assert result.accounting_closure.status == "closed_partial"


def test_typed_authorized_cli_branch_reaches_trajectory_executor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, _, request = _fixture()

    phase13_cli.run(
        argparse.Namespace(
            phase13_command="run-calibration-v2",
            config=ROOT / "configs/phase13/pre_main_calibration_v2.yaml",
            authorized_execution=AuthorizedTrajectoryExecution(
                request.verified.authorization, request
            ),
        )
    )

    assert '"status": "completed"' in capsys.readouterr().out
