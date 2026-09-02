from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memcontam.baselines.contracts import BaselineExecutionOutcome
from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.clients.base import LLMResponse
from memcontam.clients.replay import ReplayClient
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.runtime_registry import (
    PHASE13_CORE_BASELINE_REGISTRY,
    RuntimeEntry,
    RuntimeTrialResult,
)
from memcontam.memory.checkpoint_v3 import NativeState, deserialize_checkpoint, serialize_checkpoint
from memcontam.readiness.phase13_main_execution_models import MainExecutionFreeze
from memcontam.readiness.phase13_main_live_dispatch import (
    DurableMainDispatch,
    summarize_telemetry,
)
from memcontam.readiness.phase13_main_live_runtime import MainLiveRuntimeError, ProductionMainRuntime
from memcontam.readiness.phase13_main_new_mcq_runtime import (
    build_new_mcq_live_branches,
    load_new_mcq_runtime_registry,
    new_mcq_native_entries,
)
from memcontam.readiness.phase13_new_mcq_rag_models import InterventionRegistry
from memcontam.readiness.phase13_main_production import ProductionObject
from memcontam.readiness.phase13_main_production_backend import MainProductionBackend
from memcontam.readiness.phase13_main_runner import (
    MainRunBinding,
    MainRunLedger,
    enumerate_execution_units,
    run_pending,
)
from memcontam.tasks.base import TaskInstance


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path.home() / ".cache/huggingface/hub"
INTERVENTIONS = InterventionRegistry.model_validate_json(
    (ROOT / "data/phase13/rag/new_mcq/intervention_registry_v1.json").read_bytes()
)


class _ProviderFreeClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        self.calls += 1
        stage = config["method_stage"]
        if stage == "bot_problem_distill":
            content = json.dumps(
                {
                    "key_information": "provider-free preflight",
                    "restrictions": "text only",
                    "distilled_task": "return a final answer",
                }
            )
        elif stage == "bot_instantiate_solve":
            content = json.dumps(
                {
                    "selected_structure": "provider-free",
                    "solution_trace": "deterministic replay",
                    "final_answer": "final: 0",
                }
            )
        elif stage == "bot_thought_distill":
            content = json.dumps(
                {
                    "description": "provider-free procedure",
                    "template": "return the deterministic replay answer",
                    "category": "procedure-based",
                    "explicitly_used_memory_ids": [],
                }
            )
        elif stage == "reflexion_reflect":
            content = json.dumps(
                {
                    "mode": "corrective",
                    "failure_class": "incorrect_answer",
                    "reflection_text": "Use the deterministic replay answer.",
                    "explicitly_used_memory_ids": [],
                }
            )
        elif stage == "dc_rs_synthesize":
            content = "<cheatsheet>provider-free strategy</cheatsheet>"
        else:
            content = "final: 0"
        usage = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        request_contract = {
            "model": model,
            "input_sha256": hashlib.sha256(
                json.dumps(
                    messages,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode()
            ).hexdigest(),
            "temperature": config.get("temperature", 0.0),
            "top_p": config.get("top_p", 1.0),
            "reasoning": {"mode": "standard", "effort": "none", "context": "current_turn"},
            "previous_response_id": None,
            "service_tier": "default",
            "store": False,
            "tools": [],
            "max_output_tokens": config["max_output_tokens"],
        }
        authority_contract = {
            "maximum_input_tokens": config["_phase13_maximum_input_tokens"],
            "maximum_output_tokens": config["max_output_tokens"],
            "execution_envelope_id": config["_phase13_execution_envelope_id"],
            "execution_envelope_sha256": config["_phase13_execution_envelope_sha256"],
            "failure_contract_id": config["_phase13_failure_contract_id"],
            "failure_contract_sha256": config["_phase13_failure_contract_sha256"],
            "terminal_failure_contract_id": config["_phase13_terminal_failure_contract_id"],
            "terminal_failure_contract_sha256": config[
                "_phase13_terminal_failure_contract_sha256"
            ],
            "rate_card_sha256": config["_phase13_rate_card_sha256"],
        }
        return LLMResponse(
            content,
            {
                "response_id": f"provider-free-{self.calls}",
                "model": model,
                "status": "completed",
                "usage": usage,
                "attempts": 1,
                "service_tier": "default",
                "cost_usd": 0.0,
                "authoritative_provider_cost_usd": 0.0,
                "derived_cost_usd": 0.0,
                "cost_source": "AUTHORITATIVE_PROVIDER",
                "request_contract": request_contract,
                "authority_contract": authority_contract,
                "provider_free": True,
            },
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            0,
        )


@pytest.mark.parametrize(
    ("baseline", "kind", "semantic_kind", "native_component", "payload_keys"),
    (
        ("fh_bounded", "raw_interaction", "full_history_transcript", "history", {"kind", "query", "response"}),
        ("bot_style", "thought_template", "thought_template", "buffer", {"kind", "procedural_body", "retrieval_description"}),
        ("reflexion_style", "reflection", "verbal_reflection", "reflections", {"kind", "lesson"}),
        ("dc_rs", "raw_interaction", "dc_rs_io_pair", "archive", {"kind", "query", "response"}),
    ),
)
def test_new_mcq_treatments_use_each_baseline_native_carrier(
    baseline: str,
    kind: str,
    semantic_kind: str,
    native_component: str,
    payload_keys: set[str],
) -> None:
    entries = new_mcq_native_entries("mmlu_pro_engineering", baseline, INTERVENTIONS)

    assert set(entries) == {"correct", "irrelevant", "contam"}
    assert {entry.semantic_kind for entry in entries.values()} == {semantic_kind}
    assert {entry.native_component for entry in entries.values()} == {native_component}
    payloads = [json.loads(entry.content) for entry in entries.values()]
    assert all(payload_keys <= set(payload) and payload["kind"] == kind for payload in payloads)
    assert len({entry.content_hash for entry in entries.values()}) == 3
    assert all(entry.render_id for entry in entries.values())


def test_main_live_runtime_preflight_consumes_real_prefix_state_without_provider() -> None:
    package = MainExecutionFreeze.model_validate_json(
        (ROOT / "data/phase13/main/mr_p5/execution_package_v1.json").read_bytes()
    )
    unit = enumerate_execution_units(package, ROOT)[0]
    client = _ProviderFreeClient()
    runtime = ProductionMainRuntime(ROOT, CACHE_ROOT, client=client)

    runtime.preflight((unit,))

    assert client.calls == 0


def test_main_live_runtime_preflight_reports_local_cache_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = MainExecutionFreeze.model_validate_json(
        (ROOT / "data/phase13/main/mr_p5/execution_package_v1.json").read_bytes()
    )
    unit = enumerate_execution_units(package, ROOT)[0]
    runtime = ProductionMainRuntime(ROOT, CACHE_ROOT, client=ReplayClient())
    monkeypatch.setattr(
        runtime,
        "_embedder",
        lambda: (_ for _ in ()).throw(RuntimeError("pinned cache missing")),
    )

    with pytest.raises(
        MainLiveRuntimeError,
        match="MAIN_PREFIX_PREFLIGHT_INVALID:pinned cache missing",
    ):
        runtime.preflight((unit,))


@pytest.mark.parametrize(
    "baseline",
    ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style", "dc_rs"),
)
def test_production_prefix_binds_baseline_specific_arm_free_condition(
    baseline: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts: list[Game24RuntimeContext] = []
    task = TaskInstance(
        sample_id="game24-main-1",
        task_name="game24",
        input={"numbers": [1, 2, 3, 4]},
        verifier_spec={"target": 24},
    )
    state = NativeState(baseline, (), {})
    entry = RuntimeEntry(
        initial_state=lambda context: contexts.append(context) or state,
        execute_trial=lambda _context, seen: RuntimeTrialResult(
            BaselineExecutionOutcome("succeeded"), seen
        ),
        serialize_state=lambda seen: seen,
        restore_state=lambda seen, _context: seen,
        maturity_view=lambda _state, _context: None,
    )
    unit = ProductionObject(
        sequence=0,
        unit_id="0" * 64,
        kind="CLEAN_PREFIX",
        seed=0,
        task="game24",
        memory_baseline=baseline,
        arm="NOT_APPLICABLE",
        prefix_unit_id=None,
        projected_cost_krw=1,
        execution_template_id="prefix-template",
        ordered_sample_ids_sha256="1" * 64,
        registration_packet_sha256="2" * 64,
        checkpoint_registry_sha256="3" * 64,
    )
    runtime = ProductionMainRuntime.__new__(ProductionMainRuntime)
    runtime._root = ROOT
    monkeypatch.setattr(runtime, "_client", ReplayClient(), raising=False)
    monkeypatch.setattr(runtime, "_tasks", lambda *_args, **_kwargs: (task,))
    monkeypatch.setattr(runtime, "_embedder", lambda: None)
    monkeypatch.setattr(runtime, "_initial_states", lambda _task: {})
    monkeypatch.setattr(runtime, "_configs", lambda: {})
    monkeypatch.setattr(runtime, "_initial_states", lambda _task: {})
    monkeypatch.setitem(PHASE13_CORE_BASELINE_REGISTRY, baseline, entry)

    runtime.execute_prefix(unit)

    assert contexts[0].identities.condition_id == f"{baseline}-clean"


def test_all_active_prefix_paths_complete_with_provider_free_client() -> None:
    package = MainExecutionFreeze.model_validate_json(
        (ROOT / "data/phase13/main/mr_p5/execution_package_v1.json").read_bytes()
    )
    units = tuple(
        unit
        for unit in enumerate_execution_units(package, ROOT)
        if unit.kind == "CLEAN_PREFIX" and unit.seed == 0
    )
    client = _ProviderFreeClient()
    runtime = ProductionMainRuntime(ROOT, CACHE_ROOT, client=client)

    outputs = tuple(runtime.execute_prefix(unit) for unit in units)

    assert len(outputs) == 23
    assert {
        deserialize_checkpoint(output.checkpoint).baseline for output in outputs
    } == {"fh_bounded", "rag_frozen", "bot_style", "reflexion_style", "dc_rs"}
    for unit, output in zip(units, outputs, strict=True):
        if unit.memory_baseline != "reflexion_style":
            continue
        state = deserialize_checkpoint(output.checkpoint)
        expected_stages = (
            ("reflexion_generate", "reflexion_reflect")
            if state.entries
            else ("reflexion_generate",)
        )
        assert tuple(call.stage for call in output.dispatch.provider_calls) == expected_stages
    assert client.calls == sum(len(output.dispatch.provider_calls) for output in outputs)


def test_nonzero_ledger_dispatch_completes_real_production_prefix_without_network(
    tmp_path: Path,
) -> None:
    package = MainExecutionFreeze.model_validate_json(
        (ROOT / "data/phase13/main/mr_p5/execution_package_v1.json").read_bytes()
    )
    units = enumerate_execution_units(package, ROOT)
    ledger = MainRunLedger.create(
        tmp_path / "main-run-v1.sqlite3",
        MainRunBinding(
            package_id="phase13-main-a-execution-freeze-v1",
            package_sha256="1" * 64,
            package_hash="2" * 64,
            authorization_id="phase13-main-a-authorized-execution-v1",
            authorization_sha256="3" * 64,
            authorization_hash="4" * 64,
            runner_sha256="5" * 64,
        ),
        units,
    )
    evidence_root = tmp_path
    client = _ProviderFreeClient()
    runtime = ProductionMainRuntime(ROOT, CACHE_ROOT, client=client)
    backend = MainProductionBackend(
        evidence_root,
        runtime.execute_prefix,
        runtime.execute_ordinary,
        ledger.completed_evidence_sha256,
    )

    report = run_pending(
        ledger,
        DurableMainDispatch(evidence_root, backend),
        tranche_ceiling_krw=80000,
        max_units=1,
    )

    assert report.attempted_count == 1
    assert report.completed_count == 1
    assert ledger.status().pending_count == 1199
    assert summarize_telemetry(evidence_root).provider_call_count == 1


def test_durable_rag_prefix_reconciles_frozen_decoding_without_network(
    tmp_path: Path,
) -> None:
    package = MainExecutionFreeze.model_validate_json(
        (ROOT / "data/phase13/main/mr_p5/execution_package_v1.json").read_bytes()
    )
    unit = next(
        row
        for row in enumerate_execution_units(package, ROOT)
        if row.kind == "CLEAN_PREFIX" and row.memory_baseline == "rag_frozen" and row.seed == 0
    )
    runtime = ProductionMainRuntime(ROOT, CACHE_ROOT, client=_ProviderFreeClient())
    backend = MainProductionBackend(tmp_path, runtime.execute_prefix, runtime.execute_ordinary)

    completed = DurableMainDispatch(tmp_path, backend)(unit)

    assert completed.realized_cost_krw == 0


def test_failed_reflexion_prefix_cannot_create_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = MainExecutionFreeze.model_validate_json(
        (ROOT / "data/phase13/main/mr_p5/execution_package_v1.json").read_bytes()
    )
    unit = next(
        row
        for row in enumerate_execution_units(package, ROOT)
        if row.kind == "CLEAN_PREFIX" and row.memory_baseline == "reflexion_style"
    )
    runtime = ProductionMainRuntime(ROOT, CACHE_ROOT, client=ReplayClient())
    state = NativeState("reflexion_style", (), {})
    entry = RuntimeEntry(
        initial_state=lambda _context: state,
        execute_trial=lambda _context, seen: RuntimeTrialResult(
            BaselineExecutionOutcome(
                "failed",
                error_type="BaselineOutputError",
                failure_disposition="reflexion_invalid_generation",
                scientific_ineligibility_reason="invalid_reflexion_generation",
            ),
            seen,
        ),
        serialize_state=lambda seen: seen,
        restore_state=lambda seen, _context: seen,
        maturity_view=lambda _state, _context: None,
    )
    monkeypatch.setattr(runtime, "_embedder", lambda: None)
    monkeypatch.setattr(runtime, "_initial_states", lambda _task: {})
    monkeypatch.setitem(PHASE13_CORE_BASELINE_REGISTRY, "reflexion_style", entry)

    with pytest.raises(MainLiveRuntimeError, match="MAIN_PREFIX_EXECUTION_FAILED"):
        runtime.execute_prefix(unit)


def test_new_mcq_live_branch_injects_selected_h2_carrier() -> None:
    state = FullHistoryStateV3(records=[])
    serialized = PHASE13_CORE_BASELINE_REGISTRY["fh_bounded"].serialize_state(state)
    assert isinstance(serialized, NativeState)
    context = Game24RuntimeContext(
        task=TaskInstance(
            sample_id="engineering-1",
            task_name="mmlu_pro_engineering",
            input={"question": "Which carrier is active?", "options": ["A", "B", "C", "D"]},
        ),
        client=ReplayClient(responses_by_sample={}),
        model="replay",
        verifier=lambda _answer, _task: True,
        decoding={"temperature": 0.0},
        branch="clean",
        identities=RuntimeIdentities("run-1", "trial-1", 1),
        initial_states={"fh_bounded": state},
    )

    branches = build_new_mcq_live_branches(
        prefix=serialize_checkpoint(serialized),
        context=context,
        task="mmlu_pro_engineering",
        registry=load_new_mcq_runtime_registry(ROOT),
        runtime_registry=PHASE13_CORE_BASELINE_REGISTRY,
    )

    assert tuple(branches.arms) == ("clean", "correct", "irrelevant", "contam")
    assert branches.arms["clean"].root_count == 0
    assert all(branches.arms[arm].root_count == 1 for arm in ("correct", "irrelevant", "contam"))
    assert {
        event.candidate_triplet_id
        for event in branches.events
        if event.kind == "intervention_applied"
    } == {
        "phase13-new-mcq::mmlu_pro_engineering::"
        f"{INTERVENTIONS.tasks['mmlu_pro_engineering'].selected_candidate_id}"
    }
