from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.experiment import phase13_ordinary_runtime as runtime
from memcontam.readiness.phase13_authority_files import load_authority_v3
from memcontam.readiness.phase13_core_datasets import paired_trajectory_order
from memcontam.readiness.phase13_main_checkpoint import CommonCheckpointRegistry
from memcontam.readiness.phase13_production_runtime_models import ProductionOrdinaryRunIdentity
from memcontam.readiness.phase13_v3_cost import activate_policy, build_witness, freeze_base
from memcontam.readiness.phase13_v3_cost_binding import CostBoundPackageV3, LiveCosts, MRP4Costs, bind_package_costs
from memcontam.readiness.phase13_v3_cost_models import CostUnit, FinalOrder, PrefreezeBindings, StageOccurrences
from memcontam.tasks.base import TaskInstance


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, model, config):
        self.calls += 1
        return LLMResponse("final: A", {"replay": True, "attempts": 1}, {}, 0)


@pytest.fixture(scope="session")
def costs():
    authority = load_authority_v3(Path(
        "/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/References/Theoretical Artifacts"))
    base = freeze_base(activate_policy(authority), PrefreezeBindings(**{
        name: "a" * 64 for name in PrefreezeBindings.model_fields
    }), (CostUnit(unit_id="a" * 64, stages=(StageOccurrences(
        stage_id="NoMem_generation", calls=50,
    ),)),))
    bound = bind_package_costs(CostBoundPackageV3(final_order=FinalOrder(
        unit_ids=("a" * 64,), runtime_hash="b" * 64, request_hash="c" * 64, tokenizer_hash="d" * 64,
    )), MRP4Costs(policy=base.policy, base=base, witness=build_witness(base)))
    return LiveCosts(bound.package, bound.resources)


@pytest.fixture
def preloaded_run(costs):
    rows = tuple(TaskInstance(sample_id=f"mmlu_pro_physics:{index}", task_name="mmlu_pro_physics",
        input={"question": "fixture", "options": ["a", "b"]}, verifier_spec={}) for index in range(51))
    ordered = paired_trajectory_order(rows, trajectory_seed=0)
    prefix = tuple(row.sample_id for row in ordered[:1])
    suffix = tuple(row.sample_id for row in ordered[1:])
    def hash_ids(ids):
        return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    registry = CommonCheckpointRegistry.model_validate_json(json.dumps({
        "schema_version": "phase13_main_a_common_checkpoint_registry_v1",
        "task_seed_orders": {"path": "unused", "sha256": "a" * 64},
        "checkpoint_law": "tau_star=min(T_fix)", "registry_hash": "b" * 64,
        "tasks": {"mmlu_pro_physics": {"route": "3w", "L_min": 1, "H_run": 50,
            "static_route_constraints": [], "seeds": [{"seed": 0, "concrete_seed_id": "0",
                "tau_star": 2, "clean_prefix_sample_ids": prefix,
                "clean_prefix_sample_ids_sha256": hash_ids(prefix), "suffix_sample_ids": suffix,
                "suffix_sample_ids_sha256": hash_ids(suffix), "complete_order_sha256": "c" * 64}]}}}))
    raw = registry.model_dump_json().encode()
    client = FakeClient()
    resources = runtime.ValidatedOrdinaryResources(client, 8192, raw, hashlib.sha256(raw).hexdigest(), costs)
    identity = ProductionOrdinaryRunIdentity(execution_template_id="mmlu_pro_physics|nomem",
        trajectory_seed=0, concrete_seed_id="0", ordered_sample_ids_sha256=hash_ids(suffix),
        registration_packet_sha256="d" * 64, scientific_result=False,
        checkpoint_registry_sha256=resources.checkpoint_sha256)
    return runtime.ProspectiveOrdinaryRun(task_name="mmlu_pro_physics", baseline="nomem",
        run_id="fixture-only", model="gpt-5.6-luna", client=client, allow_test_client=True,
        verifier=lambda *_: True, decoding={"temperature": 0.0}, tasks=rows, trajectory_seed=0,
        production_identity=identity, validated_resources=resources)


def test_preloaded_core_order_and_context_never_read_paths(preloaded_run, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("preloaded ordinary mode reopened a pathname")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    run = replace(preloaded_run)
    tasks = runtime._ordered_tasks(run)
    runtime._validate_live_dispatch_identity(run, tasks)
    context = runtime._context(run, tasks[0], 1)
    assert len(tasks) == 50
    assert context.client is run.client


@pytest.mark.parametrize("mutation", ["path", "empty", "client", "identity", "seed"])
def test_preloaded_mode_rejects_mixed_or_partial_input(preloaded_run, mutation):
    changes = {"path": {"core_bundle": Path("forbidden")}, "empty": {"tasks": ()},
               "client": {"client": FakeClient()}, "identity": {"production_identity": None},
               "seed": {"trajectory_seed": None}}
    with pytest.raises(ValueError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
        replace(preloaded_run, **changes[mutation])


def test_preloaded_suffix_digest_cannot_be_substituted(preloaded_run):
    identity = preloaded_run.production_identity.model_copy(update={"ordered_sample_ids_sha256": "f" * 64})
    run = replace(preloaded_run, production_identity=identity)
    with pytest.raises(ValueError, match="PRODUCTION_SAMPLE_ORDER_MISMATCH"):
        runtime._ordered_tasks(run)


def test_preloaded_checkpoint_hash_cannot_be_substituted(preloaded_run):
    identity = preloaded_run.production_identity.model_copy(update={"checkpoint_registry_sha256": "f" * 64})
    run = replace(preloaded_run, production_identity=identity)
    with pytest.raises(ValueError, match="PRODUCTION_CHECKPOINT_IDENTITY_MISMATCH"):
        runtime._validate_live_dispatch_identity(run, runtime._ordered_tasks(run))


def test_preloaded_task_ownership_survives_model_mutation(preloaded_run):
    expected = tuple(row.model_dump_json() for row in runtime._ordered_tasks(preloaded_run))
    for row in preloaded_run.tasks:
        row.input["question"] = "changed after construction"
        row.sample_id = "changed"
    assert tuple(row.model_dump_json() for row in runtime._ordered_tasks(preloaded_run)) == expected
