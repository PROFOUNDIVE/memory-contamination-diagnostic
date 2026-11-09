"""Synthetic, exact-price, in-process wire transport; never production qualification."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

from pydantic import TypeAdapter

from memcontam.experiment.phase13_snapshot_diagnostic import snapshot_dispatch, snapshot_input_utf8
from memcontam.readiness.phase13_live_diagnostic import LiveDiagnosticLedger
from memcontam.readiness.phase13_live_diagnostic_models import (
    LiveDiagnosticBinding,
    LiveDiagnosticScope,
)
from memcontam.readiness.phase13_shared_provider_budget import SharedProviderBudget
from memcontam.readiness.phase13_v3_request import (
    MessageV3,
    RequestKeyV3,
    RequestMaterialV3,
    Stage,
    compile_request_bytes,
)

from .phase13_budget_fixtures import fixture as budget_fixture
from .phase13_budget_fixtures import pin, save
from .phase13_snapshot_durable_fixtures import native_snapshot
from .test_phase13_live_diagnostic_runtime import fixture as contract_fixture


def api():
    name = "memcontam.experiment.phase13_live_diagnostic_runtime"
    assert importlib.util.find_spec(name) is not None, "native Diagnostic runtime adapter missing"
    return importlib.import_module(name)


class WireFake:
    provenance = "QA_IN_PROCESS"

    def __init__(self, responses, failure=None):
        self.responses = responses
        self.failure = failure
        self.sent = []
        self.resources = None

    def send(self, operation, raw, before_egress):
        module = importlib.import_module("memcontam.readiness.phase13_diagnostic_exchange")
        assert self.resources is not None
        from memcontam.readiness.phase13_main_runner_store import connect
        with connect(self.resources.ledger.path) as connection:
            events = list(connection.execute("SELECT kind,reason FROM events ORDER BY event_sequence"))
        assert events[-1][0] == "REQUEST_INTENT"
        intent = json.loads(events[-1][1])
        assert intent["request_sha256"] == hashlib.sha256(raw).hexdigest()
        with connect(self.resources.budget.path) as connection:
            row = connection.execute("SELECT state,operation FROM reservations WHERE reservation_id=?",
                                     (intent["reservation_id"],)).fetchone()
        assert row[0] == "intent" and json.loads(row[1])["request_sha256"] == intent["request_sha256"]
        before_egress()
        self.sent.append(operation.operation)
        if self.failure == "count_timeout" and operation.operation == "count":
            raise TimeoutError("QA count timeout")
        if self.failure == "generation_timeout" and operation.operation == "generation":
            raise TimeoutError("QA generation timeout")
        if operation.operation == "count":
            body = {"object": "response.input_tokens", "input_tokens": 999999 if self.failure == "ceiling" else 10}
        else:
            index = self.sent.count("generation") - 1
            body = {"id": f"qa-response-{index}", "object": "response", "created_at": 1,
                    "model": "gpt-5.6-luna", "status": self.failure if self.failure in {"incomplete", "failed"} else "completed",
                    "service_tier": "default", "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
                    "output": [{"id": f"qa-message-{index}", "type": "message", "role": "assistant",
                                "status": "completed", "content": [{"type": "output_text", "text": self.responses[index],
                                                                       "annotations": []}]}],
                    "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20,
                              "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
                    "incomplete_details": {"reason": "max_output_tokens"} if self.failure == "incomplete" else None}
        return module.DiagnosticWireReply(status_code=200, body=json.dumps(body).encode(),
                                          provider_request_id=f"qa-http-{len(self.sent)}")


def native_fixture(root: Path, baseline="bot_style", failure=None, *, foreign_prefix=False, unparented=False, generation_bound=10000, task_name=None, truncated_schedule=False, live_candidate=False):
    module = api()
    snapshot_root = root / "snapshot"
    snapshot_root.mkdir()
    if task_name is not None:
        from memcontam.evaluation.phase13_observability_registration import (
            registered_verifier_result,
        )
        from memcontam.experiment.phase13_snapshot_diagnostic import (
            execute_snapshot_unit,
            snapshot_input_sha256,
        )
        from memcontam.tasks.base import TaskInstance

        from .phase13_snapshot_durable_fixtures import ContractTransport
        from .test_phase13_snapshot_bot import make_fixture
        from .test_phase13_snapshot_dc_rs import _native_embedder
        initial, bound, _ = make_fixture(snapshot_root, _native_embedder())
        task = (TaskInstance(sample_id="qa-meb", task_name="math_equation_balancer", input={"input": "1 ? 1 = 2"},
            verifier_spec={"target": "1 + 1 = 2", "target_value": 2}) if task_name == "math_equation_balancer" else
            TaskInstance(sample_id="qa-words", task_name="word_sorting", input={"words": ["pear", "apple"]},
                         verifier_spec={"sorted_words": ["apple", "pear"]}))
        initial = replace(initial, task_name=task_name, tasks=(task,), client=ContractTransport(initial.client),
            verifier=lambda answer, task: registered_verifier_result(task, answer).is_correct,
            decoding={"temperature": 0.0, "top_p": 1.0})
        bound = replace(bound, input_sha256=snapshot_input_sha256(initial), archive_identity=bound.archive_identity.model_copy(update={
            "checkpoint_registry_sha256": bound.state_sha256, "execution_template_id": f"{task_name}:bot_style:clean",
            "ordered_sample_ids_sha256": hashlib.sha256(json.dumps((task.sample_id,), separators=(",", ":")).encode()).hexdigest()}))
        snapshot = execute_snapshot_unit(initial, bound, snapshot_root)
    elif unparented:
        from memcontam.experiment.phase13_snapshot_diagnostic import (
            execute_snapshot_unit,
            snapshot_input_sha256,
        )
        from memcontam.tasks.base import TaskInstance

        from .phase13_snapshot_durable_fixtures import ContractTransport
        from .test_phase13_snapshot_dc_rs import _context, _shared_fixture
        task = TaskInstance(sample_id="qa-words", task_name="word_sorting", input={"words": ["pear", "apple"]},
                            verifier_spec={"sorted_words": ["apple", "pear"]})
        initial, bound = _shared_fixture(_context(task, sources=""))
        initial = replace(initial, client=ContractTransport(initial.client), decoding={"temperature": 0.0, "top_p": 1.0})
        bound = replace(bound, input_sha256=snapshot_input_sha256(initial),
                        archive_identity=bound.archive_identity.model_copy(update={"checkpoint_registry_sha256": bound.state_sha256}))
        snapshot = execute_snapshot_unit(initial, bound, snapshot_root)
    else:
        snapshot = native_snapshot(snapshot_root, baseline)
    old_unit, old_output = snapshot_dispatch(snapshot)
    run = snapshot.run
    if foreign_prefix:
        from memcontam.memory.checkpoint_v3 import serialize_checkpoint
        assert run.branch is not None
        run = replace(run, run_id="qa-foreign-suffix", branch=replace(run.branch,
            checkpoint=serialize_checkpoint(run.branch.checkpoint.state, checkpoint_index=3)))
    prep_root = root / "prep"
    prep_root.mkdir()
    contract_api, _, registration_path, contract = contract_fixture(prep_root)
    assert run.branch is not None
    state_raw = run.branch.checkpoint.canonical_bytes
    observations = [{"run_id": run.run_id, "child_execution_id": f"qa-{baseline}",
                     "input_sha256": hashlib.sha256(snapshot_input_utf8(run).encode()).hexdigest(),
                     "state_sha256": hashlib.sha256(state_raw).hexdigest(), "requests": []}]
    prices = []
    rates_pin = save(prep_root / "qa-rates.json", {"input_micros_per_token": "1", "output_micros_per_token": "1", "count_micros": 5})
    for call in old_output.provider_calls:
        assert call.call_id is not None
        native_call_id = call.call_id.replace(snapshot.run.run_id, run.run_id, 1)
        wire = compile_request_bytes(RequestKeyV3(parent_id="a" * 64, stage=TypeAdapter(Stage).validate_python(call.stage), ordinal=0),
            RequestMaterialV3(messages=tuple(MessageV3.model_validate(row) for row in call.messages), native_state=state_raw))
        body = json.loads(wire)
        count = json.dumps({name: body[name] for name in ("model", "input", "tools", "reasoning")},
                           sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        for kind, raw, suffix in (("count", count, "/responses/input_tokens"), ("generation", wire, "/responses")):
            request_id = f"{native_call_id}:{kind}"
            observations[0]["requests"].append({"request_id": request_id, "operation": kind, "stage": call.stage,
                                              "native_call_id": native_call_id if kind == "generation" else None})
            prices.append({"price_id": request_id, "operation": kind, "endpoint": contract["endpoint"] + suffix,
                           "model": "gpt-5.6-luna", "service_tier": "default", "currency": "USD",
                           "units": "USD_micros_per_bounded_operation", "upper_bound_usd_micros": generation_bound if kind == "generation" else 10000,
                           "request_sha256": hashlib.sha256(raw).hexdigest(), "as_of": "2026-10-22",
                           "valid_until": "2099-12-31", "evidence": rates_pin, "unknown_billing": False})
    budget_path = budget_fixture(root / "budget")
    budget = json.loads(budget_path.read_bytes())
    budget["allowed_runs"] = [run.run_id]
    budget["pricing"] = save(root / "budget/pricing.json", {"schema_version": "phase13_budget_pricing_v1",
                            "account_id": "qa-account", "entries": prices, "blockers": []})
    contract["budget_binding"] = save(budget_path, budget)
    if truncated_schedule:
        observations[0]["requests"] = observations[0]["requests"][:2]
    contract["observations"] = observations
    contract["rate_card"] = rates_pin
    assert run.embedding_provider is not None
    contract["embedding_metadata_sha256"] = hashlib.sha256(json.dumps(run.embedding_provider.metadata,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    contract["common_capacity_tokens"] = 8192
    if live_candidate:
        contract["qualification"] = "LIVE_REQUIRES_ADMISSION"
    if foreign_prefix:
        prefix_path = prep_root / "qa-prefix-state.json"
        prefix_path.write_bytes(state_raw)
        contract["prefix_histories"] = [{"run_id": run.run_id, "source_run_id": snapshot.run.run_id,
            "checkpoint_index": 3, "state": pin(prefix_path)}]
    contract_pin = save(prep_root / "runtime-contract.json", contract)
    registration = json.loads(registration_path.read_bytes())
    registration["runtime_contract"] = contract_pin
    registration_pin = save(registration_path, registration)
    scope_raw = old_output.model_dump(mode="json")["evidence"]["runtime_evidence"]["diagnostic_scope"]
    source_map = {row["path"]: row for row in scope_raw["source_files"]}
    assert contract_api.__file__ is not None
    for path in (Path(contract_api.__file__), *module.runtime_source_paths()):
        source_map[str(path.resolve())] = pin(path)
    for item in (contract_pin, registration_pin):
        source_map[item["path"]] = item
    if live_candidate:
        verifier_source = Path(__file__).with_name("test_phase13_diagnostic_transport.py").resolve()
        source_map[str(verifier_source)] = pin(verifier_source)
    binding = LiveDiagnosticBinding.model_validate_json(json.dumps({
        **{name: registration[name] for name in ("run_id", "source_revision", "registration_id", "authorization_id", "authorization_sha256", "scientific_disposition")},
        "ledger_path": str(root / "ledger.sqlite3"), "registration_sha256": registration_pin["sha256"],
        "source_files": list(source_map.values()), "observations": observations}))
    scope_raw.pop("real_spend_cap_krw")
    scope_raw.update(run_id=run.run_id, input_utf8=snapshot_input_utf8(run),
                     input_sha256=observations[0]["input_sha256"], state_sha256=observations[0]["state_sha256"])
    scope_raw.update(provenance="LIVE_DIAGNOSTIC", main_scope="NOT_MAIN", binding_sha256=binding.sha256(),
                     child_execution_id=observations[0]["child_execution_id"], scientific_disposition="NOT_ADOPTED",
                     source_files=list(source_map.values()))
    scope = LiveDiagnosticScope.model_validate_json(json.dumps(scope_raw))
    unit = replace(old_unit, unit_id=hashlib.sha256((run.run_id + scope.sha256()).encode()).hexdigest(),
                   diagnostic_scope_sha256=scope.sha256())
    context = contract_api.DiagnosticExecutionContext.load(binding, registration_path)
    ledger = LiveDiagnosticLedger.create(binding, (unit,))
    budget_ledger = SharedProviderBudget.create(budget_path)
    resources = module.DiagnosticRuntimeResources(context, scope, unit, ledger, budget_ledger, root / "exchanges")
    transport = WireFake([call.raw_response for call in old_output.provider_calls], failure)
    transport.resources = resources
    if live_candidate:
        return module, run, resources, transport
    admitted = module.admit_qa(run, resources, transport)
    return module, replace(run, client=admitted.execution_client, allow_test_client=False, diagnostic_context=admitted), resources, transport
