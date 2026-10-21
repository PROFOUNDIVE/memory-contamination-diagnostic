from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

if __name__ == "__main__":
    __package__ = "tests"

from memcontam.clients.base import LLMResponse
from memcontam.readiness.phase13_cost_policy import load_cost_policy_bundle
from memcontam.readiness.phase13_main_live_dispatch import MainUnitDispatchOutput
from memcontam.readiness.phase13_main_live_runtime_support import dispatch_output
from memcontam.readiness.phase13_main_production import ProductionObject
from memcontam.tasks.base import TaskInstance

from .test_phase13_snapshot_bot import make_fixture
from .test_phase13_snapshot_dc_rs import _context, _native_embedder, _shared_fixture


class ContractTransport:
    def __init__(self, client, fail_stage: str | None = None) -> None:
        self.client = client
        self.fail_stage = fail_stage
        self.calls = 0

    def chat(self, messages, model, config) -> LLMResponse:
        self.calls += 1
        if config["method_stage"] == self.fail_stage:
            raise TimeoutError("synthetic diagnostic transport timeout")
        response = self.client.chat(messages, model, config)
        policy = load_cost_policy_bundle(Path(__file__).resolve().parents[1])
        stage = next(row for row in policy.registry.stages if row.semantic_stage_id == config["method_stage"])
        request = {
            "model": model, "input_sha256": hashlib.sha256(json.dumps(
                messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
            "temperature": 0.0, "top_p": 1.0,
            "reasoning": {"mode": "standard", "effort": "none", "context": "current_turn"},
            "previous_response_id": None, "service_tier": "default", "store": False,
            "tools": [], "max_output_tokens": stage.maximum_output_tokens,
        }
        authority = {
            "maximum_input_tokens": stage.maximum_input_tokens,
            "maximum_output_tokens": stage.maximum_output_tokens,
            "execution_envelope_id": policy.registry.registry_id,
            "execution_envelope_sha256": policy.registry.registry_hash,
            "failure_contract_id": policy.retry.contract_id,
            "failure_contract_sha256": policy.retry.contract_hash,
            "terminal_failure_contract_id": policy.retry.terminal_failure_contract_id,
            "terminal_failure_contract_sha256": policy.retry.terminal_failure_contract_sha256,
            "rate_card_sha256": hashlib.sha256(json.dumps(policy.proof.rate_card.model_dump(mode="json"),
                sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        }
        return replace(response, raw={
            "attempts": 1, "status": "completed", "response_id": f"synthetic-{self.calls}",
            "usage": {"input_tokens": 1, "output_tokens": 1}, "service_tier": "default",
            "model": model, "request_contract": request, "authority_contract": authority,
            "cost_usd": 0.0, "derived_cost_usd": 0.0, "cost_source": "DERIVED_FROM_PROVIDER_USAGE",
        })


def native_snapshot(root: Path, baseline: str, fail_stage: str | None = None):
    from memcontam.experiment.phase13_snapshot_diagnostic import execute_snapshot_unit, snapshot_input_sha256

    if baseline == "bot_style":
        run, bindings, _ = make_fixture(root, _native_embedder())
    else:
        task = TaskInstance(sample_id="qa-words", task_name="word_sorting",
                            input={"words": ["pear", "apple"]}, verifier_spec={"sorted_words": ["apple", "pear"]})
        run, bindings = _shared_fixture(_context(task))
    run = replace(run, client=ContractTransport(run.client, fail_stage), decoding={"temperature": 0.0, "top_p": 1.0})
    bindings = replace(bindings, input_sha256=snapshot_input_sha256(run),
                       archive_identity=bindings.archive_identity.model_copy(update={"checkpoint_registry_sha256": bindings.state_sha256}))
    return execute_snapshot_unit(run, bindings, root)


def ordinary_output(snapshot):
    identity = snapshot.bindings.archive_identity
    checkpoint = snapshot.run.branch.checkpoint
    unit = ProductionObject(0, hashlib.sha256(snapshot.run.run_id.encode()).hexdigest(),
        "MEMORY_BEARING", 0, snapshot.run.task_name, snapshot.run.baseline, snapshot.run.arm,
        hashlib.sha256(snapshot.run.branch.prefix_identity.encode()).hexdigest(), 0,
        identity.execution_template_id, identity.ordered_sample_ids_sha256,
        identity.registration_packet_sha256, identity.checkpoint_registry_sha256)
    output = dispatch_output(unit, snapshot.ordinary.trials, identity, snapshot.archive, realized_cost_krw=0)
    return unit, MainUnitDispatchOutput(evidence={
        "evidence_kind": "MEMORY_BEARING", "prefix_unit_id": unit.prefix_unit_id,
        "consumed_checkpoint_id": checkpoint.identity.checkpoint_id,
        "consumed_checkpoint_identity_sha256": checkpoint.identity.sha256,
        "consumed_checkpoint_canonical_sha256": checkpoint.canonical_sha256,
        "runtime_evidence": output.evidence,
    }, provider_calls=output.provider_calls, realized_cost_krw=0)


if __name__ == "__main__":
    import os
    from dataclasses import asdict
    from memcontam.readiness.phase13_main_runner_ledger import MainRunLedger
    from memcontam.readiness.phase13_main_runner_models import MainRunBinding, MainRunError, InFlightEvidence
    from memcontam.readiness.phase13_main_live_evidence import MainUnitEvidence, MainReconciliationEvidence, load_durable_unit_evidence

    root = Path(os.environ["TASK14_REOPEN_ROOT"])
    payload = json.loads((root / "reopen.json").read_text())
    unit = ProductionObject(**payload["unit"])
    if os.environ.get("TASK14_LOAD_ONLY") == "1":
        path = root / "units" / f"000000-{unit.unit_id}.json"
        load_durable_unit_evidence(path, unit, hashlib.sha256(path.read_bytes()).hexdigest(), 0)
        raise SystemExit(0)
    ledger = MainRunLedger.open(root / "ledger.sqlite3", MainRunBinding(**payload["binding"]), (unit,))
    if ledger.status().in_flight_count:
        try:
            ledger.claim_dispatch(unit.unit_id, 0, 0)
        except MainRunError as error:
            assert error.code == "MAIN_RUN_IN_FLIGHT_RECONCILIATION_REQUIRED"
        else:
            raise AssertionError("redispatch allowed")
        if os.environ["TASK14_RECONCILE"] == "1":
            context = ledger.in_flight_context(unit.unit_id)
            path = root / "units" / f"000000-{unit.unit_id}.json"
            if path.exists():
                record = MainUnitEvidence.model_validate_json(path.read_bytes())
                ledger.reconcile(unit.unit_id, InFlightEvidence.completed(context,
                    hashlib.sha256(path.read_bytes()).hexdigest(), record.realized_cost_krw))
            else:
                path = root / "reconciliation" / f"{context.intent_event_hash}.json"
                record = MainReconciliationEvidence.model_validate_json(path.read_bytes())
                assert record.failure_code is not None
                ledger.reconcile(unit.unit_id, InFlightEvidence.terminal_failure(context,
                    hashlib.sha256(path.read_bytes()).hexdigest(), record.failure_code, record.realized_cost_krw))
    status = ledger.status()
    if status.completed_count:
        record = MainUnitEvidence.model_validate_json(next((root / "units").glob("*.json")).read_bytes())
    else:
        reconciliation = MainReconciliationEvidence.model_validate_json(next((root / "reconciliation").glob("*.json")).read_bytes())
        record = reconciliation.diagnostic_partial
    assert record is not None
    assert record.evidence.runtime_evidence.production_observability_archive is not None
    print(json.dumps({"pid": os.getpid(), "status": asdict(status),
        "scope_sha256": unit.diagnostic_scope_sha256,
        "archive": record.evidence.runtime_evidence.production_observability_archive.model_dump(mode="json")}))
