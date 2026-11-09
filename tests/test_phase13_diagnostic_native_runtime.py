from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from .phase13_diagnostic_runtime_fixtures import native_fixture
from .test_phase13_snapshot_bot import capacity_fixture

pytestmark = pytest.mark.usefixtures(capacity_fixture.__name__)


@pytest.mark.parametrize("baseline,count", [("bot_style", 3), ("dc_rs", 2)])
def test_registered_v4_native_output_reopens_without_v2_relabel(tmp_path: Path, baseline: str, count: int) -> None:
    module, run, resources, transport = native_fixture(tmp_path, baseline)
    result = module.execute_diagnostic(run)
    assert result.disposition == "COMPLETED"
    assert transport.sent == ["count", "generation"] * count
    assert resources.ledger.status()["completed"] == 1
    assert all(call.provider_authority_contract["execution_envelope_id"] == "CORE_EXECUTION_ENVELOPE_REGISTRY_V4"
               for call in result.provider_calls)
    assert resources.ledger.status()["known_cost_usd_micros"] == count * 25
    (tmp_path / "reopen.json").write_text(json.dumps({"binding": resources.context.binding.model_dump(mode="json"),
        "units": [asdict(resources.unit)], "digests": []}))
    child = subprocess.run([sys.executable, "-B", "-m", "tests.phase13_recovery_offline",
        str(Path(__file__).with_name("phase13_live_diagnostic_fixtures.py"))],
        env={**os.environ, "TASK8_ROOT": str(tmp_path), "TASK8_RECONCILE": "0"}, capture_output=True, text=True, timeout=30, check=False)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)["completed"] == 1
    with pytest.raises(ValueError):
        module.execute_diagnostic(run)


@pytest.mark.parametrize("failure,attempts", [("count_timeout", 0), ("ceiling", 0),
                                             ("generation_timeout", 1), ("incomplete", 1), ("failed", 1)])
@pytest.mark.parametrize("baseline", ["bot_style", "dc_rs"])
def test_terminal_and_pre_generation_holds_are_truthful(tmp_path: Path, baseline: str, failure: str, attempts: int) -> None:
    module, run, resources, transport = native_fixture(tmp_path, baseline, failure)
    result = module.execute_diagnostic(run)
    assert transport.sent.count("generation") == attempts
    assert resources.ledger.status()["completed"] == 0
    if attempts == 0:
        assert result.generation_attempts == 0
        assert not list((tmp_path / "units").glob("*.json"))
    else:
        assert result.disposition == "TERMINAL_FAILURE"
        assert len(result.provider_calls) == 1
        assert result.provider_calls[0].transport_attempts == 1
        if failure == "incomplete":
            assert result.provider_calls[0].provider_status == "incomplete"
        if failure == "failed":
            assert result.provider_calls[0].provider_status == "failed"
            assert result.realized_cost_usd_micros == 20
    with pytest.raises(ValueError):
        module.execute_diagnostic(run)
    (tmp_path / "reopen.json").write_text(json.dumps({"binding": resources.context.binding.model_dump(mode="json"),
        "units": [asdict(resources.unit)], "digests": []}))
    child = subprocess.run([sys.executable, "-B", "-m", "tests.phase13_recovery_offline",
        str(Path(__file__).with_name("phase13_live_diagnostic_fixtures.py"))],
        env={**os.environ, "TASK8_ROOT": str(tmp_path), "TASK8_RECONCILE": "0"},
        capture_output=True, text=True, timeout=30, check=False)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)["completed"] == 0


def test_diagnostic_context_excludes_main_and_test_client_escape_hatches(tmp_path: Path, monkeypatch) -> None:
    module, run, _, _ = native_fixture(tmp_path)
    from memcontam.readiness import phase13_main_checkpoint
    monkeypatch.setattr(phase13_main_checkpoint, "production_identity_from_checkpoint",
                        lambda *args, **kwargs: pytest.fail("Diagnostic consumed Main authorization"))
    with pytest.raises(ValueError):
        replace(run, allow_test_client=True)
    assert run.diagnostic_context is not None
    with pytest.raises(ValueError):
        replace(run, client=run.diagnostic_context.resources.context)
    assert module.execute_diagnostic(run).disposition == "COMPLETED"


def test_source_drift_refuses_before_count(tmp_path: Path) -> None:
    module, run, _, transport = native_fixture(tmp_path)
    path = tmp_path / "prep/qa-resource.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        module.execute_diagnostic(run)
    assert transport.sent == []


def test_qa_admission_cannot_open_live_transport(tmp_path: Path) -> None:
    module, _, resources, _ = native_fixture(tmp_path)
    with pytest.raises(ValueError, match="LIVE_ADMISSION"):
        module.open_live_transport(resources.context, None)


def test_registered_foreign_prefix_history_preserves_original_ids(tmp_path: Path) -> None:
    module, run, resources, _ = native_fixture(tmp_path, "dc_rs", foreign_prefix=True)
    before = resources.scope.state_utf8
    result = module.execute_diagnostic(run)
    assert result.disposition == "COMPLETED"
    assert "qa:trial:1:qa-prior" in before
    runtime = result.evidence["runtime_evidence"]
    assert "qa:trial:1:qa-prior" in runtime["diagnostic_post_state_utf8"]
    assert runtime["production_observability_archive"]["records"][0]["evidence"]["trial"]["checkpoint_index"] == 3


def test_semantic_failure_retains_completed_receipts_and_native_post_state(tmp_path: Path) -> None:
    module, run, _, transport = native_fixture(tmp_path, "dc_rs")
    transport.responses[-1] = "no terminal answer marker"
    result = module.execute_diagnostic(run)
    assert result.disposition == "TERMINAL_FAILURE"
    assert result.provider_calls[-1].provider_status == "completed"
    runtime = result.evidence["runtime_evidence"]
    assert runtime["diagnostic_scope"]["state_utf8"] != runtime["diagnostic_post_state_utf8"]
    assert runtime["production_observability_archive"]["records"][0]["terminal_failure_code"] == "dc_rs_invalid_final_answer"


@pytest.mark.parametrize("field,value", [("provider_response_id", "forged"), ("provider_returned_model", "forged"),
    ("token_usage", {"prompt_tokens": 99, "completion_tokens": 10}), ("provider_cost_usd", 1.0),
    ("provider_authority_contract", {"execution_envelope_id": "CORE_EXECUTION_ENVELOPE_REGISTRY_V2"})])
def test_response_identity_mutation_refuses(tmp_path: Path, field: str, value) -> None:
    module, run, resources, _ = native_fixture(tmp_path)
    output = module.execute_diagnostic(run)
    from memcontam.readiness.phase13_diagnostic_exchange_validate import validate_exchange
    call = output.provider_calls[0].model_copy(update={field: value})
    with pytest.raises(ValueError):
        validate_exchange(call, resources.context, resources.scope)


@pytest.mark.parametrize("boundary", ["reserve", "financial_intent", "diagnostic_intent"])
def test_crash_boundary_retains_hold_and_refuses_restart(tmp_path: Path, monkeypatch, boundary: str) -> None:
    module, run, resources, transport = native_fixture(tmp_path)
    from memcontam.readiness.phase13_diagnostic_exchange import DiagnosticStop
    owner, method = ((resources.budget, "reserve") if boundary == "reserve" else
                     (resources.budget, "mark_intent") if boundary == "financial_intent" else
                     (resources.ledger, "claim_request"))
    original = getattr(owner, method)
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise DiagnosticStop("QA_CRASH_" + boundary)
    monkeypatch.setattr(owner, method, crash)
    held = module.execute_diagnostic(run)
    assert held.disposition == "HOLD" and held.generation_attempts == 0
    assert resources.budget.status().reservations == 1
    assert resources.budget.status().held_usd_micros > 0
    assert transport.sent == []
    with pytest.raises(ValueError):
        module.execute_diagnostic(run)


@pytest.mark.parametrize("unparented", [False, True])
@pytest.mark.parametrize("mutation", ["parent", "curator_hash"])
def test_native_lineage_requires_real_curator_parent(tmp_path: Path, unparented: bool, mutation: str) -> None:
    module, run, resources, _ = native_fixture(tmp_path, "dc_rs", unparented=unparented)
    output = module.execute_diagnostic(run)
    curator, answer = output.provider_calls
    assert answer.source_spans[0].parent_call_id == curator.call_id
    assert answer.source_spans[0].lineage_status == ("unavailable" if unparented else "exact")
    from memcontam.readiness.phase13_live_diagnostic_evidence import validate_live_output
    from memcontam.readiness.phase13_live_diagnostic_models import LiveDiagnosticOutput
    raw = output.model_dump(mode="json")
    if mutation == "parent":
        raw["provider_calls"][1]["source_spans"][0]["parent_call_id"] = "forged-curator-call"
    else:
        raw["provider_calls"][0]["source_spans"][0]["rendered_hash"] = "f" * 64
    raw["evidence"]["runtime_evidence"]["production_observability_archive"]["records"][0]["method_calls"] = raw["provider_calls"]
    with pytest.raises(ValueError):
        validate_live_output(resources.unit, LiveDiagnosticOutput.model_validate_json(json.dumps(raw)), resources.context.binding)


def test_malformed_native_input_refuses_before_live_factory(tmp_path: Path, monkeypatch) -> None:
    module, run, resources, transport = native_fixture(tmp_path)
    from memcontam.readiness import phase13_diagnostic_transport
    monkeypatch.setattr(phase13_diagnostic_transport, "open_live_transport",
                        lambda *args: pytest.fail("live construction preceded native input validation"))
    wrong = replace(run, diagnostic_context=None, client=transport, allow_test_client=True,
                    run_id="foreign-run")
    with pytest.raises(ValueError):
        module.admit_live(wrong, resources, None)


def test_response_dependent_unpriced_second_request_holds_without_count(tmp_path: Path) -> None:
    module, run, resources, transport = native_fixture(tmp_path)
    assert transport.responses[0] is not None
    raw = json.loads(transport.responses[0])
    raw["distilled_task"] = "A completely different QA task description"
    transport.responses[0] = json.dumps(raw)
    result = module.execute_diagnostic(run)
    assert result.disposition == "HOLD" and "EXACT_PRICE" in result.code
    assert transport.sent == ["count", "generation"]
    assert result.generation_attempts == 1
    assert resources.ledger.status()["in_flight"] == 1
    assert len(result.method_calls) == 1
    retained = json.loads(Path(result.method_calls[0].path).read_bytes())
    assert retained["provider_response_id"] == "qa-response-0"


@pytest.mark.parametrize("mutation", ["wire", "count_projection", "count_response", "response_usage"])
def test_raw_exchange_mutations_refuse_even_with_rehashed_pins(tmp_path: Path, mutation: str) -> None:
    import hashlib

    from memcontam.readiness.phase13_diagnostic_exchange_validate import validate_exchange
    module, run, resources, _ = native_fixture(tmp_path)
    output = module.execute_diagnostic(run)
    call = output.provider_calls[0]
    call_contract = dict(call.provider_request_contract)
    exchange_path = Path(call_contract["exchange"]["path"])
    exchange = json.loads(exchange_path.read_bytes())
    count_path = Path(exchange["count"]["path"])
    count = json.loads(count_path.read_bytes())
    if mutation.startswith("count"):
        selected = count["request" if mutation == "count_projection" else "response"]
    else:
        selected = exchange["generation"]["request" if mutation == "wire" else "response"]
    payload_path = Path(selected["path"])
    payload = json.loads(payload_path.read_bytes())
    if mutation == "wire":
        payload["max_output_tokens"] += 1
    elif mutation == "count_projection":
        payload["input"][0]["content"] += "changed"
    elif mutation == "count_response":
        payload["input_tokens"] += 1
    else:
        payload["usage"]["output_tokens"] += 1
    payload_path.write_text(json.dumps(payload))
    selected["sha256"] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    if mutation.startswith("count"):
        count_path.write_text(json.dumps(count))
        exchange["count"]["sha256"] = hashlib.sha256(count_path.read_bytes()).hexdigest()
    exchange_path.write_text(json.dumps(exchange))
    call_contract["exchange"]["sha256"] = hashlib.sha256(exchange_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        validate_exchange(call.model_copy(update={"provider_request_contract": call_contract}), resources.context, resources.scope)


def test_insufficient_quoted_bound_refuses_generation(tmp_path: Path) -> None:
    module, run, _, transport = native_fixture(tmp_path, generation_bound=10)
    result = module.execute_diagnostic(run)
    assert result.disposition == "HOLD" and "BOUND" in result.code
    assert result.generation_attempts == 0
    assert transport.sent == ["count"]


def test_mutated_transport_cannot_reuse_admitted_context(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace
    module, run, _, transport = native_fixture(tmp_path)
    assert run.diagnostic_context is not None
    monkeypatch.setattr(run.diagnostic_context.engine, "transport", SimpleNamespace(provenance="PROVIDER", send=transport.send))
    with pytest.raises(ValueError, match="TRANSPORT"):
        module.execute_diagnostic(run)
    assert transport.sent == []


@pytest.mark.parametrize("task_name", ["math_equation_balancer", "word_sorting"])
def test_bot_intended_task_laws_execute_natively(tmp_path: Path, task_name: str) -> None:
    module, run, _, transport = native_fixture(tmp_path, task_name=task_name)
    output = module.execute_diagnostic(run)
    assert output.disposition == "COMPLETED"
    assert transport.sent == ["count", "generation"] * 3
    assert output.evidence["runtime_evidence"]["production_observability_archive"]["records"][0]["evidence"]["task"] == task_name


def test_subclass_cannot_override_diagnostic_admission(tmp_path: Path) -> None:
    from memcontam.readiness.phase13_diagnostic_admission import AdmittedDiagnosticExecution
    _, run, resources, transport = native_fixture(tmp_path)
    assert run.diagnostic_context is not None
    class ForgedContext(AdmittedDiagnosticExecution):
        def validate_run(self, run) -> None:
            return None
    forged = ForgedContext(resources, run.diagnostic_context.execution_client, run.diagnostic_context.engine)
    with pytest.raises(ValueError, match="ADMISSION"):
        replace(run, diagnostic_context=forged)
    assert transport.sent == []


def test_incomplete_registered_schedule_refuses_before_dispatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SCHEDULE"):
        native_fixture(tmp_path, truncated_schedule=True)
