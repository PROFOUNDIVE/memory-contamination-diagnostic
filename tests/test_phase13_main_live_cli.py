from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from memcontam.clients import openai_responses
from memcontam.readiness import phase13_main_live_cli
from memcontam.readiness.phase13_main_live_dispatch import MainTelemetrySummary
from memcontam.readiness.phase13_main_runner import MainRunLedger
from memcontam.readiness.phase13_main_runner_models import MainRunReport


ROOT = Path(__file__).resolve().parents[1]
P5 = ROOT / "data/phase13/main/mr_p5/execution_package_v1.json"
P6 = ROOT / "data/phase13/main/mr_p6/authorized_execution_v1.json"


class _NeverResponses:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_kwargs: str) -> None:
        self.calls += 1
        raise AssertionError("zero-unit launch must not reach the provider")


class _NoCallOpenAI:
    instance: _NoCallOpenAI

    def __init__(self, **_kwargs: str) -> None:
        self.responses = _NeverResponses()
        type(self).instance = self


def _run_argv(tmp_path: Path, run_id: str, max_units: int) -> list[str]:
    return [
        "phase13-main-a-live",
        "run",
        "--repository-root",
        str(ROOT),
        "--package",
        str(P5),
        "--authorization",
        str(P6),
        "--expected-authorization-sha256",
        hashlib.sha256(P6.read_bytes()).hexdigest(),
        "--run-root",
        str(tmp_path),
        "--run-id",
        run_id,
        "--tranche-ceiling-krw",
        "80000",
        "--max-units",
        str(max_units),
        "--cache-root",
        str(Path.home() / ".cache/huggingface/hub"),
        "--allow-live-calls",
    ]


def test_live_run_composes_authorized_zero_unit_session_without_provider_calls(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr(openai_responses, "OpenAI", _NoCallOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "no-call-construction-only")
    monkeypatch.setattr(sys, "argv", _run_argv(tmp_path, "zero-unit-no-call", 0))

    phase13_main_live_cli.main()

    report = json.loads(capsys.readouterr().out)
    assert report == {
        "attempted_count": 0,
        "completed_count": 0,
        "provider_calls_issued": 0,
        "session_state": "NOT_STARTED",
        "terminal_technical_missing_count": 0,
    }
    assert _NoCallOpenAI.instance.responses.calls == 0


def test_live_run_accepts_complete_runtime_authority_without_issuing_provider_calls(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr(openai_responses, "OpenAI", _NoCallOpenAI)
    monkeypatch.setattr(
        phase13_main_live_cli,
        "run_pending",
        lambda *_args, **_kwargs: MainRunReport("READY", 0, 0, 0),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "no-call-construction-only")
    monkeypatch.setattr(sys, "argv", _run_argv(tmp_path, "complete-main-a", 1))

    phase13_main_live_cli.main()

    assert json.loads(capsys.readouterr().out)["provider_calls_issued"] == 0
    assert _NoCallOpenAI.instance.responses.calls == 0


def test_validate_runs_provider_free_production_preflight(
    monkeypatch,
    capsys,
) -> None:
    calls: list[tuple] = []

    class CaptureRuntime:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def preflight(self, units) -> None:
            calls.append(tuple(units))

    monkeypatch.setattr(
        phase13_main_live_cli,
        "validate_main_authorization",
        lambda *_args: SimpleNamespace(
            authorization_id="phase13-main-a-authorized-execution-v1",
            main_a_status="NOT_STARTED",
        ),
    )
    monkeypatch.setattr(phase13_main_live_cli, "load_main_live_contract", lambda _path: None)
    monkeypatch.setattr(phase13_main_live_cli, "validate_main_live_contract", lambda *_args: None)
    monkeypatch.setattr(
        phase13_main_live_cli,
        "build_production_objects",
        lambda _package: (SimpleNamespace(kind="CLEAN_PREFIX"),),
    )
    monkeypatch.setattr(phase13_main_live_cli, "ProductionMainRuntime", CaptureRuntime)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase13-main-a-live",
            "validate",
            "--repository-root",
            str(ROOT),
            "--package",
            str(P5),
            "--authorization",
            str(P6),
            "--expected-authorization-sha256",
            hashlib.sha256(P6.read_bytes()).hexdigest(),
            "--cache-root",
            str(Path.home() / ".cache/huggingface/hub"),
        ],
    )

    phase13_main_live_cli.main()

    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "READY_NO_CALLS"


def test_zero_unit_run_wires_ledger_checkpoint_authority(monkeypatch, tmp_path: Path) -> None:
    captured: list[MethodType | None] = [None]

    class CaptureBackend:
        def __init__(self, _root, _prefix, _ordinary, completed_evidence_sha256=None) -> None:
            captured[0] = completed_evidence_sha256

        def __call__(self, _unit):
            raise AssertionError("zero-unit launch must not dispatch")

    monkeypatch.setattr(openai_responses, "OpenAI", _NoCallOpenAI)
    monkeypatch.setattr(phase13_main_live_cli, "MainProductionBackend", CaptureBackend)
    monkeypatch.setenv("OPENAI_API_KEY", "no-call-construction-only")
    monkeypatch.setattr(sys, "argv", _run_argv(tmp_path, "ledger-callback", 0))

    phase13_main_live_cli.main()

    callback = captured[0]
    assert isinstance(callback, MethodType)
    assert isinstance(callback.__self__, MainRunLedger)
    assert callback.__self__.status().total_count == 1200


def test_zero_unit_run_reports_durable_provider_count(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(openai_responses, "OpenAI", _NoCallOpenAI)
    monkeypatch.setattr(
        phase13_main_live_cli,
        "run_pending",
        lambda *_args, **_kwargs: MainRunReport("READY", 1, 1, 0),
    )
    monkeypatch.setattr(
        phase13_main_live_cli,
        "summarize_telemetry",
        lambda _root: MainTelemetrySummary(
            schema_version="phase13_main_telemetry_summary_v1",
            unit_count=1,
            provider_call_count=7,
            transport_attempt_count=7,
            latency_ms=1,
            token_usage={},
            provider_cost_usd="0",
            realized_cost_krw=0,
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "no-call-construction-only")
    monkeypatch.setattr(sys, "argv", _run_argv(tmp_path, "durable-telemetry", 0))

    phase13_main_live_cli.main()

    assert json.loads(capsys.readouterr().out)["provider_calls_issued"] == 7


def test_zero_unit_run_rejects_orphan_durable_telemetry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(openai_responses, "OpenAI", _NoCallOpenAI)
    monkeypatch.setattr(
        phase13_main_live_cli,
        "summarize_telemetry",
        lambda _root: MainTelemetrySummary(
            schema_version="phase13_main_telemetry_summary_v1",
            unit_count=1,
            provider_call_count=7,
            transport_attempt_count=7,
            latency_ms=1,
            token_usage={},
            provider_cost_usd="0",
            realized_cost_krw=0,
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "no-call-construction-only")
    monkeypatch.setattr(sys, "argv", _run_argv(tmp_path, "orphan-telemetry", 0))

    with pytest.raises(SystemExit, match="MAIN_LIVE_TELEMETRY_LEDGER_MISMATCH"):
        phase13_main_live_cli.main()
