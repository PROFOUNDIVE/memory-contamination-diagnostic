from __future__ import annotations

import sys
from pathlib import Path

import pytest

from memcontam.readiness import phase13_main_live_cli
from memcontam.readiness.phase13_main_production import ProductionObject
from memcontam.readiness.phase13_main_runner import MainRunBinding, MainRunLedger


ROOT = Path(__file__).resolve().parents[1]
P5 = ROOT / "data/phase13/main/mr_p5/execution_package_v1.json"
P6 = ROOT / "data/phase13/main/mr_p6/authorized_execution_v1.json"


class _ProviderFreeRuntime:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def preflight(self, _units) -> None:
        pass

    def execute_prefix(self, _unit) -> None:
        raise AssertionError("provider-free test replaces the production backend")

    def execute_ordinary(self, _request) -> None:
        raise AssertionError("provider-free test replaces the production backend")


class _PostIntentFailure(ValueError):
    pass


class _PostIntentFailingBackend:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def __call__(self, _unit) -> None:
        raise _PostIntentFailure("provider-free post-intent runtime failure")


def test_post_intent_value_error_is_currently_collapsed_to_preflight_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    unit = ProductionObject(
        sequence=0,
        unit_id="1" * 64,
        kind="NO_MEMORY_SINGLETON",
        seed=0,
        task="provider-free-g01",
        memory_baseline=None,
        arm="NOT_APPLICABLE",
        prefix_unit_id=None,
        projected_cost_krw=0,
    )
    ledger = MainRunLedger.create(
        tmp_path / "main-run-v1.sqlite3",
        MainRunBinding(
            package_id="provider-free-g01",
            package_sha256="2" * 64,
            package_hash="3" * 64,
            authorization_id="provider-free-g01",
            authorization_sha256="4" * 64,
            authorization_hash="5" * 64,
            runner_sha256="6" * 64,
        ),
        (unit,),
    )

    monkeypatch.setattr(phase13_main_live_cli, "ProductionMainRuntime", _ProviderFreeRuntime)
    monkeypatch.setattr(phase13_main_live_cli, "MainProductionBackend", _PostIntentFailingBackend)
    monkeypatch.setattr(
        phase13_main_live_cli,
        "DurableMainDispatch",
        lambda _root, backend: backend,
    )
    monkeypatch.setattr(
        phase13_main_live_cli,
        "build_production_objects",
        lambda _package: (unit,),
    )
    monkeypatch.setattr(
        phase13_main_live_cli,
        "prepare_main_run",
        lambda _request: ledger,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase13-main-a-live",
            "run",
            "--repository-root",
            str(ROOT),
            "--package",
            str(P5),
            "--authorization",
            str(P6),
            "--expected-authorization-sha256",
            "0" * 64,
            "--run-root",
            str(tmp_path),
            "--run-id",
            "provider-free-g01",
            "--tranche-ceiling-krw",
            "1",
            "--max-units",
            "1",
            "--allow-live-calls",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        phase13_main_live_cli.main()

    assert ledger.status().in_flight_count == 1
    assert ledger.in_flight_context(unit.unit_id).unit_id == unit.unit_id
    assert raised.value.code == "MAIN_LIVE_PREFLIGHT_INVALID"
