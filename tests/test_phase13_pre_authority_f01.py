from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from memcontam.clients.base import LLMClient
from memcontam.evaluation.phase13_observability_registration import (
    ObservabilityRegistrationPacket,
)
from memcontam.memory.checkpoint_v3 import Phase12Checkpoint
from memcontam.readiness import phase13_main_live_runtime
from memcontam.readiness.phase13_main_live_dispatch import MainUnitDispatchOutput
from memcontam.readiness.phase13_main_live_runtime import ProductionMainRuntime
from memcontam.readiness.phase13_main_production import ProductionObject
from memcontam.readiness.phase13_main_production_backend import (
    MainProductionBackend,
    OrdinaryRuntimeRequest,
    PrefixRuntimeOutput,
)
from memcontam.readiness.phase13_observability_models import Phase13ObservabilityFixture
from memcontam.readiness.phase13_production_observability import (
    ProductionObservabilityArchive,
    ProductionTrialRecord,
    conformance_archive,
    validate_production_archive,
)


ROOT = Path(__file__).resolve().parents[1]
OBSERVABILITY = ROOT / "data/phase13/observability"


class _Marker:
    pass


def _archive() -> tuple[ProductionObservabilityArchive, ObservabilityRegistrationPacket]:
    packet_path = OBSERVABILITY / "registration_packet_v1.json"
    packet = ObservabilityRegistrationPacket.model_validate_json(packet_path.read_bytes())
    fixture = Phase13ObservabilityFixture.model_validate_json(
        (OBSERVABILITY / "fixture_v1.json").read_bytes()
    )
    return conformance_archive(fixture, hashlib.sha256(packet_path.read_bytes()).hexdigest()), packet


def test_existing_production_archive_round_trips_and_validates_registered_fixture() -> None:
    archive, packet = _archive()

    restored = ProductionObservabilityArchive.model_validate_json(archive.model_dump_json())
    report = validate_production_archive(
        restored,
        packet,
        restored.registration_packet_sha256,
    )

    assert restored == archive
    assert report.record_count == len(archive.records)


def test_archive_record_retains_the_parsed_answer_field() -> None:
    archive, _packet = _archive()

    assert "parsed_answer" in ProductionTrialRecord.model_fields
    assert all(record.parsed_answer is None for record in archive.records)
    assert "parsed_answer" not in type(archive.records[0].evidence).model_fields


def _unit(*, memory_bearing: bool) -> ProductionObject:
    return ProductionObject(
        sequence=0,
        unit_id="1" * 64,
        kind="MEMORY_BEARING" if memory_bearing else "NO_MEMORY_SINGLETON",
        seed=0,
        task="game24",
        memory_baseline="fh_bounded" if memory_bearing else None,
        arm="positive" if memory_bearing else "NOT_APPLICABLE",
        prefix_unit_id="2" * 64 if memory_bearing else None,
        projected_cost_krw=0,
    )


def _execute_with_observed_archive(
    monkeypatch: pytest.MonkeyPatch,
    request: OrdinaryRuntimeRequest,
) -> tuple[_Marker, list[str], _Marker]:
    events: list[str] = []
    archive = _Marker()
    condensed = _Marker()
    runtime = ProductionMainRuntime(
        ROOT,
        ROOT / ".cache",
        client=cast(LLMClient, object()),
    )

    monkeypatch.setattr(ProductionMainRuntime, "_tasks", lambda *_args, **_kwargs: (object(),))
    monkeypatch.setattr(ProductionMainRuntime, "_context", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(ProductionMainRuntime, "_embedder", lambda *_args: object())
    monkeypatch.setattr(ProductionMainRuntime, "_configs", lambda *_args: {})
    monkeypatch.setattr(ProductionMainRuntime, "_initial_states", lambda *_args: {})
    monkeypatch.setattr(
        phase13_main_live_runtime,
        "build_live_reduced_main_branches",
        lambda **_kwargs: SimpleNamespace(arms={request.arm: object()}),
    )
    monkeypatch.setattr(
        phase13_main_live_runtime,
        "ProspectiveOrdinaryRun",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        phase13_main_live_runtime,
        "execute_prospective_ordinary",
        lambda _run: SimpleNamespace(trials=tuple(range(50))),
    )
    monkeypatch.setattr(
        phase13_main_live_runtime,
        "production_identity",
        lambda _unit: SimpleNamespace(registration_packet_sha256="1" * 64),
    )

    def create_archive(*_args) -> _Marker:
        events.append("archive-created")
        return archive

    def validate_archive(value: _Marker, *_args) -> None:
        assert value is archive
        events.append("archive-validated")

    def create_dispatch(_unit, trials, _identity, observed_archive) -> _Marker:
        assert len(trials) == 50
        assert observed_archive is archive
        events.append("condensed-dispatch-returned")
        return condensed

    monkeypatch.setattr(phase13_main_live_runtime, "production_archive_from_ordinary", create_archive)
    monkeypatch.setattr(phase13_main_live_runtime, "validate_production_archive", validate_archive)
    monkeypatch.setattr(phase13_main_live_runtime, "dispatch_output", create_dispatch)

    return cast(_Marker, runtime.execute_ordinary(request)), events, condensed


def test_memory_runtime_validates_archive_then_returns_condensed_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit = _unit(memory_bearing=True)
    output, events, condensed = _execute_with_observed_archive(
        monkeypatch,
        OrdinaryRuntimeRequest(
            unit,
            "fh_bounded",
            "contam",
            "positive",
            unit.prefix_unit_id,
            cast(Phase12Checkpoint, object()),
        ),
    )

    assert output is condensed
    assert events == ["archive-created", "archive-validated", "condensed-dispatch-returned"]


class _CapturedNoMemRequest(RuntimeError):
    pass


def test_nomem_backend_supplies_no_checkpoint_and_runtime_persists_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[OrdinaryRuntimeRequest] = []

    def unused_prefix(_unit: ProductionObject) -> PrefixRuntimeOutput:
        raise AssertionError("NoMem must not execute the prefix runtime")

    def capture_nomem(request: OrdinaryRuntimeRequest) -> MainUnitDispatchOutput:
        captured.append(request)
        raise _CapturedNoMemRequest

    unit = _unit(memory_bearing=False)
    backend = MainProductionBackend(tmp_path, unused_prefix, capture_nomem)
    with pytest.raises(_CapturedNoMemRequest):
        backend(unit)

    request = captured[0]
    assert request.baseline == "nomem"
    assert request.checkpoint is None
    output, events, condensed = _execute_with_observed_archive(monkeypatch, request)

    assert output is condensed
    assert events == ["archive-created", "archive-validated", "condensed-dispatch-returned"]
