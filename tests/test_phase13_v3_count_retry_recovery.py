from __future__ import annotations

import hashlib
import json
from typing import Literal, assert_never

import pytest
from pydantic import JsonValue, ValidationError

from memcontam.clients.base import LLMResponse
from memcontam.readiness.phase13_main_request_dispatch import (
    DispatchTechnicalFailureV3,
    ProductionRequestDispatcherV3,
)
from memcontam.readiness.phase13_v3_count import count_costs_krw, count_recovery_gate
from memcontam.readiness.phase13_v3_request import RequestKeyV3
from memcontam.readiness.phase13_v3_terminal_ledger import TerminalLedgerV3
from memcontam.readiness.phase13_v3_terminal_models import TerminalEvidenceError

from .phase13_count_fake import CountedProvider
from .test_phase13_v3_envelope_gate import api as api, rig as rig
from .test_phase13_v3_provider_count import CountCrash, counted as counted
from .test_phase13_v3_request_client import RetryableTimeout


def test_count_gate_batches_reads_and_rechecks_external_tamper(
    counted, monkeypatch: pytest.MonkeyPatch,
) -> None:
    counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda result: result.content)
    ledger = counted.ledger
    read = TerminalLedgerV3.count_record
    reads = 0

    def observed(self: TerminalLedgerV3, unit_id: str, role: str) -> bytes | None:
        nonlocal reads
        reads += 1
        return read(self, unit_id, role)

    monkeypatch.setattr(TerminalLedgerV3, "count_record", observed)
    count_recovery_gate(ledger)
    count_recovery_gate(ledger)
    assert reads == 0

    with ledger.connection() as connection:
        connection.execute("UPDATE provider_counts_v1 SET raw=? WHERE role='count-receipt'",
                           (b"{}",))
    with pytest.raises((TerminalEvidenceError, ValidationError)):
        count_recovery_gate(ledger)


def test_count_gate_rejects_tamper_even_after_local_count_write(counted) -> None:
    counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda result: result.content)
    ledger = counted.ledger
    count_recovery_gate(ledger)
    count_costs_krw(ledger)
    with ledger.connection() as connection:
        connection.execute("UPDATE provider_counts_v1 SET raw=? WHERE role='count-receipt'",
                           (b"{}",))
    ledger.append_count_record(counted.key.dispatch_id, "count-failure", b"{}")
    with pytest.raises((TerminalEvidenceError, ValidationError)):
        count_recovery_gate(ledger)
    with pytest.raises((TerminalEvidenceError, ValidationError)):
        count_costs_krw(ledger)


@pytest.mark.parametrize("rowid", [0, -1])
def test_count_gate_includes_nonpositive_sqlite_rowids(counted, rowid: int) -> None:
    counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda result: result.content)
    ledger = counted.ledger
    with ledger.connection() as connection:
        connection.execute("UPDATE provider_counts_v1 SET rowid=? WHERE role='count-started'",
                           (rowid,))
        connection.execute("UPDATE provider_counts_v1 SET raw=? WHERE role='count-receipt'",
                           (b"{}",))
    with pytest.raises((TerminalEvidenceError, ValidationError)):
        count_recovery_gate(ledger)
    with pytest.raises((TerminalEvidenceError, ValidationError)):
        count_costs_krw(ledger)


def test_cumulative_count_cost_batches_reads_without_skipping_tamper(
    counted, monkeypatch: pytest.MonkeyPatch,
) -> None:
    counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda result: result.content)
    ledger = counted.ledger
    expected = count_costs_krw(ledger)
    read = TerminalLedgerV3.count_record
    reads = 0

    def observed(self: TerminalLedgerV3, unit_id: str, role: str) -> bytes | None:
        nonlocal reads
        reads += 1
        return read(self, unit_id, role)

    monkeypatch.setattr(TerminalLedgerV3, "count_record", observed)
    assert count_costs_krw(ledger) == expected
    assert count_costs_krw(ledger) == expected
    assert reads == 0
    with ledger.connection() as connection:
        connection.execute("UPDATE provider_counts_v1 SET raw=? WHERE role='count-receipt'",
                           (b"{}",))
    with pytest.raises((TerminalEvidenceError, ValidationError)):
        count_costs_krw(ledger)


def test_uncounted_compile_recovers_pending_before_authoritative_overflow(rig, monkeypatch):
    rig.seen.count = 379
    original = rig.dispatcher._append

    def append(key, kind, extra=None):
        original(key, kind, extra)
        if kind == "REQUEST_COMPILED":
            raise CountCrash()

    monkeypatch.setattr(rig.dispatcher, "_append", append)
    with pytest.raises(CountCrash):
        rig.dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    reopened = TerminalLedgerV3.open(rig.ledger.path, rig.ledger.binding)
    try:
        resumed = ProductionRequestDispatcherV3(reopened, rig.binding, rig.parents,
            provider_factory=rig.dispatcher._factory)
        resumed.recover()
        assert reopened.state(rig.keys[0].dispatch_id).kind == "PENDING"
        assert rig.seen.requests == rig.seen.constructors == 0
        with pytest.raises(DispatchTechnicalFailureV3, match="MAIN_INPUT_ENVELOPE_EXCEEDED"):
            resumed.dispatch(rig.keys[0], rig.material, rig.semantic)
        assert reopened.state(rig.keys[0].dispatch_id).kind == "TERMINAL_TECHNICAL_MISSING"
        assert (rig.seen.requests, rig.seen.constructors) == (0, 1)
    finally:
        reopened.close()


@pytest.mark.parametrize("local,actual", [(379, 377), (379, 378), (378, 379)])
def test_uncounted_restart_uses_provider_gate_when_compilation_was_durable(
    counted, monkeypatch: pytest.MonkeyPatch, local: int, actual: int,
) -> None:
    import memcontam.readiness.phase13_main_request_dispatch as dispatch

    monkeypatch.setattr(dispatch, "count_prompt_tokens", lambda *_: local)
    counted.seen.tokens = actual
    append = counted.dispatcher._append

    def crash(key: RequestKeyV3, kind: str, extra: dict[str, JsonValue] | None = None) -> None:
        append(key, kind, extra)
        if kind == "REQUEST_COMPILED":
            raise CountCrash()

    monkeypatch.setattr(counted.dispatcher, "_append", crash)
    with pytest.raises(CountCrash):
        counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda result: result.content)
    original = counted.ledger.state(counted.key.dispatch_id).compiled
    assert original is not None and original.token_count == local
    assert counted.ledger.count_record(counted.key.dispatch_id, "count-started") is None
    reopened = TerminalLedgerV3.open(counted.ledger.path, counted.ledger.binding)
    try:
        resumed = ProductionRequestDispatcherV3(reopened, counted.dispatcher.binding,
            counted.dispatcher.parents, provider_factory=lambda _: counted.provider)

        resumed.recover()

        assert reopened.state(counted.key.dispatch_id).kind == "PENDING"
        assert resumed.terminal_parents == frozenset()
        assert (counted.seen.counts, counted.seen.creates) == (0, 0)
        if actual <= 378:
            assert resumed.dispatch(counted.key, lambda: counted.material,
                                    lambda result: result.content) == "final: 24"
        else:
            with pytest.raises(DispatchTechnicalFailureV3, match="MAIN_INPUT_ENVELOPE_EXCEEDED"):
                resumed.dispatch(counted.key, lambda: counted.material, lambda result: result.content)
            assert resumed.terminal_parents == frozenset({counted.key.parent_id})
        assert reopened.state(counted.key.dispatch_id).compiled == original
        assert (counted.seen.counts, counted.seen.creates) == (1, int(actual <= 378))
        before = reopened.rows()
        cost = reopened.realized_cost_krw()
        resumed.recover()
        assert reopened.rows() == before
        assert reopened.realized_cost_krw() == cost
        assert cost >= 2
    finally:
        reopened.close()


@pytest.mark.parametrize("boundary", ["count-started", "count-receipt"])
def test_authoritative_count_boundary_holds_without_new_transport_after_reopen(
    counted, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    publish = counted.dispatcher._publish_bytes

    def crash(key: RequestKeyV3, role: str, raw: bytes) -> None:
        publish(key, role, raw)
        if role == boundary:
            raise CountCrash()

    monkeypatch.setattr(counted.dispatcher, "_publish_bytes", crash)
    with pytest.raises(CountCrash):
        counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda result: result.content)
    reopened = TerminalLedgerV3.open(counted.ledger.path, counted.ledger.binding)
    try:
        resumed = ProductionRequestDispatcherV3(reopened, counted.dispatcher.binding,
            counted.dispatcher.parents, provider_factory=lambda _: counted.provider)
        before = reopened.rows()

        with pytest.raises(TerminalEvidenceError, match="MAIN_COUNT.*RECONCILIATION_REQUIRED"):
            resumed.recover()

        assert reopened.rows() == before
        assert resumed.terminal_parents == frozenset()
        with pytest.raises(TerminalEvidenceError):
            resumed.dispatch(counted.key, lambda: counted.material, lambda result: result.content)
        assert (counted.seen.counts, counted.seen.creates) == (int(boundary == "count-receipt"), 0)
    finally:
        reopened.close()


@pytest.mark.parametrize("mode", ["missing-append", "null-append", "null-reopen"])
def test_attempt_identity_is_required_when_initial_marker_is_persisted(
    counted, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    append = counted.dispatcher._append

    def crash(key: RequestKeyV3, kind: str, extra: dict[str, JsonValue] | None = None) -> None:
        if kind == "ATTEMPT_STARTED":
            raise CountCrash()
        append(key, kind, extra)

    monkeypatch.setattr(counted.dispatcher, "_append", crash)
    with pytest.raises(CountCrash):
        counted.dispatcher.dispatch(counted.key, lambda: counted.material, lambda result: result.content)
    state = counted.ledger.state(counted.key.dispatch_id)
    assert state.compiled is not None
    event = {"schema_version": "phase13_main_dispatch_evidence_v3", "kind": "ATTEMPT_STARTED",
        "unit_id": counted.key.dispatch_id, "revision": state.revision + 1,
        "previous_hash": state.event_hash, "compiled": state.compiled.model_dump(mode="json"),
        "attempt_index": 0}
    if mode != "missing-append":
        event["attempt_id"] = None
    before = counted.ledger.rows()

    if mode == "null-reopen":
        raw = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with counted.ledger.connection() as connection:
            connection.execute("INSERT INTO events(raw, event_hash) VALUES (?, ?)",
                               (raw, hashlib.sha256(raw).hexdigest()))
        with pytest.raises(TerminalEvidenceError):
            TerminalLedgerV3.open(counted.ledger.path, counted.ledger.binding)
    else:
        with pytest.raises(TerminalEvidenceError):
            counted.ledger.append(event)
        assert counted.ledger.rows() == before
    assert (counted.seen.counts, counted.seen.creates) == (1, 0)


@pytest.mark.parametrize("exhausted", [False, True])
def test_prefix_dependency_disposition_is_final_when_entitled_retry_reopens(
    rig, monkeypatch: pytest.MonkeyPatch, exhausted: bool,
) -> None:
    attempts = []

    class Provider(CountedProvider):
        def send_compiled_v3(self, compiled, before_request):
            before_request()
            attempts.append(compiled.request_bytes)
            if len(attempts) == 1 or exhausted:
                raise RetryableTimeout()
            return LLMResponse("final: 24", {"status": "completed", "model": "gpt-5.6-luna",
                "service_tier": "default", "usage": {"input_tokens": 1, "output_tokens": 1}}, {}, 0)

    dispatcher = ProductionRequestDispatcherV3(rig.ledger, rig.binding, rig.parents,
        provider_factory=lambda _: Provider(),
        retry_entitlements=frozenset({rig.keys[0].dispatch_id}))
    append = dispatcher._append

    def crash(key: RequestKeyV3, kind: str, extra: dict[str, JsonValue] | None = None) -> None:
        append(key, kind, extra)
        if kind == "RETRYABLE_ATTEMPT_FAILURE":
            raise CountCrash()

    monkeypatch.setattr(dispatcher, "_append", crash)
    with pytest.raises(CountCrash):
        dispatcher.dispatch(rig.keys[0], rig.material, rig.semantic)
    material = dispatcher.compiled_request(rig.keys[0]).material
    reopened = TerminalLedgerV3.open(rig.ledger.path, rig.ledger.binding)
    try:
        resumed = ProductionRequestDispatcherV3(reopened, rig.binding, rig.parents,
            provider_factory=lambda _: Provider(), retry_entitlements=dispatcher.retry_entitlements)

        resumed.recover()

        assert resumed.terminal_parents == frozenset()
        assert reopened.state(rig.keys[0].dispatch_id).kind == "RETRYABLE_ATTEMPT_FAILURE"
        assert len(attempts) == 1
        reopened.reconcile_cost(rig.keys[0].dispatch_id,
            {"usage": {"input_tokens": 1, "output_tokens": 0}}, "f" * 64)
        if exhausted:
            with pytest.raises(DispatchTechnicalFailureV3):
                resumed.dispatch(rig.keys[0], lambda: material, rig.semantic)
            assert resumed.terminal_parents == frozenset({"a" * 64, "d" * 64})
            for key in rig.keys[:4]:
                with pytest.raises(DispatchTechnicalFailureV3, match="MAIN_TRAJECTORY_TERMINAL"):
                    resumed.dispatch(key, lambda: material, rig.semantic)
        else:
            assert resumed.dispatch(rig.keys[0], lambda: material, rig.semantic) == "final: 24"
            assert resumed.terminal_parents == frozenset()
        assert len(attempts) == 2 and attempts[0] == attempts[1]
        assert len(reopened.count_records("count-started")) == 1
        markers = [json.loads(raw) for raw in reopened.rows() if json.loads(raw)["kind"] == "ATTEMPT_STARTED"]
        assert [row["attempt_index"] for row in markers] == [0, 1]
        assert len({row["attempt_id"] for row in markers}) == 2
        before = reopened.rows()
        resumed.recover()
        assert reopened.rows() == before
        assert len(attempts) == 2
    finally:
        reopened.close()


@pytest.mark.parametrize("tamper", ["missing", "null", "duplicate", "mismatch"])
def test_retry_marker_tamper_rejects_canonical_replay_before_new_transport(
    counted, tamper: Literal["missing", "null", "duplicate", "mismatch"],
) -> None:
    counted.seen.generation_failure = RetryableTimeout()
    dispatcher = ProductionRequestDispatcherV3(counted.ledger, counted.dispatcher.binding,
        counted.dispatcher.parents, provider_factory=lambda _: counted.provider,
        retry_entitlements=frozenset({counted.key.dispatch_id}))
    dispatcher.dispatch(counted.key, lambda: counted.material, lambda result: result.content)
    markers = [json.loads(raw) for raw in counted.ledger.rows() if json.loads(raw)["kind"] == "ATTEMPT_STARTED"]
    changed = markers[1]
    match tamper:
        case "missing":
            del changed["attempt_id"]
        case "null":
            changed["attempt_id"] = None
        case "duplicate":
            changed["attempt_id"] = markers[0]["attempt_id"]
        case "mismatch":
            changed["attempt_id"] = "f" * 64
        case unreachable:
            assert_never(unreachable)
    raw = (json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with counted.ledger.connection() as connection:
        connection.execute("UPDATE events SET raw=?, event_hash=? WHERE sequence=?",
                           (raw, hashlib.sha256(raw).hexdigest(), changed["revision"]))

    with pytest.raises(TerminalEvidenceError):
        TerminalLedgerV3.open(counted.ledger.path, counted.ledger.binding)

    assert (counted.seen.counts, counted.seen.creates) == (1, 2)
