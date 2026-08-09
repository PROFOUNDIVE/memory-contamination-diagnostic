from __future__ import annotations

# allow: SIZE_OK — Task 4's fixed QA argv keeps the complete ledger matrix in one module.

from dataclasses import replace
from pathlib import Path

import pytest

from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    RootlessContractError,
    public_key_from_seed,
    verify_object_signature,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_ledger import (
    BCT_CAP_NANOUSD,
    CUMULATIVE_CAP_NANOUSD,
    RESERVATION_NANOUSD,
    SCREENING_CAP_NANOUSD,
    GlobalLedger,
    LedgerReservation,
    ProviderUsage,
    actual_cost_nanousd,
)


SEED = bytes(range(32))
NOW = "2026-08-09T12:00:00Z"


def _reservation(slot_id: str = "slot-001") -> LedgerReservation:
    return LedgerReservation(
        slot_id=slot_id,
        idempotency_key=f"idem-{slot_id}",
        compiler_sha256="1" * 64,
        static_input_sha256="2" * 64,
        predecessor_receipt_sha256=None,
        request_sha256="3" * 64,
        request_bytes=128,
        compiled_input_tokens=100,
    )


def _ledger(root: Path, *, stage: str = "screening") -> GlobalLedger:
    root.mkdir(mode=0o700)
    return GlobalLedger(root, SEED, "attempt-001", stage)


def test_reservation_and_settlement_form_a_signed_immutable_chain(tmp_path: Path) -> None:
    # Given: an empty cooperative ledger.
    ledger = _ledger(tmp_path / "state")

    # When: one slot is reserved and settled with independently reported usage.
    reservation = ledger.reserve(_reservation(), NOW)
    settlement = ledger.settle(
        reservation.record_sha256,
        "4" * 64,
        "5" * 64,
        ProviderUsage(120, 20, 10, 130),
        "2026-08-09T12:00:01Z",
    )

    # Then: sequence, predecessor, counters, signatures, and exact cost agree.
    assert reservation.record["sequence"] == 0
    assert settlement.record["sequence"] == 1
    assert settlement.record["previous_record_sha256"] == reservation.record_sha256
    assert settlement.head["cumulative_issued"] == 1
    assert settlement.head["cumulative_settled_nanousd"] == 400_000
    public_key = public_key_from_seed(SEED)
    for domain, value in (
        ("ledger-record-v1", reservation.record),
        ("ledger-head-v1", reservation.head),
        ("ledger-record-v1", settlement.record),
        ("ledger-head-v1", settlement.head),
    ):
        unsigned = dict(value)
        signature = unsigned.pop("signature")
        assert isinstance(signature, str)
        verify_object_signature(public_key, domain, unsigned, signature)


def test_budget_formula_and_stage_cumulative_caps_never_overshoot(tmp_path: Path) -> None:
    # Given: the frozen rate card and a screening ledger near its cap.
    assert RESERVATION_NANOUSD == 16_640_000
    assert SCREENING_CAP_NANOUSD == 2_000_000_000
    assert BCT_CAP_NANOUSD == 8_000_000_000
    assert CUMULATIVE_CAP_NANOUSD == 10_000_000_000
    ledger = _ledger(tmp_path / "state")

    # When: 90 registered screening reservations are appended.
    for index in range(90):
        ledger.reserve(_reservation(f"slot-{index:03d}"), NOW)

    # Then: the registered matrix fits, while the call-count and cap boundaries fail closed.
    assert ledger.snapshot().active_unsettled_nanousd == 1_497_600_000
    with pytest.raises(RootlessContractError, match="ROOTLESS_BUDGET_CAP_EXCEEDED"):
        ledger.reserve(_reservation("slot-090"), NOW)


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (ProviderUsage(4096, 0, 640, 4736), 16_640_000),
        (ProviderUsage(100, 100, 1, 101), 260_000),
    ],
)
def test_usage_cost_charges_cached_input_at_full_rate(usage: ProviderUsage, expected: int) -> None:
    # Given/When/Then: exact integer nanousd arithmetic follows the frozen rate card.
    assert actual_cost_nanousd(usage) == expected


@pytest.mark.parametrize(
    "usage",
    [ProviderUsage(4097, 0, 0, 4097), ProviderUsage(1, 2, 0, 1), ProviderUsage(1, 0, 641, 642)],
)
def test_invalid_usage_cannot_settle_a_reservation(tmp_path: Path, usage: ProviderUsage) -> None:
    # Given: a durable reservation.
    ledger = _ledger(tmp_path / "state")
    reservation = ledger.reserve(_reservation(), NOW)

    # When/Then: invalid or above-reservation usage retains the reservation.
    with pytest.raises(RootlessContractError, match="ROOTLESS_USAGE_INVALID"):
        ledger.settle(reservation.record_sha256, "4" * 64, "5" * 64, usage, NOW)
    assert ledger.snapshot().active_unsettled_nanousd == RESERVATION_NANOUSD


@pytest.mark.parametrize(
    ("reason", "compile_status", "has_request"),
    [
        ("ROOTLESS_INPUT_CAP_EXCEEDED", "compiled", True),
        ("DOWNSTREAM_NOT_ISSUED_AFTER_PARSE_FAILURE", "blocked_predecessor", False),
        ("DOWNSTREAM_NOT_ISSUED_AFTER_PREDECESSOR_FAILURE", "blocked_predecessor", False),
    ],
)
def test_not_issued_branches_are_accounted_without_reservation(
    tmp_path: Path, reason: str, compile_status: str, has_request: bool
) -> None:
    # Given: an empty ledger and one scheduled slot.
    ledger = _ledger(tmp_path / reason)
    request = _reservation()
    if not has_request:
        request = replace(request, predecessor_receipt_sha256="6" * 64)

    # When: the broker records the slot as not issued.
    result = ledger.not_issued(
        request,
        reason,
        NOW,
        compile_status=compile_status,
        include_request=has_request,
    )

    # Then: no call or reservation is charged and the branch is immutable.
    assert result.record["record_kind"] == "not_issued"
    assert result.record["reserved_nanousd"] == 0
    assert result.head["cumulative_not_issued"] == 1
    assert ledger.snapshot().active_unsettled_nanousd == 0


def test_terminal_record_closes_stage_and_is_idempotently_recovered(tmp_path: Path) -> None:
    # Given: one settled slot and one downstream not-issued slot.
    root = tmp_path / "state"
    ledger = _ledger(root)
    reservation = ledger.reserve(_reservation(), NOW)
    ledger.settle(
        reservation.record_sha256,
        "4" * 64,
        "5" * 64,
        ProviderUsage(10, 0, 1, 11),
        NOW,
    )
    ledger.not_issued(
        replace(_reservation("slot-002"), predecessor_receipt_sha256="6" * 64),
        "DOWNSTREAM_NOT_ISSUED_AFTER_PREDECESSOR_FAILURE",
        NOW,
        compile_status="blocked_predecessor",
        include_request=False,
    )

    # When: the stage terminal is appended and the ledger is reopened.
    terminal = ledger.terminal(
        terminal_status="interrupted",
        reason_code="ROOTLESS_INTERRUPTED_UNCLEAN",
        registered_slots=2,
        created_at=NOW,
    )
    recovered = GlobalLedger(root, SEED, "attempt-001", "screening")

    # Then: recovery preserves the unique head and duplicate terminalization is idempotent.
    assert recovered.head_sha256 == terminal.head_sha256
    assert recovered.terminal(
        terminal_status="interrupted",
        reason_code="ROOTLESS_INTERRUPTED_UNCLEAN",
        registered_slots=2,
        created_at=NOW,
    ).head_sha256 == terminal.head_sha256
    with pytest.raises(RootlessContractError, match="ROOTLESS_LEDGER_TERMINAL"):
        recovered.reserve(_reservation("slot-003"), NOW)


def test_missing_head_is_reconstructed_but_tampered_record_is_rejected(tmp_path: Path) -> None:
    # Given: one record whose derived head was lost at an atomic boundary.
    root = tmp_path / "state"
    ledger = _ledger(root)
    result = ledger.reserve(_reservation(), NOW)
    result.head_path.unlink()

    # When: recovery reopens the unique valid record prefix.
    recovered = GlobalLedger(root, SEED, "attempt-001", "screening")

    # Then: only the missing head is reconstructed; record tampering is never repaired.
    assert recovered.head_sha256 == result.head_sha256
    raw = result.record_path.read_bytes()
    result.record_path.write_bytes(raw.replace(b'"request_bytes":128', b'"request_bytes":129'))
    with pytest.raises(RootlessContractError, match="ROOTLESS_LEDGER_INVALID"):
        GlobalLedger(root, SEED, "attempt-001", "screening")
