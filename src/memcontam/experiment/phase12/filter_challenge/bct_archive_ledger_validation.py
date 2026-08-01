from __future__ import annotations

from collections.abc import Mapping

from memcontam.experiment.phase12.filter_challenge.bct_archive_models import (
    LedgerError,
    ResourceBudget,
    resource_fits,
)
from memcontam.experiment.phase12.filter_challenge.bct_archive_storage import _string_value


def validate_ledger_records(
    records: list[dict[str, object]], stage_caps: Mapping[str, ResourceBudget]
) -> None:
    reservations: set[str] = set()
    settled: set[str] = set()
    for row in records:
        kind = _string_value(row["kind"])
        match kind:
            case "reserve":
                identifier = _string_value(row["reservation_id"])
                stage = _string_value(row["stage"])
                if identifier in reservations or stage not in stage_caps:
                    raise LedgerError("LEDGER_CHAIN_INVALID")
                if budget_value(row["resources"]) != stage_caps[stage]:
                    raise LedgerError("LEDGER_CHAIN_INVALID")
                reservations.add(identifier)
            case "settle":
                identifier = _string_value(row["reservation_id"])
                if (
                    identifier not in reservations
                    or identifier in settled
                    or not resource_fits(_reservation_resources(records, identifier), budget_value(row["resources"]))
                ):
                    raise LedgerError("LEDGER_CHAIN_INVALID")
                settled.add(identifier)
            case "invalidate":
                if _string_value(row["reservation_id"]) not in reservations:
                    raise LedgerError("LEDGER_CHAIN_INVALID")
            case _:
                raise LedgerError("LEDGER_CHAIN_INVALID")


def _integer_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LedgerError("LEDGER_CHAIN_INVALID")
    return value


def budget_value(value: object) -> ResourceBudget:
    if not isinstance(value, dict):
        raise LedgerError("LEDGER_CHAIN_INVALID")
    budget = ResourceBudget(
        *(
            _integer_value(value.get(name))
            for name in ("calls", "input_tokens", "output_tokens", "microusd", "wall_seconds")
        )
    )
    if min(budget.calls, budget.input_tokens, budget.output_tokens, budget.microusd, budget.wall_seconds) < 0:
        raise LedgerError("LEDGER_CHAIN_INVALID")
    return budget


def _reservation_resources(records: list[dict[str, object]], identifier: str) -> ResourceBudget:
    return next(
        budget_value(row["resources"])
        for row in records
        if row["kind"] == "reserve" and row["reservation_id"] == identifier
    )
