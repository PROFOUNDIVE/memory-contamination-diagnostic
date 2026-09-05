from __future__ import annotations

import fcntl
import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import assert_never

from memcontam.readiness.phase13_authority_files import read_regular_nofollow
from .phase13_v3_authority_models import FrozenModel
from .phase13_v3_request import STAGES, PackageBindingV3, ParentTrajectoryV3, RequestKeyV3
from .phase13_v3_cost_actual import reconcile_actual
from .phase13_v3_terminal_ledger import TerminalLedgerV3
from .phase13_v3_terminal_models import TerminalEvidenceError


class RequestIdentityReceiptV3(FrozenModel):
    binding: PackageBindingV3
    parents: tuple[ParentTrajectoryV3, ...]
    key: RequestKeyV3


@contextmanager
def request_lock(ledger: TerminalLedgerV3) -> Iterator[None]:
    with ledger.path.open("rb") as descriptor:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)


def terminal_parents(
    ledger: TerminalLedgerV3, binding: PackageBindingV3, parents: tuple[ParentTrajectoryV3, ...],
) -> frozenset[str]:
    failed: set[str] = set()
    for unit_id in ledger.binding.unit_ids:
        state = ledger.state(unit_id)
        path = ledger.path.parent / f"{unit_id}.identity.json"
        if state.revision == 0 and not path.exists():
            continue
        receipt = RequestIdentityReceiptV3.model_validate_json(read_regular_nofollow(path))
        if (receipt.binding != binding or receipt.parents != parents
                or receipt.key.dispatch_id != unit_id):
            raise TerminalEvidenceError()
        if state.kind in ("INPUT_ENVELOPE_OVERFLOW", "TERMINAL_TECHNICAL_MISSING",
                          "ATTEMPTED_PROVIDER_FAILURE", "AMBIGUOUS_ATTEMPT"):
            failed.add(receipt.key.parent_id)
    failed_prefixes = {parent.parent_id for parent in parents
                       if parent.parent_id in failed and parent.kind == "CLEAN_PREFIX"}
    failed.update(parent.parent_id for parent in parents if parent.prefix_parent_id in failed_prefixes)
    return frozenset(failed)


def recover_requests(ledger: TerminalLedgerV3) -> None:
    reopened = TerminalLedgerV3.open(ledger.path, ledger.binding)
    proof = hashlib.sha256(b"phase13-restart-v3\n" + b"\n".join(reopened.rows())).hexdigest()
    for unit_id in ledger.binding.unit_ids:
        state = reopened.state(unit_id)
        if state.kind == "REQUEST_COMPILED" and state.compiled is not None:
            receipt = RequestIdentityReceiptV3.model_validate_json(read_regular_nofollow(
                ledger.path.parent / f"{unit_id}.identity.json",
            ))
            if state.compiled.token_count > STAGES[receipt.key.stage][0]:
                reopened.append({
                    "schema_version": "phase13_main_dispatch_evidence_v3", "unit_id": unit_id,
                    "revision": state.revision + 1, "previous_hash": state.event_hash,
                    "kind": "INPUT_ENVELOPE_OVERFLOW", "compiled": state.compiled.model_dump(mode="json"),
                    "failure_code": "MAIN_INPUT_ENVELOPE_EXCEEDED", "transport_attempts": 0,
                    "realized_cost_krw": 0,
                })
        match reopened.state(unit_id).kind:
            case "DISPATCH_INTENT_PERSISTED" | "REQUEST_COMPILED" | "ATTEMPT_STARTED" | "INPUT_ENVELOPE_OVERFLOW":
                reopened.recover(unit_id, proof)
            case "PENDING" | "COMPLETED" | "TERMINAL_TECHNICAL_MISSING" | "ATTEMPTED_PROVIDER_FAILURE" | "AMBIGUOUS_ATTEMPT":
                continue
            case unreachable:
                assert_never(unreachable)


def require_known_costs(ledger: TerminalLedgerV3) -> None:
    for unit_id in ledger.binding.unit_ids:
        cost = ledger.state(unit_id).attempted_cost
        if cost is not None:
            reconcile_actual(cost)
