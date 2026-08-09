from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    parse_canonical_object,
    public_key_from_seed,
    verify_object_signature,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_execution import (
    SlotCompilation,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_firewall import (
    ROOTLESS_PROFILE_FORBIDDEN,
    has_forbidden_rootless_profile,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_ledger import GlobalLedger

PROFILE = "local_rootless_non_authoritative"


@dataclass(frozen=True, slots=True)
class RootlessArchiveValidation:
    stage: Literal["screening", "bct"]
    accounted_slots: int
    issued_slots: int
    not_issued_slots: int
    ordered_receipt_root_sha256: str


def validate_rootless_screening_archive(
    root: Path, slots: tuple[SlotCompilation, ...]
) -> RootlessArchiveValidation:
    return _validate(root, slots, "screening")


def validate_rootless_bct_archive(
    root: Path, slots: tuple[SlotCompilation, ...]
) -> RootlessArchiveValidation:
    return _validate(root, slots, "bct")


def reject_rootless_at_legacy_seam(raw: bytes | str) -> None:
    if has_forbidden_rootless_profile(raw):
        raise RootlessContractError(ROOTLESS_PROFILE_FORBIDDEN)


def _validate(
    root: Path,
    slots: tuple[SlotCompilation, ...],
    stage: Literal["screening", "bct"],
) -> RootlessArchiveValidation:
    try:
        if not slots or any(slot.stage != stage for slot in slots):
            raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID")
        attempt_id = slots[0].attempt_id
        fixture_id = root.name
        seed = hashlib.sha256(f"rootless-fixture:{fixture_id}".encode()).digest()
        public_key = public_key_from_seed(seed)
        slot_root = root / "attempts" / attempt_id / stage / "slots"
        observed_names = {path.name for path in slot_root.iterdir()}
        if observed_names != {slot.slot_id for slot in slots}:
            raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID")
        receipt_hashes: dict[str, str] = {}
        ordered_leaves = bytearray()
        issued = 0
        not_issued = 0
        receipts: list[dict[str, JsonValue]] = []
        for slot in slots:
            directory = slot_root / slot.slot_id
            receipt_raw = _read_regular(directory / "call-receipt.json")
            receipt = _verified(receipt_raw, public_key, "local-call-receipt-v1")
            receipts.append(receipt)
            predecessor = slot.predecessor_slot_ids[0] if slot.predecessor_slot_ids else None
            predecessor_hash = receipt_hashes.get(predecessor) if predecessor is not None else None
            if (
                receipt.get("profile") != PROFILE
                or receipt.get("attempt_id") != attempt_id
                or receipt.get("stage") != stage
                or receipt.get("slot_id") != slot.slot_id
                or receipt.get("compiler_sha256") != slot.compiler_sha256
                or receipt.get("static_input_sha256") != slot.static_input_sha256
                or receipt.get("predecessor_receipt_sha256") != predecessor_hash
                or receipt.get("scientific_replicate") != slot.scientific_replicate
                or receipt.get("executor_replicate_id") != slot.executor_replicate_id
            ):
                raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID")
            if receipt.get("issued") is True:
                issued += 1
                _validate_issued(directory, receipt, public_key)
            elif receipt.get("issued") is False:
                not_issued += 1
                if (
                    receipt.get("compile_status") != "blocked_predecessor"
                    or predecessor_hash is None
                    or receipt.get("not_issued_record_sha256") is None
                    or receipt.get("typed_outcome_sha256") is not None
                ):
                    raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID")
            else:
                raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID")
            receipt_hash = hashlib.sha256(receipt_raw).hexdigest()
            receipt_hashes[slot.slot_id] = receipt_hash
            leaf = hashlib.sha256(
                slot.slot_id.encode("utf-8") + b"\0" + bytes.fromhex(receipt_hash)
            ).digest()
            ordered_leaves.extend(leaf)
        ledger = GlobalLedger(root, seed, attempt_id, stage)
        snapshot = ledger.snapshot()
        if snapshot.cumulative_issued != issued or snapshot.cumulative_not_issued != not_issued:
            raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID")
        _validate_ledger_references(root, receipts)
        return RootlessArchiveValidation(
            stage,
            len(slots),
            issued,
            not_issued,
            hashlib.sha256(ordered_leaves).hexdigest(),
        )
    except (OSError, KeyError, TypeError, ValueError, RootlessContractError) as error:
        if isinstance(error, RootlessContractError) and error.code == ROOTLESS_PROFILE_FORBIDDEN:
            raise
        raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID") from error


def _validate_issued(
    directory: Path, receipt: dict[str, JsonValue], public_key: bytes
) -> None:
    required_files = {
        "request.bin",
        "response.headers",
        "response.body",
        "archive-manifest.json",
        "typed-outcome.json",
        "call-receipt.json",
    }
    if {path.name for path in directory.iterdir()} != required_files:
        raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID")
    archive_raw = _read_regular(directory / "archive-manifest.json")
    outcome_raw = _read_regular(directory / "typed-outcome.json")
    archive = _verified(archive_raw, public_key, "raw-archive-manifest-v1")
    outcome = _verified(outcome_raw, public_key, "typed-call-outcome-v1")
    if (
        receipt.get("archive_manifest_sha256") != hashlib.sha256(archive_raw).hexdigest()
        or receipt.get("typed_outcome_sha256") != hashlib.sha256(outcome_raw).hexdigest()
        or outcome.get("archive_manifest_sha256") != hashlib.sha256(archive_raw).hexdigest()
        or archive.get("request_sha256") != hashlib.sha256(
            _read_regular(directory / "request.bin")
        ).hexdigest()
        or archive.get("response_header_sha256") != hashlib.sha256(
            _read_regular(directory / "response.headers")
        ).hexdigest()
        or archive.get("response_header_bytes") != len(
            _read_regular(directory / "response.headers")
        )
        or archive.get("response_body_sha256") != hashlib.sha256(
            _read_regular(directory / "response.body")
        ).hexdigest()
        or archive.get("response_body_bytes") != len(_read_regular(directory / "response.body"))
        or archive.get("reservation_record_sha256")
        != receipt.get("reservation_record_sha256")
        or receipt.get("answer_call_id") != outcome.get("answer_call_id")
        or receipt.get("parsed_response_source_call_id")
        != outcome.get("parsed_response_source_call_id")
        or (
            receipt.get("answer_call_id") is not None
            and receipt.get("answer_call_id") != receipt.get("parsed_response_source_call_id")
        )
    ):
        raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID")


def _validate_ledger_references(root: Path, receipts: list[dict[str, JsonValue]]) -> None:
    records: dict[str, dict[str, JsonValue]] = {}
    heads: dict[str, dict[str, JsonValue]] = {}
    for path in (root / "ledger/global/records").iterdir():
        raw = _read_regular(path)
        records[hashlib.sha256(raw).hexdigest()] = parse_canonical_object(raw)
    for path in (root / "ledger/global/heads").iterdir():
        raw = _read_regular(path)
        heads[hashlib.sha256(raw).hexdigest()] = parse_canonical_object(raw)
    for receipt in receipts:
        if receipt["issued"] is True:
            reservation_hash = receipt.get("reservation_record_sha256")
            reservation = records.get(str(reservation_hash))
            if (
                reservation is None
                or reservation.get("record_kind") != "reservation"
                or heads.get(str(receipt.get("reservation_head_sha256")), {}).get(
                    "record_sha256"
                )
                != reservation_hash
            ):
                raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID")
            settlement_hash = receipt.get("settlement_record_sha256")
            if settlement_hash is not None:
                settlement = records.get(str(settlement_hash))
                if (
                    settlement is None
                    or settlement.get("record_kind") != "settlement"
                    or settlement.get("reservation_record_sha256") != reservation_hash
                    or settlement.get("archive_manifest_sha256")
                    != receipt.get("archive_manifest_sha256")
                    or settlement.get("typed_outcome_sha256") != receipt.get("typed_outcome_sha256")
                    or heads.get(str(receipt.get("settlement_head_sha256")), {}).get(
                        "record_sha256"
                    )
                    != settlement_hash
                ):
                    raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID")
        else:
            not_issued_hash = receipt.get("not_issued_record_sha256")
            not_issued = records.get(str(not_issued_hash))
            if (
                not_issued is None
                or not_issued.get("record_kind") != "not_issued"
                or heads.get(str(receipt.get("not_issued_head_sha256")), {}).get(
                    "record_sha256"
                )
                != not_issued_hash
            ):
                raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID")


def _verified(raw: bytes, public_key: bytes, domain: str) -> dict[str, JsonValue]:
    value = parse_canonical_object(raw)
    signature = value.get("signature")
    if not isinstance(signature, str):
        raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID")
    unsigned = dict(value)
    del unsigned["signature"]
    verify_object_signature(public_key, domain, unsigned, signature)
    return value


def _read_regular(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1_048_576):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


__all__ = (
    "RootlessArchiveValidation",
    "reject_rootless_at_legacy_seam",
    "validate_rootless_bct_archive",
    "validate_rootless_screening_archive",
)
