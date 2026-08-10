from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    canonical_json_value,
    parse_canonical_object,
)

PROFILE: Final = "local_rootless_non_authoritative"
MAX_FRAME_BYTES: Final = 1_048_576
_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_HEX: Final = re.compile(r"[0-9a-f]{64}\Z")
_COMMON: Final = {
    "schema_version",
    "profile",
    "message_type",
    "attempt_id",
    "stage",
    "slot_id",
    "message_sequence",
}
_VARIANT_FIELDS: Final = {
    "dispatch": {"predecessor_receipt_sha256s"},
    "accepted": {
        "reservation_record_sha256",
        "reservation_head_sha256",
        "request_sha256",
        "request_bytes",
        "compiled_input_tokens",
    },
    "response_chunk": {
        "typed_outcome_sha256",
        "total_bytes",
        "chunk_index",
        "chunk_count",
        "data_base64",
    },
    "completed": {"typed_outcome_sha256", "call_receipt_sha256", "ledger_head_sha256"},
    "accounted": {
        "call_receipt_sha256",
        "not_issued_record_sha256",
        "not_issued_head_sha256",
        "ledger_head_sha256",
    },
    "terminal": {"operational_reason", "ledger_head_sha256"},
}


def encode_frame(message: dict[str, JsonValue]) -> bytes:
    validate_message(message)
    payload = canonical_json_value(message)
    if len(payload) > MAX_FRAME_BYTES:
        raise RootlessContractError("ROOTLESS_WIRE_INVALID")
    return len(payload).to_bytes(4, "big") + payload


def decode_frame(raw: bytes) -> dict[str, JsonValue]:
    if len(raw) < 4:
        raise RootlessContractError("ROOTLESS_WIRE_INVALID")
    size = int.from_bytes(raw[:4], "big")
    if size > MAX_FRAME_BYTES or size != len(raw) - 4:
        raise RootlessContractError("ROOTLESS_WIRE_INVALID")
    try:
        value = parse_canonical_object(raw[4:] + b"\n")
    except RootlessContractError as error:
        raise RootlessContractError("ROOTLESS_WIRE_INVALID") from error
    validate_message(value)
    return value


def validate_message(message: dict[str, JsonValue]) -> None:
    message_type = message.get("message_type")
    if not isinstance(message_type, str) or message_type not in _VARIANT_FIELDS:
        raise RootlessContractError("ROOTLESS_WIRE_INVALID")
    if set(message) != _COMMON | _VARIANT_FIELDS[message_type]:
        raise RootlessContractError("ROOTLESS_WIRE_INVALID")
    sequence = message.get("message_sequence")
    if (
        message.get("schema_version") != "rootless_local_wire_v1"
        or message.get("profile") != PROFILE
        or message.get("stage") not in {"screening", "bct"}
        or not isinstance(message.get("attempt_id"), str)
        or _ID.fullmatch(str(message["attempt_id"])) is None
        or not isinstance(message.get("slot_id"), str)
        or _ID.fullmatch(str(message["slot_id"])) is None
        or not isinstance(sequence, int)
        or isinstance(sequence, bool)
        or sequence < 0
    ):
        raise RootlessContractError("ROOTLESS_WIRE_INVALID")
    for key, value in message.items():
        if key.endswith("_sha256") and (not isinstance(value, str) or _HEX.fullmatch(value) is None):
            raise RootlessContractError("ROOTLESS_WIRE_INVALID")
    if message_type == "dispatch":
        predecessors = message["predecessor_receipt_sha256s"]
        if (
            not isinstance(predecessors, list)
            or len(predecessors) > 1
            or any(not isinstance(value, str) or _HEX.fullmatch(value) is None for value in predecessors)
        ):
            raise RootlessContractError("ROOTLESS_WIRE_INVALID")


@dataclass(slots=True)
class WireProtocol:
    """Validates each per-slot branch and each direction's message sequence."""

    _next_dispatch_sequence: int = 0
    _next_result_sequence: int = 0
    _states: dict[str, str] = field(default_factory=dict)
    _closed: set[str] = field(default_factory=set)

    @property
    def closed_slots(self) -> frozenset[str]:
        return frozenset(self._closed)

    def accept(self, message: dict[str, JsonValue]) -> None:
        validate_message(message)
        sequence = message["message_sequence"]
        slot = message["slot_id"]
        message_type = message["message_type"]
        expected_sequence = (
            self._next_dispatch_sequence
            if message_type == "dispatch"
            else self._next_result_sequence
        )
        if (
            not isinstance(sequence, int)
            or not isinstance(slot, str)
            or not isinstance(message_type, str)
            or sequence != expected_sequence
            or slot in self._closed
        ):
            raise RootlessContractError("ROOTLESS_WIRE_INVALID")
        state = self._states.get(slot)
        match (state, message_type):
            case (None, "dispatch"):
                self._states[slot] = "dispatched"
            case ("dispatched", "accepted"):
                self._states[slot] = "accepted"
            case ("accepted" | "chunking", "response_chunk"):
                self._states[slot] = "chunking"
            case ("accepted" | "chunking", "completed"):
                self._states[slot] = "completed"
                self._closed.add(slot)
            case ("dispatched", "accounted" | "terminal"):
                self._states[slot] = message_type
                self._closed.add(slot)
            case _:
                raise RootlessContractError("ROOTLESS_WIRE_INVALID")
        if message_type == "dispatch":
            self._next_dispatch_sequence += 1
        else:
            self._next_result_sequence += 1


__all__ = ("WireProtocol", "decode_frame", "encode_frame", "validate_message")
