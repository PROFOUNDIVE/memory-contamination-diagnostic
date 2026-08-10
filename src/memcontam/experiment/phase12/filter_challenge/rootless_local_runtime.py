from __future__ import annotations

import base64
import hashlib
import os
import signal
import socket
from collections.abc import Callable
from typing import Final

import anyio

from memcontam.experiment.phase12.filter_challenge.rootless_local_broker import (
    BrokerRequest,
    FakeBroker,
    HTTPXTransport,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    canonical_json_file,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_execution import (
    NOW,
    SlotCompilation,
    _blocked_receipt,
    _hash_file,
    _materialize_dynamic,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_protocol import (
    MAX_FRAME_BYTES,
    WireProtocol,
    decode_frame,
    encode_frame,
)

PROFILE: Final = "local_rootless_non_authoritative"
_CHUNK_BYTES: Final = 43_008

BrokerFactory = Callable[[], FakeBroker]


def run_stage_process(slots: tuple[SlotCompilation, ...], broker_factory: BrokerFactory) -> int:
    if not slots:
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    broker_socket, worker_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    process = os.fork()
    if process == 0:
        broker_socket.close()
        os.dup2(worker_socket.fileno(), 3)
        worker_socket.close()
        for descriptor in range(4, 256):
            try:
                os.close(descriptor)
            except OSError:
                continue
        with socket.socket(fileno=3) as capability:
            os._exit(_worker(capability, slots))
    worker_socket.close()
    broker: FakeBroker | None = None
    completed = False
    try:
        broker = broker_factory()
        anyio.run(_broker, broker_socket, slots, broker)
        completed = True
    finally:
        broker_socket.close()
        if broker is not None:
            if isinstance(broker.transport, HTTPXTransport):
                anyio.run(broker.transport.aclose)
            broker.close()
        if not completed:
            os.kill(process, signal.SIGTERM)
            os.waitpid(process, 0)
    _, status = os.waitpid(process, 0)
    return os.waitstatus_to_exitcode(status)


def _worker(capability: socket.socket, slots: tuple[SlotCompilation, ...]) -> int:
    protocol = WireProtocol()
    sequence = 0
    receipt_hashes: dict[str, str] = {}
    try:
        for slot in slots:
            predecessor = slot.predecessor_slot_ids[0] if slot.predecessor_slot_ids else None
            predecessor_hash = receipt_hashes.get(predecessor) if predecessor is not None else None
            dispatch = _common(slot, "dispatch", sequence)
            dispatch["predecessor_receipt_sha256s"] = (
                [] if predecessor_hash is None else [predecessor_hash]
            )
            protocol.accept(dispatch)
            _send(capability, dispatch)
            sequence += 1
            while True:
                message = _receive(capability)
                protocol.accept(message)
                sequence += 1
                message_type = message["message_type"]
                if message_type in {"completed", "accounted"}:
                    receipt_hash = message["call_receipt_sha256"]
                    if not isinstance(receipt_hash, str):
                        raise RootlessContractError("ROOTLESS_WIRE_INVALID")
                    receipt_hashes[slot.slot_id] = receipt_hash
                    break
                if message_type == "terminal":
                    return 69
        return 0
    except (BrokenPipeError, EOFError, RootlessContractError):
        return 69


async def _broker(
    capability: socket.socket, slots: tuple[SlotCompilation, ...], broker: FakeBroker
) -> None:
    protocol = WireProtocol()
    sequence = 0
    receipt_hashes: dict[str, str] = {}
    outputs: dict[str, str] = {}
    for original in slots:
        dispatch = _receive(capability)
        protocol.accept(dispatch)
        sequence += 1
        predecessor = original.predecessor_slot_ids[0] if original.predecessor_slot_ids else None
        predecessor_hash = receipt_hashes.get(predecessor) if predecessor is not None else None
        expected_predecessors = [] if predecessor_hash is None else [predecessor_hash]
        if (
            dispatch.get("slot_id") != original.slot_id
            or dispatch.get("predecessor_receipt_sha256s") != expected_predecessors
        ):
            raise RootlessContractError("ROOTLESS_WIRE_INVALID")
        if predecessor is not None and predecessor not in outputs:
            receipt = _blocked_receipt(broker, original, predecessor_hash)
            receipt_hash = _hash_file(receipt)
            receipt_hashes[original.slot_id] = receipt_hash
            message = _common(original, "accounted", sequence)
            message.update(
                call_receipt_sha256=receipt_hash,
                not_issued_record_sha256=receipt["not_issued_record_sha256"],
                not_issued_head_sha256=receipt["not_issued_head_sha256"],
                ledger_head_sha256=receipt["not_issued_head_sha256"],
            )
            protocol.accept(message)
            _send(capability, message)
            sequence += 1
            continue
        slot = (
            _materialize_dynamic(original, outputs[predecessor])
            if original.request is None and predecessor is not None
            else original
        )
        if slot.request is None or slot.compiled_input_tokens is None:
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
        outcome = await broker.dispatch(
            BrokerRequest(
                slot.slot_id,
                f"idem-{slot.slot_id}",
                slot.compiler_sha256,
                slot.static_input_sha256,
                predecessor_hash,
                slot.request,
                slot.compiled_input_tokens,
                slot.side,
                NOW,
                slot.scientific_replicate,
                slot.executor_replicate_id,
            )
        )
        receipt_hash = _hash_file(outcome.receipt)
        receipt_hashes[slot.slot_id] = receipt_hash
        if outcome.operational_reason is not None:
            terminal = _common(slot, "terminal", sequence)
            terminal.update(
                operational_reason=outcome.operational_reason,
                ledger_head_sha256=(
                    outcome.receipt["settlement_head_sha256"]
                    or outcome.receipt["reservation_head_sha256"]
                ),
            )
            protocol.accept(terminal)
            _send(capability, terminal)
            return
        parsed = outcome.typed_outcome.get("parsed_output")
        if outcome.provider_status == "completed" and isinstance(parsed, str) and parsed:
            outputs[slot.slot_id] = parsed
        sequence = await _send_outcome(capability, protocol, slot, outcome.receipt, outcome.typed_outcome, receipt_hash, sequence)


async def _send_outcome(
    capability: socket.socket,
    protocol: WireProtocol,
    slot: SlotCompilation,
    receipt: dict[str, JsonValue],
    typed_outcome: dict[str, JsonValue],
    receipt_hash: str,
    sequence: int,
) -> int:
    accepted = _common(slot, "accepted", sequence)
    accepted.update(
        reservation_record_sha256=receipt["reservation_record_sha256"],
        reservation_head_sha256=receipt["reservation_head_sha256"],
        request_sha256=receipt["request_sha256"],
        request_bytes=receipt["request_bytes"],
        compiled_input_tokens=receipt["compiled_input_tokens"],
    )
    protocol.accept(accepted)
    _send(capability, accepted)
    sequence += 1
    raw = canonical_json_file(typed_outcome)
    outcome_hash = hashlib.sha256(raw).hexdigest()
    chunks = tuple(raw[offset : offset + _CHUNK_BYTES] for offset in range(0, len(raw), _CHUNK_BYTES))
    for index, chunk in enumerate(chunks):
        message = _common(slot, "response_chunk", sequence)
        message.update(
            typed_outcome_sha256=outcome_hash,
            total_bytes=len(raw),
            chunk_index=index,
            chunk_count=len(chunks),
            data_base64=base64.b64encode(chunk).decode("ascii"),
        )
        protocol.accept(message)
        _send(capability, message)
        sequence += 1
    completed = _common(slot, "completed", sequence)
    completed.update(
        typed_outcome_sha256=outcome_hash,
        call_receipt_sha256=receipt_hash,
        ledger_head_sha256=receipt["settlement_head_sha256"] or receipt["reservation_head_sha256"],
    )
    protocol.accept(completed)
    _send(capability, completed)
    return sequence + 1


def _common(slot: SlotCompilation, message_type: str, sequence: int) -> dict[str, JsonValue]:
    return {
        "schema_version": "rootless_local_wire_v1",
        "profile": PROFILE,
        "message_type": message_type,
        "attempt_id": slot.attempt_id,
        "stage": slot.stage,
        "slot_id": slot.slot_id,
        "message_sequence": sequence,
    }


def _send(capability: socket.socket, message: dict[str, JsonValue]) -> None:
    capability.sendall(encode_frame(message))


def _receive(capability: socket.socket) -> dict[str, JsonValue]:
    prefix = _read_exact(capability, 4)
    size = int.from_bytes(prefix, "big")
    if size > MAX_FRAME_BYTES:
        raise RootlessContractError("ROOTLESS_WIRE_INVALID")
    return decode_frame(prefix + _read_exact(capability, size))


def _read_exact(capability: socket.socket, size: int) -> bytes:
    value = bytearray()
    while len(value) < size:
        chunk = capability.recv(size - len(value))
        if not chunk:
            raise EOFError
        value.extend(chunk)
    return bytes(value)


__all__ = ("run_stage_process",)
