from __future__ import annotations

import base64
import hashlib
import os
import select
import signal
import socket
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final

import anyio
from anyio import WouldBlock
from anyio import to_thread

from memcontam.experiment.phase12.filter_challenge.rootless_local_broker import (
    BrokerRequest,
    FakeBroker,
    HTTPXTransport,
    MAX_DISPATCHES_PER_MINUTE,
    RESERVED_TOKENS_PER_MINUTE,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_archive_validator import (
    validate_rootless_bct_archive,
    validate_rootless_screening_archive,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    canonical_json_file,
    parse_canonical_object,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_closure import close_stage
from memcontam.experiment.phase12.filter_challenge.rootless_local_execution import (
    SlotCompilation,
    _blocked_receipt,
    _hash_file,
    _idempotency_key,
    _materialize_dynamic,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_native_capture import (
    NativePredecessorParseError,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_state import (
    read_private_file,
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
    broker: FakeBroker | None = None
    broker_socket: socket.socket | None = None
    process: int | None = None
    completed = False
    try:
        broker = broker_factory()
        broker_socket, worker_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        process = os.fork()
        if process == 0:
            broker_socket.close()
            os.dup2(worker_socket.fileno(), 3)
            worker_socket.close()
            _close_inherited_descriptors()
            with socket.socket(fileno=3) as capability:
                os._exit(_worker(capability, slots))
        worker_socket.close()
        try:
            anyio.run(_broker, broker_socket, slots, broker)
        except Exception:
            broker.stage_operational_reason = "ROOTLESS_INTERRUPTED_UNCLEAN"
        close_stage(broker, slots)
        completed = True
    finally:
        if broker_socket is not None:
            broker_socket.close()
        if broker is not None:
            if isinstance(broker.transport, HTTPXTransport):
                anyio.run(broker.transport.aclose)
            broker.close()
        if process is not None and not completed:
            os.kill(process, signal.SIGTERM)
            os.waitpid(process, 0)
    if process is None:
        return 69
    _, status = os.waitpid(process, 0)
    return os.waitstatus_to_exitcode(status)


def _close_inherited_descriptors() -> None:
    for name in os.listdir("/proc/self/fd"):
        try:
            descriptor = int(name)
        except ValueError:
            continue
        if descriptor <= 3:
            continue
        try:
            os.close(descriptor)
        except OSError:
            continue


def _worker(capability: socket.socket, slots: tuple[SlotCompilation, ...]) -> int:
    protocol = WireProtocol()
    dispatch_sequence = 0
    receipt_hashes: dict[str, str] = {}
    pending = list(slots)
    active: set[str] = set()

    def dispatch_ready() -> None:
        nonlocal dispatch_sequence
        while len(active) < 5:
            ready = _ready_batch(pending, receipt_hashes)
            if not ready:
                return
            slot = ready[0]
            pending.remove(slot)
            predecessor = slot.predecessor_slot_ids[0] if slot.predecessor_slot_ids else None
            predecessor_hash = receipt_hashes.get(predecessor) if predecessor is not None else None
            dispatch = _common(slot, "dispatch", dispatch_sequence)
            dispatch["predecessor_receipt_sha256s"] = (
                [] if predecessor_hash is None else [predecessor_hash]
            )
            protocol.accept(dispatch)
            _send(capability, dispatch)
            active.add(slot.slot_id)
            dispatch_sequence += 1

    try:
        dispatch_ready()
        while active:
            message = _receive(capability)
            protocol.accept(message)
            message_type = message["message_type"]
            if message_type == "terminal":
                return 69
            if message_type in {"accepted", "response_chunk"}:
                continue
            if message_type not in {"completed", "accounted"}:
                raise RootlessContractError("ROOTLESS_WIRE_INVALID")
            receipt_hash = message["call_receipt_sha256"]
            slot_id = message["slot_id"]
            if not isinstance(receipt_hash, str) or not isinstance(slot_id, str) or slot_id not in active:
                raise RootlessContractError("ROOTLESS_WIRE_INVALID")
            receipt_hashes[slot_id] = receipt_hash
            active.remove(slot_id)
            dispatch_ready()
        if pending:
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
        return 0
    except (BrokenPipeError, EOFError, RootlessContractError):
        return 69


def _ready_batch(
    pending: list[SlotCompilation], receipt_hashes: dict[str, str]
) -> tuple[SlotCompilation, ...]:
    ready = tuple(
        slot
        for slot in pending
        if not slot.predecessor_slot_ids
        or slot.predecessor_slot_ids[0] in receipt_hashes
    )
    return ready[:5]


async def _broker(
    capability: socket.socket, slots: tuple[SlotCompilation, ...], broker: FakeBroker
) -> None:
    protocol = WireProtocol()
    existing_results = _existing_results(broker, slots)
    result_sequence = 0
    receipt_hashes: dict[str, str] = {}
    outputs: dict[str, str] = {}
    pending = {slot.slot_id: slot for slot in slots}
    live_recent: list[tuple[float, int]] = []
    live_lock = anyio.Lock()
    send_results, receive_results = anyio.create_memory_object_stream[
        tuple[SlotCompilation, dict[str, JsonValue], dict[str, JsonValue] | None]
    ](5)
    active = 0

    async def execute(original: SlotCompilation) -> None:
        try:
            slot, request = _prepare_request(broker, original, receipt_hashes, outputs)
        except NativePredecessorParseError:
            predecessor = original.predecessor_slot_ids[0]
            receipt = _blocked_receipt(
                broker,
                original,
                receipt_hashes.get(predecessor),
                "DOWNSTREAM_NOT_ISSUED_AFTER_PARSE_FAILURE",
            )
            await send_results.send((original, receipt, None))
            return
        if request is None:
            receipt = _blocked_receipt(
                broker,
                original,
                receipt_hashes.get(original.predecessor_slot_ids[0]),
            )
            await send_results.send((original, receipt, None))
            return
        existing = _existing_result(existing_results, original)
        if existing is not None:
            await send_results.send((slot, *existing))
            return
        if len(request.request) > 262_144 or request.compiled_input_tokens > 3072:
            await send_results.send((slot, broker.account_input_cap(request), None))
            return
        if isinstance(broker.transport, HTTPXTransport):
            await _live_admission(live_recent, live_lock, request.compiled_input_tokens)
        outcome = await broker.dispatch(request)
        await send_results.send((slot, outcome.receipt, outcome.typed_outcome))

    async with anyio.create_task_group() as group:
        while pending or active:
            if select.select((capability,), (), (), 0)[0]:
                if active >= 5:
                    raise RootlessContractError("ROOTLESS_WIRE_INVALID")
                dispatch = await to_thread.run_sync(_receive, capability)
                protocol.accept(dispatch)
                slot_id = dispatch.get("slot_id")
                if not isinstance(slot_id, str) or slot_id not in pending:
                    raise RootlessContractError("ROOTLESS_WIRE_INVALID")
                original = pending.pop(slot_id)
                predecessor = original.predecessor_slot_ids[0] if original.predecessor_slot_ids else None
                predecessor_hash = receipt_hashes.get(predecessor) if predecessor is not None else None
                if dispatch.get("predecessor_receipt_sha256s") != (
                    [] if predecessor_hash is None else [predecessor_hash]
                ):
                    raise RootlessContractError("ROOTLESS_WIRE_INVALID")
                group.start_soon(execute, original)
                active += 1
                continue
            try:
                slot, receipt, outcome = receive_results.receive_nowait()
            except WouldBlock:
                await anyio.sleep(0.001)
                continue
            active -= 1
            if active < 0:
                raise RootlessContractError("ROOTLESS_WIRE_INVALID")
            receipt_hash = _hash_file(receipt)
            receipt_hashes[slot.slot_id] = receipt_hash
            operational_reason = receipt.get("operational_reason")
            if isinstance(operational_reason, str) and receipt.get("issued") is True:
                broker.stage_operational_reason = operational_reason
                terminal = _common(slot, "terminal", result_sequence)
                terminal.update(
                    operational_reason=operational_reason,
                    ledger_head_sha256=(
                        receipt["settlement_head_sha256"] or receipt["reservation_head_sha256"]
                    ),
                )
                protocol.accept(terminal)
                _send(capability, terminal)
                return
            if receipt.get("issued") is False:
                message = _common(slot, "accounted", result_sequence)
                message.update(
                    call_receipt_sha256=receipt_hash,
                    not_issued_record_sha256=receipt["not_issued_record_sha256"],
                    not_issued_head_sha256=receipt["not_issued_head_sha256"],
                    ledger_head_sha256=receipt["not_issued_head_sha256"],
                )
                protocol.accept(message)
                _send(capability, message)
                result_sequence += 1
                if isinstance(operational_reason, str):
                    return
                continue
            if outcome is None:
                raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID")
            parsed = outcome.get("parsed_output")
            if receipt.get("provider_status") == "completed" and isinstance(parsed, str) and parsed:
                outputs[slot.slot_id] = parsed
            result_sequence = await _send_outcome(
                capability, protocol, slot, receipt, outcome, receipt_hash, result_sequence
            )


def _prepare_request(
    broker: FakeBroker,
    original: SlotCompilation,
    receipt_hashes: dict[str, str],
    outputs: dict[str, str],
) -> tuple[SlotCompilation, BrokerRequest | None]:
    predecessor = original.predecessor_slot_ids[0] if original.predecessor_slot_ids else None
    predecessor_hash = receipt_hashes.get(predecessor) if predecessor is not None else None
    if predecessor is not None and original.request is None and predecessor not in outputs:
        return original, None
    slot = (
        _materialize_dynamic(original, outputs[predecessor])
        if original.request is None and predecessor is not None
        else original
    )
    if slot.request is None or slot.compiled_input_tokens is None:
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    return slot, BrokerRequest(
        slot_id=slot.slot_id,
        idempotency_key=_idempotency_key(slot.attempt_id, slot.slot_id),
        compiler_sha256=slot.compiler_sha256,
        static_input_sha256=slot.static_input_sha256,
        predecessor_receipt_sha256=predecessor_hash,
        request=slot.request,
        compiled_input_tokens=slot.compiled_input_tokens,
        side=slot.side,
        created_at=_utc_timestamp(),
        task=slot.task,
        baseline=slot.baseline,
        probe_id=slot.probe_id,
        native_stage=slot.native_stage,
        candidate_class=slot.candidate_class,
        scientific_replicate=slot.scientific_replicate,
        executor_replicate_id=slot.executor_replicate_id,
    )


def _existing_results(
    broker: FakeBroker, slots: tuple[SlotCompilation, ...]
) -> dict[str, tuple[dict[str, JsonValue], dict[str, JsonValue] | None]]:
    prefix: list[SlotCompilation] = []
    found_missing = False
    for slot in slots:
        receipt_path = (
            broker.root
            / "attempts"
            / broker.attempt_id
            / broker.stage
            / "slots"
            / slot.slot_id
            / "call-receipt.json"
        )
        if receipt_path.is_file():
            if found_missing:
                raise RootlessContractError("ROOTLESS_ARCHIVE_INVALID")
            prefix.append(slot)
        else:
            found_missing = True
    if not prefix:
        return {}
    validator = (
        validate_rootless_screening_archive
        if broker.stage == "screening"
        else validate_rootless_bct_archive
    )
    validator(broker.root, tuple(prefix), seed=broker.seed)
    results: dict[str, tuple[dict[str, JsonValue], dict[str, JsonValue] | None]] = {}
    for slot in prefix:
        root = broker.root / "attempts" / broker.attempt_id / broker.stage / "slots" / slot.slot_id
        receipt = parse_canonical_object(read_private_file(root / "call-receipt.json"))
        outcome = (
            parse_canonical_object(read_private_file(root / "typed-outcome.json"))
            if receipt.get("issued") is True
            else None
        )
        results[slot.slot_id] = receipt, outcome
    return results


def _existing_result(
    results: dict[str, tuple[dict[str, JsonValue], dict[str, JsonValue] | None]],
    slot: SlotCompilation,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue] | None] | None:
    return results.get(slot.slot_id)


async def _live_admission(
    recent: list[tuple[float, int]], lock: anyio.Lock, reserved_tokens: int
) -> None:
    async with lock:
        while True:
            now = anyio.current_time()
            recent[:] = [entry for entry in recent if entry[0] > now - 60.0]
            if (
                len(recent) < MAX_DISPATCHES_PER_MINUTE
                and sum(tokens for _, tokens in recent) + reserved_tokens
                <= RESERVED_TOKENS_PER_MINUTE
            ):
                recent.append((now, reserved_tokens))
                return
            await anyio.sleep(max(0.0, recent[0][0] + 60.0 - now))


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


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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
