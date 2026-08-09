from __future__ import annotations

# allow: SIZE_OK — Task 4 requires one constructor boundary and one HTTP precedence authority.

import base64
import binascii
import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, cast

import anyio
import httpx

from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
    revalidate_runtime_observations,
    validate_live_stage_binding,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    canonical_json_file,
    parse_canonical_object,
    public_key_from_seed,
    sign_object,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_ledger import (
    GlobalLedger,
    LedgerReservation,
    ProviderUsage,
    Stage,
    actual_cost_nanousd,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_state import read_private_file

PROFILE: Final = "local_rootless_non_authoritative"
EXPECTED_MODEL: Final = "gpt-4o-2024-11-20"
REQUEST_CAP: Final = 262_144
HEADER_CAP: Final = 32_768
HEADER_FIELD_CAP: Final = 100
BODY_CAP: Final = 1_048_576
TOTAL_STATE_CAP: Final = 1_661_992_960
SLOT_STATE_RESERVATION: Final = 2_621_440
FIXED_STATE_CAP: Final = 134_217_728
GENERATED_FIXED_FIXTURE_BYTES: Final = 64_245_760
_KEY_PATTERN: Final = re.compile(rb"OPENAI_API_KEY=([A-Za-z0-9_-]{20,512})\n\Z")
_HEX: Final = re.compile(r"[0-9a-f]{64}\Z")
_FIXTURE_KEYS: Final = {
    "schema_version",
    "profile",
    "kind",
    "fixture_id",
    "slot_id",
    "lifecycle_marker",
    "response_surfaced",
    "http_status",
    "headers_base64",
    "body_base64",
    "raised_exception",
}

ProviderStatus = Literal[
    "completed",
    "failed",
    "cancelled",
    "incomplete",
    "nonterminal",
    "http_error",
    "transport_error",
    "archive_error",
]


class FixtureExchange(dict[str, str | int | bool | None]):
    pass


class FixtureTransport(Protocol):
    async def exchange(self, slot_id: str, request: bytes) -> dict[str, str | int | bool | None]: ...


class HTTPXTransport:
    """Owns the sole live HTTP client and exposes observations in the fixture shape."""

    def __init__(self, secret: str) -> None:
        self._secret = secret
        self._client = httpx.AsyncClient(
            headers={}, timeout=None, trust_env=False, follow_redirects=False, http2=False
        )
        self._client.headers.clear()

    def build_request(self, body: bytes) -> httpx.Request:
        request = self._client.build_request(
            "POST",
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {self._secret}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Connection": "close",
                "User-Agent": "memcontam-rootless-local/1",
            },
            content=body,
        )
        allowed = {
            "authorization",
            "content-type",
            "accept",
            "accept-encoding",
            "connection",
            "user-agent",
            "host",
            "content-length",
        }
        if set(request.headers) != allowed or request.headers["content-length"] != str(len(body)):
            raise RootlessContractError("ROOTLESS_OUTBOUND_HEADER_INVALID")
        return request

    async def exchange(self, slot_id: str, request: bytes) -> dict[str, str | int | bool | None]:
        marker = "before_write"
        surfaced = False
        status: int | None = None
        headers = b"\0\0\0\0"
        body = bytearray()
        raised: str | None = None
        try:
            with anyio.fail_after(120):
                outbound = self.build_request(request)
                marker = "writing"
                response = await self._client.send(outbound, stream=True)
                surfaced = True
                status = response.status_code
                headers = _encode_headers(response.headers.raw)
                marker = "streaming_body"
                try:
                    async for chunk in response.aiter_raw():
                        remaining = BODY_CAP + 1 - len(body)
                        if remaining <= 0:
                            break
                        body.extend(chunk[:remaining])
                finally:
                    await response.aclose()
        except TimeoutError:
            raised = "Cancelled"
        except (httpx.ProtocolError, httpx.DecodingError) as error:
            raised = type(error).__name__
        except httpx.TimeoutException as error:
            raised = type(error).__name__
        except httpx.TransportError as error:
            raised = type(error).__name__
        return {
            "schema_version": "rootless_fake_http_exchange_v1",
            "profile": PROFILE,
            "kind": "fake_http_exchange",
            "fixture_id": "live",
            "slot_id": slot_id,
            "lifecycle_marker": marker,
            "response_surfaced": surfaced,
            "http_status": status,
            "headers_base64": base64.b64encode(headers).decode("ascii"),
            "body_base64": base64.b64encode(body).decode("ascii"),
            "raised_exception": raised,
        }

    async def aclose(self) -> None:
        self._secret = ""
        await self._client.aclose()


@dataclass(frozen=True, slots=True)
class BrokerRequest:
    slot_id: str
    idempotency_key: str
    compiler_sha256: str
    static_input_sha256: str
    predecessor_receipt_sha256: str | None
    request: bytes
    compiled_input_tokens: int
    side: Literal["control", "challenge"]
    created_at: str


@dataclass(frozen=True, slots=True)
class ReadySlot:
    slot_id: str
    predecessor_receipt_sha256: str | None
    reserved_tokens: int


@dataclass(frozen=True, slots=True)
class SchedulerState:
    active_calls: int
    recent_dispatch_monotonic_ns: tuple[int, ...]
    now_monotonic_ns: int
    accounted_receipt_sha256s: frozenset[str]


@dataclass(frozen=True, slots=True)
class BrokerOutcome:
    provider_status: ProviderStatus
    operational_reason: str | None
    http_status: int | None
    response_model: str | None
    usage: ProviderUsage | None
    actual_nanousd: int | None
    reservation_retained: bool
    response_headers: bytes
    response_body: bytes
    archive_manifest: dict[str, JsonValue]
    typed_outcome: dict[str, JsonValue]
    receipt: dict[str, JsonValue]


@dataclass(slots=True)
class RuntimeLock:
    descriptor: int

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass(frozen=True, slots=True)
class _HeaderArchive:
    raw: bytes
    encoding_status: str
    encoding_sha256: str | None
    encoding_bytes: int
    overflow: bool
    framing_valid: bool


@dataclass(frozen=True, slots=True)
class _Classification:
    provider_status: ProviderStatus
    reason: str | None
    http_status: int | None
    phase: str
    transport_error: str | None


def load_provider_key(path: Path) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > 530
        ):
            raise RootlessContractError("ROOTLESS_MISSING_SECRET")
        raw = os.read(descriptor, 531)
    finally:
        os.close(descriptor)
    match = _KEY_PATTERN.fullmatch(raw)
    if match is None:
        raise RootlessContractError("ROOTLESS_MISSING_SECRET")
    return match.group(1).decode("ascii")


def acquire_runtime_lock(path: Path) -> RuntimeLock:
    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise RootlessContractError("ROOTLESS_RUNTIME_LOCK_INVALID")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RootlessContractError("ROOTLESS_BROKER_ALREADY_RUNNING") from error
    except (OSError, RootlessContractError):
        os.close(descriptor)
        raise
    return RuntimeLock(descriptor)


def select_ready_slots(
    slots: tuple[ReadySlot, ...], state: SchedulerState
) -> tuple[ReadySlot, ...]:
    window_start = state.now_monotonic_ns - 60_000_000_000
    recent = sum(value >= window_start for value in state.recent_dispatch_monotonic_ns)
    capacity = min(5 - state.active_calls, 6 - recent)
    if capacity <= 0:
        return ()
    selected: list[ReadySlot] = []
    reserved_tokens = 0
    ready = sorted(
        (
            slot
            for slot in slots
            if slot.predecessor_receipt_sha256 is None
            or slot.predecessor_receipt_sha256 in state.accounted_receipt_sha256s
        ),
        key=lambda slot: slot.slot_id.encode(),
    )
    for slot in ready:
        if len(selected) == capacity or reserved_tokens + slot.reserved_tokens > 28_416:
            break
        selected.append(slot)
        reserved_tokens += slot.reserved_tokens
    return tuple(selected)


def revalidate_external_authority(
    runtime_manifest: Mapping[str, JsonValue],
    decoding_authority: Mapping[str, JsonValue],
    phase: Literal["before_claim", "before_dispatch"],
) -> None:
    if phase not in {"before_claim", "before_dispatch"}:
        raise RootlessContractError("ROOTLESS_EXTERNAL_AUTHORITY_IDENTITY_DRIFT")
    expected = runtime_manifest.get("ordered_external_authorities")
    if not isinstance(expected, list):
        raise RootlessContractError("ROOTLESS_EXTERNAL_AUTHORITY_REVIEW_BINDING_MISSING")
    revalidate_runtime_observations(expected, decoding_authority)


class FakeBroker:
    """Consumes canonical fixtures through the production archive/accounting path."""

    def __init__(
        self,
        binding: dict[str, JsonValue],
        transport: FixtureTransport,
        root: Path,
        seed: bytes,
        runtime_lock: RuntimeLock,
        exchange_fixture_id: str | None = None,
    ) -> None:
        self.binding = binding
        self.transport = transport
        self.root = root
        self.seed = seed
        self.runtime_lock = runtime_lock
        self.exchange_fixture_id = exchange_fixture_id or str(binding["fixture_id"])
        self.runtime_authority: tuple[Mapping[str, JsonValue], Mapping[str, JsonValue]] | None = None
        attempt_id = binding.get("attempt_id", binding.get("fixture_id"))
        stage = binding.get("stage")
        if not isinstance(attempt_id, str) or stage not in {"screening", "bct"}:
            raise RootlessContractError("ROOTLESS_BINDING_INVALID")
        self.attempt_id = attempt_id
        self.stage = cast(Stage, stage)
        self.ledger = GlobalLedger(root, seed, self.attempt_id, self.stage)

    async def dispatch(self, request: BrokerRequest) -> BrokerOutcome:
        if self.runtime_authority is not None:
            runtime_manifest, decoding_authority = self.runtime_authority
            revalidate_external_authority(runtime_manifest, decoding_authority, "before_dispatch")
        if len(request.request) > REQUEST_CAP or request.compiled_input_tokens > 3072:
            raise RootlessContractError("ROOTLESS_INPUT_CAP_EXCEEDED")
        reservation = self.ledger.reserve(
            LedgerReservation(
                request.slot_id,
                request.idempotency_key,
                request.compiler_sha256,
                request.static_input_sha256,
                request.predecessor_receipt_sha256,
                hashlib.sha256(request.request).hexdigest(),
                len(request.request),
                request.compiled_input_tokens,
            ),
            request.created_at,
        )
        slot_root = self.root / "attempts" / self.attempt_id / self.stage / "slots" / request.slot_id
        slot_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        _atomic_new(slot_root / "request.bin", request.request)
        fixture = await self.transport.exchange(request.slot_id, request.request)
        parsed = _parse_fixture(fixture, self.exchange_fixture_id, request.slot_id)
        headers = _archive_headers(parsed["headers"])
        response_body = cast(bytes, parsed["body"])
        body = response_body[:BODY_CAP]
        classification = _classify(parsed, headers, len(response_body) > BODY_CAP)
        _atomic_new(slot_root / "response.headers", headers.raw)
        _atomic_new(slot_root / "response.body", body)
        response_model, usage, parse_status, parsed_output, observed_status, response_reason = _response_values(
            body, classification, request.slot_id
        )
        if classification.reason is None and response_reason is not None:
            classification = _Classification(
                observed_status or "archive_error",
                response_reason,
                classification.http_status,
                "complete",
                None,
            )
        elif classification.reason is None and observed_status is not None:
            classification = _Classification(
                observed_status, None, classification.http_status, "complete", None
            )
        reason = classification.reason
        retained = True
        actual: int | None = None
        settlement = None
        if classification.provider_status == "completed":
            if response_model != EXPECTED_MODEL:
                reason = "ROOTLESS_WRONG_MODEL"
            try:
                if usage is None:
                    raise RootlessContractError("ROOTLESS_USAGE_INVALID")
                actual = actual_cost_nanousd(usage)
            except RootlessContractError:
                usage = None
                reason = reason or "ROOTLESS_USAGE_INVALID"
            else:
                retained = False
        archive = _signed(
            self.seed,
            "raw-archive-manifest-v1",
            {
                "schema_version": "rootless_raw_archive_manifest_v1",
                "profile": PROFILE,
                "kind": "raw_archive_manifest",
                "attempt_id": self.attempt_id,
                "stage": self.stage,
                "slot_id": request.slot_id,
                "reservation_record_sha256": reservation.record_sha256,
                "request_sha256": hashlib.sha256(request.request).hexdigest(),
                "request_bytes": len(request.request),
                "response_header_sha256": hashlib.sha256(headers.raw).hexdigest(),
                "response_header_bytes": len(headers.raw),
                "response_body_sha256": hashlib.sha256(body).hexdigest(),
                "response_body_bytes": len(body),
                "http_status": classification.http_status,
                "transport_phase": classification.phase,
                "transport_error": classification.transport_error,
                "content_encoding_status": headers.encoding_status,
                "content_encoding_sha256": headers.encoding_sha256,
                "content_encoding_bytes": headers.encoding_bytes,
                "response_model": response_model,
                "complete": classification.phase == "complete" and classification.transport_error is None,
                "started_at": request.created_at,
                "finished_at": request.created_at,
                "key_fingerprint": hashlib.sha256(public_key_from_seed(self.seed)).hexdigest(),
            },
        )
        archive_raw = canonical_json_file(archive)
        archive_hash = hashlib.sha256(archive_raw).hexdigest()
        _atomic_new(slot_root / "archive-manifest.json", archive_raw)
        typed = _signed(
            self.seed,
            "typed-call-outcome-v1",
            {
                "schema_version": "rootless_typed_call_outcome_v1",
                "profile": PROFILE,
                "kind": "typed_call_outcome",
                "attempt_id": self.attempt_id,
                "stage": self.stage,
                "slot_id": request.slot_id,
                "reservation_record_sha256": reservation.record_sha256,
                "archive_manifest_sha256": archive_hash,
                "provider_status": classification.provider_status,
                "raw_parse_status": parse_status,
                "verifier_status": "not_run",
                "parsed_output": parsed_output,
                "verifier_result": None,
                "answer_call_id": f"{request.slot_id}-answer" if parse_status != "not_run" else None,
                "parsed_response_source_call_id": f"{request.slot_id}-answer" if parse_status != "not_run" else None,
                "parser_result_id": f"{request.slot_id}-parser" if parse_status != "not_run" else None,
                "verifier_result_id": None,
                "behavioral_reason": (
                    f"{request.side.upper()}_PROVIDER_FAILURE"
                    if reason is not None
                    or classification.provider_status in {"failed", "cancelled", "incomplete"}
                    else None
                ),
                "created_at": request.created_at,
                "key_fingerprint": archive["key_fingerprint"],
            },
        )
        typed_raw = canonical_json_file(typed)
        typed_hash = hashlib.sha256(typed_raw).hexdigest()
        _atomic_new(slot_root / "typed-outcome.json", typed_raw)
        if not retained and usage is not None:
            settlement = self.ledger.settle(
                reservation.record_sha256,
                archive_hash,
                typed_hash,
                usage,
                request.created_at,
            )
        receipt = _signed(
            self.seed,
            "local-call-receipt-v1",
            {
                "schema_version": "rootless_local_call_receipt_v1",
                "profile": PROFILE,
                "kind": "local_call_receipt",
                "attempt_id": self.attempt_id,
                "stage": self.stage,
                "slot_id": request.slot_id,
                "idempotency_key": request.idempotency_key,
                "scientific_replicate": None,
                "executor_replicate_id": None,
                "issued": True,
                "compiler_sha256": request.compiler_sha256,
                "static_input_sha256": request.static_input_sha256,
                "predecessor_receipt_sha256": request.predecessor_receipt_sha256,
                "compile_status": "compiled",
                "request_sha256": hashlib.sha256(request.request).hexdigest(),
                "request_bytes": len(request.request),
                "compiled_input_tokens": request.compiled_input_tokens,
                "reservation_record_sha256": reservation.record_sha256,
                "reservation_head_sha256": reservation.head_sha256,
                "not_issued_record_sha256": None,
                "not_issued_head_sha256": None,
                "archive_manifest_sha256": archive_hash,
                "typed_outcome_sha256": typed_hash,
                "settlement_record_sha256": settlement.record_sha256 if settlement else None,
                "settlement_head_sha256": settlement.head_sha256 if settlement else None,
                "provider_status": classification.provider_status,
                "http_status": classification.http_status,
                "response_model": EXPECTED_MODEL if response_model == EXPECTED_MODEL else None,
                "usage_input_tokens": usage.input_tokens if settlement and usage else None,
                "cached_input_tokens": usage.cached_input_tokens if settlement and usage else None,
                "output_tokens": usage.output_tokens if settlement and usage else None,
                "total_tokens": usage.total_tokens if settlement and usage else None,
                "actual_nanousd": actual,
                "reservation_retained": retained,
                "answer_call_id": typed["answer_call_id"],
                "parsed_response_source_call_id": typed["parsed_response_source_call_id"],
                "parser_result_id": typed["parser_result_id"],
                "verifier_result_id": None,
                "behavioral_reason": typed["behavioral_reason"],
                "operational_reason": reason,
                "created_at": request.created_at,
                "key_fingerprint": archive["key_fingerprint"],
            },
        )
        _atomic_new(slot_root / "call-receipt.json", canonical_json_file(receipt))
        return BrokerOutcome(
            classification.provider_status,
            reason,
            classification.http_status,
            response_model,
            usage,
            actual,
            retained,
            headers.raw,
            body,
            archive,
            typed,
            receipt,
        )

    def close(self) -> None:
        self.runtime_lock.close()


def build_fake_broker_for_tests(
    binding: dict[str, JsonValue], fixture_transport: FixtureTransport, fake_root: Path
) -> FakeBroker:
    fixture_id = binding.get("fixture_id")
    if (
        binding.get("schema_version") != "rootless_fake_stage_binding_v1"
        or binding.get("profile") != PROFILE
        or binding.get("kind") != "fake_stage_binding"
        or binding.get("transport_mode") != "fake"
        or not isinstance(fixture_id, str)
        or fake_root.name != fixture_id
        or fake_root.parent.name != "fake-state"
        or fake_root.parent.parent.name != "tmp"
        or "basetemp" not in fake_root.parts
        or (fake_root / "live-attempt-claim.json").exists()
        or (fake_root / "ledger").exists()
        or (fake_root / "attempts").exists()
    ):
        raise RootlessContractError("ROOTLESS_FAKE_BOUNDARY_INVALID")
    lock_path = fake_root / "runtime.lock"
    descriptor = os.open(
        lock_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    os.fsync(descriptor)
    os.close(descriptor)
    seed = hashlib.sha256(f"rootless-fixture:{fixture_id}".encode()).digest()
    return FakeBroker(binding, fixture_transport, fake_root, seed, acquire_runtime_lock(lock_path))


def build_live_broker(
    binding: dict[str, JsonValue], state_root: Path, repository: Path
) -> FakeBroker:
    validate_live_stage_binding(binding)
    if "fake-state" in state_root.parts or binding.get("transport_mode") != "live":
        raise RootlessContractError("ROOTLESS_BINDING_INVALID")
    secret = load_provider_key(repository / ".env")
    seed = read_private_file(state_root / "keys" / "ed25519-private.key")
    attempt_id = binding.get("attempt_id")
    if not isinstance(attempt_id, str):
        raise RootlessContractError("ROOTLESS_BINDING_INVALID")
    manifest_root = state_root / "manifests" / attempt_id
    runtime_manifest = parse_canonical_object(read_private_file(manifest_root / "runtime.json"))
    decoding_authority = parse_canonical_object(
        read_private_file(manifest_root / "decoding-authority.json")
    )
    revalidate_external_authority(runtime_manifest, decoding_authority, "before_claim")
    lock = acquire_runtime_lock(state_root / "runtime.lock")
    broker = FakeBroker(binding, HTTPXTransport(secret), state_root, seed, lock, "live")
    broker.runtime_authority = (runtime_manifest, decoding_authority)
    return broker


def _parse_fixture(
    fixture: dict[str, str | int | bool | None], fixture_id: str, slot_id: str
) -> dict[str, JsonValue | bytes]:
    if (
        set(fixture) != _FIXTURE_KEYS
        or fixture.get("schema_version") != "rootless_fake_http_exchange_v1"
        or fixture.get("profile") != PROFILE
        or fixture.get("kind") != "fake_http_exchange"
        or fixture.get("fixture_id") != fixture_id
        or fixture.get("slot_id") != slot_id
        or fixture.get("lifecycle_marker")
        not in {"before_write", "writing", "awaiting_response", "streaming_body"}
        or not isinstance(fixture.get("response_surfaced"), bool)
    ):
        raise RootlessContractError("ROOTLESS_FAKE_FIXTURE_INVALID")
    try:
        headers = base64.b64decode(str(fixture["headers_base64"]), validate=True)
        body = base64.b64decode(str(fixture["body_base64"]), validate=True)
    except (ValueError, binascii.Error) as error:
        raise RootlessContractError("ROOTLESS_FAKE_FIXTURE_INVALID") from error
    return {**fixture, "headers": headers, "body": body}


def _archive_headers(attempted: JsonValue | bytes) -> _HeaderArchive:
    if not isinstance(attempted, bytes) or len(attempted) < 4:
        return _HeaderArchive(b"\0\0\0\0", "invalid", None, 0, False, False)
    count = int.from_bytes(attempted[:4], "big")
    offset = 4
    stored: list[tuple[bytes, bytes]] = []
    encoding_values: list[bytes] = []
    overflow = False
    try:
        for _ in range(count):
            if offset + 4 > len(attempted):
                raise ValueError
            name_size = int.from_bytes(attempted[offset : offset + 4], "big")
            offset += 4
            if offset + name_size + 4 > len(attempted):
                raise ValueError
            name = attempted[offset : offset + name_size]
            offset += name_size
            value_size = int.from_bytes(attempted[offset : offset + 4], "big")
            offset += 4
            if offset + value_size > len(attempted):
                raise ValueError
            value = attempted[offset : offset + value_size]
            offset += value_size
            field = (
                len(name).to_bytes(4, "big")
                + name
                + len(value).to_bytes(4, "big")
                + value
            )
            current_size = 4 + sum(8 + len(key) + len(item) for key, item in stored)
            if len(stored) == HEADER_FIELD_CAP or current_size + len(field) > HEADER_CAP:
                overflow = True
                break
            stored.append((name, value))
            if name.lower() == b"content-encoding":
                encoding_values.append(value)
        if not overflow and offset != len(attempted):
            raise ValueError
    except (ValueError, OverflowError):
        return _HeaderArchive(b"\0\0\0\0", "invalid", None, 0, False, False)
    raw = len(stored).to_bytes(4, "big") + b"".join(
        len(name).to_bytes(4, "big") + name + len(value).to_bytes(4, "big") + value
        for name, value in stored
    )
    preimage = len(encoding_values).to_bytes(4, "big") + b"".join(
        len(value).to_bytes(4, "big") + value for value in encoding_values
    )
    if overflow:
        status = "truncated"
    elif not encoding_values:
        status = "absent"
    elif any(any(byte < 0x20 or byte > 0x7E for byte in value) for value in encoding_values):
        status = "invalid"
    elif len(encoding_values) == 1 and encoding_values[0].lower() == b"identity":
        status = "identity"
    else:
        status = "unexpected"
    return _HeaderArchive(
        raw,
        status,
        hashlib.sha256(preimage).hexdigest() if encoding_values else None,
        len(preimage) if encoding_values else 0,
        overflow,
        True,
    )


def _encode_headers(fields: list[tuple[bytes, bytes]]) -> bytes:
    return len(fields).to_bytes(4, "big") + b"".join(
        len(name).to_bytes(4, "big") + name + len(value).to_bytes(4, "big") + value
        for name, value in fields
    )


def _classify(
    fixture: dict[str, JsonValue | bytes], headers: _HeaderArchive, body_overflow: bool
) -> _Classification:
    marker = str(fixture["lifecycle_marker"])
    surfaced = fixture["response_surfaced"] is True
    raised = fixture["raised_exception"]
    timeout = raised in {"PoolTimeout", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "Cancelled"}
    protocol = raised in {"ProtocolError", "RemoteProtocolError", "LocalProtocolError", "DecodingError"}
    if timeout:
        phase = {"before_write": "connect", "writing": "write", "awaiting_response": "headers", "streaming_body": "body"}[marker]
        return _Classification("transport_error", "ROOTLESS_TIMEOUT", None if not surfaced else _status(fixture), phase, "cancelled")
    if not surfaced and protocol:
        return _Classification("archive_error", "ROOTLESS_ARCHIVE_INVALID", None, "headers", "headers_failure")
    if not surfaced and raised is not None:
        phase, error = _transport_marker(marker, str(raised))
        return _Classification("transport_error", "ROOTLESS_HTTP_ERROR", None, phase, error)
    status = _status(fixture)
    if status is None or not 200 <= status <= 599:
        return _Classification("archive_error", "ROOTLESS_ARCHIVE_INVALID", None, "complete", None)
    if not headers.framing_valid:
        return _Classification("archive_error", "ROOTLESS_ARCHIVE_INVALID", status, "headers", "headers_failure")
    if headers.overflow:
        return _Classification("archive_error", "ROOTLESS_PROVIDER_RESPONSE_TOO_LARGE", status, "complete", None)
    if headers.encoding_status in {"unexpected", "invalid"}:
        return _Classification("archive_error", "ROOTLESS_UNEXPECTED_CONTENT_ENCODING", status, "complete", None)
    if body_overflow:
        return _Classification("archive_error", "ROOTLESS_PROVIDER_RESPONSE_TOO_LARGE", status, "body", None)
    if protocol:
        return _Classification("archive_error", "ROOTLESS_ARCHIVE_INVALID", status, "body", "body_failure")
    if raised is not None:
        return _Classification("transport_error", "ROOTLESS_HTTP_ERROR", status, "body", "body_failure")
    if status == 429:
        return _Classification("http_error", "ROOTLESS_RATE_LIMITED", status, "complete", None)
    if status != 200:
        return _Classification("http_error", "ROOTLESS_HTTP_ERROR", status, "complete", None)
    return _Classification("completed", None, status, "complete", None)


def _status(fixture: dict[str, JsonValue | bytes]) -> int | None:
    value = fixture["http_status"]
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _transport_marker(marker: str, raised: str) -> tuple[str, str]:
    if raised in {"ConnectError", "ProxyError", "UnsupportedProtocol"}:
        return "connect", "connect_failure"
    if raised == "WriteError":
        return "write", "write_failure"
    if raised in {"ReadError", "CloseError"}:
        return (
            ("body", "body_failure")
            if marker == "streaming_body"
            else ("headers", "headers_failure")
        )
    return {
        "before_write": ("connect", "connect_failure"),
        "writing": ("write", "write_failure"),
        "awaiting_response": ("headers", "headers_failure"),
        "streaming_body": ("body", "body_failure"),
    }[marker]


def _response_values(
    body: bytes, classification: _Classification, slot_id: str
) -> tuple[
    str | None,
    ProviderUsage | None,
    str,
    JsonValue | None,
    ProviderStatus | None,
    str | None,
]:
    if classification.reason is not None:
        return None, None, "not_run", None, None, None
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, None, "not_run", None, "archive_error", "ROOTLESS_ARCHIVE_INVALID"
    if not isinstance(value, dict) or not isinstance(value.get("status"), str):
        return None, None, "not_run", None, "archive_error", "ROOTLESS_ARCHIVE_INVALID"
    provider_status = value["status"]
    if provider_status in {"queued", "in_progress"}:
        model = value.get("model") if isinstance(value.get("model"), str) else None
        return model, None, "not_run", None, "nonterminal", "ROOTLESS_NONTERMINAL_RESPONSE"
    if provider_status not in {"completed", "failed", "cancelled", "incomplete"}:
        return None, None, "not_run", None, "archive_error", "ROOTLESS_ARCHIVE_INVALID"
    usage = _usage(value.get("usage"))
    model = value.get("model") if isinstance(value.get("model"), str) else None
    if provider_status != "completed":
        return model, usage, "not_run", None, provider_status, None
    text = value.get("output_text")
    if not isinstance(text, str):
        fragments: list[str] = []
        output = value.get("output")
        if not isinstance(output, list):
            return model, usage, "not_run", None, "archive_error", "ROOTLESS_ARCHIVE_INVALID"
        for item in output:
            if isinstance(item, dict) and isinstance(item.get("content"), list):
                for content in item["content"]:
                    if isinstance(content, dict) and isinstance(content.get("text"), str):
                        fragments.append(content["text"])
        text = "".join(fragments)
    return (
        model,
        usage,
        "success" if text else "failure",
        text if text else None,
        "completed",
        None,
    )


def _usage(value: JsonValue | object) -> ProviderUsage | None:
    if not isinstance(value, dict):
        return None
    details = value.get("input_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else 0
    fields = (value.get("input_tokens"), cached, value.get("output_tokens"), value.get("total_tokens"))
    if any(not isinstance(item, int) or isinstance(item, bool) for item in fields):
        return None
    return ProviderUsage(
        cast(int, fields[0]),
        cast(int, fields[1]),
        cast(int, fields[2]),
        cast(int, fields[3]),
    )


def _signed(seed: bytes, domain: str, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    result = dict(payload)
    result["signature"] = sign_object(seed, domain, payload)
    return result


def _atomic_new(path: Path, raw: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.rename(temporary, path)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


__all__ = (
    "BrokerOutcome",
    "BrokerRequest",
    "FixtureTransport",
    "HTTPXTransport",
    "ReadySlot",
    "SchedulerState",
    "acquire_runtime_lock",
    "build_fake_broker_for_tests",
    "build_live_broker",
    "load_provider_key",
    "revalidate_external_authority",
    "select_ready_slots",
)
