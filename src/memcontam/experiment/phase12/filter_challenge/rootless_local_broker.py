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
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, Protocol, cast

import anyio
import httpx

from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
    revalidate_runtime_observations,
    validate_live_stage_binding,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_acknowledgement import (
    parse_timestamp,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    canonical_json_file,
    parse_canonical_object,
    public_key_from_seed,
    sign_object,
    verify_object_signature,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_ledger import (
    GlobalLedger,
    LedgerReservation,
    ProviderUsage,
    Stage,
    actual_cost_nanousd,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_state import read_private_file
from memcontam.verifiers.game24 import verify_expression
from memcontam.verifiers.math_equation_balancer import verify_answer as verify_meb_answer
from memcontam.verifiers.word_sorting import verify_words

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
MAX_CONCURRENT_CALLS: Final = 5
MAX_DISPATCHES_PER_MINUTE: Final = 6
RESERVED_TOKENS_PER_MINUTE: Final = 28_416
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
    task: Literal["game24", "math_equation_balancer", "word_sorting"]
    baseline: Literal["full_history", "rag_frozen", "bot_style", "reflexion_style"]
    probe_id: str
    native_stage: Literal["answer", "bot_problem_distill", "bot_instantiate_solve"]
    candidate_class: Literal["certified_false", "correct", "irrelevant", "ordinary_false"] | None
    scientific_replicate: Literal[1, 2] | None = None
    executor_replicate_id: Literal[0, 1] | None = None


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
    recent_reserved_tokens: tuple[tuple[int, int], ...] = ()


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
    capacity = min(MAX_CONCURRENT_CALLS - state.active_calls, MAX_DISPATCHES_PER_MINUTE - recent)
    if capacity <= 0:
        return ()
    selected: list[ReadySlot] = []
    reserved_tokens = sum(
        tokens
        for timestamp, tokens in state.recent_reserved_tokens
        if timestamp >= window_start
    )
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
        if len(selected) == capacity:
            break
        if reserved_tokens + slot.reserved_tokens > RESERVED_TOKENS_PER_MINUTE:
            continue
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
        repository: Path | None = None,
    ) -> None:
        self.binding = binding
        self.transport = transport
        self.root = root
        self.seed = seed
        self.runtime_lock = runtime_lock
        self.exchange_fixture_id = exchange_fixture_id or str(binding["fixture_id"])
        self.probe_specs = (
            _load_bound_probe_specs(root, binding, seed)
            if binding.get("transport_mode") == "live"
            else _load_probe_specs(repository or Path(__file__).resolve().parents[5])
        )
        self.runtime_authority: tuple[Mapping[str, JsonValue], Mapping[str, JsonValue]] | None = None
        self.stage_clock_started = False
        self.stage_operational_reason: str | None = None
        attempt_id = binding.get("attempt_id", binding.get("fixture_id"))
        stage = binding.get("stage")
        if not isinstance(attempt_id, str) or stage not in {"screening", "bct"}:
            raise RootlessContractError("ROOTLESS_BINDING_INVALID")
        self.attempt_id = attempt_id
        self.stage: Stage = cast(Stage, stage)
        self.ledger = GlobalLedger(root, seed, self.attempt_id, self.stage)

    async def dispatch(self, request: BrokerRequest) -> BrokerOutcome:
        _validate_request_identity(self.attempt_id, request)
        if self.runtime_authority is not None:
            if not self.stage_clock_started:
                _start_stage_clock(self.root, self.attempt_id, self.stage, request.created_at, self.seed)
                self.stage_clock_started = True
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
        if classification.provider_status in {"completed", "failed", "cancelled", "incomplete"}:
            try:
                if usage is None:
                    raise RootlessContractError("ROOTLESS_USAGE_INVALID")
                actual = actual_cost_nanousd(usage)
            except RootlessContractError:
                usage = None
                reason = reason or "ROOTLESS_USAGE_INVALID"
            else:
                retained = False
            if classification.provider_status == "completed" and response_model != EXPECTED_MODEL:
                reason = "ROOTLESS_WRONG_MODEL"
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
        verifier_result = _verify_response(request, parsed_output, self.probe_specs)
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
                "verifier_status": (
                    "success"
                    if verifier_result is not None
                    else "not_run"
                ),
                "parsed_output": parsed_output,
                "verifier_result": verifier_result,
                "answer_call_id": f"{request.slot_id}-answer" if parse_status != "not_run" else None,
                "parsed_response_source_call_id": f"{request.slot_id}-answer" if parse_status != "not_run" else None,
                "parser_result_id": f"{request.slot_id}-parser" if parse_status != "not_run" else None,
                "verifier_result_id": (
                    f"{request.slot_id}-verifier"
                    if verifier_result is not None
                    else None
                ),
                "behavioral_reason": (
                    f"{request.side.upper()}_PROVIDER_FAILURE"
                    if reason is not None
                    or classification.provider_status in {"failed", "cancelled", "incomplete"}
                    else f"{request.side.upper()}_PARSE_FAILURE"
                    if classification.provider_status == "completed" and parse_status == "failure"
                    else "CONTROL_NOT_CLEAN_SOLVABLE"
                    if request.side == "control"
                    and verifier_result is False
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
                "scientific_replicate": request.scientific_replicate,
                "executor_replicate_id": request.executor_replicate_id,
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
                "verifier_result_id": typed["verifier_result_id"],
                "behavioral_reason": typed["behavioral_reason"],
                "operational_reason": reason,
                "created_at": request.created_at,
                "key_fingerprint": archive["key_fingerprint"],
            },
        )
        _atomic_new(slot_root / "call-receipt.json", canonical_json_file(receipt))
        if reason is not None:
            self.stage_operational_reason = reason
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

    def account_input_cap(self, request: BrokerRequest) -> dict[str, JsonValue]:
        _validate_request_identity(self.attempt_id, request)
        request_sha256 = hashlib.sha256(request.request).hexdigest()
        append = self.ledger.not_issued(
            LedgerReservation(
                request.slot_id,
                request.idempotency_key,
                request.compiler_sha256,
                request.static_input_sha256,
                request.predecessor_receipt_sha256,
                request_sha256,
                len(request.request),
                request.compiled_input_tokens,
            ),
            "ROOTLESS_INPUT_CAP_EXCEEDED",
            request.created_at,
            compile_status="compiled",
            include_request=True,
        )
        payload: dict[str, JsonValue] = {
            "schema_version": "rootless_local_call_receipt_v1",
            "profile": PROFILE,
            "kind": "local_call_receipt",
            "attempt_id": self.attempt_id,
            "stage": self.stage,
            "slot_id": request.slot_id,
            "idempotency_key": request.idempotency_key,
            "scientific_replicate": request.scientific_replicate,
            "executor_replicate_id": request.executor_replicate_id,
            "issued": False,
            "compiler_sha256": request.compiler_sha256,
            "static_input_sha256": request.static_input_sha256,
            "predecessor_receipt_sha256": request.predecessor_receipt_sha256,
            "compile_status": "compiled",
            "request_sha256": request_sha256,
            "request_bytes": len(request.request),
            "compiled_input_tokens": request.compiled_input_tokens,
            "reservation_record_sha256": None,
            "reservation_head_sha256": None,
            "not_issued_record_sha256": append.record_sha256,
            "not_issued_head_sha256": append.head_sha256,
            "archive_manifest_sha256": None,
            "typed_outcome_sha256": None,
            "settlement_record_sha256": None,
            "settlement_head_sha256": None,
            "provider_status": None,
            "http_status": None,
            "response_model": None,
            "usage_input_tokens": None,
            "cached_input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "actual_nanousd": None,
            "reservation_retained": False,
            "answer_call_id": None,
            "parsed_response_source_call_id": None,
            "parser_result_id": None,
            "verifier_result_id": None,
            "behavioral_reason": None,
            "operational_reason": "ROOTLESS_INPUT_CAP_EXCEEDED",
            "created_at": request.created_at,
            "key_fingerprint": hashlib.sha256(public_key_from_seed(self.seed)).hexdigest(),
        }
        receipt = _signed(self.seed, "local-call-receipt-v1", payload)
        slot_root = self.root / "attempts" / self.attempt_id / self.stage / "slots" / request.slot_id
        slot_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        _atomic_new(slot_root / "call-receipt.json", canonical_json_file(receipt))
        self.stage_operational_reason = "ROOTLESS_INPUT_CAP_EXCEEDED"
        return receipt

    def close(self) -> None:
        self.runtime_lock.close()


def build_fake_broker_for_tests(
    binding: dict[str, JsonValue], fixture_transport: FixtureTransport, fake_root: Path
) -> FakeBroker:
    if os.geteuid() == 0:
        raise RootlessContractError("ROOTLESS_ROOT_EXECUTION_FORBIDDEN")
    fixture_id = binding.get("fixture_id")
    stage = binding.get("stage")
    lock_path = fake_root / "runtime.lock"
    attempt_root = fake_root / "attempts" / str(fixture_id) / str(stage)
    continuing = lock_path.is_file() and (stage == "bct" or attempt_root.is_dir())
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
        or (stage == "bct" and not continuing)
        or (fake_root / "live-attempt-claim.json").exists()
        or (not continuing and (fake_root / "ledger").exists())
        or (not continuing and (fake_root / "attempts").exists())
        or (
            continuing
            and (fake_root / "terminals" / str(fixture_id) / f"{stage}.json").exists()
        )
    ):
        raise RootlessContractError("ROOTLESS_FAKE_BOUNDARY_INVALID")
    seed = hashlib.sha256(f"rootless-fixture:{fixture_id}".encode()).digest()
    if continuing:
        if stage == "bct":
            _validate_fake_bct_predecessor(fake_root, fixture_id, seed)
            _validate_ordinary_authority(Path(__file__).resolve().parents[5])
    else:
        keys_root = fake_root / "keys"
        keys_root.mkdir(mode=0o700)
        _atomic_new(keys_root / "ed25519-private.key", seed)
        descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        os.fsync(descriptor)
        os.close(descriptor)
    return FakeBroker(binding, fixture_transport, fake_root, seed, acquire_runtime_lock(lock_path))


def _validate_fake_bct_predecessor(root: Path, attempt_id: str, seed: bytes) -> None:
    terminal_path = root / "terminals" / attempt_id / "screening.json"
    freeze_path = root / "freeze" / attempt_id / "freeze_b.json"
    terminal = parse_canonical_object(terminal_path.read_bytes())
    freeze = parse_canonical_object(freeze_path.read_bytes())
    if (
        terminal.get("status") != "completed_estimable"
        or terminal.get("reason_code") != "SCREENING_ESTIMABLE"
        or freeze.get("screening_stage_terminal_sha256")
        != hashlib.sha256(terminal_path.read_bytes()).hexdigest()
        or freeze.get("attempt_id") != attempt_id
        or (root / "terminals" / attempt_id / "bct.json").exists()
    ):
        raise RootlessContractError("ROOTLESS_FAKE_BOUNDARY_INVALID")
    public_key = public_key_from_seed(seed)
    for domain, value in (("stage-terminal-v1", terminal), ("freeze-b-v1", freeze)):
        signature = value.get("signature")
        if not isinstance(signature, str):
            raise RootlessContractError("ROOTLESS_FAKE_BOUNDARY_INVALID")
        unsigned = dict(value)
        del unsigned["signature"]
        from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
            verify_object_signature,
        )

        verify_object_signature(public_key, domain, unsigned, signature)


def build_live_broker(
    binding: dict[str, JsonValue], state_root: Path, repository: Path
) -> FakeBroker:
    if os.geteuid() == 0:
        raise RootlessContractError("ROOTLESS_ROOT_EXECUTION_FORBIDDEN")
    validate_live_stage_binding(binding)
    if "fake-state" in state_root.parts or binding.get("transport_mode") != "live":
        raise RootlessContractError("ROOTLESS_BINDING_INVALID")
    if binding.get("stage") == "bct":
        _validate_ordinary_authority(repository)
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
    _establish_live_claim(state_root, binding, seed)
    broker = FakeBroker(binding, HTTPXTransport(secret), state_root, seed, lock, "live", repository)
    broker.runtime_authority = (runtime_manifest, decoding_authority)
    return broker


def _validate_ordinary_authority(repository: Path) -> None:
    from memcontam.experiment.phase12.filter_challenge.ordinary_authority import (
        OrdinaryAuthorityError,
        validate_ordinary_authority,
    )

    try:
        validate_ordinary_authority(repository)
    except OrdinaryAuthorityError as error:
        raise RootlessContractError(error.code) from error


def _establish_live_claim(
    root: Path, binding: Mapping[str, JsonValue], seed: bytes
) -> None:
    attempt_id = binding.get("attempt_id")
    stage = binding.get("stage")
    if (
        not isinstance(attempt_id, str)
        or stage not in {"screening", "bct"}
    ):
        raise RootlessContractError("ROOTLESS_BINDING_INVALID")
    fingerprint = hashlib.sha256(public_key_from_seed(seed)).hexdigest()
    created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    claim = _signed(
        seed,
        "live-attempt-claim-v1",
        {
            "schema_version": "rootless_live_attempt_claim_v1",
            "profile": PROFILE,
            "kind": "live_attempt_claim",
            "attempt_id": attempt_id,
            "plan_binding_sha256": binding["plan_binding_sha256"],
            "execution_commit": binding["execution_commit"],
            "created_at": created_at,
            "key_fingerprint": fingerprint,
        },
    )
    claim_raw = canonical_json_file(claim)
    claim_path = root / "live-attempt-claim.json"
    if claim_path.is_file():
        claim_value = _verified_clock_object(
            _regular_bytes(claim_path), seed, "live-attempt-claim-v1"
        )
        if (
            claim_value.get("attempt_id") != attempt_id
            or claim_value.get("plan_binding_sha256") != binding["plan_binding_sha256"]
            or claim_value.get("execution_commit") != binding["execution_commit"]
        ):
            raise RootlessContractError("ROOTLESS_CLAIM_ALREADY_EXISTS")
        checkpoints = tuple(sorted((root / "runtime-clock").glob("*.json")))
        if len(checkpoints) not in {1, 2, 3}:
            raise RootlessContractError("ROOTLESS_INTERRUPTED_UNCLEAN")
        previous_hash: str | None = None
        previous_monotonic: int | None = None
        previous_realtime = None
        current_boot_hash = _boot_id_sha256()
        for sequence, checkpoint_path in enumerate(checkpoints):
            checkpoint_raw = _regular_bytes(checkpoint_path)
            checkpoint = parse_canonical_object(checkpoint_raw)
            checkpoint_signature = checkpoint.get("signature")
            if not isinstance(checkpoint_signature, str):
                raise RootlessContractError("ROOTLESS_INTERRUPTED_UNCLEAN")
            checkpoint_unsigned = dict(checkpoint)
            del checkpoint_unsigned["signature"]
            try:
                verify_object_signature(
                    public_key_from_seed(seed),
                    "runtime-clock-checkpoint-v1",
                    checkpoint_unsigned,
                    checkpoint_signature,
                )
            except RootlessContractError as error:
                raise RootlessContractError("ROOTLESS_INTERRUPTED_UNCLEAN") from error
            if (
                checkpoint.get("sequence") != sequence
                or checkpoint.get("attempt_id") != attempt_id
                or checkpoint.get("previous_checkpoint_sha256") != previous_hash
                or checkpoint.get("boot_id_sha256") != current_boot_hash
                or checkpoint_path.name
                != f"{sequence:08d}-{hashlib.sha256(checkpoint_raw).hexdigest()}.json"
            ):
                raise RootlessContractError("ROOTLESS_INTERRUPTED_UNCLEAN")
            monotonic = checkpoint.get("checkpoint_monotonic_ns")
            realtime = checkpoint.get("checkpoint_realtime")
            if (
                not isinstance(monotonic, int)
                or isinstance(monotonic, bool)
                or not isinstance(realtime, str)
            ):
                raise RootlessContractError("ROOTLESS_INTERRUPTED_UNCLEAN")
            try:
                parsed_realtime = parse_timestamp(realtime)
            except RootlessContractError as error:
                raise RootlessContractError("ROOTLESS_INTERRUPTED_UNCLEAN") from error
            if (
                previous_monotonic is not None
                and (
                    monotonic < previous_monotonic
                    or previous_realtime is None
                    or parsed_realtime < previous_realtime
                )
            ):
                raise RootlessContractError("ROOTLESS_INTERRUPTED_UNCLEAN")
            if sequence == 0 and any(
                checkpoint.get(field) is not None
                for field in ("stage", "stage_started_at", "stage_monotonic_ns")
            ):
                raise RootlessContractError("ROOTLESS_INTERRUPTED_UNCLEAN")
            if sequence == 1 and (
                checkpoint.get("stage") not in {"screening", "bct"}
                or not isinstance(checkpoint.get("stage_started_at"), str)
                or not isinstance(checkpoint.get("stage_monotonic_ns"), int)
            ):
                raise RootlessContractError("ROOTLESS_INTERRUPTED_UNCLEAN")
            previous_hash = hashlib.sha256(checkpoint_raw).hexdigest()
            previous_monotonic = monotonic
            previous_realtime = parsed_realtime
        return
    _atomic_new(claim_path, claim_raw)
    monotonic_ns = time.monotonic_ns()
    checkpoint = _signed(
        seed,
        "runtime-clock-checkpoint-v1",
        {
            "schema_version": "rootless_runtime_clock_checkpoint_v1",
            "profile": PROFILE,
            "kind": "runtime_clock_checkpoint",
            "sequence": 0,
            "previous_checkpoint_sha256": None,
            "attempt_id": attempt_id,
            "stage": None,
            "boot_id_sha256": _boot_id_sha256(),
            "realtime_at_claim": created_at,
            "monotonic_ns_at_claim": monotonic_ns,
            "stage_started_at": None,
            "stage_monotonic_ns": None,
            "checkpoint_realtime": created_at,
            "checkpoint_monotonic_ns": monotonic_ns,
            "key_fingerprint": fingerprint,
        },
    )
    checkpoint_raw = canonical_json_file(checkpoint)
    checkpoint_hash = hashlib.sha256(checkpoint_raw).hexdigest()
    clock_root = root / "runtime-clock"
    clock_root.mkdir(mode=0o700, exist_ok=True)
    _atomic_new(clock_root / f"00000000-{checkpoint_hash}.json", checkpoint_raw)


def _regular_bytes(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise RootlessContractError("ROOTLESS_INTERRUPTED_UNCLEAN")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1_048_576):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _boot_id_sha256() -> str:
    descriptor = os.open(
        "/proc/sys/kernel/random/boot_id", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        return hashlib.sha256(os.read(descriptor, 128).strip()).hexdigest()
    finally:
        os.close(descriptor)


def _start_stage_clock(
    root: Path, attempt_id: str, stage: Stage, started_at: str, seed: bytes
) -> None:
    checkpoints = tuple(sorted((root / "runtime-clock").glob("*.json")))
    if not checkpoints:
        raise RootlessContractError("ROOTLESS_INTERRUPTED_UNCLEAN")
    if len(checkpoints) == 2:
        checkpoint = parse_canonical_object(_regular_bytes(checkpoints[1]))
        if checkpoint.get("stage") != stage:
            if stage != "bct" or checkpoint.get("stage") != "screening":
                raise RootlessContractError("ROOTLESS_INTERRUPTED_UNCLEAN")
        else:
            return
    if len(checkpoints) == 3:
        checkpoint = parse_canonical_object(_regular_bytes(checkpoints[2]))
        if checkpoint.get("stage") != stage:
            raise RootlessContractError("ROOTLESS_INTERRUPTED_UNCLEAN")
        return
    previous_raw = _regular_bytes(checkpoints[-1])
    previous = parse_canonical_object(previous_raw)
    if previous.get("boot_id_sha256") != _boot_id_sha256():
        raise RootlessContractError("ROOTLESS_INTERRUPTED_UNCLEAN")
    monotonic_ns = time.monotonic_ns()
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_runtime_clock_checkpoint_v1",
        "profile": PROFILE,
        "kind": "runtime_clock_checkpoint",
        "sequence": len(checkpoints),
        "previous_checkpoint_sha256": hashlib.sha256(previous_raw).hexdigest(),
        "attempt_id": attempt_id,
        "stage": stage,
        "boot_id_sha256": previous["boot_id_sha256"],
        "realtime_at_claim": parse_canonical_object(
            _regular_bytes(checkpoints[0])
        )["realtime_at_claim"],
        "monotonic_ns_at_claim": parse_canonical_object(
            _regular_bytes(checkpoints[0])
        )["monotonic_ns_at_claim"],
        "stage_started_at": started_at,
        "stage_monotonic_ns": monotonic_ns,
        "checkpoint_realtime": started_at,
        "checkpoint_monotonic_ns": monotonic_ns,
        "key_fingerprint": hashlib.sha256(public_key_from_seed(seed)).hexdigest(),
    }
    checkpoint = _signed(seed, "runtime-clock-checkpoint-v1", payload)
    raw = canonical_json_file(checkpoint)
    digest = hashlib.sha256(raw).hexdigest()
    _atomic_new(root / "runtime-clock" / f"{len(checkpoints):08d}-{digest}.json", raw)


def _verified_clock_object(raw: bytes, seed: bytes, domain: str) -> dict[str, JsonValue]:
    value = parse_canonical_object(raw)
    signature = value.get("signature")
    if not isinstance(signature, str):
        raise RootlessContractError("ROOTLESS_INTERRUPTED_UNCLEAN")
    unsigned = dict(value)
    del unsigned["signature"]
    try:
        verify_object_signature(public_key_from_seed(seed), domain, unsigned, signature)
    except RootlessContractError as error:
        raise RootlessContractError("ROOTLESS_INTERRUPTED_UNCLEAN") from error
    return value


def _load_probe_specs(repository: Path) -> dict[str, tuple[str, dict[str, JsonValue]]]:
    path = repository / "data/phase12/filter_v5_bct_v1/probe_construction_manifest_v1.json"
    value = parse_canonical_object(path.read_bytes())
    probes = value.get("probes")
    if not isinstance(probes, dict):
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    result: dict[str, tuple[str, dict[str, JsonValue]]] = {}
    for task, rows in probes.items():
        if not isinstance(task, str) or not isinstance(rows, list):
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
        for row in rows:
            if not isinstance(row, dict):
                raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
            probe_id = row.get("probe_id")
            certificate = row.get("certificate")
            if not isinstance(probe_id, str) or not isinstance(certificate, dict):
                raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
            result[probe_id] = task, certificate
    return result


def _load_bound_probe_specs(
    root: Path, binding: Mapping[str, JsonValue], seed: bytes
) -> dict[str, tuple[str, dict[str, JsonValue]]]:
    attempt_id = binding.get("attempt_id")
    expected_hash = binding.get("input_manifest_sha256")
    if not isinstance(attempt_id, str) or not isinstance(expected_hash, str):
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    manifest_path = root / "manifests" / attempt_id / "input.json"
    raw = read_private_file(manifest_path)
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    manifest = parse_canonical_object(raw)
    signature = manifest.get("signature")
    if not isinstance(signature, str):
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    unsigned = dict(manifest)
    del unsigned["signature"]
    from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
        verify_object_signature,
    )

    verify_object_signature(public_key_from_seed(seed), "input-manifest-v1", unsigned, signature)
    entries = manifest.get("ordered_inputs")
    if not isinstance(entries, list):
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    result: dict[str, tuple[str, dict[str, JsonValue]]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("role") not in {
            "screening-probes",
            "bct-probes",
        }:
            continue
        path_value = entry.get("absolute_path")
        digest = entry.get("sha256")
        if not isinstance(path_value, str) or not isinstance(digest, str):
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
        probe_raw = read_private_file(Path(path_value))
        if hashlib.sha256(probe_raw).hexdigest() != digest:
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
        value = parse_canonical_object(probe_raw)
        probes = value.get("probes")
        if not isinstance(probes, dict):
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
        for task, rows in probes.items():
            if not isinstance(task, str) or not isinstance(rows, list):
                raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
            for row in rows:
                if not isinstance(row, dict):
                    raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
                probe_id = row.get("probe_id")
                certificate = row.get("certificate")
                if not isinstance(probe_id, str) or not isinstance(certificate, dict):
                    raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
                result[probe_id] = (task, certificate)
    if not result:
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    return result


def _verify_response(
    request: BrokerRequest,
    parsed_output: JsonValue,
    specs: Mapping[str, tuple[str, dict[str, JsonValue]]],
) -> bool | None:
    if (
        request.native_stage == "bot_problem_distill"
        or request.task is None
        or request.probe_id is None
        or not isinstance(parsed_output, str)
    ):
        return None
    spec = specs.get(request.probe_id)
    if spec is None or spec[0] != request.task:
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    certificate = spec[1]
    if request.task == "game24":
        numbers = certificate.get("numbers")
        target = certificate.get("target")
        if not isinstance(numbers, list) or not isinstance(target, int):
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
        typed_numbers = [value for value in numbers if isinstance(value, int) and not isinstance(value, bool)]
        if len(typed_numbers) != len(numbers):
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
        return verify_expression(parsed_output, typed_numbers, target).is_correct
    if request.task == "math_equation_balancer":
        return verify_meb_answer(parsed_output, certificate).is_correct
    if request.task == "word_sorting":
        words = certificate.get("correct_order")
        if not isinstance(words, list):
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
        typed_words = [value for value in words if isinstance(value, str)]
        if len(typed_words) != len(words):
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
        return verify_words(parsed_output.split(), typed_words).is_correct
    raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")


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


def _validate_request_identity(attempt_id: str, request: BrokerRequest) -> None:
    digest = hashlib.sha256(
        attempt_id.encode() + b"\0" + request.slot_id.encode()
    ).hexdigest()
    if request.idempotency_key != f"i-{digest[:32]}":
        raise RootlessContractError("ROOTLESS_BINDING_INVALID")


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
