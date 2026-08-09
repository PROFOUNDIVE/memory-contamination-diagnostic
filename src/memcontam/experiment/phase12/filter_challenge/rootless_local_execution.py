from __future__ import annotations

# allow: SIZE_OK — the closed request compiler and its fake broker adapter share one authority.

import base64
import hashlib
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal, TypeAlias

import tiktoken

from memcontam.experiment.phase12.filter_challenge.registry_calibration import (
    BASELINES,
    CANDIDATE_CLASSES,
    TASKS,
    Baseline,
    CandidateClass,
    ScheduledCall,
    Task,
    bct_schedule,
    screening_schedule,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
    build_fake_stage_binding,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_broker import (
    BrokerRequest,
    FixtureTransport,
    build_fake_broker_for_tests,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    canonical_json_file,
    canonical_json_value,
    parse_canonical_object,
    sign_object,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_ledger import (
    LedgerReservation,
    Stage,
)

PROFILE: Final = "local_rootless_non_authoritative"
MODEL: Final = "gpt-4o-2024-11-20"
NOW: Final = "2026-08-09T12:00:00Z"
RENDERER_VERSIONS: Final = (
    "full-history-generate=rootless-adapter-v1",
    "rag-generate=rootless-adapter-v1",
    "bot-problem-distill=rootless-adapter-v1",
    "bot-instantiate-solve=rootless-adapter-v1",
    "reflexion-generate=rootless-adapter-v1",
    "responses-text-extractor=rootless-responses-text-v1",
)
_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_HEX: Final = re.compile(r"[0-9a-f]{64}\Z")
NativeStage: TypeAlias = Literal["answer", "bot_problem_distill", "bot_instantiate_solve"]
MethodStage: TypeAlias = Literal[
    "full_history_generate",
    "rag_generate",
    "bot_problem_distill",
    "bot_instantiate_solve",
    "reflexion_generate",
]
ExecutionOrder: TypeAlias = Literal["screening_control_only", "control_first", "challenge_first"]


@dataclass(frozen=True, slots=True)
class CompileContext:
    attempt_id: str
    stage: Stage
    source_manifest_sha256: str
    input_manifest_sha256: str
    compiler_sha256: str


@dataclass(frozen=True, slots=True)
class CapturedMessage:
    role: Literal["system", "user"]
    content: str

    def to_json(self) -> dict[str, JsonValue]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class SlotCompilation:
    attempt_id: str
    stage: Stage
    source_manifest_sha256: str
    input_manifest_sha256: str
    compiler_sha256: str
    task: Task
    baseline: Baseline
    probe_id: str
    side: Literal["control", "challenge"]
    candidate_class: CandidateClass | None
    scientific_replicate: Literal[1, 2] | None
    executor_replicate_id: Literal[0, 1] | None
    execution_order: ExecutionOrder
    native_stage: NativeStage
    method_stage: MethodStage
    predecessor_slot_ids: tuple[str, ...]
    static_input_sha256: str
    slot_id: str
    messages: tuple[CapturedMessage, ...]
    message_content_sha256s: tuple[str, ...] | None
    input_items_sha256: str | None
    request_sha256: str | None
    request: bytes | None
    compiled_input_tokens: int | None

    def static_input(self) -> dict[str, JsonValue]:
        return {
            "schema_version": "rootless_static_input_v1",
            "profile": PROFILE,
            "kind": "static_input",
            "attempt_id": self.attempt_id,
            "stage": self.stage,
            "source_manifest_sha256": self.source_manifest_sha256,
            "input_manifest_sha256": self.input_manifest_sha256,
            "task": self.task,
            "baseline": self.baseline,
            "probe_id": self.probe_id,
            "side": self.side,
            "candidate_class": self.candidate_class,
            "scientific_replicate": self.scientific_replicate,
            "executor_replicate_id": self.executor_replicate_id,
            "execution_order": self.execution_order,
            "native_stage": self.native_stage,
            "method_stage": self.method_stage,
            "predecessor_slot_ids": list(self.predecessor_slot_ids),
        }

    def manifest(self) -> dict[str, JsonValue]:
        return {
            "schema_version": "rootless_slot_manifest_v1",
            "profile": PROFILE,
            "kind": "slot_manifest",
            "attempt_id": self.attempt_id,
            "stage": self.stage,
            "slot_id": self.slot_id,
            "task": self.task,
            "baseline": self.baseline,
            "probe_id": self.probe_id,
            "side": self.side,
            "candidate_class": self.candidate_class,
            "scientific_replicate": self.scientific_replicate,
            "executor_replicate_id": self.executor_replicate_id,
            "native_stage": self.native_stage,
            "method_stage": self.method_stage,
            "message_roles": [message.role for message in self.messages],
            "message_content_sha256s": (
                None if self.message_content_sha256s is None else list(self.message_content_sha256s)
            ),
            "input_items_sha256": self.input_items_sha256,
            "predecessor_slot_ids": list(self.predecessor_slot_ids),
            "static_input_sha256": self.static_input_sha256,
            "request_sha256": self.request_sha256,
        }


@dataclass(frozen=True, slots=True)
class StageCompilation:
    stage: Stage
    slots: tuple[SlotCompilation, ...]
    ordered_slot_root_sha256: str

    def to_json(self) -> dict[str, JsonValue]:
        manifests: list[JsonValue] = [slot.manifest() for slot in self.slots]
        return {
            "stage": self.stage,
            "ordered_slot_root_sha256": self.ordered_slot_root_sha256,
            "slots": manifests,
        }


@dataclass(frozen=True, slots=True)
class FakeResponse:
    body: bytes
    http_status: int | None = 200
    raised_exception: str | None = None
    response_surfaced: bool = True

    @classmethod
    def completed(cls, fragments: tuple[str, ...]) -> FakeResponse:
        content: list[JsonValue] = [
            {"type": "output_text", "text": fragment} for fragment in fragments
        ]
        return cls(_response_body("completed", content))

    @classmethod
    def refusal(cls) -> FakeResponse:
        return cls(_response_body("completed", [{"type": "refusal", "refusal": "declined"}]))

    @classmethod
    def malformed(cls) -> FakeResponse:
        return cls(b"{")

    @classmethod
    def status(cls, status: str) -> FakeResponse:
        return cls(_response_body(status, []))


@dataclass(frozen=True, slots=True)
class FakeStageResult:
    receipts: tuple[dict[str, JsonValue], ...]
    outcomes: tuple[dict[str, JsonValue], ...]
    provider_calls_issued: int
    not_issued_count: int
    pilot_b_calls: Literal[0] = 0


@dataclass(frozen=True, slots=True)
class _StaticTransport(FixtureTransport):
    fixture_id: str
    response: FakeResponse

    async def exchange(self, slot_id: str, request: bytes) -> dict[str, str | int | bool | None]:
        del request
        return {
            "schema_version": "rootless_fake_http_exchange_v1",
            "profile": PROFILE,
            "kind": "fake_http_exchange",
            "fixture_id": self.fixture_id,
            "slot_id": slot_id,
            "lifecycle_marker": "streaming_body",
            "response_surfaced": self.response.response_surfaced,
            "http_status": self.response.http_status,
            "headers_base64": base64.b64encode(b"\0\0\0\0").decode("ascii"),
            "body_base64": base64.b64encode(self.response.body).decode("ascii"),
            "raised_exception": self.response.raised_exception,
        }


def build_screening_compilation(
    context: CompileContext, probes: dict[str, tuple[str, ...]]
) -> StageCompilation:
    if context.stage != "screening":
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    return _compile(context, screening_schedule(probes))


def build_bct_compilation(
    context: CompileContext, probes: dict[str, tuple[str, ...]]
) -> StageCompilation:
    if context.stage != "bct":
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    return _compile(context, bct_schedule(probes))


def rootless_request_compiler_v1(
    context: CompileContext,
    scheduled: ScheduledCall,
    predecessor_slot_ids: tuple[str, ...],
    predecessor_text: str | None = None,
) -> SlotCompilation:
    match scheduled.replicate:
        case 1:
            scientific: Literal[1, 2] | None = 1
        case 2:
            scientific = 2
        case None:
            scientific = None
        case _:
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    executor: Literal[0, 1] | None
    if context.stage == "screening":
        executor = None
        order: ExecutionOrder = "screening_control_only"
    elif scientific == 1:
        executor = 0
        order = "control_first"
    elif scientific == 2:
        executor = 1
        order = "challenge_first"
    else:
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    method = _method(scheduled.baseline, scheduled.native_stage)
    dynamic = scheduled.native_stage == "bot_instantiate_solve" and predecessor_text is None
    messages = _render_messages(scheduled, method, predecessor_text)
    static: dict[str, JsonValue] = {
        "schema_version": "rootless_static_input_v1",
        "profile": PROFILE,
        "kind": "static_input",
        "attempt_id": context.attempt_id,
        "stage": context.stage,
        "source_manifest_sha256": context.source_manifest_sha256,
        "input_manifest_sha256": context.input_manifest_sha256,
        "task": scheduled.task,
        "baseline": scheduled.baseline,
        "probe_id": scheduled.probe_id,
        "side": scheduled.side,
        "candidate_class": scheduled.candidate_class,
        "scientific_replicate": scientific,
        "executor_replicate_id": executor,
        "execution_order": order,
        "native_stage": scheduled.native_stage,
        "method_stage": method,
        "predecessor_slot_ids": list(predecessor_slot_ids),
    }
    static_hash = _hash(static)
    message_hashes, input_hash, request_hash, request, token_count = _request_values(messages, dynamic)
    preimage: dict[str, JsonValue] = {
        "schema_version": "rootless_slot_manifest_v1",
        "profile": PROFILE,
        "kind": "slot_manifest",
        "attempt_id": context.attempt_id,
        "stage": context.stage,
        "task": scheduled.task,
        "baseline": scheduled.baseline,
        "probe_id": scheduled.probe_id,
        "side": scheduled.side,
        "candidate_class": scheduled.candidate_class,
        "scientific_replicate": scientific,
        "executor_replicate_id": executor,
        "native_stage": scheduled.native_stage,
        "method_stage": method,
        "message_roles": [message.role for message in messages],
        "message_content_sha256s": None if message_hashes is None else list(message_hashes),
        "input_items_sha256": input_hash,
        "predecessor_slot_ids": list(predecessor_slot_ids),
        "static_input_sha256": static_hash,
        "request_sha256": request_hash,
    }
    slot_id = f"slot-{_hash(preimage)[:32]}"
    result = SlotCompilation(
        context.attempt_id,
        context.stage,
        context.source_manifest_sha256,
        context.input_manifest_sha256,
        context.compiler_sha256,
        scheduled.task,
        scheduled.baseline,
        scheduled.probe_id,
        scheduled.side,
        scheduled.candidate_class,
        scientific,
        executor,
        order,
        scheduled.native_stage,
        method,
        predecessor_slot_ids,
        static_hash,
        slot_id,
        messages,
        message_hashes,
        input_hash,
        request_hash,
        request,
        token_count,
    )
    validate_compilation(result)
    return result


def validate_compilation(slot: SlotCompilation) -> None:
    try:
        if (
            _ID.fullmatch(slot.attempt_id) is None
            or _ID.fullmatch(slot.probe_id) is None
            or _ID.fullmatch(slot.slot_id) is None
            or any(_ID.fullmatch(value) is None for value in slot.predecessor_slot_ids)
            or any(
                _HEX.fullmatch(value) is None
                for value in (
                    slot.source_manifest_sha256,
                    slot.input_manifest_sha256,
                    slot.compiler_sha256,
                    slot.static_input_sha256,
                )
            )
        ):
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
        if slot.static_input_sha256 != _hash(slot.static_input()):
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
        if len(slot.predecessor_slot_ids) > 1 or slot.slot_id in slot.predecessor_slot_ids:
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
        slot_preimage = slot.manifest()
        del slot_preimage["slot_id"]
        if slot.slot_id != f"slot-{_hash(slot_preimage)[:32]}":
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
        dynamic = slot.native_stage == "bot_instantiate_solve" and slot.request is None
        values = _request_values(slot.messages, dynamic)
        if values != (
            slot.message_content_sha256s,
            slot.input_items_sha256,
            slot.request_sha256,
            slot.request,
            slot.compiled_input_tokens,
        ):
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    except (TypeError, ValueError) as error:
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID") from error


def compile_request_goldens(probes: dict[str, tuple[str, ...]]) -> dict[str, JsonValue]:
    screening = screening_schedule(probes)
    bct = _bct_golden_schedule(probes)
    entries: list[JsonValue] = [_golden_entry(call) for call in (*screening, *bct)]
    return {
        "schema_version": "rootless_request_golden_manifest_v1",
        "profile": PROFILE,
        "kind": "request_goldens",
        "ordered_entries": entries,
    }


def load_probe_ids(repository_root: Path) -> dict[str, tuple[str, ...]]:
    path = repository_root / "data/phase12/filter_v5_bct_v1/probe_construction_manifest_v1.json"
    payload = parse_canonical_object(path.read_bytes())
    raw_probes = payload.get("probes")
    if not isinstance(raw_probes, dict) or tuple(raw_probes) != TASKS:
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    result: dict[str, tuple[str, ...]] = {}
    for task in TASKS:
        rows = raw_probes.get(task)
        if not isinstance(rows, list) or len(rows) != 6:
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
        probe_ids: list[str] = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("probe_id"), str):
                raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
            probe_id = row.get("probe_id")
            if not isinstance(probe_id, str):
                raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
            probe_ids.append(probe_id)
        if len(set(probe_ids)) != 6:
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
        result[task] = tuple(probe_ids)
    return result


def write_request_goldens(repository_root: Path) -> str:
    probes = load_probe_ids(repository_root)
    first = canonical_json_file(compile_request_goldens(probes))
    second = canonical_json_file(compile_request_goldens(probes))
    if first != second:
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    path = repository_root / "data/phase12/filter_v5_rootless_local/request_goldens.json"
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != first:
            raise RootlessContractError("ROOTLESS_REQUEST_GOLDEN_MISMATCH")
        return hashlib.sha256(first).hexdigest()
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    try:
        offset = 0
        while offset < len(first):
            offset += os.write(descriptor, first[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.rename(temporary, path)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    if path.read_bytes() != second:
        raise RootlessContractError("ROOTLESS_REQUEST_GOLDEN_MISMATCH")
    return hashlib.sha256(first).hexdigest()


async def execute_fake_stage(
    slots: tuple[SlotCompilation, ...], response: FakeResponse, temporary_root: Path
) -> FakeStageResult:
    if not slots:
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    stage = slots[0].stage
    fixture_id = slots[0].attempt_id
    root = temporary_root / "basetemp" / "t5" / "tmp" / "fake-state" / fixture_id
    root.mkdir(mode=0o700, parents=True)
    binding = build_fake_stage_binding(
        fixture_id=fixture_id,
        stage=stage,
        source_manifest_sha256=slots[0].source_manifest_sha256,
        input_manifest_sha256=slots[0].input_manifest_sha256,
        compiler_sha256=slots[0].compiler_sha256,
        schedule_sha256=_schedule_root(slots),
        fake_scenario_sha256=hashlib.sha256(response.body).hexdigest(),
    )
    broker = build_fake_broker_for_tests(binding, _StaticTransport(fixture_id, response), root)
    receipts: list[dict[str, JsonValue]] = []
    outcomes: list[dict[str, JsonValue]] = []
    receipt_hashes: dict[str, str] = {}
    outputs: dict[str, str] = {}
    issued = 0
    blocked = 0
    try:
        for original in slots:
            predecessor_id = original.predecessor_slot_ids[0] if original.predecessor_slot_ids else None
            predecessor_hash = receipt_hashes.get(predecessor_id) if predecessor_id is not None else None
            if predecessor_id is not None and predecessor_id not in outputs:
                receipt = _blocked_receipt(broker, original, predecessor_hash)
                receipts.append(receipt)
                receipt_hashes[original.slot_id] = _hash_file(receipt)
                blocked += 1
                continue
            slot = (
                _materialize_dynamic(original, outputs[predecessor_id])
                if original.request is None and predecessor_id is not None
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
            issued += 1
            receipts.append(outcome.receipt)
            outcomes.append(outcome.typed_outcome)
            receipt_hashes[slot.slot_id] = _hash_file(outcome.receipt)
            parsed = outcome.typed_outcome.get("parsed_output")
            if outcome.provider_status == "completed" and isinstance(parsed, str) and parsed:
                outputs[slot.slot_id] = parsed
    finally:
        broker.close()
    return FakeStageResult(tuple(receipts), tuple(outcomes), issued, blocked)


def _compile(context: CompileContext, schedule: tuple[ScheduledCall, ...]) -> StageCompilation:
    slots: list[SlotCompilation] = []
    groups: dict[tuple[str, ...], str] = {}
    for call in schedule:
        group = (
            call.task,
            call.baseline,
            call.probe_id,
            str(call.candidate_class),
            str(call.replicate),
        )
        predecessor = groups.get(group)
        predecessor_ids = () if predecessor is None else (predecessor,)
        slot = rootless_request_compiler_v1(context, call, predecessor_ids)
        slots.append(slot)
        if call.baseline == "bot_style" or context.stage == "bct":
            groups[group] = slot.slot_id
    result = tuple(slots)
    return StageCompilation(context.stage, result, _schedule_root(result))


def _method(baseline: Baseline, native: str) -> MethodStage:
    match (baseline, native):
        case ("full_history", "answer"):
            return "full_history_generate"
        case ("rag_frozen", "answer"):
            return "rag_generate"
        case ("bot_style", "bot_problem_distill"):
            return "bot_problem_distill"
        case ("bot_style", "bot_instantiate_solve"):
            return "bot_instantiate_solve"
        case ("reflexion_style", "answer"):
            return "reflexion_generate"
        case _:
            raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")


def _render_messages(
    call: ScheduledCall, method: MethodStage, predecessor_text: str | None
) -> tuple[CapturedMessage, ...]:
    system = CapturedMessage("system", f"{method}:rootless-adapter-v1")
    user_value: dict[str, JsonValue] = {
        "task": call.task,
        "probe_id": call.probe_id,
        "side": call.side,
        "candidate_class": call.candidate_class,
        "predecessor_output": predecessor_text,
    }
    return system, CapturedMessage("user", canonical_json_value(user_value).decode("utf-8"))


def _request_values(
    messages: tuple[CapturedMessage, ...], dynamic: bool
) -> tuple[tuple[str, ...] | None, str | None, str | None, bytes | None, int | None]:
    if dynamic:
        return None, None, None, None, None
    hashes = tuple(hashlib.sha256(message.content.encode()).hexdigest() for message in messages)
    items: list[JsonValue] = [
        {
            "role": message.role,
            "content": [{"type": "input_text", "text": message.content}],
        }
        for message in messages
    ]
    items_raw = canonical_json_value(items)
    request_value: dict[str, JsonValue] = {
        "input": items,
        "max_output_tokens": 512,
        "model": MODEL,
        "service_tier": "default",
        "temperature": 0,
    }
    request = canonical_json_value(request_value)
    return (
        hashes,
        hashlib.sha256(items_raw).hexdigest(),
        hashlib.sha256(request).hexdigest(),
        request,
        len(tiktoken.get_encoding("o200k_base").encode(items_raw.decode("utf-8"))),
    )


def _materialize_dynamic(slot: SlotCompilation, predecessor_text: str) -> SlotCompilation:
    call = ScheduledCall(
        call_id="dynamic",
        task=slot.task,
        baseline=slot.baseline,
        probe_id=slot.probe_id,
        side=slot.side,
        candidate_class=slot.candidate_class,
        replicate=slot.scientific_replicate,
        native_stage=slot.native_stage,
    )
    messages = _render_messages(call, slot.method_stage, predecessor_text)
    hashes, input_hash, request_hash, request, tokens = _request_values(messages, False)
    return replace(
        slot,
        messages=messages,
        message_content_sha256s=hashes,
        input_items_sha256=input_hash,
        request_sha256=request_hash,
        request=request,
        compiled_input_tokens=tokens,
    )


def _schedule_root(slots: tuple[SlotCompilation, ...]) -> str:
    leaves = b"".join(bytes.fromhex(_hash(slot.manifest())) for slot in slots)
    return hashlib.sha256(leaves).hexdigest()


def _hash(value: JsonValue) -> str:
    return hashlib.sha256(canonical_json_value(value)).hexdigest()


def _hash_file(value: dict[str, JsonValue]) -> str:
    return hashlib.sha256(canonical_json_file(value)).hexdigest()


def _response_body(status: str, content: list[JsonValue]) -> bytes:
    value: dict[str, JsonValue] = {
        "id": "resp-t5",
        "model": MODEL,
        "object": "response",
        "output": [
            {
                "content": content,
                "id": "msg-t5",
                "role": "assistant",
                "status": status,
                "type": "message",
            }
        ],
        "status": status,
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 5,
            "total_tokens": 105,
        },
    }
    return canonical_json_value(value)


def _blocked_receipt(broker, slot: SlotCompilation, predecessor_hash: str | None) -> dict[str, JsonValue]:
    if predecessor_hash is None:
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    request = LedgerReservation(
        slot.slot_id,
        f"idem-{slot.slot_id}",
        slot.compiler_sha256,
        slot.static_input_sha256,
        predecessor_hash,
        hashlib.sha256(b"").hexdigest(),
        0,
        0,
    )
    append = broker.ledger.not_issued(
        request,
        "DOWNSTREAM_NOT_ISSUED_AFTER_PREDECESSOR_FAILURE",
        NOW,
        compile_status="blocked_predecessor",
        include_request=False,
    )
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_local_call_receipt_v1",
        "profile": PROFILE,
        "kind": "local_call_receipt",
        "attempt_id": broker.attempt_id,
        "stage": slot.stage,
        "slot_id": slot.slot_id,
        "idempotency_key": f"idem-{slot.slot_id}",
        "scientific_replicate": slot.scientific_replicate,
        "executor_replicate_id": slot.executor_replicate_id,
        "issued": False,
        "compiler_sha256": slot.compiler_sha256,
        "static_input_sha256": slot.static_input_sha256,
        "predecessor_receipt_sha256": predecessor_hash,
        "compile_status": "blocked_predecessor",
        "request_sha256": None,
        "request_bytes": None,
        "compiled_input_tokens": None,
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
        "behavioral_reason": "DOWNSTREAM_NOT_ISSUED_AFTER_PREDECESSOR_FAILURE",
        "operational_reason": None,
        "created_at": NOW,
        "key_fingerprint": hashlib.sha256(broker.seed).hexdigest(),
    }
    payload["signature"] = sign_object(broker.seed, "local-call-receipt-v1", payload)
    slot_root = broker.root / "attempts" / broker.attempt_id / slot.stage / "slots" / slot.slot_id
    slot_root.mkdir(mode=0o700, parents=True)
    raw = canonical_json_file(payload)
    descriptor = os.open(slot_root / "call-receipt.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def _golden_entry(call: ScheduledCall) -> dict[str, JsonValue]:
    context = CompileContext("golden", "screening" if call.candidate_class is None else "bct", "1" * 64, "2" * 64, "3" * 64)
    slot = rootless_request_compiler_v1(context, call, ())
    selector: dict[str, JsonValue] = {
        "stage": context.stage,
        "task": call.task,
        "baseline": call.baseline,
        "probe_id": call.probe_id,
        "side": call.side,
        "candidate_class": call.candidate_class,
        "scientific_replicate": call.replicate,
        "executor_replicate_id": slot.executor_replicate_id,
        "native_stage": call.native_stage,
        "method_stage": slot.method_stage,
    }
    return {
        "golden_key": f"g-{_hash(selector)[:32]}",
        **selector,
        "message_roles": [message.role for message in slot.messages],
        "message_content_sha256s": (
            None if slot.message_content_sha256s is None else list(slot.message_content_sha256s)
        ),
        "input_items_sha256": slot.input_items_sha256,
        "request_sha256": slot.request_sha256,
    }


def _bct_golden_schedule(probes: dict[str, tuple[str, ...]]) -> tuple[ScheduledCall, ...]:
    if tuple(probes) != TASKS or any(len(probes[task]) != 6 for task in TASKS):
        raise RootlessContractError("ROOTLESS_COMPILATION_INVALID")
    calls: list[ScheduledCall] = []
    for task in TASKS:
        for baseline in BASELINES:
            stages: tuple[NativeStage, ...] = (
                ("bot_problem_distill", "bot_instantiate_solve")
                if baseline == "bot_style"
                else ("answer",)
            )
            for probe in probes[task]:
                for candidate in CANDIDATE_CLASSES:
                    for replicate in (1, 2):
                        sides: tuple[Literal["control", "challenge"], ...] = (
                            ("control", "challenge")
                            if replicate == 1
                            else ("challenge", "control")
                        )
                        for side in sides:
                            for ordinal, native in enumerate(stages, start=1):
                                calls.append(
                                    ScheduledCall(
                                        call_id=f"golden-{len(calls)}-{ordinal}",
                                        task=task,
                                        baseline=baseline,
                                        probe_id=probe,
                                        side=side,
                                        candidate_class=candidate,
                                        replicate=replicate,
                                        native_stage=native,
                                    )
                                )
    return tuple(calls)


__all__ = (
    "CompileContext",
    "FakeResponse",
    "FakeStageResult",
    "RENDERER_VERSIONS",
    "SlotCompilation",
    "StageCompilation",
    "build_bct_compilation",
    "build_screening_compilation",
    "compile_request_goldens",
    "execute_fake_stage",
    "load_probe_ids",
    "rootless_request_compiler_v1",
    "validate_compilation",
    "write_request_goldens",
)
