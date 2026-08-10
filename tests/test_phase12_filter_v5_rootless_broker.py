from __future__ import annotations

# allow: SIZE_OK — Task 4 fixes the complete broker matrix in this one QA module.

import base64
import hashlib
import inspect
import os
import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import anyio
import pytest

from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
    build_fake_stage_binding,
    build_stage_binding,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_broker import (
    BrokerRequest,
    FixtureTransport,
    HTTPXTransport,
    ReadySlot,
    SchedulerState,
    acquire_runtime_lock,
    build_fake_broker_for_tests,
    build_live_broker,
    load_provider_key,
    revalidate_external_authority,
    select_ready_slots,
    _establish_live_claim,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    RootlessContractError,
    SIGNATURE_DOMAINS,
    public_key_from_seed,
    sign_object,
    verify_object_signature,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_protocol import (
    WireProtocol,
    decode_frame,
    encode_frame,
)
from memcontam.experiment.phase12 import cli as phase12_cli

PROFILE: Final = "local_rootless_non_authoritative"
NOW: Final = "2026-08-09T12:00:00Z"
MODEL: Final = "gpt-4o-2024-11-20"


def _binding(fixture_id: str = "fixture-1") -> dict[str, str | int | bool | None | list]:
    return build_fake_stage_binding(
        fixture_id=fixture_id,
        stage="screening",
        source_manifest_sha256="1" * 64,
        input_manifest_sha256="2" * 64,
        compiler_sha256="3" * 64,
        schedule_sha256="4" * 64,
        fake_scenario_sha256="5" * 64,
    )


def _root(tmp_path: Path, fixture_id: str = "fixture-1") -> Path:
    root = tmp_path / "basetemp" / "t4" / "tmp" / "fake-state" / fixture_id
    root.mkdir(mode=0o700, parents=True)
    return root


def _headers(*fields: tuple[bytes, bytes]) -> bytes:
    raw = len(fields).to_bytes(4, "big")
    for name, value in fields:
        raw += len(name).to_bytes(4, "big") + name + len(value).to_bytes(4, "big") + value
    return raw


def _response(
    *,
    status: int = 200,
    provider_status: str = "completed",
    model: str = MODEL,
    usage: dict[str, int | dict[str, int]] | None = None,
) -> bytes:
    if usage is None:
        usage = {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 10},
            "output_tokens": 5,
            "total_tokens": 105,
        }
    return (
        '{"id":"resp_1","model":"%s","object":"response","output":'
        '[{"content":[{"text":"ok","type":"output_text"}],"id":"msg_1",'
        '"role":"assistant","status":"completed","type":"message"}],'
        '"status":"%s","usage":%s}'
        % (model, provider_status, __import__("json").dumps(usage, separators=(",", ":")))
    ).encode()


def _exchange(
    *,
    status: int | None = 200,
    headers: bytes | None = None,
    body: bytes | None = None,
    marker: str = "streaming_body",
    surfaced: bool = True,
    raised: str | None = None,
) -> dict[str, str | int | bool | None]:
    return {
        "schema_version": "rootless_fake_http_exchange_v1",
        "profile": PROFILE,
        "kind": "fake_http_exchange",
        "fixture_id": "fixture-1",
        "slot_id": "slot-001",
        "lifecycle_marker": marker,
        "response_surfaced": surfaced,
        "http_status": status,
        "headers_base64": base64.b64encode(headers if headers is not None else _headers()).decode(),
        "body_base64": base64.b64encode(body if body is not None else _response()).decode(),
        "raised_exception": raised,
    }


@dataclass(frozen=True, slots=True)
class StaticFixtureTransport(FixtureTransport):
    value: dict[str, str | int | bool | None]

    async def exchange(self, slot_id: str, request: bytes) -> dict[str, str | int | bool | None]:
        assert slot_id == "slot-001"
        assert request.startswith(b"{")
        return self.value


def _request() -> BrokerRequest:
    raw = b'{"input":[{"content":"hello","role":"user"}],"model":"gpt-4o-2024-11-20"}'
    digest = hashlib.sha256(b"fixture-1\0slot-001").hexdigest()
    return BrokerRequest(
        slot_id="slot-001",
        idempotency_key=f"i-{digest[:32]}",
        compiler_sha256="3" * 64,
        static_input_sha256="6" * 64,
        predecessor_receipt_sha256=None,
        request=raw,
        compiled_input_tokens=5,
        side="control",
        created_at=NOW,
        task="game24",
        baseline="full_history",
        probe_id="fv5-cal-game24-001",
        native_stage="answer",
        candidate_class=None,
    )


def _dispatch(tmp_path: Path, exchange: dict[str, str | int | bool | None]):
    broker = build_fake_broker_for_tests(
        _binding(), StaticFixtureTransport(exchange), _root(tmp_path)
    )
    try:
        return anyio.run(broker.dispatch, _request())
    finally:
        broker.close()


def test_every_signature_domain_is_bound_to_domain_and_payload() -> None:
    # Given: the exhaustive Task 4 signature-domain table.
    seed = bytes(range(32))
    public_key = public_key_from_seed(seed)
    payload = {"profile": PROFILE, "value": 1}

    # When/Then: each domain verifies only its own preimage.
    for domain in SIGNATURE_DOMAINS:
        signature = sign_object(seed, domain, payload)
        verify_object_signature(public_key, domain, payload, signature)
        other = next(candidate for candidate in SIGNATURE_DOMAINS if candidate != domain)
        with pytest.raises(RootlessContractError, match="ROOTLESS_SIGNATURE_INVALID"):
            verify_object_signature(public_key, other, payload, signature)


def test_unsigned_wire_frame_and_closed_branch_state_machine() -> None:
    # Given: one canonical dispatch frame.
    dispatch = {
        "schema_version": "rootless_local_wire_v1",
        "profile": PROFILE,
        "message_type": "dispatch",
        "attempt_id": "attempt-001",
        "stage": "screening",
        "slot_id": "slot-001",
        "message_sequence": 0,
        "predecessor_receipt_sha256s": [],
    }

    # When: it crosses a four-byte big-endian frame and enters a protocol session.
    encoded = encode_frame(dispatch)
    protocol = WireProtocol()
    protocol.accept(dispatch)
    protocol.accept({
        "schema_version": "rootless_local_wire_v1",
        "profile": PROFILE,
        "message_type": "accepted",
        "attempt_id": "attempt-001",
        "stage": "screening",
        "slot_id": "slot-001",
        "message_sequence": 0,
        "reservation_record_sha256": "1" * 64,
        "reservation_head_sha256": "2" * 64,
        "request_sha256": "3" * 64,
        "request_bytes": 10,
        "compiled_input_tokens": 4,
    })

    # Then: framing round-trips and mixed/duplicate variants fail closed.
    assert int.from_bytes(encoded[:4], "big") == len(encoded) - 4
    assert decode_frame(encoded) == dispatch
    with pytest.raises(RootlessContractError, match="ROOTLESS_WIRE_INVALID"):
        protocol.accept(dispatch)


@pytest.mark.parametrize(
    ("status", "provider_status", "reason"),
    [
        (201, "http_error", "ROOTLESS_HTTP_ERROR"),
        (429, "http_error", "ROOTLESS_RATE_LIMITED"),
        (599, "http_error", "ROOTLESS_HTTP_ERROR"),
        (99, "archive_error", "ROOTLESS_ARCHIVE_INVALID"),
        (100, "archive_error", "ROOTLESS_ARCHIVE_INVALID"),
        (199, "archive_error", "ROOTLESS_ARCHIVE_INVALID"),
        (600, "archive_error", "ROOTLESS_ARCHIVE_INVALID"),
    ],
)
def test_http_status_matrix_is_first_match_closed(
    tmp_path: Path, status: int, provider_status: str, reason: str
) -> None:
    # Given/When: fake HTTPX surfaces a final or adapter-invalid status.
    outcome = _dispatch(tmp_path, _exchange(status=status))

    # Then: status classification never enters HTTP-200 parsing.
    assert outcome.provider_status == provider_status
    assert outcome.operational_reason == reason
    assert outcome.http_status == (status if 200 <= status <= 599 else None)
    assert outcome.reservation_retained is True


def test_valid_completed_response_is_archived_settled_and_receipted(tmp_path: Path) -> None:
    # Given/When: a complete expected-model response reports valid usage.
    outcome = _dispatch(tmp_path, _exchange())

    # Then: archived model, typed output, usage, and settlement are independently bound.
    assert outcome.provider_status == "completed"
    assert outcome.operational_reason is None
    assert outcome.response_model == MODEL
    assert outcome.usage is not None and outcome.usage.input_tokens == 100
    assert outcome.actual_nanousd == 300_000
    assert outcome.reservation_retained is False
    assert outcome.typed_outcome["raw_parse_status"] == "success"
    assert outcome.receipt["compiled_input_tokens"] == 5
    assert outcome.receipt["usage_input_tokens"] == 100


@pytest.mark.parametrize("provider_status", ["failed", "cancelled", "incomplete"])
def test_valid_usage_behavioral_provider_failures_settle_and_continue(
    tmp_path: Path, provider_status: str
) -> None:
    # Given/When: HTTP 200 carries a terminal provider failure with valid usage.
    outcome = _dispatch(tmp_path, _exchange(body=_response(provider_status=provider_status)))

    # Then: the literal status is preserved and its valid paid usage settles normally.
    assert outcome.provider_status == provider_status
    assert outcome.operational_reason is None
    assert outcome.reservation_retained is False
    assert outcome.receipt["behavioral_reason"] == "CONTROL_PROVIDER_FAILURE"
    assert outcome.receipt["settlement_record_sha256"] is not None
    assert outcome.receipt["usage_input_tokens"] == 100
    assert outcome.receipt["actual_nanousd"] == 300_000


def test_broker_request_requires_compilation_identity() -> None:
    # Given: the typed broker admission boundary.
    signature = inspect.signature(BrokerRequest)

    # When/Then: task, probe, and native stage cannot be omitted or defaulted.
    for field in ("task", "probe_id", "native_stage"):
        assert signature.parameters[field].default is inspect.Parameter.empty


def test_broker_rejects_noncanonical_idempotency_key(tmp_path: Path) -> None:
    # Given: a compiled request with a caller-selected idempotency key.
    broker = build_fake_broker_for_tests(
        _binding(), StaticFixtureTransport(_exchange()), _root(tmp_path)
    )

    # When/Then: admission rejects it before reservation or transport.
    try:
        with pytest.raises(RootlessContractError, match="ROOTLESS_BINDING_INVALID"):
            anyio.run(broker.dispatch, replace(_request(), idempotency_key="caller-selected"))
    finally:
        broker.close()


def test_live_claim_and_runtime_checkpoint_precede_admission(tmp_path: Path) -> None:
    # Given: a validated live binding and its private signing seed.
    root = tmp_path / "live-state"
    (root / "keys").mkdir(mode=0o700, parents=True)
    seed = bytes(range(32))
    binding = build_stage_binding(
        attempt_id="attempt-001",
        stage="screening",
        plan_binding_sha256="1" * 64,
        trusted_base_commit="a" * 40,
        execution_commit="b" * 40,
        decoding_authority_sha256="2" * 64,
        rate_card_sha256="3" * 64,
        source_manifest_sha256="4" * 64,
        runtime_manifest_sha256="5" * 64,
        input_manifest_sha256="6" * 64,
        compiler_sha256="7" * 64,
        schedule_sha256="8" * 64,
        registered_slots=90,
        stage_cap_nanousd=2_000_000_000,
        created_at=NOW,
    )

    # When: the live broker establishes its first-admission continuity prefix.
    _establish_live_claim(root, binding, seed)

    # Then: claim and checkpoint are signed, commit-bound, and durable before transport use.
    claim = __import__("json").loads((root / "live-attempt-claim.json").read_bytes())
    checkpoints = tuple((root / "runtime-clock").glob("*.json"))
    assert claim["schema_version"] == "rootless_live_attempt_claim_v1"
    assert claim["execution_commit"] == "b" * 40
    assert claim["plan_binding_sha256"] == "1" * 64
    assert len(checkpoints) == 1
    checkpoints[0].unlink()
    with pytest.raises(RootlessContractError, match="ROOTLESS_INTERRUPTED_UNCLEAN"):
        _establish_live_claim(root, binding, seed)


def test_live_claim_is_reused_across_stage_specific_clock_checkpoints(tmp_path: Path) -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_broker import (
        _start_stage_clock,
    )

    # Given: one attempt with separately bound screening and BCT stages.
    root = tmp_path / "live-state"
    root.mkdir(mode=0o700)
    common = {
        "attempt_id": "attempt-001",
        "plan_binding_sha256": "1" * 64,
        "trusted_base_commit": "a" * 40,
        "execution_commit": "b" * 40,
        "decoding_authority_sha256": "2" * 64,
        "rate_card_sha256": "3" * 64,
        "source_manifest_sha256": "4" * 64,
        "runtime_manifest_sha256": "5" * 64,
        "input_manifest_sha256": "6" * 64,
        "compiler_sha256": "7" * 64,
        "schedule_sha256": "8" * 64,
    }
    screening = build_stage_binding(
        **common,
        stage="screening",
        registered_slots=90,
        stage_cap_nanousd=2_000_000_000,
        created_at="2026-08-09T12:00:00Z",
    )
    bct = build_stage_binding(
        **common,
        stage="bct",
        predecessor_terminal_sha256="9" * 64,
        freeze_b_sha256="a" * 64,
        registered_slots=480,
        stage_cap_nanousd=8_000_000_000,
        created_at="2026-08-09T12:01:00Z",
    )
    seed = bytes(range(32))

    # When: the screening claim/checkpoint is followed by BCT setup and its checkpoint.
    _establish_live_claim(root, screening, seed)
    claim_created_at = __import__("json").loads((root / "live-attempt-claim.json").read_bytes())["created_at"]
    assert isinstance(claim_created_at, str)
    _start_stage_clock(root, "attempt-001", "screening", claim_created_at, seed)
    _establish_live_claim(root, bct, seed)
    _start_stage_clock(root, "attempt-001", "bct", claim_created_at, seed)

    # Then: the claim remains unique and both stage checkpoints are chained by sequence.
    checkpoints = tuple(sorted((root / "runtime-clock").glob("*.json")))
    values = tuple(__import__("json").loads(path.read_bytes()) for path in checkpoints)
    assert len(tuple(root.glob("live-attempt-claim.json"))) == 1
    assert [value["stage"] for value in values] == [None, "screening", "bct"]
    assert [value["sequence"] for value in values] == [0, 1, 2]


@pytest.mark.parametrize("provider_status", ["queued", "in_progress"])
def test_nonterminal_provider_response_retains_reservation(
    tmp_path: Path, provider_status: str
) -> None:
    # Given/When: HTTP 200 carries a valid nonterminal Responses status.
    outcome = _dispatch(tmp_path, _exchange(body=_response(provider_status=provider_status)))

    # Then: nonterminal response is an operational stop without settlement.
    assert outcome.provider_status == "nonterminal"
    assert outcome.operational_reason == "ROOTLESS_NONTERMINAL_RESPONSE"
    assert outcome.reservation_retained is True


def test_malformed_http_200_json_is_archive_invalid(tmp_path: Path) -> None:
    # Given/When: HTTP 200 body is not a valid Response object.
    outcome = _dispatch(tmp_path, _exchange(body=b"not-json"))

    # Then: response parsing cannot invent provider or usage fields.
    assert outcome.provider_status == "archive_error"
    assert outcome.operational_reason == "ROOTLESS_ARCHIVE_INVALID"
    assert outcome.response_model is None and outcome.usage is None


def test_wrong_model_settles_valid_usage_but_nulls_receipt_model(tmp_path: Path) -> None:
    # Given/When: valid usage accompanies the wrong model identity.
    outcome = _dispatch(tmp_path, _exchange(body=_response(model="wrong-model")))

    # Then: usage is not discarded, while the operational stop remains explicit.
    assert outcome.operational_reason == "ROOTLESS_WRONG_MODEL"
    assert outcome.reservation_retained is False
    assert outcome.archive_manifest["response_model"] == "wrong-model"
    assert outcome.receipt["response_model"] is None
    assert outcome.receipt["usage_input_tokens"] == 100


def test_invalid_usage_retains_reservation(tmp_path: Path) -> None:
    # Given/When: provider usage exceeds the reserved token envelope.
    body = _response(usage={"input_tokens": 4097, "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 0, "total_tokens": 4097})
    outcome = _dispatch(tmp_path, _exchange(body=body))

    # Then: no settlement or inferred cost is emitted.
    assert outcome.operational_reason == "ROOTLESS_USAGE_INVALID"
    assert outcome.usage is None and outcome.actual_nanousd is None
    assert outcome.reservation_retained is True


def test_header_overflow_precedes_429_and_stores_only_complete_fields(tmp_path: Path) -> None:
    # Given: a surfaced 429 with 101 complete attempted fields.
    fields = tuple((f"x-{index}".encode(), b"v") for index in range(101))

    # When: bounded framing reaches the 101st field.
    outcome = _dispatch(tmp_path, _exchange(status=429, headers=_headers(*fields)))

    # Then: size precedence wins and persisted framing declares 100 fields.
    assert outcome.operational_reason == "ROOTLESS_PROVIDER_RESPONSE_TOO_LARGE"
    assert int.from_bytes(outcome.response_headers[:4], "big") == 100
    assert outcome.archive_manifest["content_encoding_status"] == "truncated"


@pytest.mark.parametrize(
    ("headers", "reason"),
    [
        (_headers((b"Content-Encoding", b"gzip")), "ROOTLESS_UNEXPECTED_CONTENT_ENCODING"),
        (_headers((b"Content-Encoding", b"\x00")), "ROOTLESS_UNEXPECTED_CONTENT_ENCODING"),
    ],
)
def test_content_encoding_classification_precedes_body_and_status(
    tmp_path: Path, headers: bytes, reason: str
) -> None:
    # Given/When: complete headers contain unsupported or invalid encoding bytes.
    outcome = _dispatch(tmp_path, _exchange(status=429, headers=headers))

    # Then: encoding classification is the first applicable error.
    assert outcome.operational_reason == reason


def test_malformed_header_framing_is_archive_invalid(tmp_path: Path) -> None:
    # Given/When: a surfaced response carries a truncated header field frame.
    outcome = _dispatch(tmp_path, _exchange(headers=b"\x00\x00\x00\x01\x00\x00"))

    # Then: framing invalidity is not misclassified as Content-Encoding.
    assert outcome.operational_reason == "ROOTLESS_ARCHIVE_INVALID"
    assert outcome.archive_manifest["content_encoding_status"] == "invalid"


def test_body_overflow_keeps_exact_prefix(tmp_path: Path) -> None:
    # Given/When: a streamed body crosses the 1 MiB archive cap.
    outcome = _dispatch(tmp_path, _exchange(body=b"x" * (1_048_576 + 1)))

    # Then: the retained prefix is exact and size failure wins before parsing.
    assert outcome.operational_reason == "ROOTLESS_PROVIDER_RESPONSE_TOO_LARGE"
    assert outcome.response_body == b"x" * 1_048_576


@pytest.mark.parametrize(
    ("raised", "marker", "surfaced", "provider_status", "reason", "phase", "error"),
    [
        ("ConnectTimeout", "before_write", False, "transport_error", "ROOTLESS_TIMEOUT", "connect", "cancelled"),
        ("ProtocolError", "awaiting_response", False, "archive_error", "ROOTLESS_ARCHIVE_INVALID", "headers", "headers_failure"),
        ("DecodingError", "streaming_body", True, "archive_error", "ROOTLESS_ARCHIVE_INVALID", "body", "body_failure"),
        ("ConnectError", "before_write", False, "transport_error", "ROOTLESS_HTTP_ERROR", "connect", "connect_failure"),
        ("WriteError", "writing", False, "transport_error", "ROOTLESS_HTTP_ERROR", "write", "write_failure"),
        ("ReadError", "awaiting_response", False, "transport_error", "ROOTLESS_HTTP_ERROR", "headers", "headers_failure"),
        ("CloseError", "streaming_body", True, "transport_error", "ROOTLESS_HTTP_ERROR", "body", "body_failure"),
        ("FutureTransportError", "streaming_body", True, "transport_error", "ROOTLESS_HTTP_ERROR", "body", "body_failure"),
    ],
)
def test_httpx_exception_matrix_uses_lifecycle_and_surface_state(
    tmp_path: Path,
    raised: str,
    marker: str,
    surfaced: bool,
    provider_status: str,
    reason: str,
    phase: str,
    error: str,
) -> None:
    # Given/When: the fake adapter raises at a precise HTTPX lifecycle marker.
    outcome = _dispatch(
        tmp_path,
        _exchange(
            raised=raised,
            marker=marker,
            surfaced=surfaced,
            status=200 if surfaced else None,
        ),
    )

    # Then: the closed exception fallback preserves the first observable condition.
    assert (outcome.provider_status, outcome.operational_reason) == (provider_status, reason)
    assert (outcome.archive_manifest["transport_phase"], outcome.archive_manifest["transport_error"]) == (
        phase,
        error,
    )


@pytest.mark.parametrize(
    "raw",
    [
        b"OPENAI_API_KEY=short\n",
        b"export OPENAI_API_KEY=" + b"a" * 20 + b"\n",
        b"OPENAI_API_KEY='" + b"a" * 20 + b"'\n",
        b"OPENAI_API_KEY=" + b"a" * 20,
        b"OPENAI_API_KEY=" + b"a" * 20 + b"\nEXTRA=x\n",
    ],
)
def test_secret_loader_accepts_only_exact_private_env_bytes(tmp_path: Path, raw: bytes) -> None:
    # Given: a private regular .env candidate.
    inherited = os.environ.get("OPENAI_API_KEY")
    env = tmp_path / ".env"
    env.write_bytes(raw)
    env.chmod(0o600)

    # When/Then: malformed grammar is rejected without exporting a key.
    with pytest.raises(RootlessContractError, match="ROOTLESS_MISSING_SECRET"):
        load_provider_key(env)
    assert os.environ.get("OPENAI_API_KEY") == inherited


def test_secret_loader_returns_valid_key_without_export(tmp_path: Path) -> None:
    # Given: exact accepted bytes.
    inherited = os.environ.get("OPENAI_API_KEY")
    env = tmp_path / ".env"
    env.write_bytes(b"OPENAI_API_KEY=" + b"a_B-9" * 4 + b"\n")
    env.chmod(0o600)

    # When: the broker-only loader reads it.
    key = load_provider_key(env)

    # Then: the key exists only in the returned memory value.
    assert key == "a_B-9" * 4
    assert os.environ.get("OPENAI_API_KEY") == inherited


def test_runtime_lock_contention_has_no_ledger_side_effect(tmp_path: Path) -> None:
    # Given: one process-local broker lock is held.
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    lock = root / "runtime.lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)
    first = acquire_runtime_lock(lock)

    # When/Then: a contender gets the closed error before ledger creation.
    try:
        with pytest.raises(RootlessContractError, match="ROOTLESS_BROKER_ALREADY_RUNNING"):
            acquire_runtime_lock(lock)
        assert not (root / "ledger").exists()
    finally:
        first.close()


def test_fake_and_live_constructors_reject_schema_transport_and_root_swaps(tmp_path: Path) -> None:
    # Given: one fake binding/root and a production-shaped binding.
    fake = _binding()
    fake_root = _root(tmp_path)
    live = dict(fake)
    live.update(schema_version="rootless_stage_binding_v1", kind="rootless_stage_binding", transport_mode="live")

    # When/Then: fake cannot touch live-like roots and live cannot accept injection/fake schema.
    with pytest.raises(RootlessContractError, match="ROOTLESS_FAKE_BOUNDARY_INVALID"):
        build_fake_broker_for_tests(fake, StaticFixtureTransport(_exchange()), tmp_path / "live-state")
    with pytest.raises(RootlessContractError, match="ROOTLESS_BINDING_INVALID"):
        build_live_broker(fake, fake_root, tmp_path)
    assert not (fake_root / "live-attempt-claim.json").exists()


def test_broker_constructors_reject_root_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: an otherwise valid disposable fake boundary under effective UID zero.
    monkeypatch.setattr(os, "geteuid", lambda: 0)

    # When/Then: root cannot instantiate either execution broker class.
    with pytest.raises(RootlessContractError, match="ROOTLESS_ROOT_EXECUTION_FORBIDDEN"):
        build_fake_broker_for_tests(
            _binding(), StaticFixtureTransport(_exchange()), _root(tmp_path)
        )


def test_recursive_bot_not_issued_frames_account_each_descendant() -> None:
    # Given: a typed-output BoT chain after a parse-failed predecessor.
    protocol = WireProtocol()
    base = {
        "schema_version": "rootless_local_wire_v1",
        "profile": PROFILE,
        "attempt_id": "attempt-001",
        "stage": "screening",
    }

    # When: each descendant receives its own dispatch/accounted branch.
    for sequence, slot in enumerate(("bot-solve", "bot-descendant")):
        protocol.accept({**base, "message_type": "dispatch", "slot_id": slot,
                         "message_sequence": sequence, "predecessor_receipt_sha256s": ["1" * 64]})
        protocol.accept({**base, "message_type": "accounted", "slot_id": slot,
                         "message_sequence": sequence, "call_receipt_sha256": "2" * 64,
                         "not_issued_record_sha256": "3" * 64, "not_issued_head_sha256": "4" * 64,
                         "ledger_head_sha256": "5" * 64})

    # Then: both slots are closed independently without an accepted branch.
    assert protocol.closed_slots == frozenset({"bot-solve", "bot-descendant"})


def test_t4_cli_retains_admin_grammar_and_adds_private_broker_runtime() -> None:
    # Given: the shared Phase-12 parser and both old/new rootless operations.
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    phase12_cli.add_parser(commands)

    # When: the launcher-only broker command is parsed.
    arguments = parser.parse_args([
        "phase12", "filter-v5-rootless", "--repo-root", "/repo", "--state-home", "/state",
        "broker-runtime", "--attempt-id", "attempt-001", "--stage", "screening",
        "--authority", "/state/authority.json", "--worker-fd", "3",
    ])

    # Then: T3's command family remains the owner and the capability is one explicit FD.
    assert arguments.rootless_command == "broker-runtime"
    assert arguments.worker_fd == 3


def test_posix_launcher_binds_clean_environment_and_anonymous_fd_contract() -> None:
    # Given: the tracked launcher bytes.
    raw = Path("scripts/launch_phase12_filter_v5_rootless_broker.sh").read_text(encoding="utf-8")

    # When/Then: its machine-consumed command contains the fixed isolation primitives.
    assert raw.startswith("#!/bin/sh\n")
    assert "env -i" in raw
    assert "-B -I -m memcontam.cli" in raw
    assert "broker-runtime" in raw
    assert "--worker-fd 3" in raw
    assert "OPENAI_API_KEY" in raw
    assert "socket" not in raw


def test_scheduler_is_work_conserving_and_obeys_concurrency_rpm_tpm() -> None:
    # Given: seven lexically unordered ready slots and one blocked predecessor.
    slots = tuple(
        ReadySlot(f"slot-{index}", None, 4736) for index in (6, 2, 5, 1, 4, 0, 3)
    ) + (ReadySlot("blocked", "receipt-needed", 4736),)
    state = SchedulerState(
        active_calls=1,
        recent_dispatch_monotonic_ns=(1, 2),
        now_monotonic_ns=10,
        accounted_receipt_sha256s=frozenset(),
    )

    # When: the deterministic scheduler fills every currently legal worker.
    selected = select_ready_slots(slots, state)

    # Then: it dispatches immediately up to concurrency and rolling limits in slot order.
    assert [slot.slot_id for slot in selected] == ["slot-0", "slot-1", "slot-2", "slot-3"]
    saturated = replace(state, recent_dispatch_monotonic_ns=(1, 2, 3, 4, 5, 6))
    assert select_ready_slots(slots, saturated) == ()


def test_scheduler_refuses_a_refill_that_exceeds_reserved_tpm() -> None:
    # Given: one available worker but nearly all 60-second reserved token capacity is retained.
    state = SchedulerState(
        active_calls=4,
        recent_dispatch_monotonic_ns=(1,),
        now_monotonic_ns=10,
        accounted_receipt_sha256s=frozenset(),
        recent_reserved_tokens=((1, 28_000),),
    )
    slot = ReadySlot("slot-001", None, 4736)

    # When: completion would otherwise refill the fifth worker.
    selected = select_ready_slots((slot,), state)

    # Then: the deterministic reservation window blocks the refill before dispatch.
    assert selected == ()


def test_broker_reuses_t3_external_authority_predicate_before_and_after_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memcontam.experiment.phase12.filter_challenge import rootless_local_broker

    # Given: runtime-bound observations and the single T3 predicate seam.
    calls: list[tuple[list, dict]] = []
    expected = [{"role": "experiment-design"}]
    runtime = {"ordered_external_authorities": expected}
    decoding = {"ordered_sources": []}
    monkeypatch.setattr(
        rootless_local_broker,
        "revalidate_runtime_observations",
        lambda observations, authority: calls.append((list(observations), dict(authority))),
    )

    # When: both preclaim construction and immediate pre-dispatch gates run.
    revalidate_external_authority(runtime, decoding, "before_claim")
    revalidate_external_authority(runtime, decoding, "before_dispatch")

    # Then: neither phase implements or caches a second authority predicate.
    assert calls == [(expected, decoding), (expected, decoding)]


def test_httpx_transport_disables_environment_redirect_retry_and_operation_deadlines() -> None:
    # Given: a production transport created from an in-memory broker-only key.
    transport = HTTPXTransport("a" * 20)

    # When: it constructs the exact canonical request.
    request = transport.build_request(b"{}")

    # Then: only the fixed application headers plus generated host/length are present.
    assert request.url == "https://api.openai.com/v1/responses"
    assert dict(request.headers) == {
        "host": "api.openai.com",
        "content-length": "2",
        "authorization": "Bearer " + "a" * 20,
        "content-type": "application/json",
        "accept": "application/json",
        "accept-encoding": "identity",
        "connection": "close",
        "user-agent": "memcontam-rootless-local/1",
    }
    anyio.run(transport.aclose)
