from __future__ import annotations

# allow: SIZE_OK — the closed acknowledgement and skip schemas share one signing authority.

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final, Sequence

from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    canonical_json_value,
    public_key_from_seed,
    sign_object,
    verify_object_signature,
)

PROFILE: Final = "local_rootless_non_authoritative"
ID_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
HEX_PATTERN: Final = re.compile(r"[0-9a-f]{64}\Z")
TIMESTAMP_PATTERN: Final = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")


@dataclass(frozen=True, slots=True)
class AcknowledgementClock:
    now: datetime

    def __post_init__(self) -> None:
        if self.now.tzinfo is None or self.now.utcoffset() != timedelta(0) or self.now.microsecond:
            raise RootlessContractError("ROOTLESS_TIMESTAMP_INVALID")


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_timestamp(value: str) -> datetime:
    if TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise RootlessContractError("ROOTLESS_TIMESTAMP_INVALID")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise RootlessContractError("ROOTLESS_TIMESTAMP_INVALID") from error


def _require_id(value: str) -> None:
    if ID_PATTERN.fullmatch(value) is None:
        raise RootlessContractError("ROOTLESS_ACKNOWLEDGEMENT_INVALID")


def _require_hash(value: str) -> None:
    if HEX_PATTERN.fullmatch(value) is None:
        raise RootlessContractError("ROOTLESS_ACKNOWLEDGEMENT_INVALID")


def _signed(seed: bytes, domain: str, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    result = dict(payload)
    result["signature"] = sign_object(seed, domain, payload)
    return result


def _common(
    attempt_id: str,
    set_id: str,
    operator_label: str,
    operator_index: int,
    plan_binding_sha256: str,
    nonce: str,
    seed: bytes,
    clock: AcknowledgementClock,
) -> dict[str, JsonValue]:
    for value in (attempt_id, set_id, operator_label):
        _require_id(value)
    if operator_index not in {1, 2} or set_id not in {"plan", "screening", "bct"}:
        raise RootlessContractError("ROOTLESS_ACKNOWLEDGEMENT_INVALID")
    _require_hash(plan_binding_sha256)
    _require_hash(nonce)
    public_key = public_key_from_seed(seed)
    issued_at = clock.now
    return {
        "schema_version": "rootless_local_acknowledgement_v1",
        "profile": PROFILE,
        "kind": "local_operator_acknowledgement",
        "set_id": set_id,
        "operator_label": operator_label,
        "acknowledgement_id": f"{set_id}-operator-{operator_index}",
        "plan_binding_sha256": plan_binding_sha256,
        "key_fingerprint": hashlib.sha256(public_key).hexdigest(),
        "issued_at": _timestamp(issued_at),
        "expires_at": _timestamp(issued_at + timedelta(hours=24)),
        "nonce": nonce,
    }


def create_plan_acknowledgement(
    *,
    attempt_id: str,
    set_id: str,
    operator_label: str,
    operator_index: int,
    plan_binding_sha256: str,
    nonce: str,
    seed: bytes,
    clock: AcknowledgementClock,
) -> dict[str, JsonValue]:
    if set_id != "plan":
        raise RootlessContractError("ROOTLESS_ACKNOWLEDGEMENT_INVALID")
    payload = _common(
        attempt_id, set_id, operator_label, operator_index, plan_binding_sha256, nonce, seed, clock
    )
    payload["scope"] = "plan"
    return _signed(seed, "plan-acknowledgement-v1", payload)


def create_stage_acknowledgement(
    *,
    attempt_id: str,
    stage: str,
    set_id: str,
    operator_label: str,
    operator_index: int,
    plan_binding_sha256: str,
    stage_binding_sha256: str,
    nonce: str,
    seed: bytes,
    clock: AcknowledgementClock,
) -> dict[str, JsonValue]:
    if stage not in {"screening", "bct"} or set_id != stage:
        raise RootlessContractError("ROOTLESS_ACKNOWLEDGEMENT_INVALID")
    _require_hash(stage_binding_sha256)
    payload = _common(
        attempt_id, set_id, operator_label, operator_index, plan_binding_sha256, nonce, seed, clock
    )
    payload["scope"] = stage
    payload["stage_binding_sha256"] = stage_binding_sha256
    return _signed(seed, "stage-acknowledgement-v1", payload)


def create_rate_acknowledgement(
    *,
    attempt_id: str,
    provider_account_label: str,
    rpm_limit: int,
    tpm_limit: int,
    observed_at: str,
    nonce: str,
    seed: bytes,
    clock: AcknowledgementClock,
) -> dict[str, JsonValue]:
    _require_id(attempt_id)
    _require_id(provider_account_label)
    _require_hash(nonce)
    observed = parse_timestamp(observed_at)
    if observed > clock.now + timedelta(seconds=60) or observed < clock.now - timedelta(seconds=300):
        raise RootlessContractError("ROOTLESS_RATE_ACKNOWLEDGEMENT_STALE")
    if rpm_limit < 6 or tpm_limit < 30_000:
        raise RootlessContractError("ROOTLESS_RATE_ACKNOWLEDGEMENT_INVALID")
    public_key = public_key_from_seed(seed)
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_local_rate_acknowledgement_v1",
        "profile": PROFILE,
        "kind": "local_rate_capability",
        "provider_account_label": provider_account_label,
        "rpm_limit": rpm_limit,
        "tpm_limit": tpm_limit,
        "source": "operator_dashboard",
        "observed_at": observed_at,
        "issued_at": _timestamp(clock.now),
        "expires_at": _timestamp(clock.now + timedelta(hours=24)),
        "key_fingerprint": hashlib.sha256(public_key).hexdigest(),
        "nonce": nonce,
    }
    return _signed(seed, "rate-acknowledgement-v1", payload)


def object_sha256(value: dict[str, JsonValue]) -> str:
    return hashlib.sha256(canonical_json_value(value)).hexdigest()


def create_skip_receipt(
    *,
    reason: str,
    missing_input_role: str | None,
    attempt_id: str | None,
    reviewed_plan_sha256: str | None,
    created_at: str,
    external_authority_diagnostic: str | None = None,
    failed_command: str | None = None,
    observed_exit: int | None = None,
    seed: bytes | None = None,
) -> dict[str, JsonValue]:
    reasons = {
        "ROOTLESS_MISSING_EXTERNAL_INPUT",
        "ROOTLESS_MISSING_SECRET",
        "ROOTLESS_RATE_CAPABILITY_MISSING",
        "ROOTLESS_PAID_EGRESS_NOT_ENABLED",
        "ROOTLESS_PRECLAIM_ADMIN_FAILED",
    }
    diagnostics = {
        "ROOTLESS_EXTERNAL_AUTHORITY_REVIEW_BINDING_MISSING",
        "ROOTLESS_EXTERNAL_AUTHORITY_PATH_MISMATCH",
        "ROOTLESS_EXTERNAL_AUTHORITY_DESCRIPTOR_UNSAFE",
        "ROOTLESS_EXTERNAL_AUTHORITY_MOUNT_NOT_READ_ONLY",
        "ROOTLESS_EXTERNAL_AUTHORITY_HASH_MISMATCH",
        "ROOTLESS_EXTERNAL_AUTHORITY_IDENTITY_DRIFT",
    }
    external_roles = {
        "ROOTLESS_THEORETICAL_EXPERIMENT_DESIGN",
        "ROOTLESS_THEORETICAL_FILTER_V5_AMENDMENT",
        "ROOTLESS_THEORETICAL_AUTHORITY_AGENTS",
    }
    if reason not in reasons or (attempt_id is not None and ID_PATTERN.fullmatch(attempt_id) is None):
        raise RootlessContractError("ROOTLESS_SKIP_RECEIPT_INVALID")
    if reviewed_plan_sha256 is not None:
        _require_hash(reviewed_plan_sha256)
    is_external = missing_input_role in external_roles
    if is_external != (external_authority_diagnostic in diagnostics):
        raise RootlessContractError("ROOTLESS_SKIP_RECEIPT_INVALID")
    is_admin = reason == "ROOTLESS_PRECLAIM_ADMIN_FAILED"
    if is_admin != (failed_command is not None and observed_exit is not None):
        raise RootlessContractError("ROOTLESS_SKIP_RECEIPT_INVALID")
    if observed_exit is not None and not 1 <= observed_exit <= 255:
        raise RootlessContractError("ROOTLESS_SKIP_RECEIPT_INVALID")
    parse_timestamp(created_at)
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_local_skip_receipt_v1",
        "profile": PROFILE,
        "kind": "zero_call_skip",
        "reason": reason,
        "missing_input_role": missing_input_role,
        "external_authority_diagnostic": external_authority_diagnostic,
        "failed_command": failed_command,
        "observed_exit": observed_exit,
        "attempt_id": attempt_id,
        "reviewed_plan_sha256": reviewed_plan_sha256,
        "provider_calls_issued": 0,
        "created_at": created_at,
        "key_fingerprint": None,
    }
    if seed is None:
        payload["signature"] = None
        return payload
    payload["key_fingerprint"] = hashlib.sha256(public_key_from_seed(seed)).hexdigest()
    result = dict(payload)
    result["signature"] = sign_object(seed, "zero-call-skip-v1", payload)
    return result


def _verify_signed(value: dict[str, JsonValue], public_key: bytes, domain: str) -> None:
    signature = value.get("signature")
    if not isinstance(signature, str):
        raise RootlessContractError("ROOTLESS_ACKNOWLEDGEMENT_INVALID")
    unsigned = dict(value)
    del unsigned["signature"]
    verify_object_signature(public_key, domain, unsigned, signature)


def validate_acknowledgement_pair(
    pair: Sequence[dict[str, JsonValue]],
    public_key: bytes,
    now: datetime,
) -> None:
    if len(pair) != 2:
        raise RootlessContractError("ROOTLESS_ACKNOWLEDGEMENT_SET_INVALID")
    raw_labels = [value.get("operator_label") for value in pair]
    raw_identifiers = [value.get("acknowledgement_id") for value in pair]
    raw_nonces = [value.get("nonce") for value in pair]
    if not all(isinstance(value, str) for value in (*raw_labels, *raw_identifiers, *raw_nonces)):
        raise RootlessContractError("ROOTLESS_ACKNOWLEDGEMENT_SET_INVALID")
    labels = [value for value in raw_labels if isinstance(value, str)]
    identifiers = [value for value in raw_identifiers if isinstance(value, str)]
    nonces = [value for value in raw_nonces if isinstance(value, str)]
    if labels != sorted(labels) or len(set(labels)) != 2 or len(set(identifiers)) != 2 or len(set(nonces)) != 2:
        raise RootlessContractError("ROOTLESS_ACKNOWLEDGEMENT_SET_INVALID")
    for value in pair:
        scope = value.get("scope")
        common_keys = {
            "schema_version",
            "profile",
            "kind",
            "scope",
            "set_id",
            "operator_label",
            "acknowledgement_id",
            "plan_binding_sha256",
            "key_fingerprint",
            "issued_at",
            "expires_at",
            "nonce",
            "signature",
        }
        expected_keys = common_keys if scope == "plan" else common_keys | {"stage_binding_sha256"}
        if (
            set(value) != expected_keys
            or value.get("schema_version") != "rootless_local_acknowledgement_v1"
            or value.get("profile") != PROFILE
            or value.get("kind") != "local_operator_acknowledgement"
            or value.get("key_fingerprint") != hashlib.sha256(public_key).hexdigest()
        ):
            raise RootlessContractError("ROOTLESS_ACKNOWLEDGEMENT_INVALID")
        domain = "plan-acknowledgement-v1" if scope == "plan" else "stage-acknowledgement-v1"
        _verify_signed(value, public_key, domain)
        issued = value.get("issued_at")
        expires = value.get("expires_at")
        if not isinstance(issued, str) or not isinstance(expires, str):
            raise RootlessContractError("ROOTLESS_ACKNOWLEDGEMENT_INVALID")
        issued_time = parse_timestamp(issued)
        if issued_time > now + timedelta(seconds=60) or issued_time < now - timedelta(seconds=300):
            raise RootlessContractError("ROOTLESS_ACKNOWLEDGEMENT_STALE")
        if parse_timestamp(expires) != issued_time + timedelta(hours=24) or parse_timestamp(expires) <= now:
            raise RootlessContractError("ROOTLESS_ACKNOWLEDGEMENT_EXPIRED")


def validate_rate_acknowledgement(
    value: dict[str, JsonValue],
    public_key: bytes,
    now: datetime,
    *,
    check_issue_age: bool = True,
) -> None:
    expected = {
        "schema_version",
        "profile",
        "kind",
        "provider_account_label",
        "rpm_limit",
        "tpm_limit",
        "source",
        "observed_at",
        "issued_at",
        "expires_at",
        "key_fingerprint",
        "nonce",
        "signature",
    }
    if (
        set(value) != expected
        or value.get("schema_version") != "rootless_local_rate_acknowledgement_v1"
        or value.get("profile") != PROFILE
        or value.get("kind") != "local_rate_capability"
        or value.get("source") != "operator_dashboard"
        or value.get("key_fingerprint") != hashlib.sha256(public_key).hexdigest()
    ):
        raise RootlessContractError("ROOTLESS_RATE_ACKNOWLEDGEMENT_INVALID")
    _verify_signed(value, public_key, "rate-acknowledgement-v1")
    issued_raw, observed_raw, expires_raw = (
        value.get("issued_at"),
        value.get("observed_at"),
        value.get("expires_at"),
    )
    if not isinstance(issued_raw, str) or not isinstance(observed_raw, str) or not isinstance(expires_raw, str):
        raise RootlessContractError("ROOTLESS_RATE_ACKNOWLEDGEMENT_INVALID")
    issued, observed, expires = (
        parse_timestamp(issued_raw),
        parse_timestamp(observed_raw),
        parse_timestamp(expires_raw),
    )
    if observed > issued or observed < issued - timedelta(seconds=300) or expires != issued + timedelta(hours=24):
        raise RootlessContractError("ROOTLESS_RATE_ACKNOWLEDGEMENT_INVALID")
    if check_issue_age and (issued > now + timedelta(seconds=60) or issued < now - timedelta(seconds=300)):
        raise RootlessContractError("ROOTLESS_ACKNOWLEDGEMENT_STALE")
    if expires <= now:
        raise RootlessContractError("ROOTLESS_ACKNOWLEDGEMENT_EXPIRED")


__all__ = (
    "AcknowledgementClock",
    "create_plan_acknowledgement",
    "create_rate_acknowledgement",
    "create_skip_receipt",
    "create_stage_acknowledgement",
    "object_sha256",
    "parse_timestamp",
    "validate_acknowledgement_pair",
    "validate_rate_acknowledgement",
)
