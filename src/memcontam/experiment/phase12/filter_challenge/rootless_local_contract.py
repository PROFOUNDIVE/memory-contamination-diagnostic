from __future__ import annotations

import json
import unicodedata
from base64 import b64decode, b64encode
from typing import Final, TypeAlias

JsonValue: TypeAlias = str | int | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class RootlessContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)

    def __str__(self) -> str:
        return self.code


class _DuplicateKeyError(ValueError):
    pass


SIGNATURE_DOMAINS: Final = frozenset(
    {
        "attempt-terminal-v1",
        "bct-family-evidence-v1",
        "bct-projection-manifest-v1",
        "bct-result-manifest-v1",
        "call-receipt-manifest-v1",
        "freeze-b-v1",
        "input-manifest-v1",
        "ledger-head-v1",
        "ledger-record-v1",
        "live-attempt-claim-v1",
        "local-call-receipt-v1",
        "plan-acknowledgement-v1",
        "publication-receipt-v1",
        "rate-acknowledgement-v1",
        "raw-archive-manifest-v1",
        "receipt-manifest-v1",
        "request-compiler-manifest-v1",
        "revocation-v1",
        "runtime-clock-checkpoint-v1",
        "runtime-manifest-v1",
        "schedule-manifest-v1",
        "search-config-rows-v1",
        "source-manifest-v1",
        "stage-acknowledgement-v1",
        "stage-execution-authority-v1",
        "stage-terminal-v1",
        "state-inventory-v1",
        "typed-call-outcome-v1",
        "zero-call-skip-v1",
    }
)


def canonical_json_file(value: JsonValue) -> bytes:
    _validate_json_value(value)
    return _canonical_json(value) + b"\n"


def canonical_json_value(value: JsonValue) -> bytes:
    _validate_json_value(value)
    return _canonical_json(value)


def parse_canonical_object(raw: bytes) -> dict[str, JsonValue]:
    try:
        decoded = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKeyError, RootlessContractError) as error:
        raise RootlessContractError("ROOTLESS_JSON_INVALID") from error
    if not isinstance(decoded, dict):
        raise RootlessContractError("ROOTLESS_JSON_INVALID")
    _validate_json_value(decoded)
    if canonical_json_file(decoded) != raw:
        raise RootlessContractError("ROOTLESS_JSON_NONCANONICAL")
    return decoded


def public_key_from_seed(seed: bytes) -> bytes:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        return Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    except (ImportError, ValueError) as error:
        raise RootlessContractError("ROOTLESS_PRIVATE_KEY_INVALID") from error


def sign_object(seed: bytes, domain: str, value: JsonValue) -> str:
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        signature = Ed25519PrivateKey.from_private_bytes(seed).sign(_signature_preimage(domain, value))
    except (ImportError, ValueError) as error:
        raise RootlessContractError("ROOTLESS_PRIVATE_KEY_INVALID") from error
    return b64encode(signature).decode("ascii")


def verify_object_signature(public_key: bytes, domain: str, value: JsonValue, signature: str) -> None:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as error:
        raise RootlessContractError("ROOTLESS_SIGNATURE_INVALID") from error
    try:
        encoded = signature.encode("ascii")
        decoded = b64decode(encoded, validate=True)
        if len(decoded) != 64 or b64encode(decoded) != encoded:
            raise ValueError
        Ed25519PublicKey.from_public_bytes(public_key).verify(decoded, _signature_preimage(domain, value))
    except (InvalidSignature, ValueError) as error:
        raise RootlessContractError("ROOTLESS_SIGNATURE_INVALID") from error


def _signature_preimage(domain: str, value: JsonValue) -> bytes:
    if domain not in SIGNATURE_DOMAINS:
        raise RootlessContractError("ROOTLESS_SIGNATURE_INVALID")
    return domain.encode("ascii") + b"\x00" + canonical_json_value(value)


def _canonical_json(value: JsonValue) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _reject_float(_: str) -> None:
    raise RootlessContractError("ROOTLESS_JSON_INVALID")


def _reject_constant(_: str) -> None:
    raise RootlessContractError("ROOTLESS_JSON_INVALID")


def _validate_json_value(value: JsonValue) -> None:
    if isinstance(value, str):
        if unicodedata.normalize("NFC", value) != value or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise RootlessContractError("ROOTLESS_JSON_INVALID")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_json_value(key)
            _validate_json_value(item)
        return
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return
    raise RootlessContractError("ROOTLESS_JSON_INVALID")
