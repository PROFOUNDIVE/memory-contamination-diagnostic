from __future__ import annotations

import base64
import binascii
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from memcontam.experiment.phase12.filter_challenge.rootless_local_acknowledgement import (
    parse_timestamp,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
    validate_live_stage_binding,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    canonical_json_value,
    parse_canonical_object,
    verify_object_signature,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_state import read_private_file

_AUTHORITY_FIELDS: Final = {
    "schema_version",
    "profile",
    "kind",
    "attempt_id",
    "stage",
    "stage_binding_sha256",
    "stage_acknowledgement_sha256s",
    "plan_acknowledgement_sha256s",
    "rate_acknowledgement_sha256",
    "execution_commit",
    "source_manifest_sha256",
    "runtime_manifest_sha256",
    "input_manifest_sha256",
    "issued_at",
    "expires_at",
    "key_fingerprint",
    "signature",
}
_BINDING_FIELDS: Final = {
    "schema_version",
    "profile",
    "kind",
    "transport_mode",
    "attempt_id",
    "stage",
    "plan_binding_sha256",
    "trusted_base_commit",
    "execution_commit",
    "decoding_authority_sha256",
    "rate_card_sha256",
    "source_manifest_sha256",
    "runtime_manifest_sha256",
    "input_manifest_sha256",
    "compiler_sha256",
    "schedule_sha256",
    "predecessor_terminal_sha256",
    "freeze_b_sha256",
    "registered_slots",
    "stage_cap_nanousd",
    "created_at",
}
_PUBLIC_KEY_FIELDS: Final = {
    "algorithm",
    "created_at",
    "kind",
    "profile",
    "public_key_base64",
    "key_fingerprint",
    "schema_version",
}


def load_public_key(state_root: Path) -> bytes:
    metadata = parse_canonical_object(
        read_private_file(state_root / "keys" / "ed25519-public.json")
    )
    encoded = metadata.get("public_key_base64")
    if (
        set(metadata) != _PUBLIC_KEY_FIELDS
        or metadata.get("schema_version") != "rootless_local_public_key_v1"
        or metadata.get("profile") != "local_rootless_non_authoritative"
        or metadata.get("kind") != "public_key"
        or metadata.get("algorithm") != "ed25519"
        or not isinstance(encoded, str)
    ):
        raise RootlessContractError("ROOTLESS_EXECUTION_AUTHORITY_INVALID")
    try:
        public_key = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise RootlessContractError("ROOTLESS_EXECUTION_AUTHORITY_INVALID") from error
    if (
        len(public_key) != 32
        or base64.b64encode(public_key).decode("ascii") != encoded
        or hashlib.sha256(public_key).hexdigest() != metadata.get("key_fingerprint")
    ):
        raise RootlessContractError("ROOTLESS_EXECUTION_AUTHORITY_INVALID")
    return public_key


def validate_execution_authority(
    authority: dict[str, JsonValue],
    binding: dict[str, JsonValue],
    public_key: bytes,
) -> None:
    validate_live_stage_binding(binding)
    signature = authority.get("signature")
    current = datetime.now(UTC).replace(microsecond=0)
    issued_at = authority.get("issued_at")
    expires_at = authority.get("expires_at")
    if (
        set(authority) != _AUTHORITY_FIELDS
        or set(binding) != _BINDING_FIELDS
        or authority.get("schema_version") != "rootless_stage_execution_authority_v1"
        or authority.get("profile") != "local_rootless_non_authoritative"
        or authority.get("kind") != "rootless_stage_execution_authority"
        or authority.get("attempt_id") != binding.get("attempt_id")
        or authority.get("stage") != binding.get("stage")
        or not isinstance(signature, str)
        or not isinstance(issued_at, str)
        or not isinstance(expires_at, str)
        or authority.get("key_fingerprint") != hashlib.sha256(public_key).hexdigest()
        or authority.get("stage_binding_sha256")
        != hashlib.sha256(canonical_json_value(binding)).hexdigest()
        or any(
            authority.get(name) != binding.get(name)
            for name in (
                "execution_commit",
                "source_manifest_sha256",
                "runtime_manifest_sha256",
                "input_manifest_sha256",
            )
        )
    ):
        raise RootlessContractError("ROOTLESS_EXECUTION_AUTHORITY_INVALID")
    if parse_timestamp(expires_at) <= current or parse_timestamp(issued_at) > current:
        raise RootlessContractError("ROOTLESS_EXECUTION_AUTHORITY_INVALID")
    unsigned = dict(authority)
    del unsigned["signature"]
    verify_object_signature(
        public_key,
        "stage-execution-authority-v1",
        unsigned,
        signature,
    )


__all__ = ("load_public_key", "validate_execution_authority")
