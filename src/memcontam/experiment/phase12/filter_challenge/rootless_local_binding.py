from __future__ import annotations

# allow: SIZE_OK — the approved Task-3 plan requires one binding/predicate authority module.

import base64
import hashlib
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from memcontam.experiment.phase12.filter_challenge.rootless_local_acknowledgement import (
    object_sha256,
    parse_timestamp,
    validate_acknowledgement_pair,
    validate_rate_acknowledgement,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    canonical_json_value,
    public_key_from_seed,
    sign_object,
)

PROFILE: Final = "local_rootless_non_authoritative"
_HEX: Final = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA: Final = re.compile(r"[0-9a-f]{40}\Z")
_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_DIR_FLAGS: Final = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS: Final = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
_CONFIG_DIGESTS: Final = {
    "decoding_authority": "d2cbe904e95cfb2892887b35493ca18158b7b15130744e7d7b086520ef95dc36",
    "rate_card": "1d339d378bbc0a22bc52dc6eafba336521967135ae395023b78883a342354493",
    "screening": "92fe6bfb97abb8b5a4124388184f09528d6189588c2baf202a404a9b6976c485",
    "bct": "59669e0127eb1e904f3744386328209ae0640c38237a00feee9b517c5512ea06",
}
_CONFIG_FILENAMES: Final = {
    "decoding_authority": "decoding_authority.json",
    "rate_card": "rate_card.json",
    "screening": "screening.yaml",
    "bct": "bct.yaml",
}
_EXTERNAL_ROLES: Final = ("experiment-design", "filter-v5-amendment", "authority-agents")
_EXTERNAL_INPUT_ROLES: Final = {
    "experiment-design": "ROOTLESS_THEORETICAL_EXPERIMENT_DESIGN",
    "filter-v5-amendment": "ROOTLESS_THEORETICAL_FILTER_V5_AMENDMENT",
    "authority-agents": "ROOTLESS_THEORETICAL_AUTHORITY_AGENTS",
}
_ESCAPES: Final = {b"040": b" ", b"011": b"\t", b"012": b"\n", b"134": b"\\"}


@dataclass(frozen=True, slots=True)
class MountRecord:
    mount_id: int
    parent_id: int
    major: int
    minor: int
    root: bytes
    mount_point: bytes
    options: tuple[bytes, ...]
    raw_sha256: str


@dataclass(frozen=True, slots=True)
class RuntimeInstallationEvidence:
    python_path: str
    python_version: str
    pip_version: str
    memcontam_import_path: str
    repo_root_mode_bits: int
    editable_direct_url_sha256: str
    distribution_record_sha256: str
    native_extension_hashes: tuple[str, ...]
    requirements_lock_sha256: str
    requirements_dev_lock_sha256: str
    tiktoken_version: str
    tokenizer_source_sha256: str


class ExternalAuthorityObservationError(RootlessContractError):
    def __init__(self, code: str, missing_input_role: str) -> None:
        super().__init__(code)
        self.missing_input_role = missing_input_role


def _external_error(code: str = "ROOTLESS_EXTERNAL_AUTHORITY_MOUNT_NOT_READ_ONLY") -> RootlessContractError:
    return RootlessContractError(code)


def _decode_mount_field(raw: bytes) -> bytes:
    output = bytearray()
    index = 0
    while index < len(raw):
        if raw[index] != 0x5C:
            output.append(raw[index])
            index += 1
            continue
        escape = raw[index + 1 : index + 4]
        if len(escape) != 3 or escape not in _ESCAPES:
            raise _external_error()
        output.extend(_ESCAPES[escape])
        index += 4
    return bytes(output)


def _decimal(raw: bytes, *, positive: bool) -> int:
    if not raw or not raw.isdigit() or (len(raw) > 1 and raw.startswith(b"0")):
        raise _external_error()
    value = int(raw)
    if positive and value == 0:
        raise _external_error()
    return value


def parse_mountinfo(raw: bytes) -> tuple[MountRecord, ...]:
    if not raw or not raw.endswith(b"\n") or b"\x00" in raw or b"\r" in raw:
        raise _external_error()
    records: list[MountRecord] = []
    for line_with_lf in raw.splitlines(keepends=True):
        fields = line_with_lf[:-1].split(b" ")
        if b"" in fields or fields.count(b"-") != 1:
            raise _external_error()
        separator = fields.index(b"-")
        if separator < 6 or len(fields) - separator - 1 != 3:
            raise _external_error()
        device = fields[2].split(b":")
        if len(device) != 2:
            raise _external_error()
        options = tuple(fields[5].split(b","))
        if not options or any(not option for option in options):
            raise _external_error()
        records.append(
            MountRecord(
                _decimal(fields[0], positive=True),
                _decimal(fields[1], positive=True),
                _decimal(device[0], positive=False),
                _decimal(device[1], positive=False),
                _decode_mount_field(fields[3]),
                _decode_mount_field(fields[4]),
                options,
                hashlib.sha256(line_with_lf).hexdigest(),
            )
        )
    return tuple(records)


def _component_match(path: bytes, mount_point: bytes) -> bool:
    return mount_point == b"/" or path == mount_point or path.startswith(mount_point + b"/")


def select_mount_record(path: bytes, device: int, records: Sequence[MountRecord]) -> MountRecord:
    candidates = [record for record in records if _component_match(path, record.mount_point)]
    if not candidates:
        raise _external_error()
    longest = max(len(record.mount_point) for record in candidates)
    selected = [
        record
        for record in candidates
        if len(record.mount_point) == longest
        and record.major == os.major(device)
        and record.minor == os.minor(device)
    ]
    if len(selected) != 1 or b"ro" not in selected[0].options:
        raise _external_error()
    return selected[0]


def _lexical_absolute(raw: str) -> tuple[bytes, ...]:
    encoded = os.fsencode(raw)
    if not encoded.startswith(b"/") or encoded.startswith(b"//") or b"\x00" in encoded:
        raise _external_error("ROOTLESS_EXTERNAL_AUTHORITY_DESCRIPTOR_UNSAFE")
    components = tuple(encoded[1:].split(b"/"))
    if not components or any(component in {b"", b".", b".."} for component in components):
        raise _external_error("ROOTLESS_EXTERNAL_AUTHORITY_DESCRIPTOR_UNSAFE")
    return components


def _open_external(raw_path: str) -> int:
    components = _lexical_absolute(raw_path)
    descriptor = os.open(b"/", _DIR_FLAGS)
    try:
        for component in components[:-1]:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        result = os.open(components[-1], _FILE_FLAGS, dir_fd=descriptor)
    except OSError as error:
        raise _external_error("ROOTLESS_EXTERNAL_AUTHORITY_DESCRIPTOR_UNSAFE") from error
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(os.fstat(result).st_mode):
        os.close(result)
        raise _external_error("ROOTLESS_EXTERNAL_AUTHORITY_DESCRIPTOR_UNSAFE")
    return result


def _source_fields(source: Mapping[str, JsonValue]) -> tuple[str, str, str, list[dict[str, JsonValue]]]:
    if set(source) != {"role", "absolute_path", "full_sha256", "ordered_spans"}:
        raise _external_error("ROOTLESS_EXTERNAL_AUTHORITY_REVIEW_BINDING_MISSING")
    role = source["role"]
    path = source["absolute_path"]
    full_hash = source["full_sha256"]
    spans = source["ordered_spans"]
    if (
        not isinstance(role, str)
        or role not in _EXTERNAL_ROLES
        or not isinstance(path, str)
        or not isinstance(full_hash, str)
        or _HEX.fullmatch(full_hash) is None
        or not isinstance(spans, list)
        or not all(isinstance(span, dict) for span in spans)
    ):
        raise _external_error("ROOTLESS_EXTERNAL_AUTHORITY_REVIEW_BINDING_MISSING")
    typed_spans = [span for span in spans if isinstance(span, dict)]
    return role, path, full_hash, typed_spans


def _read_descriptor(descriptor: int) -> tuple[bytes, os.stat_result, os.stat_result]:
    before = os.fstat(descriptor)
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1_048_576):
        chunks.append(chunk)
    after = os.fstat(descriptor)
    return b"".join(chunks), before, after


def _read_mountinfo() -> bytes:
    descriptor = os.open(b"/proc/self/mountinfo", _FILE_FLAGS)
    try:
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1_048_576):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _span_hashes(raw: bytes, spans: Sequence[dict[str, JsonValue]]) -> list[str]:
    lines = raw.splitlines(keepends=True)
    hashes: list[str] = []
    for span in spans:
        if set(span) != {"start_line", "end_line", "sha256"}:
            raise _external_error("ROOTLESS_EXTERNAL_AUTHORITY_REVIEW_BINDING_MISSING")
        start, end, expected = span["start_line"], span["end_line"], span["sha256"]
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not isinstance(expected, str)
            or start < 1
            or end < start
            or end > len(lines)
            or _HEX.fullmatch(expected) is None
        ):
            raise _external_error("ROOTLESS_EXTERNAL_AUTHORITY_REVIEW_BINDING_MISSING")
        hashes.append(hashlib.sha256(b"".join(lines[start - 1 : end])).hexdigest())
    return hashes


def _flags(value: int | os.statvfs_result) -> int:
    return value if isinstance(value, int) else value.f_flag


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size


def _mount_identity(record: MountRecord) -> tuple[int, int, int, int, bytes, bytes, str]:
    return (
        record.mount_id,
        record.parent_id,
        record.major,
        record.minor,
        record.root,
        record.mount_point,
        record.raw_sha256,
    )


def observe_external_authority(
    source: Mapping[str, JsonValue],
    *,
    requested_path: str | None = None,
    mountinfo_reader: Callable[[], bytes] | None = None,
    fstatvfs_reader: Callable[[int], int | os.statvfs_result] = os.fstatvfs,
) -> dict[str, JsonValue]:
    role, path, expected_full, spans = _source_fields(source)
    if requested_path is not None and requested_path != path:
        raise _external_error("ROOTLESS_EXTERNAL_AUTHORITY_PATH_MISMATCH")
    read_mountinfo = mountinfo_reader or _read_mountinfo
    first_mount_raw = read_mountinfo()
    first_descriptor = _open_external(path)
    try:
        first_info = os.fstat(first_descriptor)
        first_mount = select_mount_record(os.fsencode(path), first_info.st_dev, parse_mountinfo(first_mount_raw))
        first_flags = _flags(fstatvfs_reader(first_descriptor))
        if first_flags & os.ST_RDONLY == 0:
            raise _external_error()
        raw, before, after = _read_descriptor(first_descriptor)
    finally:
        os.close(first_descriptor)
    second_mount_raw = read_mountinfo()
    second_descriptor = _open_external(path)
    try:
        second_info = os.fstat(second_descriptor)
        second_mount = select_mount_record(
            os.fsencode(path), second_info.st_dev, parse_mountinfo(second_mount_raw)
        )
        second_flags = _flags(fstatvfs_reader(second_descriptor))
    finally:
        os.close(second_descriptor)
    if (
        second_flags & os.ST_RDONLY == 0
        or _identity(before) != _identity(after)
        or _identity(after) != _identity(second_info)
        or _mount_identity(first_mount) != _mount_identity(second_mount)
    ):
        raise _external_error("ROOTLESS_EXTERNAL_AUTHORITY_IDENTITY_DRIFT")
    observed_spans = _span_hashes(raw, spans)
    expected_spans = [span["sha256"] for span in spans]
    observed_full = hashlib.sha256(raw).hexdigest()
    if observed_full != expected_full or observed_spans != expected_spans:
        raise _external_error("ROOTLESS_EXTERNAL_AUTHORITY_HASH_MISMATCH")
    span_values: list[JsonValue] = [*observed_spans]
    return {
        "role": role,
        "absolute_path": path,
        "review_binding_sha256": hashlib.sha256(canonical_json_value(dict(source))).hexdigest(),
        "mount_id": first_mount.mount_id,
        "parent_mount_id": first_mount.parent_id,
        "mount_device_major": first_mount.major,
        "mount_device_minor": first_mount.minor,
        "mount_root_base64": base64.b64encode(first_mount.root).decode("ascii"),
        "mount_point_base64": base64.b64encode(first_mount.mount_point).decode("ascii"),
        "mountinfo_line_sha256": first_mount.raw_sha256,
        "mount_options_read_only": True,
        "file_st_dev": before.st_dev,
        "file_st_ino": before.st_ino,
        "file_mode_bits": stat.S_IMODE(before.st_mode),
        "file_nlink": before.st_nlink,
        "file_size": before.st_size,
        "fstatvfs_read_only": True,
        "full_sha256": observed_full,
        "ordered_span_sha256s": span_values,
    }


def observe_external_authorities(
    decoding_authority: Mapping[str, JsonValue],
) -> list[JsonValue]:
    sources = decoding_authority.get("ordered_sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise _external_error("ROOTLESS_EXTERNAL_AUTHORITY_REVIEW_BINDING_MISSING")
    observations: list[JsonValue] = []
    for expected_role, source in zip(_EXTERNAL_ROLES, sources, strict=True):
        if not isinstance(source, dict) or source.get("role") != expected_role:
            raise _external_error("ROOTLESS_EXTERNAL_AUTHORITY_REVIEW_BINDING_MISSING")
        try:
            observations.append(observe_external_authority(source))
        except RootlessContractError as error:
            raise ExternalAuthorityObservationError(
                error.code, _EXTERNAL_INPUT_ROLES[expected_role]
            ) from error
    return observations


def validate_rootless_configs(repository: Path) -> Mapping[str, str]:
    config_root = repository / "configs" / "phase12" / "filter_v5_rootless_local"
    digests: dict[str, str] = {}
    from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import parse_canonical_object

    for name, filename in _CONFIG_FILENAMES.items():
        raw = (config_root / filename).read_bytes()
        parse_canonical_object(raw)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != _CONFIG_DIGESTS[name]:
            raise RootlessContractError("ROOTLESS_CONFIG_DIGEST_INVALID")
        digests[name] = digest
    return digests


def _require_hash(value: str) -> None:
    if _HEX.fullmatch(value) is None:
        raise RootlessContractError("ROOTLESS_BINDING_INVALID")


def build_stage_binding(
    *,
    attempt_id: str,
    stage: str,
    plan_binding_sha256: str,
    trusted_base_commit: str,
    execution_commit: str,
    decoding_authority_sha256: str,
    rate_card_sha256: str,
    source_manifest_sha256: str,
    runtime_manifest_sha256: str,
    input_manifest_sha256: str,
    compiler_sha256: str,
    schedule_sha256: str,
    registered_slots: int,
    stage_cap_nanousd: int,
    created_at: str,
    predecessor_terminal_sha256: str | None = None,
    freeze_b_sha256: str | None = None,
) -> dict[str, JsonValue]:
    if _ID.fullmatch(attempt_id) is None or stage not in {"screening", "bct"}:
        raise RootlessContractError("ROOTLESS_BINDING_INVALID")
    for value in (
        plan_binding_sha256,
        decoding_authority_sha256,
        rate_card_sha256,
        source_manifest_sha256,
        runtime_manifest_sha256,
        input_manifest_sha256,
        compiler_sha256,
        schedule_sha256,
    ):
        _require_hash(value)
    if (
        _GIT_SHA.fullmatch(trusted_base_commit) is None
        or _GIT_SHA.fullmatch(execution_commit) is None
        or trusted_base_commit == "0" * 40
        or execution_commit == "0" * 40
    ):
        raise RootlessContractError("ROOTLESS_BINDING_INVALID")
    if stage == "screening" and (predecessor_terminal_sha256 is not None or freeze_b_sha256 is not None):
        raise RootlessContractError("ROOTLESS_BINDING_INVALID")
    if stage == "bct":
        if predecessor_terminal_sha256 is None or freeze_b_sha256 is None:
            raise RootlessContractError("ROOTLESS_BINDING_INVALID")
        _require_hash(predecessor_terminal_sha256)
        _require_hash(freeze_b_sha256)
    parse_timestamp(created_at)
    return {
        "schema_version": "rootless_stage_binding_v1",
        "profile": PROFILE,
        "kind": "rootless_stage_binding",
        "transport_mode": "live",
        "attempt_id": attempt_id,
        "stage": stage,
        "plan_binding_sha256": plan_binding_sha256,
        "trusted_base_commit": trusted_base_commit,
        "execution_commit": execution_commit,
        "decoding_authority_sha256": decoding_authority_sha256,
        "rate_card_sha256": rate_card_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "compiler_sha256": compiler_sha256,
        "schedule_sha256": schedule_sha256,
        "predecessor_terminal_sha256": predecessor_terminal_sha256,
        "freeze_b_sha256": freeze_b_sha256,
        "registered_slots": registered_slots,
        "stage_cap_nanousd": stage_cap_nanousd,
        "created_at": created_at,
    }


def build_fake_stage_binding(
    *,
    fixture_id: str,
    stage: str,
    source_manifest_sha256: str,
    input_manifest_sha256: str,
    compiler_sha256: str,
    schedule_sha256: str,
    fake_scenario_sha256: str,
) -> dict[str, JsonValue]:
    if _ID.fullmatch(fixture_id) is None or stage not in {"screening", "bct"}:
        raise RootlessContractError("ROOTLESS_BINDING_INVALID")
    for value in (
        source_manifest_sha256,
        input_manifest_sha256,
        compiler_sha256,
        schedule_sha256,
        fake_scenario_sha256,
    ):
        _require_hash(value)
    return {
        "schema_version": "rootless_fake_stage_binding_v1",
        "profile": PROFILE,
        "kind": "fake_stage_binding",
        "transport_mode": "fake",
        "fixture_id": fixture_id,
        "stage": stage,
        "source_manifest_sha256": source_manifest_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "compiler_sha256": compiler_sha256,
        "schedule_sha256": schedule_sha256,
        "fake_scenario_sha256": fake_scenario_sha256,
    }


def validate_live_stage_binding(binding: Mapping[str, JsonValue]) -> None:
    if (
        binding.get("schema_version") != "rootless_stage_binding_v1"
        or binding.get("profile") != PROFILE
        or binding.get("kind") != "rootless_stage_binding"
        or binding.get("transport_mode") != "live"
    ):
        raise RootlessContractError("ROOTLESS_BINDING_INVALID")


def _unsigned(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: item for key, item in value.items() if key != "signature"}


def _signed_manifest(
    payload: dict[str, JsonValue],
    *,
    seed: bytes,
    domain: str,
) -> dict[str, JsonValue]:
    result = dict(payload)
    result["signature"] = sign_object(seed, domain, payload)
    return result


def build_runtime_manifest(
    evidence: RuntimeInstallationEvidence,
    ordered_external_authorities: Sequence[JsonValue],
    *,
    seed: bytes,
    created_at: str,
) -> dict[str, JsonValue]:
    parse_timestamp(created_at)
    if evidence.repo_root_mode_bits != 0o755 or len(ordered_external_authorities) != 3:
        raise RootlessContractError("ROOTLESS_RUNTIME_MANIFEST_INVALID")
    roles = [entry.get("role") if isinstance(entry, dict) else None for entry in ordered_external_authorities]
    if roles != list(_EXTERNAL_ROLES):
        raise RootlessContractError("ROOTLESS_RUNTIME_MANIFEST_INVALID")
    for value in (
        evidence.editable_direct_url_sha256,
        evidence.distribution_record_sha256,
        *evidence.native_extension_hashes,
        evidence.requirements_lock_sha256,
        evidence.requirements_dev_lock_sha256,
        evidence.tokenizer_source_sha256,
    ):
        _require_hash(value)
    public_key = public_key_from_seed(seed)
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_runtime_manifest_v1",
        "profile": PROFILE,
        "kind": "rootless_runtime_manifest",
        "python_path": evidence.python_path,
        "python_version": evidence.python_version,
        "pip_version": evidence.pip_version,
        "bootstrap_index_url": "https://pypi.org/simple/",
        "bootstrap_egress_policy": "public_pypi_hash_pinned",
        "memcontam_import_path": evidence.memcontam_import_path,
        "repo_root_mode_bits": evidence.repo_root_mode_bits,
        "editable_direct_url_sha256": evidence.editable_direct_url_sha256,
        "distribution_record_sha256": evidence.distribution_record_sha256,
        "native_extension_hashes": list(evidence.native_extension_hashes),
        "requirements_lock_sha256": evidence.requirements_lock_sha256,
        "requirements_dev_lock_sha256": evidence.requirements_dev_lock_sha256,
        "tiktoken_version": evidence.tiktoken_version,
        "tokenizer_source_sha256": evidence.tokenizer_source_sha256,
        "ordered_external_authorities": list(ordered_external_authorities),
        "created_at": created_at,
        "key_fingerprint": hashlib.sha256(public_key).hexdigest(),
    }
    return _signed_manifest(payload, seed=seed, domain="runtime-manifest-v1")


def build_source_manifest(
    execution_commit: str,
    ordered_files: Sequence[JsonValue],
    *,
    seed: bytes,
    created_at: str,
) -> dict[str, JsonValue]:
    if _GIT_SHA.fullmatch(execution_commit) is None or not ordered_files:
        raise RootlessContractError("ROOTLESS_SOURCE_MANIFEST_INVALID")
    paths = [entry.get("repo_relative_path") if isinstance(entry, dict) else None for entry in ordered_files]
    if paths != sorted(paths, key=lambda value: str(value).encode("utf-8")) or len(set(paths)) != len(paths):
        raise RootlessContractError("ROOTLESS_SOURCE_MANIFEST_INVALID")
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_source_manifest_v1",
        "profile": PROFILE,
        "kind": "rootless_source_manifest",
        "execution_commit": execution_commit,
        "ordered_files": list(ordered_files),
        "created_at": created_at,
        "key_fingerprint": hashlib.sha256(public_key_from_seed(seed)).hexdigest(),
    }
    return _signed_manifest(payload, seed=seed, domain="source-manifest-v1")


def build_input_manifest(
    ordered_inputs: Sequence[JsonValue],
    decoding_authority_sha256: str,
    rate_card_sha256: str,
    *,
    seed: bytes,
    created_at: str,
) -> dict[str, JsonValue]:
    _require_hash(decoding_authority_sha256)
    _require_hash(rate_card_sha256)
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_input_manifest_v1",
        "profile": PROFILE,
        "kind": "rootless_input_manifest",
        "ordered_inputs": list(ordered_inputs),
        "decoding_authority_sha256": decoding_authority_sha256,
        "rate_card_sha256": rate_card_sha256,
        "created_at": created_at,
        "key_fingerprint": hashlib.sha256(public_key_from_seed(seed)).hexdigest(),
    }
    return _signed_manifest(payload, seed=seed, domain="input-manifest-v1")


def build_compiler_manifest(
    execution_commit: str,
    ordered_source_roles: Sequence[str],
    ordered_source_hashes: Sequence[str],
    *,
    seed: bytes,
    created_at: str,
) -> dict[str, JsonValue]:
    if (
        _GIT_SHA.fullmatch(execution_commit) is None
        or not ordered_source_roles
        or len(ordered_source_roles) != len(ordered_source_hashes)
        or any(_HEX.fullmatch(value) is None for value in ordered_source_hashes)
    ):
        raise RootlessContractError("ROOTLESS_COMPILER_MANIFEST_INVALID")
    renderers = [
        "full-history-generate=rootless-adapter-v1",
        "rag-generate=rootless-adapter-v1",
        "bot-problem-distill=rootless-adapter-v1",
        "bot-instantiate-solve=rootless-adapter-v1",
        "reflexion-generate=rootless-adapter-v1",
        "responses-text-extractor=rootless-responses-text-v1",
    ]
    renderer_values: list[JsonValue] = [*renderers]
    role_values: list[JsonValue] = [*ordered_source_roles]
    hash_values: list[JsonValue] = [*ordered_source_hashes]
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_request_compiler_manifest_v1",
        "profile": PROFILE,
        "kind": "rootless_request_compiler_manifest",
        "execution_commit": execution_commit,
        "ordered_source_roles": role_values,
        "ordered_source_hashes": hash_values,
        "renderer_versions": renderer_values,
        "canonicalizer_version": "rootless_local_contract_v1",
        "created_at": created_at,
        "key_fingerprint": hashlib.sha256(public_key_from_seed(seed)).hexdigest(),
    }
    return _signed_manifest(payload, seed=seed, domain="request-compiler-manifest-v1")


def build_schedule_manifest(
    attempt_id: str,
    stage: str,
    slots: Sequence[Mapping[str, JsonValue]],
    *,
    seed: bytes,
    created_at: str,
) -> dict[str, JsonValue]:
    if _ID.fullmatch(attempt_id) is None or stage not in {"screening", "bct"}:
        raise RootlessContractError("ROOTLESS_SCHEDULE_MANIFEST_INVALID")
    leaves = [hashlib.sha256(canonical_json_value(dict(slot))).digest() for slot in slots]
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_schedule_manifest_v1",
        "profile": PROFILE,
        "kind": "rootless_schedule_manifest",
        "attempt_id": attempt_id,
        "stage": stage,
        "slot_count": len(slots),
        "ordered_slot_root_sha256": hashlib.sha256(b"".join(leaves)).hexdigest(),
        "created_at": created_at,
        "key_fingerprint": hashlib.sha256(public_key_from_seed(seed)).hexdigest(),
    }
    return _signed_manifest(payload, seed=seed, domain="schedule-manifest-v1")


def build_execution_authority(
    binding: Mapping[str, JsonValue],
    plan_acknowledgements: Sequence[dict[str, JsonValue]],
    stage_acknowledgements: Sequence[dict[str, JsonValue]],
    rate_acknowledgement: dict[str, JsonValue],
    *,
    seed: bytes,
    issued_at: str,
) -> dict[str, JsonValue]:
    if len(plan_acknowledgements) != 2 or len(stage_acknowledgements) != 2:
        raise RootlessContractError("ROOTLESS_EXECUTION_AUTHORITY_INVALID")
    issued = parse_timestamp(issued_at)
    binding_hash = hashlib.sha256(canonical_json_value(dict(binding))).hexdigest()
    if any(value.get("stage_binding_sha256") != binding_hash for value in stage_acknowledgements):
        raise RootlessContractError("ROOTLESS_EXECUTION_AUTHORITY_INVALID")
    expiries: list[str] = []
    for value in (*plan_acknowledgements, *stage_acknowledgements, rate_acknowledgement):
        expiry = value.get("expires_at")
        if not isinstance(expiry, str):
            raise RootlessContractError("ROOTLESS_EXECUTION_AUTHORITY_INVALID")
        expiries.append(expiry)
    expires_at = min(expiries, key=parse_timestamp)
    if parse_timestamp(expires_at) <= issued:
        raise RootlessContractError("ROOTLESS_EXECUTION_AUTHORITY_INVALID")
    public_key = public_key_from_seed(seed)
    validate_acknowledgement_pair(plan_acknowledgements, public_key, issued)
    validate_acknowledgement_pair(stage_acknowledgements, public_key, issued)
    validate_rate_acknowledgement(rate_acknowledgement, public_key, issued)
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_stage_execution_authority_v1",
        "profile": PROFILE,
        "kind": "rootless_stage_execution_authority",
        "attempt_id": binding["attempt_id"],
        "stage": binding["stage"],
        "stage_binding_sha256": binding_hash,
        "stage_acknowledgement_sha256s": [object_sha256(value) for value in stage_acknowledgements],
        "plan_acknowledgement_sha256s": [object_sha256(value) for value in plan_acknowledgements],
        "rate_acknowledgement_sha256": object_sha256(rate_acknowledgement),
        "execution_commit": binding["execution_commit"],
        "source_manifest_sha256": binding["source_manifest_sha256"],
        "runtime_manifest_sha256": binding["runtime_manifest_sha256"],
        "input_manifest_sha256": binding["input_manifest_sha256"],
        "issued_at": issued_at,
        "expires_at": expires_at,
        "key_fingerprint": hashlib.sha256(public_key).hexdigest(),
    }
    result = dict(payload)
    result["signature"] = sign_object(seed, "stage-execution-authority-v1", payload)
    return result


def revalidate_runtime_observations(
    expected: Sequence[JsonValue],
    decoding_authority: Mapping[str, JsonValue],
) -> None:
    if list(expected) != observe_external_authorities(decoding_authority):
        raise _external_error("ROOTLESS_EXTERNAL_AUTHORITY_IDENTITY_DRIFT")


__all__ = (
    "ExternalAuthorityObservationError",
    "MountRecord",
    "RuntimeInstallationEvidence",
    "build_fake_stage_binding",
    "build_compiler_manifest",
    "build_execution_authority",
    "build_input_manifest",
    "build_runtime_manifest",
    "build_schedule_manifest",
    "build_source_manifest",
    "build_stage_binding",
    "observe_external_authorities",
    "observe_external_authority",
    "parse_mountinfo",
    "revalidate_runtime_observations",
    "select_mount_record",
    "validate_live_stage_binding",
    "validate_rootless_configs",
)
