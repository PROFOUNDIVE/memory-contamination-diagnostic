from __future__ import annotations

import base64
import binascii
import hashlib
import os
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    canonical_json_file,
    parse_canonical_object,
    public_key_from_seed,
)


_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_DIR_FLAGS: Final = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS: Final = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


@dataclass(frozen=True, slots=True)
class InitStateRequest:
    state_home: Path
    plan_source: Path
    plan_descriptor: Path
    review_metadata: Path
    attempt_id: str


@dataclass(frozen=True, slots=True)
class InitializedState:
    state_root: Path
    plan_binding: Path
    private_seed: Path
    public_metadata: Path
    runtime_lock: Path
    tokenizer_cache: Path
    plan_sha256: str


def initialize_state(request: InitStateRequest) -> InitializedState:
    if _ID.fullmatch(request.attempt_id) is None:
        raise RootlessContractError("ROOTLESS_ATTEMPT_ID_INVALID")
    state_home = _require_private_directory(request.state_home)
    plan = _read_private_file(request.plan_source)
    plan_sha256 = hashlib.sha256(plan).hexdigest()
    _validate_reviewed_plan(request, plan_sha256)
    state_root = state_home / "memcontam" / "phase12-filter-v5-rootless-local"
    _mkdir_chain(state_home, ("memcontam", "phase12-filter-v5-rootless-local"))
    for relative in (
        "keys",
        "manifests",
        "acknowledgements",
        "bindings",
        "authorities",
        "revocations",
        "tokenizer",
        "tokenizer/cache",
    ):
        _mkdir_chain(state_root, (relative,))
    plan_binding = state_root / "plan-bind.md"
    private_seed = state_root / "keys" / "ed25519-private.key"
    public_metadata = state_root / "keys" / "ed25519-public.json"
    runtime_lock = state_root / "runtime.lock"
    tokenizer_cache = state_root / "tokenizer" / "cache"
    _write_new(runtime_lock, b"")
    _write_new(plan_binding, plan)
    seed = os.urandom(32)
    _write_new(private_seed, seed)
    public_key = public_key_from_seed(seed)
    payload: dict[str, JsonValue] = {
        "algorithm": "ed25519",
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "key_fingerprint": hashlib.sha256(public_key).hexdigest(),
        "kind": "public_key",
        "profile": "local_rootless_non_authoritative",
        "public_key_base64": base64.b64encode(public_key).decode("ascii"),
        "schema_version": "rootless_local_public_key_v1",
    }
    _write_new(public_metadata, canonical_json_file(payload))
    return InitializedState(
        state_root,
        plan_binding,
        private_seed,
        public_metadata,
        runtime_lock,
        tokenizer_cache,
        plan_sha256,
    )


def read_private_file(path: Path) -> bytes:
    return _read_private_file(path)


def state_root(state_home: Path) -> Path:
    return state_home / "memcontam" / "phase12-filter-v5-rootless-local"


def cache_tokenizer_source(
    source: Path,
    cache_directory: Path,
    *,
    expected_sha256: str = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    expected_rank_count: int | None = None,
) -> Path:
    raw = _read_private_file(source)
    if hashlib.sha256(raw).hexdigest() != expected_sha256 or not raw.endswith(b"\n"):
        raise RootlessContractError("ROOTLESS_TOKENIZER_SOURCE_INVALID")
    tokens: set[bytes] = set()
    ranks: set[int] = set()
    for line in raw.splitlines():
        fields = line.split(b" ")
        if len(fields) != 2 or not fields[1].isdigit() or (len(fields[1]) > 1 and fields[1].startswith(b"0")):
            raise RootlessContractError("ROOTLESS_TOKENIZER_SOURCE_INVALID")
        try:
            token = base64.b64decode(fields[0], validate=True)
        except (ValueError, binascii.Error) as error:
            raise RootlessContractError("ROOTLESS_TOKENIZER_SOURCE_INVALID") from error
        rank = int(fields[1])
        if base64.b64encode(token) != fields[0] or token in tokens or rank in ranks:
            raise RootlessContractError("ROOTLESS_TOKENIZER_SOURCE_INVALID")
        tokens.add(token)
        ranks.add(rank)
    if ranks != set(range(len(ranks))) or (expected_rank_count is not None and len(ranks) != expected_rank_count):
        raise RootlessContractError("ROOTLESS_TOKENIZER_SOURCE_INVALID")
    cache = _require_private_directory(cache_directory)
    destination = cache / "fb374d419588a4632f3f557e76b4b70aebbca790"
    _write_new(destination, raw)
    if _read_private_file(destination) != raw:
        raise RootlessContractError("ROOTLESS_TOKENIZER_SOURCE_INVALID")
    return destination


def acknowledgement_path(
    root: Path,
    attempt_id: str,
    scope: str,
    set_id: str,
    operator_index: int,
) -> Path:
    if _ID.fullmatch(attempt_id) is None or _ID.fullmatch(set_id) is None or operator_index not in {1, 2}:
        raise RootlessContractError("ROOTLESS_ACKNOWLEDGEMENT_PATH_INVALID")
    if scope == "plan":
        return root / "acknowledgements" / "plan" / attempt_id / set_id / f"operator-{operator_index}.json"
    if scope in {"screening", "bct"}:
        return root / "acknowledgements" / "stage" / attempt_id / scope / set_id / f"operator-{operator_index}.json"
    raise RootlessContractError("ROOTLESS_ACKNOWLEDGEMENT_PATH_INVALID")


def write_canonical_new(path: Path, value: JsonValue) -> None:
    _mkdir_chain(path.parents[1], (path.parent.name,)) if not path.parent.exists() else None
    _write_new(path, canonical_json_file(value))


def _validate_reviewed_plan(request: InitStateRequest, plan_sha256: str) -> None:
    descriptor = _read_private_file(request.plan_descriptor)
    expected = f"{plan_sha256}  phase12-filter-v5-rootless-local-execution.md\n".encode("ascii")
    if descriptor != expected:
        raise RootlessContractError("ROOTLESS_REVIEWED_PLAN_DESCRIPTOR_INVALID")
    metadata = parse_canonical_object(_read_private_file(request.review_metadata))
    if metadata not in ({},):
        required = {
            "created_at",
            "kind",
            "momus_launch_id",
            "momus_session_id",
            "momus_verdict",
            "oracle_launch_id",
            "oracle_session_id",
            "oracle_verdict",
            "plan_sha256",
            "profile",
            "round_id",
            "schema_version",
        }
        if set(metadata) != required or metadata.get("plan_sha256") != plan_sha256:
            raise RootlessContractError("ROOTLESS_REVIEW_METADATA_INVALID")


def _require_private_directory(path: Path) -> Path:
    raw = os.fspath(path)
    if not raw.startswith("/") or raw != os.path.normpath(raw):
        raise RootlessContractError("ROOTLESS_STATE_PATH_INVALID")
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise RootlessContractError("ROOTLESS_STATE_PATH_UNSAFE")
    return path


def _read_private_file(path: Path) -> bytes:
    parent = _require_private_directory(path.parent)
    parent_descriptor = os.open(parent, _DIR_FLAGS)
    try:
        descriptor = os.open(path.name, _FILE_FLAGS, dir_fd=parent_descriptor)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600:
                raise RootlessContractError("ROOTLESS_STATE_FILE_UNSAFE")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1_048_576):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _mkdir_chain(root: Path, parts: tuple[str, ...]) -> None:
    current = os.open(root, _DIR_FLAGS)
    try:
        for name in parts:
            try:
                os.mkdir(name, 0o700, dir_fd=current)
                os.fsync(current)
            except FileExistsError:
                pass
            child = os.open(name, _DIR_FLAGS, dir_fd=current)
            os.close(current)
            current = child
            info = os.fstat(current)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise RootlessContractError("ROOTLESS_STATE_PATH_UNSAFE")
    finally:
        os.close(current)


def _write_new(path: Path, raw: bytes) -> None:
    parent = os.open(path.parent, _DIR_FLAGS)
    try:
        descriptor = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=parent)
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(descriptor, raw[offset:])
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent)
    except FileExistsError as error:
        raise RootlessContractError("ROOTLESS_STATE_IMMUTABLE") from error
    finally:
        os.close(parent)
