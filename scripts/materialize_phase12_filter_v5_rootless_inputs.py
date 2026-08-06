from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import stat
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Never, TypeAlias


TRUSTED_BASE: Final = "c057fb1adf9571ef21cd19fa2733c5ac47b40798"
PROFILE: Final = "local_rootless_non_authoritative"
MANIFEST_RELATIVE_PATH: Final = Path("configs/phase12/filter_v5_rootless_local/external_inputs.json")
MANIFEST_DESCRIPTOR_RELATIVE_PATH: Final = Path(
    "docs/evidence/phase12-filter-v5-rootless-local/legacy-input-manifest.sha256"
)
PLAN_DESCRIPTOR_FILENAME: Final = "phase12-filter-v5-rootless-local-execution.md"
ID_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
HEX_PATTERN: Final = re.compile(r"[0-9a-f]{64}\Z")
TIMESTAMP_PATTERN: Final = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
ROUND_PATTERN: Final = re.compile(
    r"phase12-filter-v5-rootless-local-execution-r([1-9][0-9]*)-([0-9a-f]{8})\Z"
)
REVIEW_METADATA_FORGEABILITY: Final = (
    "same-UID operators can forge operator_asserted_review_metadata; it is not review authority"
)
PROTECTED_PATHS: Final = (
    ".sisyphus/evidence/phase12-filter-v5-build-v1",
    "docs/evidence/phase12-filter-v5-bct-v1",
    "data/phase12/filter_v5_bct_v1",
    "configs/phase12/filter_v5_bct_calibration.yaml",
)
GIT_CONTEXT_VALIDATOR: Final = Path(__file__).with_name("validate_phase12_filter_v5_rootless_git_context.py")
JsonValue: TypeAlias = str | int | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class LegacyFenceError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class InputPin:
    role: str
    destination: Path
    size_bytes: int
    sha256: str


INPUT_PINS: Final = (
    InputPin(
        "ROOTLESS_HISTORICAL_SCREENING_PLAN",
        Path(".omo/plans/phase12-filter-v5-screening-bct-execution.md"),
        144691,
        "9270d31770eb97e732602cfe85a250111208afeae293b0a20ab618baadb43317",
    ),
    InputPin(
        "ROOTLESS_HISTORICAL_SCREENING_DESCRIPTOR",
        Path(".omo/approvals/phase12-filter-v5-screening-bct-execution.plan.sha256"),
        65,
        "92c6d30f026a10f47067e5467c0e9e0abc35b653385f4f08ad7d301838e06160",
    ),
    InputPin(
        "ROOTLESS_HISTORICAL_POST_DESCRIPTOR",
        Path(".omo/approvals/phase12-post-filter-v5-calibration-readiness.plan.sha256"),
        65,
        "7b878988972b5bc3c1a2ba24785b978cc26b973e1e44e8059ff8d3133227842e",
    ),
)


def _fail(code: str) -> Never:
    raise LegacyFenceError(code)


def _absolute(path: Path) -> Path:
    text = os.fspath(path)
    if not text.startswith("/") or text.startswith("//") or text != os.path.normpath(text) or any(
        part in {".", ".."} for part in path.parts
    ):
        _fail("ROOTLESS_LEGACY_PATH_INVALID")
    return path


def _parse_cli_path(raw: str, *, file_source: bool) -> Path:
    if raw == "/":
        if file_source:
            _fail("ROOTLESS_LEGACY_PATH_INVALID")
        return Path(raw)
    if not raw.startswith("/") or raw.startswith("//") or raw.endswith("/"):
        _fail("ROOTLESS_LEGACY_PATH_INVALID")
    if any(component in {"", ".", ".."} for component in raw[1:].split("/")):
        _fail("ROOTLESS_LEGACY_PATH_INVALID")
    return _absolute(Path(raw))


def _safe_directory(info: os.stat_result, current_uid: int, require_private: bool = True) -> None:
    if not stat.S_ISDIR(info.st_mode) or info.st_uid not in {0, current_uid}:
        _fail("ROOTLESS_LEGACY_PATH_UNSAFE")
    if require_private and stat.S_IMODE(info.st_mode) & 0o022:
        _fail("ROOTLESS_LEGACY_PATH_UNSAFE")


def _read_regular(path: Path, required_mode: int | None, current_uid_only: bool) -> bytes:
    target = _absolute(path)
    current_uid = os.getuid()
    parts = tuple(part for part in target.parts if part != "/")
    if not parts:
        _fail("ROOTLESS_LEGACY_PATH_INVALID")
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        _safe_directory(os.fstat(directory), current_uid)
        for component in parts[:-1]:
            next_directory = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory,
            )
            os.close(directory)
            directory = next_directory
            _safe_directory(os.fstat(directory), current_uid)
        file_descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory
        )
        try:
            info = os.fstat(file_descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                _fail("ROOTLESS_LEGACY_FILE_UNSAFE")
            if current_uid_only and info.st_uid != current_uid:
                _fail("ROOTLESS_LEGACY_FILE_UNSAFE")
            if not current_uid_only and info.st_uid not in {0, current_uid}:
                _fail("ROOTLESS_LEGACY_FILE_UNSAFE")
            if required_mode is not None and stat.S_IMODE(info.st_mode) != required_mode:
                _fail("ROOTLESS_LEGACY_FILE_UNSAFE")
            chunks: list[bytes] = []
            while chunk := os.read(file_descriptor, 1_048_576):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(file_descriptor)
    except OSError as error:
        raise LegacyFenceError("ROOTLESS_LEGACY_OPEN_FAILED") from error
    finally:
        os.close(directory)


def _canonical_json(value: JsonValue) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _reject_pairs(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            _fail("ROOTLESS_LEGACY_JSON_INVALID")
        result[key] = value
    return result


def _reject_float(_: str) -> None:
    _fail("ROOTLESS_LEGACY_JSON_INVALID")


def _validate_json_value(value: JsonValue) -> None:
    if isinstance(value, str):
        if unicodedata.normalize("NFC", value) != value or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            _fail("ROOTLESS_LEGACY_JSON_INVALID")
    elif isinstance(value, list):
        for item in value:
            _validate_json_value(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail("ROOTLESS_LEGACY_JSON_INVALID")
            _validate_json_value(key)
            _validate_json_value(item)
    elif value is None or isinstance(value, bool) or (isinstance(value, int) and not isinstance(value, bool)):
        return
    else:
        _fail("ROOTLESS_LEGACY_JSON_INVALID")


def _parse_canonical_object(raw: bytes) -> dict[str, JsonValue]:
    try:
        value = json.loads(raw, object_pairs_hook=_reject_pairs, parse_float=_reject_float)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LegacyFenceError("ROOTLESS_LEGACY_JSON_INVALID") from error
    if not isinstance(value, dict):
        _fail("ROOTLESS_LEGACY_JSON_INVALID")
    _validate_json_value(value)
    if _canonical_json(value) != raw:
        _fail("ROOTLESS_LEGACY_JSON_NONCANONICAL")
    return value


def _require_string(value: JsonValue) -> str:
    if not isinstance(value, str):
        _fail("ROOTLESS_LEGACY_SCHEMA_INVALID")
    return value


def _require_hash(value: JsonValue) -> str:
    result = _require_string(value)
    if HEX_PATTERN.fullmatch(result) is None:
        _fail("ROOTLESS_LEGACY_SCHEMA_INVALID")
    return result


def _require_id(value: JsonValue) -> str:
    result = _require_string(value)
    if ID_PATTERN.fullmatch(result) is None:
        _fail("ROOTLESS_LEGACY_SCHEMA_INVALID")
    return result


def _require_txt256(value: JsonValue) -> str:
    result = _require_string(value)
    if not 1 <= len(result) <= 256 or unicodedata.normalize("NFC", result) != result or any(
        0xD800 <= ord(character) <= 0xDFFF for character in result
    ):
        _fail("ROOTLESS_REVIEW_METADATA_INVALID")
    return result


def _require_timestamp(value: JsonValue) -> None:
    timestamp = _require_string(value)
    if TIMESTAMP_PATTERN.fullmatch(timestamp) is None:
        _fail("ROOTLESS_LEGACY_SCHEMA_INVALID")
    try:
        datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise LegacyFenceError("ROOTLESS_LEGACY_SCHEMA_INVALID") from error


def _verify_manifest(repository_root: Path) -> str:
    manifest = _read_regular(repository_root / MANIFEST_RELATIVE_PATH, None, False)
    descriptor = _read_regular(repository_root / MANIFEST_DESCRIPTOR_RELATIVE_PATH, None, False)
    expected_descriptor = (
        hashlib.sha256(manifest).hexdigest().encode("ascii")
        + b"  configs/phase12/filter_v5_rootless_local/external_inputs.json\n"
    )
    if not hmac.compare_digest(descriptor, expected_descriptor):
        _fail("ROOTLESS_LEGACY_MANIFEST_DESCRIPTOR_INVALID")
    payload = _parse_canonical_object(manifest)
    if set(payload) != {"schema_version", "profile", "kind", "ordered_inputs"}:
        _fail("ROOTLESS_LEGACY_MANIFEST_INVALID")
    ordered_inputs = payload["ordered_inputs"]
    if (
        payload["schema_version"] != "rootless_external_input_manifest_v1"
        or payload["profile"] != PROFILE
        or payload["kind"] != "external_input_manifest"
        or not isinstance(ordered_inputs, list)
        or len(ordered_inputs) != len(INPUT_PINS)
    ):
        _fail("ROOTLESS_LEGACY_MANIFEST_INVALID")
    for record, pin in zip(ordered_inputs, INPUT_PINS, strict=True):
        if not isinstance(record, dict) or set(record) != {
            "role",
            "repo_relative_destination",
            "size_bytes",
            "sha256",
        }:
            _fail("ROOTLESS_LEGACY_MANIFEST_INVALID")
        size = record["size_bytes"]
        if (
            record["role"] != pin.role
            or record["repo_relative_destination"] != str(pin.destination)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size != pin.size_bytes
            or record["sha256"] != pin.sha256
        ):
            _fail("ROOTLESS_LEGACY_MANIFEST_INVALID")
    return hashlib.sha256(manifest).hexdigest()


def _git(repository_root: Path, *arguments: str) -> bytes:
    validation = subprocess.run(
        [
            os.path.realpath(sys.executable),
            "-B",
            "-I",
            "-S",
            str(GIT_CONTEXT_VALIDATOR),
            "--repo-root",
            str(repository_root),
        ],
        check=False,
        capture_output=True,
        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    if validation.returncode != 0 or validation.stdout or validation.stderr:
        _fail("ROOTLESS_LEGACY_GIT_INVALID")
    result = subprocess.run(
        [
            "/usr/bin/git",
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.fileMode=true",
            "-c",
            "core.ignoreCase=false",
            "-c",
            "core.precomposeUnicode=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.excludesFile=/dev/null",
            "-c",
            "core.attributesFile=/dev/null",
            "-c",
            "core.bare=false",
            "-c",
            f"core.worktree={repository_root}",
            "-c",
            "status.relativePaths=false",
            "-c",
            "submodule.recurse=false",
            "-c",
            "diff.ignoreSubmodules=none",
            "-C",
            str(repository_root),
            *arguments,
        ],
        check=False,
        capture_output=True,
        env={
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_NO_REPLACE_OBJECTS": "1",
        },
    )
    if result.returncode != 0 or result.stderr:
        _fail("ROOTLESS_LEGACY_GIT_INVALID")
    return result.stdout


def _verify_protected_base(repository_root: Path, require_clean: bool) -> None:
    _git(repository_root, "merge-base", "--is-ancestor", TRUSTED_BASE, "HEAD")
    if require_clean and _git(
        repository_root, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"
    ):
        _fail("ROOTLESS_LEGACY_WORKTREE_DIRTY")
    records = _git(repository_root, "ls-tree", "-r", "-z", TRUSTED_BASE, "--", *PROTECTED_PATHS)
    if not records:
        _fail("ROOTLESS_LEGACY_BASE_INVALID")
    for record in records.rstrip(b"\0").split(b"\0"):
        try:
            metadata, encoded_path = record.split(b"\t", 1)
            mode, kind, _ = metadata.split(b" ")
            relative_path = encoded_path.decode("utf-8")
        except ValueError as error:
            raise LegacyFenceError("ROOTLESS_LEGACY_BASE_INVALID") from error
        if mode != b"100644" or kind != b"blob" or relative_path.startswith("/") or ".." in Path(relative_path).parts:
            _fail("ROOTLESS_LEGACY_BASE_INVALID")
        expected = _git(repository_root, "cat-file", "blob", f"{TRUSTED_BASE}:{relative_path}")
        observed = _read_regular(repository_root / relative_path, None, False)
        if not hmac.compare_digest(observed, expected):
            _fail("ROOTLESS_LEGACY_PROTECTED_BYTES_DRIFT")


def _open_or_create_directory(parent: int, name: str) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent)
    except FileExistsError:
        pass
    else:
        try:
            os.fsync(parent)
        except OSError as error:
            raise LegacyFenceError("ROOTLESS_LEGACY_DESTINATION_UNSAFE") from error
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
    except OSError as error:
        raise LegacyFenceError("ROOTLESS_LEGACY_DESTINATION_UNSAFE") from error
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        os.close(descriptor)
        _fail("ROOTLESS_LEGACY_DESTINATION_UNSAFE")
    return descriptor


def _write_once(directory: int, name: str, raw: bytes) -> None:
    temporary = f".{name}.tmp"
    try:
        os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        _fail("ROOTLESS_LEGACY_REMATERIALIZATION_REJECTED")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
    except OSError as error:
        raise LegacyFenceError("ROOTLESS_LEGACY_DESTINATION_UNSAFE") from error
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        os.fsync(directory)
        os.unlink(temporary, dir_fd=directory)
        os.fsync(directory)
    except OSError as error:
        raise LegacyFenceError("ROOTLESS_LEGACY_DESTINATION_UNSAFE") from error


def _require_absent(directory: int, name: str) -> None:
    try:
        os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return
    _fail("ROOTLESS_LEGACY_REMATERIALIZATION_REJECTED")


def _read_pinned_sources(sources: tuple[Path, Path, Path]) -> tuple[bytes, bytes, bytes]:
    values: list[bytes] = []
    for source, pin in zip(sources, INPUT_PINS, strict=True):
        raw = _read_regular(source, 0o600, True)
        if len(raw) != pin.size_bytes or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), pin.sha256):
            _fail("ROOTLESS_LEGACY_SOURCE_PIN_MISMATCH")
        values.append(raw)
    return values[0], values[1], values[2]


def _verify_materialized_inputs(root: Path, require_clean: bool) -> str:
    _verify_manifest(root)
    _verify_protected_base(root, require_clean)
    for pin in INPUT_PINS:
        raw = _read_regular(root / pin.destination, 0o600, True)
        if len(raw) != pin.size_bytes or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), pin.sha256):
            _fail("ROOTLESS_LEGACY_DESTINATION_MISMATCH")
    return _verify_manifest(root)


def validate_legacy_fence(repository_root: Path) -> str:
    return _verify_materialized_inputs(_absolute(repository_root), True)


def validate_final_index_legacy_manifest_sha256(value: str, repository_root: Path) -> None:
    if not hmac.compare_digest(value, _verify_manifest(_absolute(repository_root))):
        _fail("ROOTLESS_LEGACY_FINAL_INDEX_HASH_INVALID")


def validate_reviewed_plan(plan_source: Path, descriptor_path: Path, metadata_path: Path) -> str:
    plan = _read_regular(plan_source, 0o600, True)
    digest = hashlib.sha256(plan).hexdigest()
    expected_descriptor = f"{digest}  {PLAN_DESCRIPTOR_FILENAME}\n".encode("ascii")
    if not hmac.compare_digest(_read_regular(descriptor_path, 0o600, True), expected_descriptor):
        _fail("ROOTLESS_REVIEWED_PLAN_DESCRIPTOR_INVALID")
    metadata = _parse_canonical_object(_read_regular(metadata_path, 0o600, True))
    expected_keys = {
        "schema_version",
        "profile",
        "kind",
        "plan_sha256",
        "round_id",
        "momus_launch_id",
        "momus_session_id",
        "momus_verdict",
        "oracle_launch_id",
        "oracle_session_id",
        "oracle_verdict",
        "created_at",
    }
    if set(metadata) != expected_keys or metadata["schema_version"] != "rootless_operator_asserted_review_metadata_v1" or metadata["profile"] != PROFILE or metadata["kind"] != "operator_asserted_dual_review":
        _fail("ROOTLESS_REVIEW_METADATA_INVALID")
    if _require_hash(metadata["plan_sha256"]) != digest or metadata["momus_verdict"] != "OKAY" or metadata["oracle_verdict"] != "OKAY":
        _fail("ROOTLESS_REVIEW_METADATA_INVALID")
    round_id = _require_string(metadata["round_id"])
    match = ROUND_PATTERN.fullmatch(round_id)
    if match is None or match.group(2) != digest[:8]:
        _fail("ROOTLESS_REVIEW_METADATA_INVALID")
    number, prefix = match.groups()
    if _require_id(metadata["momus_launch_id"]) != f"momus-r{number}-{prefix}" or _require_id(metadata["oracle_launch_id"]) != f"oracle-r{number}-{prefix}":
        _fail("ROOTLESS_REVIEW_METADATA_INVALID")
    _require_txt256(metadata["round_id"])
    _require_txt256(metadata["momus_session_id"])
    _require_txt256(metadata["oracle_session_id"])
    _require_timestamp(metadata["created_at"])
    return digest


def materialize(repository_root: Path, sources: tuple[Path, Path, Path]) -> None:
    root = _absolute(repository_root)
    _verify_manifest(root)
    _verify_protected_base(root, False)
    raw_sources = _read_pinned_sources(sources)
    repository_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        omo = _open_or_create_directory(repository_descriptor, ".omo")
        try:
            plans = _open_or_create_directory(omo, "plans")
            approvals = _open_or_create_directory(omo, "approvals")
            try:
                for directory, pin in zip((plans, approvals, approvals), INPUT_PINS, strict=True):
                    _require_absent(directory, pin.destination.name)
                for directory, pin, raw in zip((plans, approvals, approvals), INPUT_PINS, raw_sources, strict=True):
                    _write_once(directory, pin.destination.name, raw)
            finally:
                os.close(plans)
                os.close(approvals)
        finally:
            os.close(omo)
    finally:
        os.close(repository_descriptor)
    _verify_materialized_inputs(root, False)


def main() -> int:
    parser = argparse.ArgumentParser(description=REVIEW_METADATA_FORGEABILITY)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--historical-screening-plan", required=True)
    parser.add_argument("--historical-screening-descriptor", required=True)
    parser.add_argument("--historical-post-descriptor", required=True)
    arguments = parser.parse_args()
    try:
        materialize(
            _parse_cli_path(arguments.repo_root, file_source=False),
            (
                _parse_cli_path(arguments.historical_screening_plan, file_source=True),
                _parse_cli_path(arguments.historical_screening_descriptor, file_source=True),
                _parse_cli_path(arguments.historical_post_descriptor, file_source=True),
            ),
        )
    except LegacyFenceError as error:
        print(error.code, file=sys.stderr)
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
