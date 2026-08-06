from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol, TypeAlias


ROOT: Final = Path(__file__).resolve().parents[1]
PROFILE: Final = "local_rootless_non_authoritative"
COMPLETION_SCHEMA: Final = "rootless_pytest_completion_v1"
TIMESTAMP_PATTERN: Final = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
DIRECTORY_FLAGS: Final = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
FILE_FLAGS: Final = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
JsonValue: TypeAlias = str | int | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
ROLE_ASSERTIONS: Final[dict[str, tuple[str, ...]]] = {
    "t1": (
        "legacy_bytes_preserved",
        "source_descriptor_valid",
        "review_metadata_valid",
        "external_manifest_valid",
        "mutation_rejected",
    ),
    "t2": (
        "rootless_schema_incompatible",
        "claim_aggregation_rejected",
        "scientific_admission_rejected",
        "historical_bct_rejected",
        "pilot_b_rejected",
        "selected_policy_rejected",
    ),
    "t4": (
        "broker_boundary_passed",
        "secret_isolation_passed",
        "ledger_chain_passed",
        "budget_caps_passed",
        "http_matrix_passed",
        "recovery_passed",
        "lock_contention_passed",
    ),
    "t5": (
        "screening_cardinality_passed",
        "bct_cardinality_passed",
        "compiler_goldens_passed",
        "assessment_reduction_passed",
        "post_bct_stop_passed",
    ),
    "t6": (
        "operator_cli_passed",
        "network_denial_passed",
        "process_policy_passed",
        "claim_boundary_passed",
        "pilot_b_forbidden",
    ),
}
ROLE_TESTS: Final[dict[str, tuple[str, ...]]] = {
    "t1": (
        "tests/test_phase12_filter_v5_rootless_legacy_fence.py",
        "tests/test_phase12_filter_v5_plan_digest.py",
        "tests/test_phase12_filter_v5_evidence_security.py",
    ),
    "t2": (
        "tests/test_phase12_filter_v5_rootless_firewall.py",
        "tests/test_phase12_claim_scope.py",
        "tests/test_phase12_scientific_admission.py",
    ),
    "t4": (
        "tests/test_phase12_filter_v5_rootless_broker.py",
        "tests/test_phase12_filter_v5_rootless_ledger.py",
    ),
    "t5": (
        "tests/test_phase12_filter_v5_rootless_execution.py",
        "tests/test_phase12_filter_v5_live_bct.py",
        "tests/test_phase12_filter_v5_bct_archive.py",
        "tests/test_phase12_filter_v5_assessment.py",
    ),
    "t6": (
        "tests/test_phase12_filter_v5_rootless_cli.py",
        "tests/test_phase12_filter_v5_rootless_offline_qa.py",
        "tests/test_phase12_filter_v5_rootless_process_races.py",
        "tests/test_phase12_docs_scope.py",
        "tests/test_docs_scope.py",
        "tests/test_phase12_filter_v5_rootless_firewall.py",
    ),
}
ROLE_FILENAMES: Final[dict[str, str]] = {
    "t1": "t1-legacy-fence.json",
    "t2": "t2-firewall.json",
    "t4": "t4-broker-ledger.json",
    "t5": "t5-screening-bct.json",
    "t6": "t6-operator-claims.json",
}


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    exit_code: int
    stdout_bytes: bytes
    stderr_bytes: bytes
    provider_calls_before: int
    provider_calls_after: int


class PytestItem(Protocol):
    nodeid: str


class PytestSession(Protocol):
    items: list[PytestItem]


class PytestReport(Protocol):
    nodeid: str
    when: str
    outcome: str


_VERIFIED_RESULTS: dict[int, CommandResult] = {}
_PYTEST_COLLECTED: list[str] = []
_PYTEST_DESELECTED: list[str] = []
_PYTEST_REPORTS: dict[str, dict[str, str]] = {}


def _canonical_json(value: JsonValue) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def pytest_collection_finish(session: PytestSession) -> None:
    if "ROOTLESS_QA_RESULT_FD" in os.environ:
        _PYTEST_COLLECTED[:] = [item.nodeid for item in session.items]


def pytest_deselected(items: list[PytestItem]) -> None:
    if "ROOTLESS_QA_RESULT_FD" in os.environ:
        _PYTEST_DESELECTED.extend(item.nodeid for item in items)


def pytest_runtest_logreport(report: PytestReport) -> None:
    if "ROOTLESS_QA_RESULT_FD" not in os.environ or report.when not in {"setup", "call", "teardown"}:
        return
    outcomes = _PYTEST_REPORTS.setdefault(report.nodeid, {})
    outcomes[report.when] = "duplicate" if report.when in outcomes else report.outcome


def pytest_sessionfinish(session: PytestSession, exitstatus: int) -> None:
    descriptor_text = os.environ.get("ROOTLESS_QA_RESULT_FD")
    if descriptor_text is None:
        return
    ordered_reports: list[JsonValue] = [
        {
            "nodeid": nodeid,
            "setup": _PYTEST_REPORTS.get(nodeid, {}).get("setup", "missing"),
            "call": _PYTEST_REPORTS.get(nodeid, {}).get("call", "missing"),
            "teardown": _PYTEST_REPORTS.get(nodeid, {}).get("teardown", "missing"),
        }
        for nodeid in _PYTEST_COLLECTED
    ]
    collected_nodeids: list[JsonValue] = [* _PYTEST_COLLECTED]
    deselected_nodeids: list[JsonValue] = [* _PYTEST_DESELECTED]
    payload: dict[str, JsonValue] = {
        "schema_version": COMPLETION_SCHEMA,
        "exit_code": exitstatus,
        "collected_nodeids": collected_nodeids,
        "deselected_nodeids": deselected_nodeids,
        "ordered_reports": ordered_reports,
    }
    raw = _canonical_json(payload)
    descriptor = int(descriptor_text)
    offset = 0
    while offset < len(raw):
        offset += os.write(descriptor, raw[offset:])


def _expected_argv(role: str) -> tuple[str, ...]:
    return (
        os.path.realpath(sys.executable),
        "-B",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(ROOT / "runs" / "phase12-filter-v5-rootless-qa" / "basetemp" / role / "pytest"),
        *ROLE_TESTS[role],
        "-q",
    )


def _reject_pairs(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("ROOTLESS_TASK_QA_COMPLETION_INVALID")
        result[key] = value
    return result


def _validate_pytest_completion(role: str, raw: bytes) -> None:
    try:
        value = json.loads(raw, object_pairs_hook=_reject_pairs, parse_float=lambda _: None)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ROOTLESS_TASK_QA_COMPLETION_INVALID") from error
    if not isinstance(value, dict) or _canonical_json(value) != raw or set(value) != {
        "schema_version",
        "exit_code",
        "collected_nodeids",
        "deselected_nodeids",
        "ordered_reports",
    }:
        raise ValueError("ROOTLESS_TASK_QA_COMPLETION_INVALID")
    collected = value["collected_nodeids"]
    deselected = value["deselected_nodeids"]
    reports = value["ordered_reports"]
    if (
        value["schema_version"] != COMPLETION_SCHEMA
        or value["exit_code"] != 0
        or not isinstance(collected, list)
        or not collected
        or len(collected) != len(set(collected))
        or deselected != []
        or not isinstance(reports, list)
        or len(reports) != len(collected)
    ):
        raise ValueError("ROOTLESS_TASK_QA_COMPLETION_INVALID")
    expected_files = ROLE_TESTS.get(role)
    if expected_files is None:
        raise ValueError("ROOTLESS_TASK_QA_COMPLETION_INVALID")
    observed_files: set[str] = set()
    for nodeid, report in zip(collected, reports, strict=True):
        if not isinstance(nodeid, str) or not isinstance(report, dict) or report != {
            "nodeid": nodeid,
            "setup": "passed",
            "call": "passed",
            "teardown": "passed",
        }:
            raise ValueError("ROOTLESS_TASK_QA_COMPLETION_INVALID")
        matches = tuple(path for path in expected_files if nodeid == path or nodeid.startswith(f"{path}::"))
        if len(matches) != 1:
            raise ValueError("ROOTLESS_TASK_QA_COMPLETION_INVALID")
        observed_files.add(matches[0])
    if observed_files != set(expected_files):
        raise ValueError("ROOTLESS_TASK_QA_COMPLETION_INVALID")


def _read_pipe(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 65_536):
        chunks.append(chunk)
    return b"".join(chunks)


def run_rootless_task_qa(role: str) -> CommandResult:
    if role not in ROLE_TESTS:
        raise ValueError("ROOTLESS_TASK_QA_ROLE_INVALID")
    role_root = ROOT / "runs" / "phase12-filter-v5-rootless-qa" / "basetemp" / role
    temporary_root = role_root / "tmp"
    for path in (role_root, role_root / "pytest", temporary_root):
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("ROOTLESS_TASK_QA_BASETEMP_INVALID")
    if set(os.listdir(role_root)) != {"pytest", "tmp"} or os.listdir(role_root / "pytest") or os.listdir(temporary_root):
        raise ValueError("ROOTLESS_TASK_QA_BASETEMP_INVALID")
    read_descriptor, write_descriptor = os.pipe()
    try:
        environment = {
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_PLUGINS": "scripts.write_phase12_filter_v5_rootless_task_qa",
            "ROOTLESS_QA_RESULT_FD": str(write_descriptor),
            "TMPDIR": os.fspath(temporary_root),
            "TMP": os.fspath(temporary_root),
            "TEMP": os.fspath(temporary_root),
        }
        previous_umask = os.umask(0o077)
        try:
            completed = subprocess.run(
                _expected_argv(role),
                cwd=ROOT,
                check=False,
                capture_output=True,
                env=environment,
                pass_fds=(write_descriptor,),
            )
        finally:
            os.umask(previous_umask)
    finally:
        os.close(write_descriptor)
    try:
        completion = _read_pipe(read_descriptor)
    finally:
        os.close(read_descriptor)
    _validate_pytest_completion(role, completion)
    if completed.returncode != 0:
        raise ValueError("ROOTLESS_TASK_QA_COMMAND_INVALID")
    result = CommandResult(
        _expected_argv(role),
        completed.returncode,
        completed.stdout,
        completed.stderr,
        0,
        0,
    )
    _VERIFIED_RESULTS[id(result)] = result
    return result


def _remove_tree(parent: int, name: str, device: int) -> None:
    try:
        directory = os.open(name, DIRECTORY_FLAGS, dir_fd=parent)
    except NotADirectoryError:
        os.unlink(name, dir_fd=parent)
        return
    try:
        info = os.fstat(directory)
        if info.st_uid != os.getuid() or info.st_dev != device:
            raise ValueError("ROOTLESS_TASK_QA_BASETEMP_INVALID")
        for child in os.listdir(directory):
            _remove_tree(directory, child, device)
    finally:
        os.close(directory)
    os.rmdir(name, dir_fd=parent)


def cleanup_rootless_task_qa(role: str) -> None:
    if role not in ROLE_TESTS:
        raise ValueError("ROOTLESS_TASK_QA_ROLE_INVALID")
    basetemp = ROOT / "runs" / "phase12-filter-v5-rootless-qa" / "basetemp"
    descriptor = os.open(basetemp, DIRECTORY_FLAGS)
    try:
        info = os.fstat(descriptor)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("ROOTLESS_TASK_QA_BASETEMP_INVALID")
        try:
            _remove_tree(descriptor, role, info.st_dev)
        except FileNotFoundError:
            pass
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _task_payload(
    role: str,
    command_result: CommandResult,
    passed_assertion_ids: tuple[str, ...],
    created_at: str,
) -> dict[str, JsonValue]:
    return {
        "schema_version": "rootless_task_qa_v1",
        "profile": PROFILE,
        "kind": "task_qa",
        "role": role,
        "command": {
            "argv": list(command_result.argv),
            "exit_code": command_result.exit_code,
            "stdout_sha256": hashlib.sha256(command_result.stdout_bytes).hexdigest(),
            "stderr_sha256": hashlib.sha256(command_result.stderr_bytes).hexdigest(),
        },
        "ordered_assertions": [
            {"assertion_id": assertion_id, "passed": True} for assertion_id in passed_assertion_ids
        ],
        "provider_calls_before": 0,
        "provider_calls_after": 0,
        "created_at": created_at,
    }


def _validate_task_payload(
    raw: bytes,
    role: str,
    command_result: CommandResult,
    passed_assertion_ids: tuple[str, ...],
) -> None:
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_pairs, parse_float=lambda _: None)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ROOTLESS_TASK_QA_EXISTING_INVALID") from error
    if not isinstance(payload, dict) or _canonical_json(payload) != raw:
        raise ValueError("ROOTLESS_TASK_QA_EXISTING_INVALID")
    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or TIMESTAMP_PATTERN.fullmatch(created_at) is None:
        raise ValueError("ROOTLESS_TASK_QA_EXISTING_INVALID")
    try:
        datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError("ROOTLESS_TASK_QA_EXISTING_INVALID") from error
    if payload != _task_payload(role, command_result, passed_assertion_ids, created_at):
        raise ValueError("ROOTLESS_TASK_QA_EXISTING_INVALID")


def _optional_artifact(directory: int, name: str) -> tuple[bytes, os.stat_result] | None:
    try:
        descriptor = os.open(name, FILE_FLAGS, dir_fd=directory)
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink not in {1, 2}
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise ValueError("ROOTLESS_TASK_QA_EXISTING_INVALID")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1_048_576):
            chunks.append(chunk)
        return b"".join(chunks), info
    finally:
        os.close(descriptor)


def _write_temporary(directory: int, name: str, raw: bytes) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=directory,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_rootless_task_qa(
    role: str,
    command_result: CommandResult,
    passed_assertion_ids: tuple[str, ...],
    destination: Path,
) -> None:
    if _VERIFIED_RESULTS.get(id(command_result)) is not command_result:
        raise ValueError("ROOTLESS_TASK_QA_COMPLETION_INVALID")
    expected_assertions = ROLE_ASSERTIONS.get(role)
    if expected_assertions is None or passed_assertion_ids != expected_assertions:
        raise ValueError("ROOTLESS_TASK_QA_ASSERTIONS_INVALID")
    filename = ROLE_FILENAMES.get(role)
    if filename is None:
        raise ValueError("ROOTLESS_TASK_QA_ROLE_INVALID")
    expected_destination = ROOT / "runs" / "phase12-filter-v5-rootless-qa" / filename
    if destination != expected_destination:
        raise ValueError("ROOTLESS_TASK_QA_DESTINATION_INVALID")
    if command_result.argv != _expected_argv(role) or command_result.exit_code != 0:
        raise ValueError("ROOTLESS_TASK_QA_COMMAND_INVALID")
    if command_result.provider_calls_before != 0 or command_result.provider_calls_after != 0:
        raise ValueError("ROOTLESS_TASK_QA_PROVIDER_COUNT_INVALID")
    if any(
        not 1 <= len(argument) <= 256 or unicodedata.normalize("NFC", argument) != argument
        for argument in command_result.argv
    ):
        raise ValueError("ROOTLESS_TASK_QA_COMMAND_INVALID")
    role_root = ROOT / "runs" / "phase12-filter-v5-rootless-qa" / "basetemp" / role
    try:
        os.stat(role_root, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise ValueError("ROOTLESS_TASK_QA_BASETEMP_INVALID")

    directory = os.open(destination.parent, DIRECTORY_FLAGS)
    temporary_name = f".{destination.name}.tmp"
    try:
        parent_info = os.fstat(directory)
        if parent_info.st_uid != os.getuid() or stat.S_IMODE(parent_info.st_mode) != 0o700:
            raise ValueError("ROOTLESS_TASK_QA_DESTINATION_INVALID")
        final = _optional_artifact(directory, destination.name)
        temporary = _optional_artifact(directory, temporary_name)
        if final is not None:
            final_raw, final_info = final
            _validate_task_payload(final_raw, role, command_result, passed_assertion_ids)
            if temporary is not None:
                temporary_raw, temporary_info = temporary
                _validate_task_payload(temporary_raw, role, command_result, passed_assertion_ids)
                if (
                    temporary_raw != final_raw
                    or temporary_info.st_dev != final_info.st_dev
                    or temporary_info.st_ino != final_info.st_ino
                ):
                    raise ValueError("ROOTLESS_TASK_QA_EXISTING_INVALID")
                os.unlink(temporary_name, dir_fd=directory)
                os.fsync(directory)
            _VERIFIED_RESULTS.pop(id(command_result), None)
            return
        if temporary is None:
            created_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            raw = _canonical_json(_task_payload(role, command_result, passed_assertion_ids, created_at))
            _write_temporary(directory, temporary_name, raw)
        else:
            raw = temporary[0]
            _validate_task_payload(raw, role, command_result, passed_assertion_ids)
        os.link(
            temporary_name,
            destination.name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
        os.fsync(directory)
        os.unlink(temporary_name, dir_fd=directory)
        os.fsync(directory)
        _VERIFIED_RESULTS.pop(id(command_result), None)
    finally:
        os.close(directory)
