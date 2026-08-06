from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final


ROOT: Final = Path(__file__).resolve().parents[1]
PROFILE: Final = "local_rootless_non_authoritative"
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


def _canonical_json(value: dict[str, str | int | list[dict[str, str | bool] | str]]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


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


def write_rootless_task_qa(
    role: str,
    command_result: CommandResult,
    passed_assertion_ids: tuple[str, ...],
    destination: Path,
) -> None:
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
    payload = {
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
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    raw = _canonical_json(payload)
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            fd = os.open(
                destination.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=descriptor,
            )
        except FileExistsError as error:
            raise ValueError("ROOTLESS_TASK_QA_DESTINATION_EXISTS") from error
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(fd, raw[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
