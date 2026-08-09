from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Final

from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_operator import (
    PROFILE,
    qa_root,
    read_canonical,
    write_anchor,
    write_new_or_same,
)


_REPORTS: Final = {
    "pre_f1": "f1-plan-compliance.json",
    "pre_f2": "f2-broker-security.json",
    "pre_f3": "f3-cli-rehearsal.json",
}
_TESTS: Final = {
    "pre_f1": (
        "tests/test_phase12_filter_v5_rootless_legacy_fence.py",
        "tests/test_phase12_filter_v5_rootless_binding.py",
        "tests/test_phase12_filter_v5_rootless_external_authority.py",
        "tests/test_phase12_filter_v5_rootless_firewall.py",
    ),
    "pre_f2": (
        "tests/test_phase12_filter_v5_rootless_broker.py",
        "tests/test_phase12_filter_v5_rootless_ledger.py",
    ),
    "pre_f3": (
        "tests/test_phase12_filter_v5_rootless_cli.py",
        "tests/test_phase12_filter_v5_rootless_execution.py",
        "tests/test_phase12_filter_v5_rootless_post_bct.py",
        "tests/test_phase12_filter_v5_rootless_process_races.py",
    ),
}


def run_pre_egress(repository: Path, role: str, execution_commit: str, created_at: str) -> str:
    write_anchor(repository, execution_commit)
    role_root = qa_root(repository) / "basetemp" / role
    pytest_root = role_root / "pytest"
    temporary = role_root / "tmp"
    role_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    pytest_root.mkdir(mode=0o700)
    temporary.mkdir(mode=0o700)
    argv = (
        os.path.realpath(sys.executable), "-B", "-I", "-m", "pytest", "-p", "no:cacheprovider",
        "--basetemp", os.fspath(pytest_root), *_TESTS[role], "-q",
    )
    environment = {
        "LC_ALL": "C", "PATH": "/usr/bin:/bin", "TMPDIR": os.fspath(temporary),
        "TMP": os.fspath(temporary), "TEMP": os.fspath(temporary),
    }
    try:
        completed = subprocess.run(
            argv, cwd=repository, env=environment, capture_output=True, check=False, close_fds=True
        )
    finally:
        shutil.rmtree(role_root)
    if completed.returncode:
        raise RootlessContractError("ROOTLESS_PRE_EGRESS_QA_FAILED")
    value: dict[str, JsonValue] = {
        "schema_version": "rootless_pre_egress_qa_v1", "profile": PROFILE, "role": role,
        "execution_commit": execution_commit, "command_argv": list(argv), "exit_code": 0,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "provider_calls_before": 0, "provider_calls_after": 0, "created_at": created_at,
    }
    return write_new_or_same(qa_root(repository) / "pre-egress" / _REPORTS[role], value)


def verify_pre_egress(repository: Path, execution_commit: str) -> None:
    anchor = read_canonical(qa_root(repository) / "pre-egress/execution-anchor.json")
    if anchor.get("execution_commit") != execution_commit:
        raise RootlessContractError("ROOTLESS_PRE_EGRESS_QA_INVALID")
    for role, filename in _REPORTS.items():
        report = read_canonical(qa_root(repository) / "pre-egress" / filename)
        if report.get("role") != role or report.get("execution_commit") != execution_commit or report.get("exit_code") != 0:
            raise RootlessContractError("ROOTLESS_PRE_EGRESS_QA_INVALID")
