from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field


Sha256 = str


class EvidenceError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TrackedArtifact(_StrictModel):
    path: str
    sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")


class CommandResult(_StrictModel):
    command: str
    exit_code: int
    stdout_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    terminal_lines: tuple[str, ...]


class DirtySnapshot(_StrictModel):
    entry_count: int = Field(ge=1)
    content_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    combined_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    untracked: Literal[True]
    staged: Literal[False]


class ProtectedDirtyRoot(_StrictModel):
    path: str
    before: DirtySnapshot
    after: DirtySnapshot


class EvidenceReport(_StrictModel):
    schema_version: Literal["phase13_authority_sync_calibration_v2_evidence_v1"]
    build_terminal: Literal["DETERMINISTIC_AUTHORITY_SYNC_COMPLETE"]
    calibration_terminal: Literal["CALIBRATION_V2_EXTERNAL_BLOCK"]
    main_terminal: Literal["MAIN_A_EXECUTION_FORBIDDEN"]
    provider_calls: Literal[0]
    calibration_ran: Literal[False]
    scientific_evidence: Literal[False]
    archive_status: Literal["absent"]
    claim_status: Literal["absent"]
    external_blockers: tuple[
        Literal["authenticated_structural_checkpoint_authority_incomplete"],
        Literal["runtime_archive_cardinality_contract_incompatible"],
    ]
    tracked_artifacts: tuple[TrackedArtifact, ...]
    command_results: tuple[CommandResult, ...]
    protected_dirty_roots: tuple[ProtectedDirtyRoot, ...]
    main_a_artifact_paths: tuple[()]
    report_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")


class VerifiedEvidence(_StrictModel):
    terminal: Literal["CALIBRATION_V2_EXTERNAL_BLOCK"]
    provider_calls: Literal[0]


_BLOCKERS: Final = (
    "authenticated_structural_checkpoint_authority_incomplete",
    "runtime_archive_cardinality_contract_incompatible",
)
_PROTECTED_ROOTS: Final = (
    "$tmp_dir/",
    "data/embedding_cache/bfv2-source-contract-replay-test/",
    "data/embedding_cache/bfv2-structural-replay-test/",
    "oh-my-opencode.json",
)
_COMMAND_RESULTS: Final = (
    (
        "PYTHONPATH=src python -m memcontam.cli phase13 validate-calibration-v2 --config configs/phase13/pre_main_calibration_v2.yaml",
        0,
        ("DETERMINISTIC_AUTHORITY_SYNC_COMPLETE",),
    ),
    (
        "PYTHONPATH=src python -m memcontam.cli phase13 prepare-calibration-v2 --config configs/phase13/pre_main_calibration_v2.yaml",
        0,
        ("DETERMINISTIC_AUTHORITY_SYNC_COMPLETE", "AWAITING_CALIBRATION_V2_AUTHORIZATION"),
    ),
    (
        "PYTHONPATH=src python -m memcontam.cli phase13 run-calibration-v2 --config configs/phase13/pre_main_calibration_v2.yaml",
        1,
        ("CALIBRATION_V2_EXTERNAL_BLOCK", "MAIN_A_EXECUTION_FORBIDDEN"),
    ),
    (
        "PYTHONPATH=src python -m memcontam.cli phase13 main-a",
        1,
        ("MAIN_A_EXECUTION_FORBIDDEN",),
    ),
)


def _fail(code: str) -> None:
    raise EvidenceError(code)


def _safe_tracked_path(repository_root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    private_parts = {".omo", ".env", "credentials", "cache", "authorization", "request"}
    if path.is_absolute() or ".." in path.parts or private_parts.intersection(path.parts):
        _fail("TRACKED_ARTIFACT_PATH_INVALID")
    resolved = repository_root.joinpath(*path.parts)
    if not resolved.is_file() or resolved.is_symlink():
        _fail("TRACKED_ARTIFACT_PATH_INVALID")
    tracked = subprocess.run(
        ("git", "-C", str(repository_root), "ls-files", "--error-unmatch", "--", relative),
        check=False,
        capture_output=True,
        env={**os.environ, "GIT_MASTER": "1"},
    )
    if tracked.returncode != 0:
        _fail("TRACKED_ARTIFACT_PATH_INVALID")
    return resolved


def _verify_commands(repository_root: Path, results: tuple[CommandResult, ...]) -> None:
    observed = tuple((row.command, row.exit_code, row.terminal_lines) for row in results)
    if observed != _COMMAND_RESULTS:
        _fail("COMMAND_OBSERVATION_INVALID")
    if any(
        row.stdout_sha256
        != hashlib.sha256(("\n".join(row.terminal_lines) + "\n").encode()).hexdigest()
        for row in results
    ):
        _fail("COMMAND_OBSERVATION_INVALID")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    for name in ("OPENAI_API_KEY", "BGE_M3_CACHE_PATH"):
        environment.pop(name, None)
    for row in results:
        arguments = shlex.split(row.command)
        actual = subprocess.run(
            (sys.executable, *arguments[2:]),
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        lines = tuple((*actual.stdout.splitlines(), *actual.stderr.splitlines()))
        if actual.returncode != row.exit_code or lines != row.terminal_lines:
            _fail("COMMAND_OBSERVATION_INVALID")


def verify_evidence_report(repository_root: Path, report_path: Path) -> VerifiedEvidence:
    try:
        report_bytes = report_path.read_bytes()
        raw = json.loads(report_bytes)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise EvidenceError("EVIDENCE_REPORT_INVALID") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("report_sha256"), str):
        _fail("EVIDENCE_REPORT_INVALID")
    payload = dict(raw)
    declared_hash = payload.pop("report_sha256")
    actual_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if declared_hash != actual_hash:
        _fail("REPORT_HASH_MISMATCH")
    if raw.get("calibration_terminal") != "CALIBRATION_V2_EXTERNAL_BLOCK":
        _fail("TERMINAL_INTERPRETATION_INVALID")
    if tuple(raw.get("external_blockers", ())) != _BLOCKERS:
        _fail("EXTERNAL_BLOCKERS_INVALID")
    if raw.get("main_a_artifact_paths") != []:
        _fail("MAIN_A_ARTIFACT_FORBIDDEN")
    try:
        report = EvidenceReport.model_validate_json(report_bytes)
    except ValueError as error:
        raise EvidenceError("EVIDENCE_REPORT_INVALID") from error
    for artifact in report.tracked_artifacts:
        actual = hashlib.sha256(_safe_tracked_path(repository_root, artifact.path).read_bytes()).hexdigest()
        if actual != artifact.sha256:
            _fail("TRACKED_ARTIFACT_HASH_MISMATCH")
    _verify_commands(repository_root, report.command_results)
    if tuple(root.path for root in report.protected_dirty_roots) != _PROTECTED_ROOTS:
        _fail("PROTECTED_DIRTY_INVENTORY_MISMATCH")
    if any(root.before != root.after for root in report.protected_dirty_roots):
        _fail("PROTECTED_DIRTY_STATE_MISMATCH")
    if (repository_root / "runs/phase13-main-a").exists():
        _fail("MAIN_A_ARTIFACT_FORBIDDEN")
    return VerifiedEvidence(terminal=report.calibration_terminal, provider_calls=report.provider_calls)


__all__ = ("EvidenceError", "VerifiedEvidence", "verify_evidence_report")
