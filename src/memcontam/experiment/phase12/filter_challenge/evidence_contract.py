from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue


EVIDENCE_FILENAMES: Final[tuple[str, ...]] = (
    "implementation_manifest.json",
    "policy_schema_hashes.json",
    "mft_fv5_report.json",
    "information_boundary_report.json",
    "route_invariance_report.json",
    "answer_call_provenance_report.json",
    "archive_validation_report.json",
    "test_lint_typecheck_report.json",
    "bct_readiness_report.json",
)
NON_MANIFEST_FILENAMES: Final[tuple[str, ...]] = EVIDENCE_FILENAMES[1:]
AMENDMENT: Final[dict[str, str]] = {
    "sha256": "d75d4ad5d5b13a057fc5cf49fd02f00277b83c4863f39b2aa1975660f5fecfee",
    "version": "phase12-filter-v5-amendment-v1.3",
}
AUTHORITY_HASHES: Final[dict[str, str]] = {
    "accepted_erratum": "362f3ba6c51dec7ebfd61b68a9c908e64ef84858e93f796bf5de6b40fb70cd46",
    "baseline": "8a279c1a644e84adc508c87f12b02009e22555072379252f29af06b8878fcae9",
    "experiment_design": "984fe2881690d93a8ccced87abf03de4bf0012158462cf07ed23505414073eb0",
    "protocol": "9dcd3855f3f65b8d623b3d3c3600a6a7b0231228c19231359d883e3f40b1959a",
    "theory": "56c42b1af761c2f3838638316823d0ce394c6c543f17aff97f4657721c964983",
}
AUTHORITY_BINDINGS: Final = (
    (
        "accepted_erratum",
        Path("/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/References/Theoretical Artifacts/AGENTS.md"),
        AUTHORITY_HASHES["accepted_erratum"],
    ),
    (
        "amendment",
        Path("/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/References/Theoretical Artifacts/Phase 12 Filter-v5 Verifier-Backed Challenge Amendment.md"),
        AMENDMENT["sha256"],
    ),
    (
        "theory",
        Path("/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/References/Theoretical Artifacts/Phase 12 — THEORETICAL ARTIFACT.md"),
        AUTHORITY_HASHES["theory"],
    ),
    (
        "baseline",
        Path("/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/References/Theoretical Artifacts/Phase 12-Compatible Baseline Memory and Lightweight Filter Design revised-v3.md"),
        AUTHORITY_HASHES["baseline"],
    ),
    (
        "protocol",
        Path("/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/References/Theoretical Artifacts/Phase 12-Compatible Contamination Construction Intervention Timing and Sensitivity Protocol.md"),
        AUTHORITY_HASHES["protocol"],
    ),
    (
        "experiment_design",
        Path("/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/References/Theoretical Artifacts/Phase 12-Compatible Pilot Main and Exploratory Experiment Design.md"),
        AUTHORITY_HASHES["experiment_design"],
    ),
)
POLICY: Final[dict[str, str]] = {
    "canonical_patch_status": "pending_before_provider_backed_pilot_b",
    "claim_boundary": "build_layer_implementation_and_state_transition_only",
    "identity": "Filter-Challenge-v1",
    "schema_version": "filter_challenge_domain_v1",
}
@dataclass(frozen=True, slots=True)
class EvidenceBuildError(ValueError):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class DescriptorHash:
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    header: dict[str, JsonValue]
    implementation_manifest_sha256: str
    root: Path


def canonical_json_bytes(value: JsonValue) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def descriptor_sha256(path: Path) -> DescriptorHash:
    if not path.is_absolute():
        raise EvidenceBuildError("DESCRIPTOR_PATH_ABSOLUTE_REQUIRED")
    if any(part in {".", ".."} for part in path.parts):
        raise EvidenceBuildError("DESCRIPTOR_PATH_COMPONENT_INVALID")
    parts = tuple(part for part in path.parts if part != "/")
    if not parts:
        raise EvidenceBuildError("DESCRIPTOR_PATH_INVALID")
    root_fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    directory_fd = root_fd
    try:
        for component in parts[:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                os.close(next_fd)
                raise EvidenceBuildError("DESCRIPTOR_ANCESTOR_INVALID")
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        try:
            info = os.fstat(file_fd)
            if not stat.S_ISREG(info.st_mode):
                raise EvidenceBuildError("DESCRIPTOR_FINAL_NOT_REGULAR")
            digest = hashlib.sha256()
            while chunk := os.read(file_fd, 1_048_576):
                digest.update(chunk)
            return DescriptorHash(byte_count=info.st_size, sha256=digest.hexdigest())
        finally:
            os.close(file_fd)
    except OSError as error:
        raise EvidenceBuildError("DESCRIPTOR_OPEN_FAILED") from error
    finally:
        os.close(directory_fd)


def json_value_from_bytes(raw: bytes, error_code: str) -> JsonValue:
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceBuildError(error_code) from error


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def require_clean_repository(root: Path, implementation_commit: str) -> None:
    if not root.is_absolute():
        raise EvidenceBuildError("REPOSITORY_ROOT_ABSOLUTE_REQUIRED")
    top_level = _git(root, "rev-parse", "--show-toplevel", "REPOSITORY_ROOT_INVALID")
    if top_level != str(root):
        raise EvidenceBuildError("REPOSITORY_ROOT_INVALID")
    if _git(root, "status", "--porcelain=v1", "REPOSITORY_ROOT_INVALID"):
        raise EvidenceBuildError("REPOSITORY_DIRTY")
    head = _git(root, "rev-parse", "HEAD", "REPOSITORY_ROOT_INVALID")
    if head != implementation_commit:
        raise EvidenceBuildError("IMPLEMENTATION_COMMIT_MISMATCH")


def _git(root: Path, *arguments: str) -> str:
    *command, error_code = arguments
    result = subprocess.run(
        ["git", "-C", str(root), *command], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise EvidenceBuildError(error_code)
    return result.stdout.strip()
