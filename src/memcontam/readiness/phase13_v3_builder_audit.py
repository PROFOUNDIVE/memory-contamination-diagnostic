from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter

from .phase13_authority_files import authority_directory
from .phase13_main_checkpoint import ArtifactIdentity
from .phase13_v3_builder_inputs import PREFIX, artifact_raw
from .phase13_v3_publication import ArtifactError, OUTPUT_PATHS
from .phase13_v3_resource_files import read_files
from .phase13_v3_source_closure import _git

RECEIPTS: Final = {
    ".omo/evidence/phase13-initial-head.txt": "ad82ecc044e9e7456ee05fbca21b6135d5ffd2f67d28ce395ccc93b3eb49eb72",
    ".omo/evidence/phase13-initial-status.bin": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ".omo/evidence/phase13-historical-baseline.json": "7b892e10d4e44cff4115ebf23810dbdfff4757910cbc6307c4f54be8ccc52ce0",
}
_BASELINE: Final[TypeAdapter[tuple[ArtifactIdentity, ...]]] = TypeAdapter(tuple[ArtifactIdentity, ...])
ALLOW: Final = (
    "src/memcontam/readiness/phase13_*.py", "src/memcontam/clients/openai_responses.py",
    "src/memcontam/experiment/phase13_ordinary_runtime.py", "src/memcontam/verifiers/math_equation_balancer.py",
    "scripts/build_phase13_corrected_main_closure.py", "scripts/diagnose_phase13_mr_p5_closure.py",
    "scripts/build_phase13_main_registries.py", "tests/phase13_v3_fixtures.py", "tests/test_phase13_*.py",
    "tests/test_bot_retrieval_decision.py", ".omo/evidence/*", *(PREFIX + name for name in OUTPUT_PATHS),
)


def audit_scope(repository: Path, output: Path, compare: Path | None) -> None:
    if compare is not None:
        for name in OUTPUT_PATHS:
            if artifact_raw(output, name) != artifact_raw(compare, name):
                raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
        actual = {path.relative_to(compare).as_posix() for path in compare.rglob("*") if not path.is_dir()}
        if actual != set(OUTPUT_PATHS):
            raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
        return
    receipts = {row.binding.path: row for row in read_files(repository, tuple(RECEIPTS))}
    if any(receipts[name].binding.sha256 != expected for name, expected in RECEIPTS.items()):
        raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
    baseline = _BASELINE.validate_json(receipts[".omo/evidence/phase13-historical-baseline.json"].raw)
    if len(baseline) != 147:
        raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
    current = {row.binding.path: row.binding.sha256 for row in read_files(repository, tuple(row.path for row in baseline))}
    if any(current[row.path] != row.sha256 for row in baseline):
        raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
    initial = receipts[".omo/evidence/phase13-initial-head.txt"].raw.decode().strip()
    with authority_directory(repository) as directory:
        diff = _git(directory, ("diff", "--name-status", "--no-renames", initial, "HEAD")).decode()
        for line in diff.splitlines():
            status, path = line.split("\t", 1)
            if status not in {"A", "M"} or not any(fnmatch.fnmatchcase(path, rule) for rule in ALLOW):
                raise ArtifactError("MAIN_GOVERNED_SOURCE_DRIFT")
        for filename in ("pyproject.toml", "requirements.lock", "requirements-dev.lock"):
            row, = read_files(repository, (filename,))
            if row.raw != _git(directory, ("show", f"{initial}:{filename}")):
                raise ArtifactError("MAIN_GOVERNED_SOURCE_DRIFT")
    baseline_names = {row.path.removeprefix(PREFIX) for row in baseline if row.path.startswith(PREFIX)}
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if not path.is_dir()}
    if actual != baseline_names | set(OUTPUT_PATHS):
        raise ArtifactError("MAIN_ARTIFACT_BINDING_MISMATCH")
    read_files(output, tuple(actual))
