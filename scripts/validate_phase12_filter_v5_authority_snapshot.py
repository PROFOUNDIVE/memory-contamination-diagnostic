from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Final


ROOT: Final = Path(__file__).resolve().parents[1]
AUTHORITY_ROOT: Final = Path(
    "/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/References/Theoretical Artifacts"
)
STARTING_HEAD: Final = "32efbe15fc645de84585acb55b78946cfcd4298a"
TREE_BINDINGS: Final = {
    "src/memcontam/experiment/phase12/filter_challenge": "9d3641dfe5c7481d51b12d66486160875f2109ec",
    ".sisyphus/evidence/phase12-filter-v5-build-v1": "1c4f11a5fffb52c8bb2e80bfd0a1722449df936c",
}
AUTHORITIES: Final = (
    ("A3", "Phase 12 Filter-v5 Verifier-Backed Challenge Amendment.md", "d75d4ad5d5b13a057fc5cf49fd02f00277b83c4863f39b2aa1975660f5fecfee"),
    ("A4", "Phase 12 — THEORETICAL ARTIFACT.md", "56c42b1af761c2f3838638316823d0ce394c6c543f17aff97f4657721c964983"),
    ("A5", "Phase 12-Compatible Baseline Memory and Lightweight Filter Design revised-v3.md", "8a279c1a644e84adc508c87f12b02009e22555072379252f29af06b8878fcae9"),
    ("A6", "Phase 12-Compatible Contamination Construction Intervention Timing and Sensitivity Protocol.md", "9dcd3855f3f65b8d623b3d3c3600a6a7b0231228c19231359d883e3f40b1959a"),
    ("A7", "Phase 12-Compatible Pilot Main and Exploratory Experiment Design.md", "984fe2881690d93a8ccced87abf03de4bf0012158462cf07ed23505414073eb0"),
    ("A8", "AGENTS.md", "362f3ba6c51dec7ebfd61b68a9c908e64ef84858e93f796bf5de6b40fb70cd46"),
)
SEMANTIC_A3: Final = (
    "strict_primary_eligible", "candidate-absent control screening", "Pilot-A instances and their canonical variants remain excluded", "pooling_allowed: false", "not_contradicted -> active",
)


class ValidationError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _absolute(path: Path) -> Path:
    value = path if path.is_absolute() else Path.cwd() / path
    if any(part in {".", ".."} for part in value.parts):
        raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
    return value


def _read_nofollow(path: Path) -> tuple[bytes, os.stat_result]:
    target = _absolute(path)
    parts = tuple(part for part in target.parts if part != "/")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        file_descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
        try:
            info = os.fstat(file_descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
            chunks: list[bytes] = []
            while chunk := os.read(file_descriptor, 1_048_576):
                chunks.append(chunk)
            return b"".join(chunks), info
        finally:
            os.close(file_descriptor)
    except OSError as error:
        raise ValidationError("THEORETICAL_AUTHORITY_DRIFT") from error
    finally:
        os.close(descriptor)


def _json(path: Path) -> dict[str, object]:
    raw, _ = _read_nofollow(path)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError("THEORETICAL_AUTHORITY_DRIFT") from error
    if not isinstance(value, dict):
        raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
    return value


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments), capture_output=True, check=False, text=True,
        env={**os.environ, "GIT_MASTER": "1"},
    )
    if result.returncode:
        raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
    return result.stdout.strip()


def _plan_digest(repository_root: Path) -> str:
    plan, _ = _read_nofollow(repository_root / ".omo/plans/phase12-post-filter-v5-calibration-readiness.md")
    descriptor, _ = _read_nofollow(repository_root / ".omo/approvals/phase12-post-filter-v5-calibration-readiness.plan.sha256")
    if descriptor != hashlib.sha256(plan).hexdigest().encode("ascii") + b"\n":
        raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
    return descriptor.decode("ascii").strip()


def _validate_authorities(manifest: dict[str, object], authority_root: Path) -> list[dict[str, object]]:
    records = manifest.get("external_authorities")
    if not isinstance(records, list) or len(records) != len(AUTHORITIES):
        raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
    by_id = {item.get("authority_id"): item for item in records if isinstance(item, dict)}
    observed: list[dict[str, object]] = []
    for authority_id, relative_path, expected_hash in AUTHORITIES:
        record = by_id.get(authority_id)
        if not isinstance(record, dict) or record.get("relative_path") != relative_path or record.get("sha256") != expected_hash:
            raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
        raw, info = _read_nofollow(authority_root / relative_path)
        identity = {"device": info.st_dev, "inode": info.st_ino, "mode": stat.S_IMODE(info.st_mode)}
        if hashlib.sha256(raw).hexdigest() != expected_hash or record.get("byte_count") != len(raw) or record.get("identity") != identity:
            raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
        observed.append({"authority_id": authority_id, "sha256": expected_hash, "byte_count": len(raw), "identity": identity})
        if authority_id == "A3" and any(token.encode("utf-8") not in raw for token in SEMANTIC_A3):
            raise ValidationError("THEORETICAL_AMENDMENT_REQUIRED")
    return observed


def _validate_repository(manifest: dict[str, object], root: Path) -> None:
    start = manifest.get("starting_repository")
    if not isinstance(start, dict) or start.get("head") != STARTING_HEAD or start.get("worktree_clean") is not True:
        raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
    trees = manifest.get("git_trees")
    if trees != TREE_BINDINGS or _git(root, "cat-file", "-e", f"{STARTING_HEAD}^{{commit}}") != "":
        raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
    for path, expected in TREE_BINDINGS.items():
        if _git(root, "rev-parse", f"{STARTING_HEAD}:{path}") != expected or _git(root, "rev-parse", f"HEAD:{path}") != expected:
            raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
    sealed = manifest.get("sealed_v1_evidence")
    if not isinstance(sealed, dict) or sealed.get("git_tree") != TREE_BINDINGS[".sisyphus/evidence/phase12-filter-v5-build-v1"]:
        raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
    files = sealed.get("files")
    if not isinstance(files, dict) or set(files) != {path.name for path in (root / ".sisyphus/evidence/phase12-filter-v5-build-v1").glob("*.json")}:
        raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
    for name, expected_hash in files.items():
        if not isinstance(name, str) or not isinstance(expected_hash, str):
            raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
        raw, _ = _read_nofollow(root / ".sisyphus/evidence/phase12-filter-v5-build-v1" / name)
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")


def _validate_source_universe(manifest: dict[str, object], root: Path, authority_root: Path) -> None:
    binding = manifest.get("source_universe")
    if not isinstance(binding, dict) or binding.get("path") != "data/phase12/filter_v5_bct_v1/source_universe_v1.json":
        raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
    raw, _ = _read_nofollow(root / binding["path"])
    if binding.get("sha256") != hashlib.sha256(raw).hexdigest():
        raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
    universe = _json(root / binding["path"])
    files, spans = universe.get("source_files"), universe.get("worked_spans")
    if not isinstance(files, dict) or len(files) != 17 or not isinstance(spans, list) or len(spans) != 3:
        raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
    for path, expected_hash in files.items():
        if not isinstance(path, str) or not isinstance(expected_hash, str):
            raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
        source, _ = _read_nofollow(root / path)
        if hashlib.sha256(source).hexdigest() != expected_hash:
            raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
    records = {item[0]: item[1] for item in AUTHORITIES}
    for span in spans:
        if not isinstance(span, dict) or span.get("authority_id") not in records:
            raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
        source, _ = _read_nofollow(authority_root / records[span["authority_id"]])
        start, end = span.get("start_line"), span.get("end_line")
        if not isinstance(start, int) or not isinstance(end, int) or hashlib.sha256(b"".join(source.splitlines(keepends=True)[start - 1:end])).hexdigest() != span.get("sha256"):
            raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")


def validate(manifest_path: Path, output: Path, authority_root: Path, repository_root: Path) -> dict[str, object]:
    manifest = _json(manifest_path)
    digest = _plan_digest(repository_root)
    if manifest.get("schema_version") != "phase12_fv5_authority_transition_manifest_v1" or manifest.get("approved_plan_sha256") != digest or manifest.get("external_files_changed") is not False:
        raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
    _validate_repository(manifest, repository_root)
    observed = _validate_authorities(manifest, authority_root)
    _validate_source_universe(manifest, repository_root, authority_root)
    bindings = manifest.get("consumer_bindings")
    if not isinstance(bindings, dict) or bindings.get("historical") != ["pilot_a_v1", "filter_v5_build_v1"] or bindings.get("new") != ["filter_v5_bct_calibration_v1", "exploratory_code_source_fidelity_v2"]:
        raise ValidationError("THEORETICAL_AUTHORITY_DRIFT")
    report = {"approved_plan_sha256": digest, "external_authorities": observed, "external_files_changed": False, "provider_calls_issued": 0, "schema_version": "phase12_fv5_authority_snapshot_validation_v1"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authority-root", type=Path, default=AUTHORITY_ROOT)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    try:
        validate(arguments.manifest, arguments.output, arguments.authority_root, arguments.repository_root)
    except ValidationError as error:
        print(error.code)
        return 2
    print("AUTHORITY_SNAPSHOT_VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
