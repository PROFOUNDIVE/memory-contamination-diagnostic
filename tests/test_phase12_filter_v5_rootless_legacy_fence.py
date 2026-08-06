from __future__ import annotations

import importlib.util
import ast
import json
import os
import stat
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import TypedDict

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRUSTED_INPUT_ROOT = ROOT.parent / "memory-contamination-diagnostic-filter-v5"
IGNORED_INPUT_ROOT = ROOT
MATERIALIZER = ROOT / "scripts" / "materialize_phase12_filter_v5_rootless_inputs.py"
SETUP = ROOT / "scripts" / "setup_phase12_filter_v5_rootless_t1_inputs.py"
GIT_VALIDATOR = ROOT / "scripts" / "validate_phase12_filter_v5_rootless_git_context.py"
QA_WRITER = ROOT / "scripts" / "write_phase12_filter_v5_rootless_task_qa.py"
MANIFEST = ROOT / "configs" / "phase12" / "filter_v5_rootless_local" / "external_inputs.json"
MANIFEST_DESCRIPTOR = (
    ROOT / "docs" / "evidence" / "phase12-filter-v5-rootless-local" / "legacy-input-manifest.sha256"
)
PUBLICATION = "docs/evidence/phase12-filter-v5-rootless-local/rehearsal-publication.json"
INPUTS = (
    (
        "ROOTLESS_HISTORICAL_SCREENING_PLAN",
        ".omo/plans/phase12-filter-v5-screening-bct-execution.md",
        144691,
        "9270d31770eb97e732602cfe85a250111208afeae293b0a20ab618baadb43317",
    ),
    (
        "ROOTLESS_HISTORICAL_SCREENING_DESCRIPTOR",
        ".omo/approvals/phase12-filter-v5-screening-bct-execution.plan.sha256",
        65,
        "92c6d30f026a10f47067e5467c0e9e0abc35b653385f4f08ad7d301838e06160",
    ),
    (
        "ROOTLESS_HISTORICAL_POST_DESCRIPTOR",
        ".omo/approvals/phase12-post-filter-v5-calibration-readiness.plan.sha256",
        65,
        "7b878988972b5bc3c1a2ba24785b978cc26b973e1e44e8059ff8d3133227842e",
    ),
)
SETUP_INPUTS = {
    ".omo/plans/phase12-filter-v5-screening-bct-execution.md": (
        144691,
        "9270d31770eb97e732602cfe85a250111208afeae293b0a20ab618baadb43317",
    ),
    ".omo/approvals/phase12-filter-v5-screening-bct-execution.plan.sha256": (
        65,
        "92c6d30f026a10f47067e5467c0e9e0abc35b653385f4f08ad7d301838e06160",
    ),
    ".omo/plans/phase12-post-filter-v5-calibration-readiness.md": (
        95737,
        "d7109bffe61d5a82ccbd5300e0cca0da9d4411b681ff3358c702024c4074879d",
    ),
    ".omo/approvals/phase12-post-filter-v5-calibration-readiness.plan.sha256": (
        65,
        "7b878988972b5bc3c1a2ba24785b978cc26b973e1e44e8059ff8d3133227842e",
    ),
    ".omo/evidence/phase12-post-filter-v5-calibration-readiness/task-3-screening-stage-result.json": (
        246,
        "583d1bd5a579af84b00ded45e67b66f491940237c4e708027d9da827b4bbb8f7",
    ),
    ".omo/evidence/phase12-post-filter-v5-calibration-readiness/task-5-bct-stage-result.json": (
        240,
        "3d7b04540abb253583c345ba66a15163468e69ca5bdfc50ccdb4b68fa99d6792",
    ),
    ".omo/evidence/phase12-post-filter-v5-calibration-readiness/task-6-pilot-b-readiness-stage-result.json": (
        254,
        "f595fc0a17e330e387e96b7506b65bd2631285e3506ec68afcef8a9294261fbe",
    ),
}


def _canonical_json(value: dict[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _manifest_bytes() -> bytes:
    return _canonical_json(
        {
            "schema_version": "rootless_external_input_manifest_v1",
            "profile": "local_rootless_non_authoritative",
            "kind": "external_input_manifest",
            "ordered_inputs": [
                {
                    "role": role,
                    "repo_relative_destination": destination,
                    "size_bytes": size,
                    "sha256": digest,
                }
                for role, destination, size, digest in INPUTS
            ],
        }
    )


def _load_materializer() -> ModuleType:
    assert MATERIALIZER.is_file(), "T1 materializer is required"
    specification = importlib.util.spec_from_file_location("rootless_legacy_fence", MATERIALIZER)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _load_script(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, check=False, text=True)


def _make_clean_fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / "fixture-repository"
    cloned = _run(["git", "clone", "--no-local", "--quiet", str(ROOT), str(fixture)], ROOT)
    assert cloned.returncode == 0, cloned.stderr
    os.chmod(fixture, 0o755)
    os.chmod(fixture / ".git", 0o700)
    os.chmod(fixture / ".git" / "info", 0o700)
    os.chmod(fixture / ".git" / "config", 0o600)
    os.chmod(fixture / ".git" / "info" / "exclude", 0o600)
    manifest_path = fixture / MANIFEST.relative_to(ROOT)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(_manifest_bytes())
    descriptor_path = fixture / MANIFEST_DESCRIPTOR.relative_to(ROOT)
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_path.write_bytes(
        f"{sha256(_manifest_bytes()).hexdigest()}  configs/phase12/filter_v5_rootless_local/external_inputs.json\n".encode(
            "ascii"
        )
    )
    ignore_path = fixture / ".gitignore"
    ignore_path.write_bytes((ROOT / ".gitignore").read_bytes())
    os.chmod(fixture / ".omo", 0o700)
    os.chmod(fixture / ".omo" / "plans", 0o700)
    committed = _run(
        [
            "git",
            "-c",
            "user.name=T1 Test",
            "-c",
            "user.email=t1@example.invalid",
            "add",
            str(manifest_path.relative_to(fixture)),
            str(descriptor_path.relative_to(fixture)),
            ".gitignore",
        ],
        fixture,
    )
    assert committed.returncode in {0, 1}, committed.stderr
    clean = _run(["git", "status", "--porcelain=v1"], fixture)
    assert clean.returncode == 0 and clean.stdout == "", clean.stderr
    return fixture


def _external_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "external-inputs"
    source_paths: list[Path] = []
    for _, destination, _, _ in INPUTS:
        source = IGNORED_INPUT_ROOT / destination
        copied = source_root / source.name
        copied.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw = _read_test_source(source)
        except OSError as error:
            raise RuntimeError("ROOTLESS_T1_INPUT_SOURCE_MISSING") from error
        descriptor = os.open(
            copied,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(descriptor, raw[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        source_paths.append(copied)
    assert len(source_paths) == 3
    return source_paths[0], source_paths[1], source_paths[2]


def _materialize(fixture: Path, sources: tuple[Path, Path, Path]) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            sys.executable,
            "-B",
            str(MATERIALIZER),
            "--repo-root",
            str(fixture),
            "--historical-screening-plan",
            str(sources[0]),
            "--historical-screening-descriptor",
            str(sources[1]),
            "--historical-post-descriptor",
            str(sources[2]),
        ],
        ROOT,
    )


def _run_validator(repository: Path) -> subprocess.CompletedProcess[str]:
    return _run(
        [sys.executable, "-B", "-I", "-S", str(GIT_VALIDATOR), "--repo-root", str(repository)],
        ROOT,
    )


def _make_git_fixture(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(tmp_path, 0o700)
    fixture = tmp_path / "git-context-fixture"
    cloned = _run(["git", "clone", "--no-local", "--quiet", str(ROOT), str(fixture)], ROOT)
    assert cloned.returncode == 0, cloned.stderr
    os.chmod(fixture, 0o755)
    os.chmod(fixture / ".git", 0o700)
    os.chmod(fixture / ".git" / "config", 0o600)
    os.chmod(fixture / ".git" / "info", 0o700)
    os.chmod(fixture / ".git" / "info" / "exclude", 0o600)
    return fixture


def _review_fixture(tmp_path: Path) -> tuple[ModuleType, Path, Path, Path, dict[str, object]]:
    module = _load_materializer()
    plan = tmp_path / "plan.md"
    descriptor = tmp_path / "plan.sha256"
    metadata = tmp_path / "review.json"
    plan.write_bytes(b"rootless plan\n")
    os.chmod(plan, 0o600)
    plan_hash = sha256(b"rootless plan\n").hexdigest()
    descriptor.write_bytes(f"{plan_hash}  phase12-filter-v5-rootless-local-execution.md\n".encode())
    os.chmod(descriptor, 0o600)
    payload: dict[str, object] = {
        "schema_version": "rootless_operator_asserted_review_metadata_v1",
        "profile": "local_rootless_non_authoritative",
        "kind": "operator_asserted_dual_review",
        "plan_sha256": plan_hash,
        "round_id": f"phase12-filter-v5-rootless-local-execution-r1-{plan_hash[:8]}",
        "momus_launch_id": f"momus-r1-{plan_hash[:8]}",
        "momus_session_id": "momus-session",
        "momus_verdict": "OKAY",
        "oracle_launch_id": f"oracle-r1-{plan_hash[:8]}",
        "oracle_session_id": "oracle-session",
        "oracle_verdict": "OKAY",
        "created_at": "2026-08-06T00:00:00Z",
    }
    metadata.write_bytes(_canonical_json(payload))
    os.chmod(metadata, 0o600)
    return module, plan, descriptor, metadata, payload


def test_manifest_and_descriptor_have_exact_canonical_bytes() -> None:
    # Given: the three T1 historical input pins.
    expected_manifest = _manifest_bytes()

    # When: the tracked manifest and descriptor are read.
    # Then: canonical bytes and the descriptor hash bind the same manifest bytes.
    assert MANIFEST.read_bytes() == expected_manifest
    assert MANIFEST_DESCRIPTOR.read_bytes() == (
        f"{sha256(expected_manifest).hexdigest()}  configs/phase12/filter_v5_rootless_local/external_inputs.json\n"
    ).encode("ascii")


def test_materializer_creates_only_verified_destinations_once(tmp_path: Path) -> None:
    # Given: a clean descendant of the trusted base and detached 0600 historical sources.
    fixture = _make_clean_fixture(tmp_path)
    sources = _external_sources(tmp_path)

    # When: the one-time materializer runs.
    first = _materialize(fixture, sources)

    # Then: it writes exact private regular files, and a second invocation cannot mutate them.
    assert first.returncode == 0, first.stdout + first.stderr
    destinations = tuple(fixture / destination for _, destination, _, _ in INPUTS)
    before = tuple(destination.read_bytes() for destination in destinations)
    for destination, expected, (_, _, size, digest) in zip(destinations, before, INPUTS, strict=True):
        info = destination.stat()
        assert stat.S_IMODE(info.st_mode) == 0o600
        assert info.st_uid == os.getuid()
        assert info.st_nlink == 1
        assert len(expected) == size
        assert sha256(expected).hexdigest() == digest
    second = _materialize(fixture, sources)
    assert second.returncode != 0
    assert tuple(destination.read_bytes() for destination in destinations) == before


@pytest.mark.parametrize(
    "relative_argument",
    (
        "repo-root",
        "historical-screening-plan",
        "historical-screening-descriptor",
        "historical-post-descriptor",
    ),
)
def test_materializer_cli_rejects_every_relative_input_path(
    tmp_path: Path, relative_argument: str
) -> None:
    fixture = _make_clean_fixture(tmp_path)
    sources = _external_sources(tmp_path)
    arguments = {
        "repo-root": fixture,
        "historical-screening-plan": sources[0],
        "historical-screening-descriptor": sources[1],
        "historical-post-descriptor": sources[2],
    }
    arguments[relative_argument] = Path(os.path.relpath(arguments[relative_argument], ROOT))

    result = _run(
        [
            sys.executable,
            "-B",
            str(MATERIALIZER),
            "--repo-root",
            str(arguments["repo-root"]),
            "--historical-screening-plan",
            str(arguments["historical-screening-plan"]),
            "--historical-screening-descriptor",
            str(arguments["historical-screening-descriptor"]),
            "--historical-post-descriptor",
            str(arguments["historical-post-descriptor"]),
        ],
        ROOT,
    )

    assert result.returncode == 64
    assert result.stderr.strip() == "ROOTLESS_LEGACY_PATH_INVALID"


@pytest.mark.parametrize("relative_argument", ("plan", "descriptor", "metadata"))
def test_reviewed_plan_api_rejects_every_relative_input_path(
    tmp_path: Path, relative_argument: str
) -> None:
    module, plan, descriptor, metadata, _ = _review_fixture(tmp_path)
    arguments = {"plan": plan, "descriptor": descriptor, "metadata": metadata}
    arguments[relative_argument] = Path(os.path.relpath(arguments[relative_argument], ROOT))

    with pytest.raises(module.LegacyFenceError, match="ROOTLESS_LEGACY_PATH_INVALID"):
        module.validate_reviewed_plan(
            arguments["plan"], arguments["descriptor"], arguments["metadata"]
        )


def test_materializer_rejects_noncanonical_absolute_path() -> None:
    module = _load_materializer()

    with pytest.raises(module.LegacyFenceError, match="ROOTLESS_LEGACY_PATH_INVALID"):
        module._absolute(Path("/tmp/rootless-segment/../rootless-input"))


def test_materializer_rejects_unsafe_source_and_base_blob_drift_before_writes(tmp_path: Path) -> None:
    # Given: a clean fixture with an unsafe external source.
    fixture = _make_clean_fixture(tmp_path)
    sources = _external_sources(tmp_path)
    os.chmod(sources[0], 0o644)

    # When: the materializer checks the unsafe source.
    unsafe = _materialize(fixture, sources)

    # Then: it rejects before writing its fixed destination.
    assert unsafe.returncode != 0
    assert not (fixture / INPUTS[0][1]).exists()

    # Given: safe sources and a clean descendant that mutates a protected tracked byte.
    os.chmod(sources[0], 0o600)
    protected = fixture / "configs/phase12/filter_v5_bct_calibration.yaml"
    protected.write_bytes(protected.read_bytes() + b"# drift\n")
    committed = _run(
        [
            "git",
            "-c",
            "user.name=T1 Test",
            "-c",
            "user.email=t1@example.invalid",
            "add",
            str(protected.relative_to(fixture)),
        ],
        fixture,
    )
    assert committed.returncode == 0, committed.stderr
    committed = _run(
        [
            "git",
            "-c",
            "user.name=T1 Test",
            "-c",
            "user.email=t1@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "mutate protected input",
        ],
        fixture,
    )
    assert committed.returncode == 0, committed.stderr

    # When: the historical-base fence runs before another materialization.
    drift = _materialize(fixture, sources)

    # Then: a descendant may exist, but protected bytes must still equal the pinned base blobs.
    assert drift.returncode != 0
    assert not (fixture / INPUTS[0][1]).exists()


def test_review_metadata_requires_closed_grammar_but_discloses_same_uid_forgeability(
    tmp_path: Path,
) -> None:
    # Given: a descriptor-bound plan and an operator-asserted dual-review record.
    module = _load_materializer()
    plan = tmp_path / "plan.md"
    descriptor = tmp_path / "plan.sha256"
    metadata = tmp_path / "review.json"
    plan.write_bytes(b"rootless plan\n")
    os.chmod(plan, 0o600)
    plan_hash = sha256(plan.read_bytes()).hexdigest()
    descriptor.write_bytes(f"{plan_hash}  phase12-filter-v5-rootless-local-execution.md\n".encode("ascii"))
    os.chmod(descriptor, 0o600)
    metadata.write_bytes(
        _canonical_json(
            {
                "schema_version": "rootless_operator_asserted_review_metadata_v1",
                "profile": "local_rootless_non_authoritative",
                "kind": "operator_asserted_dual_review",
                "plan_sha256": plan_hash,
                "round_id": f"phase12-filter-v5-rootless-local-execution-r1-{plan_hash[:8]}",
                "momus_launch_id": f"momus-r1-{plan_hash[:8]}",
                "momus_session_id": "momus-session",
                "momus_verdict": "OKAY",
                "oracle_launch_id": f"oracle-r1-{plan_hash[:8]}",
                "oracle_session_id": "oracle-session",
                "oracle_verdict": "OKAY",
                "created_at": "2026-08-06T00:00:00Z",
            }
        )
    )
    os.chmod(metadata, 0o600)

    # When: the review metadata is validated.
    validated = module.validate_reviewed_plan(plan, descriptor, metadata)

    # Then: only the matching closed grammar is accepted, without treating it as trusted authority.
    assert validated == plan_hash
    assert "same-UID operators can forge" in module.REVIEW_METADATA_FORGEABILITY
    metadata.write_bytes(metadata.read_bytes().replace(b"momus-r1-", b"momus-r2-", 1))
    with pytest.raises(module.LegacyFenceError):
        module.validate_reviewed_plan(plan, descriptor, metadata)


def test_manifest_and_gitignore_reject_drift_without_hiding_rootless_evidence(tmp_path: Path) -> None:
    # Given: a materialized fixture and its exact manifest descriptor.
    module = _load_materializer()
    fixture = _make_clean_fixture(tmp_path)
    sources = _external_sources(tmp_path)
    materialized = _materialize(fixture, sources)
    assert materialized.returncode == 0, materialized.stdout + materialized.stderr

    # When: the fence checks materialized bytes and the final-index manifest target.
    manifest_hash = module.validate_legacy_fence(fixture)

    # Then: the manifest hash is accepted, descriptor substitution and destination mutation are rejected.
    assert manifest_hash == sha256(_manifest_bytes()).hexdigest()
    with pytest.raises(module.LegacyFenceError):
        module.validate_final_index_legacy_manifest_sha256(
            sha256((fixture / MANIFEST_DESCRIPTOR.relative_to(ROOT)).read_bytes()).hexdigest(), fixture
        )
    destination = fixture / INPUTS[1][1]
    destination.write_bytes(destination.read_bytes().replace(b"9", b"8", 1))
    with pytest.raises(module.LegacyFenceError):
        module.validate_legacy_fence(fixture)

    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".omo/" in ignored
    assert ".env" in ignored
    assert "runs/*" in ignored and "!runs/.gitkeep" in ignored
    assert ignored.count(f"/{PUBLICATION}") == 1
    assert _run(["git", "check-ignore", "-q", "--no-index", "--", PUBLICATION], ROOT).returncode == 0
    for visible_path in (
        "docs/evidence/phase12-filter-v5-rootless-local/final-verification-index.json",
        "docs/evidence/phase12-filter-v5-rootless-local/claim-boundary.md",
    ):
        assert _run(["git", "check-ignore", "-q", "--no-index", "--", visible_path], ROOT).returncode == 1


def test_materializer_rejects_dirty_worktree_and_symlinked_materialization(tmp_path: Path) -> None:
    # Given: a clean fixture with detached historical inputs.
    module = _load_materializer()
    fixture = _make_clean_fixture(tmp_path)
    sources = _external_sources(tmp_path)
    materialized = _materialize(fixture, sources)
    assert materialized.returncode == 0, materialized.stdout + materialized.stderr
    (fixture / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    # When: a dirty worktree reaches the later authority fence.
    with pytest.raises(module.LegacyFenceError):
        module.validate_legacy_fence(fixture)

    # Then: it rejects before an authority can consume materialized bytes.
    (fixture / "untracked.txt").unlink()
    destination = fixture / INPUTS[0][1]
    destination.unlink()
    destination.symlink_to(sources[0])
    with pytest.raises(module.LegacyFenceError):
        module.validate_legacy_fence(fixture)


def test_task_qa_writer_binds_only_the_fixed_t1_command_and_assertions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a private T1 QA root and the exact completed pytest command record.
    writer_path = ROOT / "scripts" / "write_phase12_filter_v5_rootless_task_qa.py"
    specification = importlib.util.spec_from_file_location("rootless_task_qa", writer_path)
    assert specification is not None and specification.loader is not None
    writer = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = writer
    specification.loader.exec_module(writer)
    monkeypatch.setattr(writer, "ROOT", tmp_path)
    destination = tmp_path / "runs" / "phase12-filter-v5-rootless-qa" / "t1-legacy-fence.json"
    destination.parent.mkdir(parents=True, mode=0o700)
    argv = (
        os.path.realpath(sys.executable),
        "-B",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(tmp_path / "runs" / "phase12-filter-v5-rootless-qa" / "basetemp" / "t1" / "pytest"),
        "tests/test_phase12_filter_v5_rootless_legacy_fence.py",
        "tests/test_phase12_filter_v5_plan_digest.py",
        "tests/test_phase12_filter_v5_evidence_security.py",
        "-q",
    )
    command = writer.CommandResult(argv, 0, b"5 passed\n", b"", 0, 0)
    writer._VERIFIED_RESULTS[id(command)] = command

    # When: the writer seals the fixed command and ordered assertions.
    writer.write_rootless_task_qa("t1", command, writer.ROLE_ASSERTIONS["t1"], destination)

    # Then: canonical evidence is private, complete, and cannot be overwritten.
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["role"] == "t1"
    assert [item["assertion_id"] for item in payload["ordered_assertions"]] == list(
        writer.ROLE_ASSERTIONS["t1"]
    )
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    with pytest.raises(ValueError):
        writer.write_rootless_task_qa("t1", command, writer.ROLE_ASSERTIONS["t1"], destination)


def test_external_source_boundary_rejects_happy_materialization_when_detached_inputs_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a fresh clone without the optional detached historical-input checkout.
    monkeypatch.setattr(sys.modules[__name__], "IGNORED_INPUT_ROOT", tmp_path / "absent-checkout")

    # When: the happy-path fixture asks for detached bytes.
    # Then: the missing detached authority is a nonzero setup failure, never a skip.
    with pytest.raises(RuntimeError, match="ROOTLESS_T1_INPUT_SOURCE_MISSING"):
        _external_sources(tmp_path)


def test_git_wrapper_sets_closed_t1_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a materializer with its Git subprocess replaced by an observable fake.
    module = _load_materializer()
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if command[0] == "/usr/bin/git":
            observed["command"] = command
            observed["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    # When: the trusted-base wrapper runs a read-only Git operation.
    module._git(ROOT, "status")

    # Then: closed environment and config controls cannot inherit local Git behavior.
    environment = observed["environment"]
    command = observed["command"]
    assert isinstance(environment, dict) and environment["GIT_ATTR_NOSYSTEM"] == "1"
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"
    assert isinstance(command, list)
    for control in (
        "core.fileMode=true",
        "core.ignoreCase=false",
        "core.precomposeUnicode=false",
        "core.excludesFile=/dev/null",
        "core.attributesFile=/dev/null",
        "core.bare=false",
        "status.relativePaths=false",
        "submodule.recurse=false",
        "diff.ignoreSubmodules=none",
    ):
        assert control in command


def test_safe_directory_rejects_group_or_other_writable_mode() -> None:
    # Given: directory metadata whose owner is trusted but mode permits group writes.
    module = _load_materializer()
    unsafe = os.stat_result((stat.S_IFDIR | 0o775, 0, 0, 0, os.getuid(), 0, 0, 0, 0, 0))

    # When: the ancestor validation receives that metadata.
    # Then: the global safe-chain rule rejects it.
    with pytest.raises(module.LegacyFenceError):
        module._safe_directory(unsafe, os.getuid())


def test_partial_materialization_temp_rejects_without_destination_overwrite(tmp_path: Path) -> None:
    # Given: an interrupted fixed temporary write with no published destination.
    module = _load_materializer()
    directory = tmp_path / "destination"
    directory.mkdir(mode=0o700)
    temporary = directory / ".input.tmp"
    temporary.write_bytes(b"partial")
    os.chmod(temporary, 0o600)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    # When: materialization attempts to publish the same fixed destination.
    try:
        with pytest.raises(module.LegacyFenceError):
            module._write_once(descriptor, "input", b"complete")
    finally:
        os.close(descriptor)

    # Then: it cannot claim success or overwrite a destination after interruption.
    assert not (directory / "input").exists()
    assert temporary.read_bytes() == b"partial"


def test_materializer_fsyncs_parent_after_mkdir_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_materializer()
    parent_path = tmp_path / "materializer-parent"
    parent_path.mkdir(mode=0o700)
    parent = os.open(parent_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    events: list[str] = []
    real_mkdir = module.os.mkdir
    real_fsync = module.os.fsync
    real_open = module.os.open

    def recording_mkdir(path: str, mode: int, *, dir_fd: int) -> None:
        events.append(f"mkdir:{path}")
        real_mkdir(path, mode, dir_fd=dir_fd)

    def recording_fsync(descriptor: int) -> None:
        events.append("fsync:parent")
        real_fsync(descriptor)

    def recording_open(path: str, flags: int, *args: object, **kwargs: object) -> int:
        events.append(f"open:{path}")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "mkdir", recording_mkdir)
    monkeypatch.setattr(module.os, "fsync", recording_fsync)
    monkeypatch.setattr(module.os, "open", recording_open)
    try:
        child = module._open_or_create_directory(parent, "child")
        os.close(child)
    finally:
        os.close(parent)

    assert events == ["mkdir:child", "fsync:parent", "open:child"]


def test_real_read_rejects_group_writable_ancestor(tmp_path: Path) -> None:
    module = _load_materializer()
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o700)
    source = unsafe / "source"
    source.write_bytes(b"source")
    os.chmod(source, 0o600)
    os.chmod(unsafe, 0o770)

    with pytest.raises(module.LegacyFenceError, match="ROOTLESS_LEGACY_PATH_UNSAFE"):
        module._read_regular(source, 0o600, True)


@pytest.mark.parametrize(
    ("metadata_change", "descriptor_change"),
    (
        ({"plan_sha256": "0" * 64}, None),
        ({"momus_session_id": "x" * 257}, None),
        ({"oracle_session_id": "e\u0301"}, None),
        ({"round_id": "phase12-filter-v5-rootless-local-execution-r01-deadbeef"}, None),
        ({"momus_launch_id": "momus-r2-deadbeef"}, None),
        ({}, b"0" * 64 + b"  phase12-filter-v5-rootless-local-execution.md\n"),
        ({}, b"0" * 64 + b" phase12-filter-v5-rootless-local-execution.md\n"),
        ({}, b"0" * 64 + b"  wrong.md\n"),
        ({}, b"0" * 64 + b"  phase12-filter-v5-rootless-local-execution.md"),
    ),
)
def test_review_metadata_rejects_every_closed_grammar_drift(
    tmp_path: Path,
    metadata_change: dict[str, object],
    descriptor_change: bytes | None,
) -> None:
    module, plan, descriptor, metadata, payload = _review_fixture(tmp_path)
    payload.update(metadata_change)
    metadata.write_bytes(_canonical_json(payload))
    if descriptor_change is not None:
        descriptor.write_bytes(descriptor_change)

    with pytest.raises(module.LegacyFenceError):
        module.validate_reviewed_plan(plan, descriptor, metadata)


@pytest.mark.parametrize("mutation", ("missing", "extra", "reordered", "alternate"))
def test_manifest_rejects_record_set_and_order_drift(tmp_path: Path, mutation: str) -> None:
    module = _load_materializer()
    fixture = tmp_path / "manifest-fixture"
    manifest_path = fixture / MANIFEST.relative_to(ROOT)
    descriptor_path = fixture / MANIFEST_DESCRIPTOR.relative_to(ROOT)
    manifest_path.parent.mkdir(parents=True)
    descriptor_path.parent.mkdir(parents=True)
    payload = json.loads(_manifest_bytes())
    records = payload["ordered_inputs"]
    assert isinstance(records, list)
    if mutation == "missing":
        records.pop()
    elif mutation == "extra":
        records.append(records[-1])
    elif mutation == "reordered":
        records[0], records[1] = records[1], records[0]
    else:
        records[0]["repo_relative_destination"] = ".omo/plans/alternate.md"
    raw = _canonical_json(payload)
    manifest_path.write_bytes(raw)
    descriptor_path.write_bytes(
        f"{sha256(raw).hexdigest()}  configs/phase12/filter_v5_rootless_local/external_inputs.json\n".encode()
    )

    with pytest.raises(module.LegacyFenceError, match="ROOTLESS_LEGACY_MANIFEST_INVALID"):
        module._verify_manifest(fixture)


@pytest.mark.parametrize(
    "descriptor",
    (
        b"0" * 64 + b" configs/phase12/filter_v5_rootless_local/external_inputs.json\n",
        b"0" * 64 + b"  alternate.json\n",
        b"0" * 64 + b"  configs/phase12/filter_v5_rootless_local/external_inputs.json",
        b"F" * 64 + b"  configs/phase12/filter_v5_rootless_local/external_inputs.json\n",
    ),
)
def test_manifest_descriptor_rejects_filename_spacing_lf_and_hash_drift(
    tmp_path: Path, descriptor: bytes
) -> None:
    module = _load_materializer()
    fixture = tmp_path / "descriptor-fixture"
    manifest_path = fixture / MANIFEST.relative_to(ROOT)
    descriptor_path = fixture / MANIFEST_DESCRIPTOR.relative_to(ROOT)
    manifest_path.parent.mkdir(parents=True)
    descriptor_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(_manifest_bytes())
    descriptor_path.write_bytes(descriptor)

    with pytest.raises(module.LegacyFenceError, match="ROOTLESS_LEGACY_MANIFEST_DESCRIPTOR_INVALID"):
        module._verify_manifest(fixture)


def test_materializer_preflights_every_destination_before_first_write(tmp_path: Path) -> None:
    fixture = _make_clean_fixture(tmp_path)
    sources = _external_sources(tmp_path)
    approvals = fixture / ".omo" / "approvals"
    approvals.mkdir(mode=0o700)
    occupied = approvals / Path(INPUTS[1][1]).name
    occupied.write_bytes(b"occupied")
    os.chmod(occupied, 0o600)

    result = _materialize(fixture, sources)

    assert result.returncode != 0
    assert not (fixture / INPUTS[0][1]).exists()
    assert occupied.read_bytes() == b"occupied"


@pytest.mark.parametrize("attack", ("missing", "symlink", "hardlink", "mode", "mismatch"))
def test_materializer_rejects_every_unsafe_source_class_before_writes(
    tmp_path: Path, attack: str
) -> None:
    fixture = _make_clean_fixture(tmp_path)
    sources = list(_external_sources(tmp_path))
    attacked = sources[0]
    if attack == "missing":
        attacked.unlink()
    elif attack == "symlink":
        target = attacked.with_name("symlink-target")
        target.write_bytes(_read_test_source(IGNORED_INPUT_ROOT / INPUTS[0][1]))
        os.chmod(target, 0o600)
        attacked.unlink()
        attacked.symlink_to(target)
    elif attack == "hardlink":
        os.link(attacked, attacked.with_name("second-link"))
    elif attack == "mode":
        os.chmod(attacked, 0o640)
    else:
        raw = attacked.read_bytes()
        attacked.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        os.chmod(attacked, 0o600)

    result = _materialize(fixture, (sources[0], sources[1], sources[2]))

    assert result.returncode != 0
    assert not (fixture / INPUTS[0][1]).exists()


def test_materializer_rejects_non_current_uid_source() -> None:
    module = _load_materializer()
    root_owned = Path("/etc/passwd")
    assert os.lstat(root_owned).st_uid == 0

    with pytest.raises(module.LegacyFenceError, match="ROOTLESS_LEGACY_FILE_UNSAFE"):
        module._read_regular(root_owned, None, True)


@pytest.mark.parametrize("attack", ("hardlink", "mode"))
def test_legacy_fence_rejects_unsafe_materialized_destination(
    tmp_path: Path, attack: str
) -> None:
    module = _load_materializer()
    fixture = _make_clean_fixture(tmp_path)
    sources = _external_sources(tmp_path)
    result = _materialize(fixture, sources)
    assert result.returncode == 0, result.stdout + result.stderr
    destination = fixture / INPUTS[0][1]
    if attack == "hardlink":
        os.link(destination, destination.with_name("second-link"))
    else:
        os.chmod(destination, 0o640)

    with pytest.raises(module.LegacyFenceError):
        module.validate_legacy_fence(fixture)


def test_setup_uses_only_descriptor_reads_for_authority_paths() -> None:
    tree = ast.parse(SETUP.read_text(encoding="utf-8"))
    forbidden = {
        "resolve",
        "exists",
        "is_dir",
        "is_file",
        "read_bytes",
        "copyfile",
        "chmod",
        "stat",
    }
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert calls.isdisjoint(forbidden)


def test_setup_reads_open_descriptor_when_source_name_is_swapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _load_script(SETUP, "rootless_t1_setup_descriptor_race")
    _, source = _setup_roots(tmp_path)
    pin = setup.INPUT_PINS[0]
    attacked = source / pin.relative_path
    original = attacked.read_bytes()
    root_descriptor = setup._open_root(source)
    real_open = setup.os.open
    swapped = False

    def swapping_open(path: str, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == pin.relative_path.name and not swapped:
            swapped = True
            attacked.rename(attacked.with_name("opened-original"))
            attacked.symlink_to("opened-original")
        return descriptor

    monkeypatch.setattr(setup.os, "open", swapping_open)
    try:
        observed = setup._read_relative(root_descriptor, pin.relative_path)
    finally:
        os.close(root_descriptor)

    assert observed == original


def test_setup_has_closed_pins_for_every_fixed_input() -> None:
    setup = _load_script(SETUP, "rootless_t1_setup")
    observed = {str(pin.relative_path): (pin.size_bytes, pin.sha256) for pin in setup.INPUT_PINS}
    assert observed == SETUP_INPUTS


def test_setup_rejects_unsafe_root_without_chmod_side_effect(tmp_path: Path) -> None:
    setup = _load_script(SETUP, "rootless_t1_setup_unsafe_root")
    repository = tmp_path / "repository"
    source = tmp_path / "source"
    repository.mkdir(mode=0o700)
    source.mkdir(mode=0o700)
    os.chmod(repository, 0o770)

    with pytest.raises(setup.SetupError):
        setup.setup_inputs(repository, source)

    assert stat.S_IMODE(os.lstat(repository).st_mode) == 0o770


def _read_test_source(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1_048_576):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _setup_roots(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "setup-destination"
    source = tmp_path / "setup-source"
    repository.mkdir(mode=0o700)
    source.mkdir(mode=0o700)
    for relative in SETUP_INPUTS:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(target.parent, 0o700)
        target.write_bytes(_read_test_source(TRUSTED_INPUT_ROOT / relative))
        os.chmod(target, 0o600)
    return repository, source


def test_setup_materializes_fixed_private_inputs_and_is_idempotent(tmp_path: Path) -> None:
    setup = _load_script(SETUP, "rootless_t1_setup_success")
    repository, source = _setup_roots(tmp_path)

    setup.setup_inputs(repository, source)
    before = {relative: (repository / relative).read_bytes() for relative in SETUP_INPUTS}
    setup.setup_inputs(repository, source)

    assert set(before) == set(SETUP_INPUTS)
    for relative, raw in before.items():
        info = os.lstat(repository / relative)
        assert stat.S_IMODE(info.st_mode) == 0o600
        assert info.st_uid == os.getuid()
        assert info.st_nlink == 1
        assert len(raw) == SETUP_INPUTS[relative][0]
        assert sha256(raw).hexdigest() == SETUP_INPUTS[relative][1]


def test_setup_fsyncs_each_parent_after_mkdir_before_descending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _load_script(SETUP, "rootless_t1_setup_mkdir_fsync")
    repository = tmp_path / "setup-fsync"
    repository.mkdir(mode=0o700)
    root = os.open(repository, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    events: list[str] = []
    real_mkdir = setup.os.mkdir
    real_fsync = setup.os.fsync
    real_open = setup.os.open

    def recording_mkdir(path: str, mode: int, *, dir_fd: int) -> None:
        events.append(f"mkdir:{path}")
        real_mkdir(path, mode, dir_fd=dir_fd)

    def recording_fsync(descriptor: int) -> None:
        events.append("fsync:parent")
        real_fsync(descriptor)

    def recording_open(path: str, flags: int, *args: object, **kwargs: object) -> int:
        events.append(f"open:{path}")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(setup.os, "mkdir", recording_mkdir)
    monkeypatch.setattr(setup.os, "fsync", recording_fsync)
    monkeypatch.setattr(setup.os, "open", recording_open)
    try:
        parent = setup._destination_parent(root, Path("first/second/input"), create=True)
        assert parent is not None
        os.close(parent)
    finally:
        os.close(root)

    assert events == [
        "mkdir:first",
        "fsync:parent",
        "open:first",
        "mkdir:second",
        "fsync:parent",
        "open:second",
    ]


@pytest.mark.parametrize("attack", ("missing", "symlink", "hardlink", "mode", "mismatch", "ancestor"))
def test_setup_rejects_every_unsafe_source_before_destination_writes(
    tmp_path: Path, attack: str
) -> None:
    setup = _load_script(SETUP, f"rootless_t1_setup_source_{attack}")
    repository, source = _setup_roots(tmp_path)
    relative = next(iter(SETUP_INPUTS))
    attacked = source / relative
    if attack == "missing":
        attacked.unlink()
    elif attack == "symlink":
        attacked.unlink()
        attacked.symlink_to(TRUSTED_INPUT_ROOT / relative)
    elif attack == "hardlink":
        os.link(attacked, attacked.with_name("second-link"))
    elif attack == "mode":
        os.chmod(attacked, 0o640)
    elif attack == "mismatch":
        raw = attacked.read_bytes()
        attacked.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        os.chmod(attacked, 0o600)
    else:
        os.chmod(source, 0o770)

    with pytest.raises(setup.SetupError):
        setup.setup_inputs(repository, source)

    assert not (repository / ".omo").exists()


@pytest.mark.parametrize("attack", ("symlink", "hardlink", "mode", "mismatch"))
def test_setup_rejects_unsafe_existing_destination_without_overwrite(
    tmp_path: Path, attack: str
) -> None:
    setup = _load_script(SETUP, f"rootless_t1_setup_destination_{attack}")
    repository, source = _setup_roots(tmp_path)
    setup.setup_inputs(repository, source)
    relative = next(iter(SETUP_INPUTS))
    attacked = repository / relative
    if attack == "symlink":
        attacked.unlink()
        attacked.symlink_to(source / relative)
    elif attack == "hardlink":
        os.link(attacked, attacked.with_name("second-link"))
    elif attack == "mode":
        os.chmod(attacked, 0o640)
    else:
        raw = attacked.read_bytes()
        attacked.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        os.chmod(attacked, 0o600)

    with pytest.raises(setup.SetupError):
        setup.setup_inputs(repository, source)

    if attack == "mismatch":
        assert attacked.read_bytes() != (source / relative).read_bytes()


def test_git_validator_uses_only_descriptor_reads_for_git_authority() -> None:
    tree = ast.parse(GIT_VALIDATOR.read_text(encoding="utf-8"))
    forbidden = {"resolve", "exists", "is_dir", "is_file", "read_bytes", "iterdir", "lstat"}
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert calls.isdisjoint(forbidden)


def test_git_validator_reads_open_descriptor_when_config_name_is_swapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = _load_script(GIT_VALIDATOR, "rootless_git_descriptor_race")
    fixture = _make_git_fixture(tmp_path)
    git_directory = validator._open_absolute_directory(fixture / ".git", current_uid_only=True)
    config = fixture / ".git" / "config"
    original = config.read_bytes()
    real_open = validator.os.open
    swapped = False

    def swapping_open(path: str, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == "config" and not swapped:
            swapped = True
            config.rename(config.with_name("opened-config"))
            config.write_text('[include]\n\tpath = /tmp/hostile\n', encoding="utf-8")
            os.chmod(config, 0o600)
        return descriptor

    monkeypatch.setattr(validator.os, "open", swapping_open)
    try:
        observed = validator._required_file(git_directory, "config")
    finally:
        os.close(git_directory)

    assert observed == original


@pytest.mark.parametrize(
    "config_append",
    (
        '\n[branch "main"]\n\tvscode-merge-base = origin/main\n',
        '\n[remote "origin"]\n\tpush = refs/heads/main\n',
        '\n[include]\n\tpath = /tmp/hostile\n',
        '\n[includeIf "gitdir:/tmp/**"]\n\tpath = /tmp/hostile\n',
        '\n[core]\n\tbare = false\n',
        '\n[core]\n\tunknown = value\n',
    ),
)
def test_git_validator_rejects_forbidden_include_duplicate_and_suffix_keys(
    tmp_path: Path, config_append: str
) -> None:
    fixture = _make_git_fixture(tmp_path)
    config = fixture / ".git" / "config"
    config.write_text(config.read_text(encoding="utf-8") + config_append, encoding="utf-8")

    result = _run_validator(fixture)

    assert result.returncode == 64
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("relative", "raw"),
    (
        ("info/exclude", b"ignored-pattern\n"),
        ("info/exclude", b" # leading whitespace is a pattern\n"),
        ("info/attributes", b"*.txt text\n"),
        ("objects/info/alternates", b"/tmp/alternate\n"),
    ),
)
def test_git_validator_rejects_info_rules_attributes_and_alternates(
    tmp_path: Path, relative: str, raw: bytes
) -> None:
    fixture = _make_git_fixture(tmp_path)
    target = fixture / ".git" / relative
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    target.write_bytes(raw)
    os.chmod(target, 0o600)

    assert _run_validator(fixture).returncode == 64


def test_git_validator_rejects_replace_refs(tmp_path: Path) -> None:
    fixture = _make_git_fixture(tmp_path)
    replace = _run(["git", "replace", "HEAD", "HEAD^"], fixture)
    assert replace.returncode == 0, replace.stderr

    assert _run_validator(fixture).returncode == 64


def test_git_validator_rejects_packed_replace_refs(tmp_path: Path) -> None:
    fixture = _make_git_fixture(tmp_path)
    object_id = _run(["git", "rev-parse", "HEAD"], fixture)
    assert object_id.returncode == 0, object_id.stderr
    packed_refs = fixture / ".git" / "packed-refs"
    packed_refs.write_text(
        f"# pack-refs with: peeled fully-peeled sorted\n{object_id.stdout.strip()} refs/replace/{object_id.stdout.strip()}\n",
        encoding="ascii",
    )
    os.chmod(packed_refs, 0o600)

    assert _run_validator(fixture).returncode == 64


def test_git_validator_rejects_forbidden_worktree_config(tmp_path: Path) -> None:
    fixture = _make_git_fixture(tmp_path)
    config = fixture / ".git" / "config"
    config.write_text(
        config.read_text(encoding="utf-8") + "\n[extensions]\n\tworktreeConfig = true\n",
        encoding="utf-8",
    )
    worktree_config = fixture / ".git" / "config.worktree"
    worktree_config.write_text('[branch "main"]\n\tvscode-merge-base = origin/main\n', encoding="utf-8")
    os.chmod(worktree_config, 0o600)

    assert _run_validator(fixture).returncode == 64


@pytest.mark.parametrize("location", ("common", "git"))
def test_git_validator_rejects_common_and_git_directory_alternates(
    tmp_path: Path, location: str
) -> None:
    common = _make_git_fixture(tmp_path / "common")
    linked = tmp_path / "linked"
    added = _run(["git", "worktree", "add", "--detach", "--quiet", str(linked), "HEAD"], common)
    assert added.returncode == 0, added.stderr
    os.chmod(linked, 0o755)
    git_pointer = (linked / ".git").read_text(encoding="utf-8").removeprefix("gitdir: ").strip()
    git_directory = Path(git_pointer)
    os.chmod(git_directory, 0o700)
    target_root = common / ".git" if location == "common" else git_directory
    alternates = target_root / "objects" / "info" / "alternates"
    alternates.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(alternates.parent, 0o700)
    alternates.write_text("/tmp/alternate\n", encoding="utf-8")
    os.chmod(alternates, 0o600)

    assert _run_validator(linked).returncode == 64


def test_git_validator_rejects_git_directory_replace_refs(tmp_path: Path) -> None:
    common = _make_git_fixture(tmp_path / "common")
    linked = tmp_path / "linked"
    added = _run(["git", "worktree", "add", "--detach", "--quiet", str(linked), "HEAD"], common)
    assert added.returncode == 0, added.stderr
    git_directory = Path((linked / ".git").read_text(encoding="utf-8").removeprefix("gitdir: ").strip())
    os.chmod(git_directory, 0o700)
    replacement = git_directory / "refs" / "replace" / "attacker"
    replacement.parent.mkdir(parents=True, mode=0o700)
    os.chmod(replacement.parent, 0o700)
    replacement.write_text("attacker\n", encoding="utf-8")
    os.chmod(replacement, 0o600)

    assert _run_validator(linked).returncode == 64


def test_git_validator_rejects_worktree_common_directory_drift(tmp_path: Path) -> None:
    common = _make_git_fixture(tmp_path / "common")
    linked = tmp_path / "linked"
    added = _run(["git", "worktree", "add", "--detach", "--quiet", str(linked), "HEAD"], common)
    assert added.returncode == 0, added.stderr
    git_directory = Path((linked / ".git").read_text(encoding="utf-8").removeprefix("gitdir: ").strip())
    os.chmod(git_directory, 0o700)
    commondir = git_directory / "commondir"
    commondir.write_text("../invalid-common\n", encoding="utf-8")
    os.chmod(commondir, 0o600)

    assert _run_validator(linked).returncode == 64


@pytest.mark.parametrize("flag", ("--assume-unchanged", "--skip-worktree"))
def test_git_validator_rejects_index_flags(tmp_path: Path, flag: str) -> None:
    fixture = _make_git_fixture(tmp_path)
    changed = _run(["git", "update-index", flag, ".gitignore"], fixture)
    assert changed.returncode == 0, changed.stderr

    assert _run_validator(fixture).returncode == 64


def test_git_validator_rejects_unsafe_ancestor_and_metadata_mode(tmp_path: Path) -> None:
    ancestor = tmp_path / "unsafe-ancestor"
    ancestor.mkdir(mode=0o700)
    fixture = _make_git_fixture(ancestor)
    os.chmod(ancestor, 0o770)
    assert _run_validator(fixture).returncode == 64

    os.chmod(ancestor, 0o700)
    os.chmod(fixture / ".git" / "config", 0o660)
    assert _run_validator(fixture).returncode == 64


def test_git_validator_rejects_gitignore_blob_and_rule_drift(tmp_path: Path) -> None:
    fixture = _make_git_fixture(tmp_path)
    ignore = fixture / ".gitignore"
    ignore.write_bytes(ignore.read_bytes() + b"docs/evidence/phase12-filter-v5-rootless-local/*\n")
    assert _run_validator(fixture).returncode == 64

    committed = _run(["git", "add", ".gitignore"], fixture)
    assert committed.returncode == 0, committed.stderr
    committed = _run(
        [
            "git",
            "-c",
            "user.name=T1 Test",
            "-c",
            "user.email=t1@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "hostile ignore rule",
        ],
        fixture,
    )
    assert committed.returncode == 0, committed.stderr
    assert _run_validator(fixture).returncode == 64


def test_git_validator_accepts_publication_only_and_keeps_every_sibling_visible(tmp_path: Path) -> None:
    fixture = _make_git_fixture(tmp_path)
    publication = fixture / PUBLICATION
    publication.parent.mkdir(parents=True, exist_ok=True)
    publication.write_text("{}\n", encoding="utf-8")
    os.chmod(publication, 0o600)

    assert _run_validator(fixture).returncode == 0
    status = _run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"],
        fixture,
    )
    assert status.returncode == 0 and status.stdout == ""
    assert _run(["git", "check-ignore", "-q", "--no-index", "--", PUBLICATION], fixture).returncode == 0
    for sibling in (
        "docs/evidence/phase12-filter-v5-rootless-local/final-verification-index.json",
        "docs/evidence/phase12-filter-v5-rootless-local/claim-boundary.md",
        "docs/evidence/phase12-filter-v5-rootless-local/f1-plan-compliance.json",
        "docs/evidence/phase12-filter-v5-rootless-local/rehearsal-publication.json.bak",
        "docs/evidence/phase12-filter-v5-rootless-local/alternate-publication.json",
    ):
        assert _run(["git", "check-ignore", "-q", "--no-index", "--", sibling], fixture).returncode == 1

def test_git_validator_accepts_only_exact_structural_config_values(tmp_path: Path) -> None:
    fixture = _make_git_fixture(tmp_path)
    config = fixture / ".git" / "config"
    config.write_bytes(config.read_bytes().replace(b"bare = false", b"bare = true"))

    assert _run_validator(fixture).returncode == 64


def test_each_authority_git_operation_has_immediate_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_materializer()
    calls: list[tuple[str, ...]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._git(ROOT, "rev-parse", "HEAD")
    module._git(ROOT, "status", "--porcelain=v1")

    assert len(calls) == 4
    assert calls[0][0] == os.path.realpath(sys.executable)
    assert calls[1][0] == "/usr/bin/git"
    assert calls[2][0] == os.path.realpath(sys.executable)
    assert calls[3][0] == "/usr/bin/git"


def _completion_report(
    writer: ModuleType,
    *,
    call_outcome: str = "passed",
    deselected: tuple[str, ...] = (),
    exit_code: int = 0,
) -> bytes:
    nodeids = tuple(f"{path}::test_complete" for path in writer.ROLE_TESTS["t1"])
    return _canonical_json(
        {
            "schema_version": "rootless_pytest_completion_v1",
            "exit_code": exit_code,
            "collected_nodeids": list(nodeids),
            "deselected_nodeids": list(deselected),
            "ordered_reports": [
                {
                    "nodeid": nodeid,
                    "setup": "passed",
                    "call": call_outcome,
                    "teardown": "passed",
                }
                for nodeid in nodeids
            ],
        }
    )


class CompletionOverrides(TypedDict, total=False):
    call_outcome: str
    deselected: tuple[str, ...]
    exit_code: int


@pytest.mark.parametrize(
    "report",
    (
        {"call_outcome": "skipped"},
        {"deselected": ("tests/test_phase12_filter_v5_rootless_legacy_fence.py::test_hidden",)},
        {"exit_code": 1},
    ),
)
def test_qa_writer_rejects_skipped_deselected_or_failed_completion(report: CompletionOverrides) -> None:
    writer = _load_script(QA_WRITER, "rootless_task_qa_completion")
    raw = _completion_report(writer, **report)

    with pytest.raises(ValueError, match="ROOTLESS_TASK_QA_COMPLETION_INVALID"):
        writer._validate_pytest_completion("t1", raw)


def _registered_command(writer: ModuleType, tmp_path: Path) -> object:
    argv = (
        os.path.realpath(sys.executable),
        "-B",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(tmp_path / "runs/phase12-filter-v5-rootless-qa/basetemp/t1/pytest"),
        *writer.ROLE_TESTS["t1"],
        "-q",
    )
    command = writer.CommandResult(argv, 0, b"all passed\n", b"", 0, 0)
    writer._VERIFIED_RESULTS[id(command)] = command
    return command


def test_qa_writer_rejects_caller_fabricated_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writer = _load_script(QA_WRITER, "rootless_task_qa_fabricated")
    monkeypatch.setattr(writer, "ROOT", tmp_path)
    destination = tmp_path / "runs/phase12-filter-v5-rootless-qa/t1-legacy-fence.json"
    destination.parent.mkdir(parents=True, mode=0o700)
    command = writer.CommandResult(writer._expected_argv("t1"), 0, b"7 passed, 19 skipped\n", b"", 0, 0)

    with pytest.raises(ValueError, match="ROOTLESS_TASK_QA_COMPLETION_INVALID"):
        writer.write_rootless_task_qa("t1", command, writer.ROLE_ASSERTIONS["t1"], destination)


def test_qa_writer_retries_interruption_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _load_script(QA_WRITER, "rootless_task_qa_before_publish")
    monkeypatch.setattr(writer, "ROOT", tmp_path)
    destination = tmp_path / "runs/phase12-filter-v5-rootless-qa/t1-legacy-fence.json"
    destination.parent.mkdir(parents=True, mode=0o700)
    command = _registered_command(writer, tmp_path)
    real_link = writer.os.link
    interrupted = False

    def interrupt_link(*args: object, **kwargs: object) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise InterruptedError
        real_link(*args, **kwargs)

    monkeypatch.setattr(writer.os, "link", interrupt_link)
    with pytest.raises(InterruptedError):
        writer.write_rootless_task_qa("t1", command, writer.ROLE_ASSERTIONS["t1"], destination)
    assert not destination.exists()

    writer.write_rootless_task_qa("t1", command, writer.ROLE_ASSERTIONS["t1"], destination)
    assert destination.is_file()
    assert not destination.with_name(f".{destination.name}.tmp").exists()


def test_qa_writer_retries_interruption_after_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _load_script(QA_WRITER, "rootless_task_qa_after_publish")
    monkeypatch.setattr(writer, "ROOT", tmp_path)
    destination = tmp_path / "runs/phase12-filter-v5-rootless-qa/t1-legacy-fence.json"
    destination.parent.mkdir(parents=True, mode=0o700)
    command = _registered_command(writer, tmp_path)
    real_unlink = writer.os.unlink
    interrupted = False

    def interrupt_unlink(*args: object, **kwargs: object) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise InterruptedError
        real_unlink(*args, **kwargs)

    monkeypatch.setattr(writer.os, "unlink", interrupt_unlink)
    with pytest.raises(InterruptedError):
        writer.write_rootless_task_qa("t1", command, writer.ROLE_ASSERTIONS["t1"], destination)
    before = destination.read_bytes()
    temporary = destination.with_name(f".{destination.name}.tmp")
    final_info = os.lstat(destination)
    temporary_info = os.lstat(temporary)
    assert final_info.st_nlink == temporary_info.st_nlink == 2
    assert (final_info.st_dev, final_info.st_ino) == (temporary_info.st_dev, temporary_info.st_ino)

    writer.write_rootless_task_qa("t1", command, writer.ROLE_ASSERTIONS["t1"], destination)
    assert destination.read_bytes() == before
    assert os.lstat(destination).st_nlink == 1
    assert not temporary.exists()


def test_qa_writer_rejects_final_hard_link_to_unrelated_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _load_script(QA_WRITER, "rootless_task_qa_final_alias")
    monkeypatch.setattr(writer, "ROOT", tmp_path)
    destination = tmp_path / "runs/phase12-filter-v5-rootless-qa/t1-legacy-fence.json"
    destination.parent.mkdir(parents=True, mode=0o700)
    command = _registered_command(writer, tmp_path)
    writer.write_rootless_task_qa("t1", command, writer.ROLE_ASSERTIONS["t1"], destination)
    os.link(destination, destination.with_name("unrelated-final-alias.json"))
    command = _registered_command(writer, tmp_path)

    with pytest.raises(ValueError, match="ROOTLESS_TASK_QA_EXISTING_INVALID"):
        writer.write_rootless_task_qa("t1", command, writer.ROLE_ASSERTIONS["t1"], destination)


def test_qa_writer_rejects_temporary_hard_link_to_unrelated_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _load_script(QA_WRITER, "rootless_task_qa_temporary_alias")
    monkeypatch.setattr(writer, "ROOT", tmp_path)
    destination = tmp_path / "runs/phase12-filter-v5-rootless-qa/t1-legacy-fence.json"
    destination.parent.mkdir(parents=True, mode=0o700)
    temporary = destination.with_name(f".{destination.name}.tmp")
    command = _registered_command(writer, tmp_path)
    raw = writer._canonical_json(
        writer._task_payload(
            "t1", command, writer.ROLE_ASSERTIONS["t1"], "2026-08-06T00:00:00Z"
        )
    )
    temporary.write_bytes(raw)
    os.chmod(temporary, 0o600)
    os.link(temporary, destination.with_name("unrelated-temporary-alias.json"))

    with pytest.raises(ValueError, match="ROOTLESS_TASK_QA_EXISTING_INVALID"):
        writer.write_rootless_task_qa("t1", command, writer.ROLE_ASSERTIONS["t1"], destination)


def test_qa_writer_rejects_final_and_temporary_on_different_inodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _load_script(QA_WRITER, "rootless_task_qa_different_inodes")
    monkeypatch.setattr(writer, "ROOT", tmp_path)
    destination = tmp_path / "runs/phase12-filter-v5-rootless-qa/t1-legacy-fence.json"
    destination.parent.mkdir(parents=True, mode=0o700)
    temporary = destination.with_name(f".{destination.name}.tmp")
    command = _registered_command(writer, tmp_path)
    raw = writer._canonical_json(
        writer._task_payload(
            "t1", command, writer.ROLE_ASSERTIONS["t1"], "2026-08-06T00:00:00Z"
        )
    )
    destination.write_bytes(raw)
    temporary.write_bytes(raw)
    os.chmod(destination, 0o600)
    os.chmod(temporary, 0o600)

    with pytest.raises(ValueError, match="ROOTLESS_TASK_QA_EXISTING_INVALID"):
        writer.write_rootless_task_qa("t1", command, writer.ROLE_ASSERTIONS["t1"], destination)


def test_qa_writer_rejects_artifact_with_more_than_two_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _load_script(QA_WRITER, "rootless_task_qa_excess_links")
    monkeypatch.setattr(writer, "ROOT", tmp_path)
    destination = tmp_path / "runs/phase12-filter-v5-rootless-qa/t1-legacy-fence.json"
    destination.parent.mkdir(parents=True, mode=0o700)
    command = _registered_command(writer, tmp_path)
    writer.write_rootless_task_qa("t1", command, writer.ROLE_ASSERTIONS["t1"], destination)
    os.link(destination, destination.with_name("first-alias.json"))
    os.link(destination, destination.with_name("second-alias.json"))
    command = _registered_command(writer, tmp_path)

    with pytest.raises(ValueError, match="ROOTLESS_TASK_QA_EXISTING_INVALID"):
        writer.write_rootless_task_qa("t1", command, writer.ROLE_ASSERTIONS["t1"], destination)
