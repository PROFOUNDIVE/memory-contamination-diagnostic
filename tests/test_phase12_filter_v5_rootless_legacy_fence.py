from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_CHECKOUT = ROOT.parent / "memory-contamination-diagnostic-filter-v5"
MATERIALIZER = ROOT / "scripts" / "materialize_phase12_filter_v5_rootless_inputs.py"
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


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, check=False, text=True)


def _make_clean_fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / "fixture-repository"
    cloned = _run(["git", "clone", "--no-local", "--quiet", str(ROOT), str(fixture)], ROOT)
    assert cloned.returncode == 0, cloned.stderr
    manifest_path = fixture / MANIFEST.relative_to(ROOT)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(_manifest_bytes())
    descriptor_path = fixture / MANIFEST_DESCRIPTOR.relative_to(ROOT)
    descriptor_path.parent.mkdir(parents=True)
    descriptor_path.write_bytes(
        f"{sha256(_manifest_bytes()).hexdigest()}  configs/phase12/filter_v5_rootless_local/external_inputs.json\n".encode(
            "ascii"
        )
    )
    ignore_path = fixture / ".gitignore"
    ignore_path.write_bytes((ROOT / ".gitignore").read_bytes())
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
            "add rootless external manifest fixture",
        ],
        fixture,
    )
    assert committed.returncode == 0, committed.stderr
    return fixture


def _external_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    assert PRIMARY_CHECKOUT.is_dir(), "the primary checkout supplies detached historical inputs"
    source_root = tmp_path / "external-inputs"
    source_paths: list[Path] = []
    for _, destination, _, _ in INPUTS:
        source = PRIMARY_CHECKOUT / destination
        assert source.is_file(), f"missing detached source: {source}"
        copied = source_root / source.name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copied)
        os.chmod(copied, 0o600)
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
