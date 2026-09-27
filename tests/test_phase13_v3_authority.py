import hashlib
import json
import os
from pathlib import Path
from typing import Final

import pytest

from memcontam.readiness import phase13_authority_files as files
from .phase13_corrective_identity import corrective_identity

ROOT: Final = Path("/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/References/Theoretical Artifacts")
NAMES: Final = (
    "Phase 13 \u2014 THEORETICAL ARTIFACT revised-v1.md",
    "Phase 13-Compatible Baseline Memory and Filter Design revised-v5.md",
    "Phase 13-Compatible Contamination Construction Intervention Timing and Sensitivity Protocol revised-v9.md",
    "2026-08-24_Phase13_MainA_PostCutoff_Acceleration_Addendum_revised-v5.md",
    "Phase 13-Compatible Pilot Main and Exploratory Experiment Design revised-v14.md",
    "2026-09-03_Phase13_MainA_Corrective_Scientific_Decision_Authority.md",
    "AGENTS.md",
)
MANIFEST: Final = "2026-09-05_Phase13_Input_Envelope_Authority_Revision_Manifest.md"
V2_MANIFEST: Final = "2026-09-23_Phase13_Game24_WS_Retry_Authority_Revision_Manifest.md"


@pytest.fixture(scope="session")
def authority_bytes() -> tuple[bytes, ...]:
    return tuple(
        files.read_regular_nofollow(ROOT / name)
        for name in (*NAMES, MANIFEST, V2_MANIFEST)
    )


@pytest.fixture
def authority_root(tmp_path: Path, authority_bytes: tuple[bytes, ...]) -> Path:
    root = tmp_path / "inputs"
    root.mkdir()
    for name, raw in zip((*NAMES, MANIFEST, V2_MANIFEST), authority_bytes, strict=True):
        (root / name).write_bytes(raw)
    return root


def test_baseline_regular_bytes(tmp_path: Path) -> None:
    target = tmp_path / "authority.md"
    target.write_bytes(b"authority\x00\n")
    actual = files.read_regular_nofollow(target)
    assert actual == b"authority\x00\n"


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink"])
def test_baseline_rejects_nonregular(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "authority.md"
    if kind == "directory":
        target.mkdir()
    if kind == "symlink":
        target.symlink_to(tmp_path)
    with pytest.raises(files.AuthorityFileError) as caught:
        files.read_regular_nofollow(target)
    assert caught.value.code == "AUTHORITY_FILE_NOT_REGULAR"


def test_snapshot_binds_exact_routed_stack(authority_root: Path) -> None:
    snapshot = files.load_authority_v3(authority_root, identity=corrective_identity())
    assert tuple(row.filename for row in snapshot.documents) == NAMES
    assert tuple(row.role for row in snapshot.documents) == (
        "theory", "baseline", "contamination_protocol", "narrow_addendum",
        "experiment", "corrective_scientific", "router",
    )
    for row in snapshot.documents:
        raw = (authority_root / row.filename).read_bytes()
        assert (row.size, row.sha256) == (len(raw), hashlib.sha256(raw).hexdigest())
    assert snapshot.provenance.filename == MANIFEST
    assert snapshot.provenance.role == "provenance_only"
    assert snapshot.revision_manifest.filename == V2_MANIFEST
    assert snapshot.revision_manifest.sha256 == "a123b5ce1076d78d12482617c26e3344c37449a07e622f620b66139d46012055"


def test_registry_values_and_identities(authority_root: Path) -> None:
    snapshot = files.load_authority_v3(authority_root, identity=corrective_identity())
    assert [(stage.stage_id, stage.maximum_output_tokens, stage.maximum_input_tokens)
            for stage in snapshot.registry.stages] == [
        ("FH_generation", 512, 9330), ("RAG_generation", 512, 378),
        ("BoT_problem_distillation", 384, 1177), ("BoT_solve", 512, 1949),
        ("BoT_thought_distillation", 384, 2545),
        ("Reflexion_actor_generation", 512, 2282), ("Reflexion_reflection", 384, 3349),
        ("DC_RS_generation", 512, 9212), ("DC_RS_writer_synthesis", 8192, 13521),
        ("NoMem_generation", 512, 1160),
    ]
    assert snapshot.registry.sha256 == "5796df90795ff7f753aad753abc1a70499c083fdb40dabb7a26afe011e58be38"
    assert snapshot.registry.transport_contract_sha256 == "664e36f7fc74d74c640f3c41924e81d31ae01672f285fbdc18cff6f602bdc155"
    assert snapshot.registry.retry_allocation_registry_sha256 == "0dcab38c3fb9efa55b1370e18dbea8544af28d3f09b330405b286809239cb1c9"
    assert snapshot.terminal.sha256 == "599eb322efdfea397c227fdad25f2d9371c444392eb68ef943a04748467d16bd"
    assert snapshot.retry.sha256 == "0dcab38c3fb9efa55b1370e18dbea8544af28d3f09b330405b286809239cb1c9"
    assert snapshot.transport.sha256 == "664e36f7fc74d74c640f3c41924e81d31ae01672f285fbdc18cff6f602bdc155"
    assert (
        snapshot.registry.default_max_transport_attempts,
        snapshot.registry.entitled_eligible_max_transport_attempts,
        snapshot.registry.maximum_retries_after_initial_attempt,
        snapshot.capacity.B_mem_tokens,
    ) == (1, 2, 1, 8192)
    assert snapshot.identity == corrective_identity()
    assert snapshot.predecessor_rag_input_tokens == 378
    assert snapshot.repository_344_status == "STALE_IMPLEMENTATION_HISTORY"


@pytest.mark.parametrize("old,new", [
    (b"RAG_generation|512|378", b"RAG_generation|512|290"),
    (b"FH_generation|512|9330", b"FH_generation|512|9331"),
    (b"default_max_transport_attempts=1", b"default_max_transport_attempts=2"),
    (b"maximum_retries_after_initial_attempt=1", b"maximum_retries_after_initial_attempt=2"),
    (b"terminal_trigger=UNCLASSIFIED_PROVIDER_FAILURE", b"terminal_trigger=UNKNOWN"),
    (b"END_CORE_EXECUTION_ENVELOPE_REGISTRY_V4", b"END_BROKEN"),
])
def test_altered_block_rejected(authority_root: Path, old: bytes, new: bytes) -> None:
    target = authority_root / NAMES[4]
    target.write_bytes(target.read_bytes().replace(old, new))
    with pytest.raises(files.AuthorityFileError, match="MAIN_ENVELOPE_REGISTRY_MISMATCH"):
        files.load_authority_v3(authority_root, identity=corrective_identity())


def test_v3_rejects_repository_344_projection(authority_root: Path, tmp_path: Path) -> None:
    target = authority_root / NAMES[4]
    target.write_bytes(target.read_bytes().replace(b"RAG_generation|512|378", b"RAG_generation|512|344"))
    with pytest.raises(files.AuthorityFileError, match="MAIN_ENVELOPE_REGISTRY_MISMATCH"):
        files.build_authority_v3(authority_root, tmp_path, corrective_identity())
    assert not (tmp_path / "current_authority_v3.json").exists()


def test_wrong_whole_file_hash_rejected(authority_root: Path) -> None:
    expected = files.load_authority_v3(authority_root, identity=corrective_identity())
    target = authority_root / NAMES[0]
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(files.AuthorityFileError, match="MAIN_AUTHORITY_BINDING_MISMATCH"):
        files.load_authority_v3(authority_root, expected)


def test_altered_router_target_rejected(authority_root: Path) -> None:
    router = authority_root / "AGENTS.md"
    router.write_bytes(router.read_bytes().replace(NAMES[4].encode(), b"stale.md", 1))
    with pytest.raises(files.AuthorityFileError, match="MAIN_AUTHORITY_BINDING_MISMATCH"):
        files.load_authority_v3(authority_root, identity=corrective_identity())


def test_authority_loader_rejects_symlinked_parent(authority_root: Path, tmp_path: Path) -> None:
    parent = tmp_path / "linked"
    parent.symlink_to(authority_root, target_is_directory=True)
    with pytest.raises(files.AuthorityFileError, match="MAIN_PATH_UNSAFE"):
        files.build_authority_v3(parent, tmp_path, corrective_identity())
    assert not (tmp_path / "current_authority_v3.json").exists()


def test_authority_loader_rejects_symlinked_final(authority_root: Path) -> None:
    target = authority_root / NAMES[0]
    target.unlink()
    target.symlink_to(ROOT / NAMES[0])
    with pytest.raises(files.AuthorityFileError, match="MAIN_PATH_UNSAFE"):
        files.load_authority_v3(authority_root, identity=corrective_identity())


def test_authority_loader_rejects_escape(authority_root: Path) -> None:
    with pytest.raises(files.AuthorityFileError, match="MAIN_PATH_UNSAFE"):
        files.load_authority_v3(authority_root / ".." / "inputs", identity=corrective_identity())


def test_unstable_read_rejected(authority_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = os.read
    target = authority_root / NAMES[0]
    def changing_read(descriptor: int, count: int) -> bytes:
        raw = original(descriptor, count)
        if raw and b"# Phase 13" in raw:
            with target.open("ab") as stream:
                stream.write(b"changed")
        return raw
    monkeypatch.setattr(files.os, "read", changing_read)
    with pytest.raises(files.AuthorityFileError, match="MAIN_PATH_UNSAFE"):
        files.load_authority_v3(authority_root, identity=corrective_identity())


def test_builder_emits_json_without_external_mutation(tmp_path: Path) -> None:
    before = tuple(
        files.read_regular_nofollow(ROOT / name)
        for name in (*NAMES, MANIFEST, V2_MANIFEST)
    )
    snapshot = files.build_authority_v3(ROOT, tmp_path, corrective_identity())
    raw = (tmp_path / "current_authority_v3.json").read_bytes()
    assert json.loads(raw)["schema_version"] == "phase13_main_authority_snapshot_v3"
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert files.load_authority_v3(ROOT, snapshot) == snapshot
    assert tuple(
        files.read_regular_nofollow(ROOT / name)
        for name in (*NAMES, MANIFEST, V2_MANIFEST)
    ) == before


def test_publication_never_replaces(authority_root: Path, tmp_path: Path) -> None:
    target = tmp_path / "current_authority_v3.json"
    target.write_bytes(b"existing")
    with pytest.raises(files.AuthorityFileError, match="MAIN_AUTHORITY_BINDING_MISMATCH"):
        files.build_authority_v3(authority_root, tmp_path, corrective_identity())
    assert target.read_bytes() == b"existing"


def test_capacity_drift_rejected(authority_root: Path) -> None:
    router = authority_root / "AGENTS.md"
    router.write_bytes(router.read_bytes().replace(b"B_mem_tokens=8192", b"B_mem_tokens=4096"))
    with pytest.raises(files.AuthorityFileError, match="MAIN_ENVELOPE_REGISTRY_MISMATCH"):
        files.load_authority_v3(authority_root, identity=corrective_identity())


@pytest.mark.parametrize("attempt", range(3))
def test_interruption_closes_descriptors(authority_root: Path, monkeypatch: pytest.MonkeyPatch, attempt: int) -> None:
    before = len(tuple(Path("/proc/self/fd").iterdir()))
    def interrupted_read(descriptor: int, count: int) -> bytes:
        raise InterruptedError(attempt)
    monkeypatch.setattr(files.os, "read", interrupted_read)
    with pytest.raises(files.AuthorityFileError, match="MAIN_PATH_UNSAFE"):
        files.load_authority_v3(authority_root, identity=corrective_identity())
    assert len(tuple(Path("/proc/self/fd").iterdir())) == before


def test_ancestor_swap_during_read_rejected(authority_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = os.read
    moved = authority_root.with_name("moved")
    def swapping_read(descriptor: int, count: int) -> bytes:
        raw = original(descriptor, count)
        if not moved.exists():
            authority_root.rename(moved)
            authority_root.mkdir()
        return raw
    monkeypatch.setattr(files.os, "read", swapping_read)
    with pytest.raises(files.AuthorityFileError, match="MAIN_PATH_UNSAFE"):
        files.load_authority_v3(authority_root, identity=corrective_identity())


def test_self_rehashed_semantic_projection_rejected(authority_root: Path) -> None:
    expected = files.load_authority_v3(authority_root, identity=corrective_identity())
    payload = json.loads(expected.model_dump_json())
    payload["registry"]["stages"][1]["maximum_input_tokens"] = 344
    payload["registry"]["sha256"] = hashlib.sha256(json.dumps(payload["registry"], sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError):
        type(expected).model_validate_json(json.dumps(payload))


def test_duplicate_block_rejected(authority_root: Path) -> None:
    target = authority_root / NAMES[4]
    raw = target.read_bytes()
    begin = raw.index(b"\nBEGIN_CORE_EXECUTION_ENVELOPE_REGISTRY_V4\n")
    end = raw.index(b"\nEND_CORE_EXECUTION_ENVELOPE_REGISTRY_V4\n", begin)
    target.write_bytes(raw + raw[begin:end] + b"\nEND_CORE_EXECUTION_ENVELOPE_REGISTRY_V4\n")
    with pytest.raises(files.AuthorityFileError, match="MAIN_ENVELOPE_REGISTRY_MISMATCH"):
        files.load_authority_v3(authority_root, identity=corrective_identity())


def test_router_decoy_does_not_override_precedence(authority_root: Path) -> None:
    router = authority_root / "AGENTS.md"
    raw = router.read_text()
    raw = raw.replace("Theory \u2192 Baseline", "Baseline \u2192 Theory", 1)
    decoy = f"Theory \u2192 Baseline \u2192 Contamination Protocol \u2192 `{NAMES[3]}` \u2192 Experiment Design\n"
    router.write_text(raw.replace("## Ownership", decoy + "## Ownership"))
    with pytest.raises(files.AuthorityFileError, match="MAIN_AUTHORITY_BINDING_MISMATCH"):
        files.load_authority_v3(authority_root, identity=corrective_identity())


def test_output_ancestor_swap_leaves_no_snapshot(authority_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    moved = tmp_path / "moved-output"
    original = os.link
    def swapping_link(src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int, follow_symlinks: bool) -> None:
        original(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd, follow_symlinks=follow_symlinks)
        output.rename(moved)
        output.mkdir()
    monkeypatch.setattr(files.os, "link", swapping_link)
    with pytest.raises(files.AuthorityFileError, match="MAIN_PATH_UNSAFE"):
        files.build_authority_v3(authority_root, output, corrective_identity())
    assert not tuple(moved.iterdir())
    assert not tuple(output.iterdir())
