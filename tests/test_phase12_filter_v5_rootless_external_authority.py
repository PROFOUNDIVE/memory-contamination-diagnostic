from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

import pytest

from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
)


def _mountinfo(path: Path, *, options: bytes = b"ro", mount_id: int = 7) -> bytes:
    info = path.stat()
    escaped = os.fsencode(path.parent).replace(b" ", b"\\040")
    return (
        str(mount_id).encode()
        + b" 1 "
        + f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}".encode()
        + b" / "
        + escaped
        + b" "
        + options
        + b" - ext4 /dev/test ro\n"
    )


def _source(path: Path) -> dict[str, JsonValue]:
    raw = path.read_bytes()
    return {
        "role": "experiment-design",
        "absolute_path": os.fspath(path),
        "full_sha256": sha256(raw).hexdigest(),
        "ordered_spans": [{"start_line": 2, "end_line": 2, "sha256": sha256(b"two\n").hexdigest()}],
    }


def test_mountinfo_parser_decodes_all_legal_octal_escapes() -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import parse_mountinfo

    # Given: one well-formed record containing every legal mountinfo escape.
    raw = b"9 1 8:1 /root\\040x\\011y\\012z\\134w /mnt\\040disk ro - ext4 /dev/sda ro\n"

    # When: the Linux mount snapshot is parsed.
    records = parse_mountinfo(raw)

    # Then: root and mount point contain the exact decoded bytes.
    assert records[0].root == b"/root x\ty\nz\\w"
    assert records[0].mount_point == b"/mnt disk"


@pytest.mark.parametrize("raw", [
    b"9 1 8:1 / /mnt\\ ro - ext4 x ro\n",
    b"9 1 8:1 / /mnt\\041 ro - ext4 x ro\n",
    b"9 1 8:1 / /mnt\\\\040 ro - ext4 x ro\n",
    b"9 1 8:1 / /mnt ro ext4 x ro\n",
])
def test_mountinfo_parser_rejects_malformed_records(raw: bytes) -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import parse_mountinfo

    # Given/When/Then: malformed syntax cannot become mount authority.
    with pytest.raises(RootlessContractError, match="ROOTLESS_EXTERNAL_AUTHORITY_MOUNT_NOT_READ_ONLY"):
        parse_mountinfo(raw)


def test_mount_selection_uses_longest_component_prefix_and_device(tmp_path: Path) -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
        parse_mountinfo,
        select_mount_record,
    )

    # Given: root, sibling-prefix, and exact-parent mount records.
    target = tmp_path / "authority.md"
    target.write_text("authority", encoding="utf-8")
    info = target.stat()
    device = f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}".encode()
    mount = os.fsencode(tmp_path).replace(b" ", b"\\040")
    raw = b"1 1 " + device + b" / / ro - ext4 x ro\n"
    raw += b"2 1 " + device + b" / " + mount + b"-sibling ro - ext4 x ro\n"
    raw += b"3 1 " + device + b" / " + mount + b" ro - ext4 x ro\n"

    # When: the target mount is selected.
    selected = select_mount_record(os.fsencode(target), info.st_dev, parse_mountinfo(raw))

    # Then: only the longest component-boundary match wins.
    assert selected.mount_id == 3


def test_external_observation_binds_hash_spans_mount_and_identity(tmp_path: Path) -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
        observe_external_authority,
    )

    # Given: a stable synthetic source and two identical read-only mount snapshots.
    source_path = tmp_path / "authority.md"
    source_path.write_bytes(b"one\ntwo\nthree")
    mount = _mountinfo(source_path)

    # When: the descriptor is observed twice with independent read-only evidence.
    observation = observe_external_authority(
        _source(source_path),
        mountinfo_reader=iter((mount, mount)).__next__,
        fstatvfs_reader=lambda _fd: os.ST_RDONLY,
    )

    # Then: the closed observation binds bytes, span, mount, and descriptor identity.
    assert observation["full_sha256"] == sha256(source_path.read_bytes()).hexdigest()
    assert observation["ordered_span_sha256s"] == [sha256(b"two\n").hexdigest()]
    assert observation["mount_options_read_only"] is True
    assert observation["fstatvfs_read_only"] is True
    assert observation["file_st_ino"] == source_path.stat().st_ino


def test_mode_0444_on_writable_mount_is_not_external_authority(tmp_path: Path) -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
        observe_external_authority,
    )

    # Given: immutable-looking mode bits but writable mount observations.
    source_path = tmp_path / "authority.md"
    source_path.write_bytes(b"one\ntwo\n")
    source_path.chmod(0o444)
    mount = _mountinfo(source_path, options=b"rw")

    # When/Then: file mode cannot substitute for both runtime RO checks.
    with pytest.raises(RootlessContractError, match="ROOTLESS_EXTERNAL_AUTHORITY_MOUNT_NOT_READ_ONLY"):
        observe_external_authority(
            _source(source_path),
            mountinfo_reader=iter((mount, mount)).__next__,
            fstatvfs_reader=lambda _fd: 0,
        )


def test_external_observation_detects_mount_identity_and_hash_drift(tmp_path: Path) -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
        observe_external_authority,
    )

    # Given: one source with either a changed mount snapshot or wrong expected bytes.
    source_path = tmp_path / "authority.md"
    source_path.write_bytes(b"one\ntwo\n")
    first = _mountinfo(source_path, mount_id=7)
    second = _mountinfo(source_path, mount_id=8)

    # When/Then: mount drift precedes stable-byte hash mismatch.
    with pytest.raises(RootlessContractError, match="ROOTLESS_EXTERNAL_AUTHORITY_IDENTITY_DRIFT"):
        observe_external_authority(
            _source(source_path),
            mountinfo_reader=iter((first, second)).__next__,
            fstatvfs_reader=lambda _fd: os.ST_RDONLY,
        )
    bad_source = _source(source_path)
    bad_source["full_sha256"] = "0" * 64
    with pytest.raises(RootlessContractError, match="ROOTLESS_EXTERNAL_AUTHORITY_HASH_MISMATCH"):
        observe_external_authority(
            bad_source,
            mountinfo_reader=iter((first, first)).__next__,
            fstatvfs_reader=lambda _fd: os.ST_RDONLY,
        )


def test_external_observation_preserves_six_distinct_diagnostics(tmp_path: Path) -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
        observe_external_authority,
    )

    # Given: malformed binding, path disagreement, and a symlink final component.
    source_path = tmp_path / "authority.md"
    source_path.write_bytes(b"one\ntwo\n")
    source = _source(source_path)
    mount = _mountinfo(source_path)
    symlink = tmp_path / "authority-link.md"
    symlink.symlink_to(source_path)
    symlink_source = dict(source)
    symlink_source["absolute_path"] = os.fspath(symlink)

    # When/Then: pre-read failures retain their exact diagnostic identity.
    with pytest.raises(RootlessContractError, match="ROOTLESS_EXTERNAL_AUTHORITY_REVIEW_BINDING_MISSING"):
        observe_external_authority({"role": "experiment-design"})
    with pytest.raises(RootlessContractError, match="ROOTLESS_EXTERNAL_AUTHORITY_PATH_MISMATCH"):
        observe_external_authority(source, requested_path=os.fspath(source_path) + "-other")
    with pytest.raises(RootlessContractError, match="ROOTLESS_EXTERNAL_AUTHORITY_DESCRIPTOR_UNSAFE"):
        observe_external_authority(
            symlink_source,
            mountinfo_reader=iter((mount, mount)).__next__,
            fstatvfs_reader=lambda _fd: os.ST_RDONLY,
        )


def test_real_external_authorities_are_observed_without_mutation() -> None:
    from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
        observe_external_authorities,
    )
    from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
        parse_canonical_object,
    )

    # Given: exact metadata snapshots for the three configured external sources.
    config = parse_canonical_object(
        Path("configs/phase12/filter_v5_rootless_local/decoding_authority.json").read_bytes()
    )
    sources = config["ordered_sources"]
    assert isinstance(sources, list)
    source_objects = [source for source in sources if isinstance(source, dict)]
    raw_paths = [source.get("absolute_path") for source in source_objects]
    assert all(isinstance(path, str) for path in raw_paths)
    paths = [Path(path) for path in raw_paths if isinstance(path, str)]
    before = [(path.stat(), sha256(path.read_bytes()).hexdigest()) for path in paths]

    # When: the production two-snapshot predicate observes the current mount namespace.
    observations = observe_external_authorities(config)

    # Then: all authority bytes and descriptor metadata remain unchanged.
    after = [(path.stat(), sha256(path.read_bytes()).hexdigest()) for path in paths]
    observation_objects = [entry for entry in observations if isinstance(entry, dict)]
    assert len(observation_objects) == 3
    assert [entry["role"] for entry in observation_objects] == [
        "experiment-design",
        "filter-v5-amendment",
        "authority-agents",
    ]
    assert [
        (item.st_dev, item.st_ino, item.st_mode, item.st_nlink, item.st_size, digest)
        for item, digest in before
    ] == [
        (item.st_dev, item.st_ino, item.st_mode, item.st_nlink, item.st_size, digest)
        for item, digest in after
    ]
