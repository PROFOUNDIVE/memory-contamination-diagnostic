from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
import stat
from uuid import uuid4

from memcontam.readiness.phase13_v3_authority_models import (
    PROVENANCE_FILENAME, ROUTED_DOCUMENTS, AuthoritySnapshotV3, DocumentBinding, V3Identity,
)
from memcontam.readiness.phase13_v3_authority_parser import (
    parse_capacity, parse_registry, parse_terminal, validate_router,
)


class AuthorityFileError(ValueError):
    def __init__(self, code: str = "AUTHORITY_FILE_NOT_REGULAR") -> None:
        super().__init__(code)
        self.code = code


def read_regular_nofollow(path: Path) -> bytes:
    try:
        with authority_directory(path.parent) as directory:
            return read_authority_at(directory, path.name)
    except (OSError, AuthorityFileError) as error:
        raise AuthorityFileError() from error


@contextmanager
def authority_directory(path: Path) -> Iterator[int]:
    """Hold every no-follow ancestor until the directory operation finishes."""
    target = path if path.is_absolute() else Path.cwd() / path
    if any(part in {".", ".."} for part in target.parts):
        raise AuthorityFileError("MAIN_PATH_UNSAFE")
    parts = tuple(part for part in target.parts if part != "/")
    with ExitStack() as resources:
        try:
            directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            resources.callback(os.close, directory)
            ancestors: list[tuple[int, str, int]] = []
            for component in parts:
                parent = directory
                directory = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                resources.callback(os.close, directory)
                ancestors.append((parent, component, directory))
                if not stat.S_ISDIR(os.fstat(directory).st_mode):
                    raise AuthorityFileError("MAIN_PATH_UNSAFE")
            yield directory
            for parent, component, descriptor in ancestors:
                current = os.stat(component, dir_fd=parent, follow_symlinks=False)
                opened = os.fstat(descriptor)
                if (current.st_dev, current.st_ino, current.st_mode) != (opened.st_dev, opened.st_ino, opened.st_mode):
                    raise AuthorityFileError("MAIN_PATH_UNSAFE")
        except OSError as error:
            raise AuthorityFileError("MAIN_PATH_UNSAFE") from error


def read_authority_at(directory: int, filename: str) -> bytes:
    """Read a contained regular file once, rejecting changes during that read."""
    if not filename or filename in {".", ".."} or "/" in filename or "\x00" in filename:
        raise AuthorityFileError("MAIN_PATH_UNSAFE")
    with ExitStack() as resources:
        try:
            descriptor = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            resources.callback(os.close, descriptor)
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise AuthorityFileError("MAIN_PATH_UNSAFE")
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1_048_576):
                chunks.append(chunk)
            raw = b"".join(chunks)
            after = os.fstat(descriptor)
            named = os.stat(filename, dir_fd=directory, follow_symlinks=False)
            fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
            if len(raw) != before.st_size or any(
                getattr(before, field) != getattr(after, field) or getattr(after, field) != getattr(named, field)
                for field in fields
            ):
                raise AuthorityFileError("MAIN_PATH_UNSAFE")
            return raw
        except OSError as error:
            raise AuthorityFileError("MAIN_PATH_UNSAFE") from error


def load_authority_v3(root: Path, expected: AuthoritySnapshotV3 | None = None, *, identity: V3Identity | None = None) -> AuthoritySnapshotV3:
    selected_identity = expected.identity if expected is not None else identity
    if selected_identity is None or (identity is not None and identity != selected_identity):
        raise AuthorityFileError("MAIN_AUTHORITY_BINDING_MISMATCH")
    with authority_directory(root) as directory:
        raw_documents = tuple(read_authority_at(directory, filename) for _, filename in ROUTED_DOCUMENTS)
        provenance_raw = read_authority_at(directory, PROVENANCE_FILENAME)
    try:
        validate_router(raw_documents[6])
    except (ValueError, IndexError) as error:
        raise AuthorityFileError("MAIN_AUTHORITY_BINDING_MISMATCH") from error
    try:
        registry = parse_registry(raw_documents[4])
        terminal = parse_terminal(raw_documents[4])
        capacity = parse_capacity(raw_documents[6])
    except (ValueError, IndexError, KeyError) as error:
        raise AuthorityFileError("MAIN_ENVELOPE_REGISTRY_MISMATCH") from error
    snapshot = AuthoritySnapshotV3(
        identity=selected_identity,
        documents=tuple(DocumentBinding(filename=filename, role=role, size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                        for (role, filename), raw in zip(ROUTED_DOCUMENTS, raw_documents, strict=True)),
        provenance=DocumentBinding(filename=PROVENANCE_FILENAME, role="provenance_only", size=len(provenance_raw), sha256=hashlib.sha256(provenance_raw).hexdigest()),
        registry=registry, terminal=terminal, capacity=capacity,
    )
    if expected is not None and snapshot != expected:
        raise AuthorityFileError("MAIN_AUTHORITY_BINDING_MISMATCH")
    return snapshot


def build_authority_v3(root: Path, output_root: Path, identity: V3Identity) -> AuthoritySnapshotV3:
    """Publish only the authority snapshot into an existing output directory."""
    snapshot = load_authority_v3(root, identity=identity)
    raw = (json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with ExitStack() as cleanup:
        published = False
        completed = False
        with authority_directory(output_root) as directory:
            cleanup_directory = os.dup(directory)
            cleanup.callback(os.close, cleanup_directory)
            temporary = f".authority-{uuid4().hex}.tmp"
            target = "current_authority_v3.json"
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)

            def rollback_publication() -> None:
                if published and not completed:
                    try:
                        os.unlink(target, dir_fd=cleanup_directory)
                        os.fsync(cleanup_directory)
                    except OSError as error:
                        raise AuthorityFileError("MAIN_PATH_UNSAFE") from error

            cleanup.callback(rollback_publication)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                    try:
                        os.link(temporary, target, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
                    except FileExistsError as error:
                        raise AuthorityFileError("MAIN_AUTHORITY_BINDING_MISMATCH") from error
                    published = True
                    os.fsync(stream.fileno())
                    os.fsync(directory)
            except OSError as error:
                raise AuthorityFileError("MAIN_PATH_UNSAFE") from error
            finally:
                os.unlink(temporary, dir_fd=directory)
        completed = True
    return snapshot
