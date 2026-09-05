from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import Final
from uuid import uuid4

from .phase13_authority_files import AuthorityFileError, authority_directory, read_authority_at
from .phase13_v3_resource_files import ValidatedResource, _signature, read_files


LEDGER_FILENAME: Final = "main_run_ledger_v3.sqlite3"


class EntrypointPathError(ValueError):
    def __init__(self, code: str = "MAIN_PATH_UNSAFE") -> None:
        self.code = code
        super().__init__(code)


def relative_path(root: Path, path: Path) -> str:
    if ".." in path.parts:
        raise EntrypointPathError()
    try:
        return str(path.absolute().relative_to(root.absolute()))
    except ValueError as error:
        raise EntrypointPathError() from error


def read_authorization_digest(root: Path, path: Path) -> str:
    resource, = read_files(root, (relative_path(root, path),))
    return parse_authorization_digest(resource.raw)


def parse_authorization_digest(raw: bytes) -> str:
    if re.fullmatch(rb"[0-9a-f]{64}\n", raw) is None:
        raise EntrypointPathError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
    return raw[:-1].decode("ascii")


def verify_resource_namespace(root: Path, resources: tuple[ValidatedResource, ...]) -> None:
    try:
        for resource in resources:
            relative = Path(resource.binding.path)
            with authority_directory(root / relative.parent) as descriptor:
                info = os.stat(relative.name, dir_fd=descriptor, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or _signature(info) != resource.signature:
                    raise EntrypointPathError()
                if resource.descriptor is None:
                    raise EntrypointPathError()
                before = os.fstat(resource.descriptor)
                raw = os.pread(resource.descriptor, len(resource.raw) + 1, 0)
                after = os.fstat(resource.descriptor)
                if (_signature(before) != resource.signature or _signature(after) != resource.signature
                    or raw != resource.raw
                    or _signature(os.stat(relative.name, dir_fd=descriptor, follow_symlinks=False)) != resource.signature):
                    raise EntrypointPathError()
    except (OSError, AuthorityFileError) as error:
        raise EntrypointPathError() from error


def _identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_uid, info.st_mode


def _require(info: os.stat_result, directory: bool = False) -> None:
    expected = stat.S_IFDIR | 0o700 if directory else stat.S_IFREG | 0o600
    if info.st_uid != os.getuid() or info.st_mode != expected or (not directory and info.st_nlink != 1):
        raise EntrypointPathError()


@dataclass(frozen=True, slots=True)
class PrivateLedger:
    directory: Path
    directory_fd: int
    database_fd: int
    directory_identity: tuple[int, int, int, int]
    database_identity: tuple[int, int, int, int]

    @property
    def path(self) -> Path:
        return Path(f"/proc/self/fd/{self.directory_fd}/{LEDGER_FILENAME}")

    def check(self) -> None:
        try:
            with authority_directory(self.directory) as current:
                infos = (os.fstat(current), os.fstat(self.directory_fd))
                for info in infos:
                    _require(info, directory=True)
                    if _identity(info) != self.directory_identity:
                        raise EntrypointPathError()
            for info in (os.fstat(self.database_fd), os.stat(
                LEDGER_FILENAME, dir_fd=self.directory_fd, follow_symlinks=False,
            )):
                _require(info)
                if _identity(info) != self.database_identity:
                    raise EntrypointPathError()
            for suffix in ("-wal", "-shm"):
                try:
                    info = os.stat(LEDGER_FILENAME + suffix, dir_fd=self.directory_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                _require(info)
        except (OSError, AuthorityFileError) as error:
            raise EntrypointPathError() from error

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.check()
        before = self.journal_identities()
        with closing(sqlite3.connect(f"{self.path.as_uri()}?mode=rw", uri=True)) as connection:
            self.check()
            after_open = self.journal_identities()
            if any(identity is not None and identity != after_open[index] for index, identity in enumerate(before)):
                raise EntrypointPathError()
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            self.check()
            journals = self.journal_identities()
            with connection:
                yield connection
                self.check()
                if self.journal_identities() != journals:
                    raise EntrypointPathError()
            self.check()
            if self.journal_identities() != journals:
                raise EntrypointPathError()
        self.check()
        self.sync()

    def journal_identities(self) -> tuple[tuple[int, int, int, int] | None, ...]:
        identities: list[tuple[int, int, int, int] | None] = []
        for suffix in ("-wal", "-shm"):
            try:
                info = os.stat(LEDGER_FILENAME + suffix, dir_fd=self.directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                identities.append(None)
            else:
                _require(info)
                identities.append(_identity(info))
        return tuple(identities)

    def sync(self) -> None:
        self.check()
        os.fsync(self.database_fd)
        os.fsync(self.directory_fd)

    @contextmanager
    def lock_descriptor(self) -> Iterator[int]:
        self.check()
        descriptor = os.open(LEDGER_FILENAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.directory_fd)
        try:
            info = os.fstat(descriptor)
            _require(info)
            if _identity(info) != self.database_identity:
                raise EntrypointPathError()
            self.check()
            yield descriptor
            self.check()
        finally:
            os.close(descriptor)

    def read_record(self, name: str) -> bytes:
        self.check()
        try:
            raw = read_authority_at(self.directory_fd, name)
        except AuthorityFileError as error:
            raise EntrypointPathError() from error
        self.check()
        return raw

    def publish_record(self, name: str, raw: bytes) -> None:
        if not name or "/" in name or name in {".", ".."}:
            raise EntrypointPathError()
        self.check()
        temporary = ".receipt-" + uuid4().hex
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.directory_fd)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, name, src_dir_fd=self.directory_fd, dst_dir_fd=self.directory_fd, follow_symlinks=False)
            except FileExistsError:
                if self.read_record(name) != raw:
                    raise EntrypointPathError()
            os.fsync(self.directory_fd)
        finally:
            os.unlink(temporary, dir_fd=self.directory_fd)
        self.check()


@contextmanager
def private_ledger(directory: Path, *, create: bool) -> Iterator[PrivateLedger]:
    if ".." in directory.parts:
        raise EntrypointPathError()
    try:
        with ExitStack() as stack:
            parent = stack.enter_context(authority_directory(directory.parent))
            if create:
                os.mkdir(directory.name, mode=0o700, dir_fd=parent)
                os.fsync(parent)
            descriptor = stack.enter_context(authority_directory(directory))
            _require(os.fstat(descriptor), directory=True)
            flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
            if create:
                flags |= os.O_CREAT | os.O_EXCL
            database = os.open(LEDGER_FILENAME, flags, 0o600, dir_fd=descriptor)
            stack.callback(os.close, database)
            _require(os.fstat(database))
            ledger = PrivateLedger(directory, descriptor, database,
                                   _identity(os.fstat(descriptor)), _identity(os.fstat(database)))
            ledger.check()
            ledger.sync()
            yield ledger
    except (OSError, AuthorityFileError) as error:
        raise EntrypointPathError() from error
