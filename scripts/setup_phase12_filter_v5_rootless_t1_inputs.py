from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final


DIRECTORY_FLAGS: Final = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
FILE_FLAGS: Final = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


class SetupError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class InputPin:
    relative_path: Path
    size_bytes: int
    sha256: str


INPUT_PINS: Final = (
    InputPin(
        Path(".omo/plans/phase12-filter-v5-screening-bct-execution.md"),
        144691,
        "9270d31770eb97e732602cfe85a250111208afeae293b0a20ab618baadb43317",
    ),
    InputPin(
        Path(".omo/approvals/phase12-filter-v5-screening-bct-execution.plan.sha256"),
        65,
        "92c6d30f026a10f47067e5467c0e9e0abc35b653385f4f08ad7d301838e06160",
    ),
    InputPin(
        Path(".omo/plans/phase12-post-filter-v5-calibration-readiness.md"),
        95737,
        "d7109bffe61d5a82ccbd5300e0cca0da9d4411b681ff3358c702024c4074879d",
    ),
    InputPin(
        Path(".omo/approvals/phase12-post-filter-v5-calibration-readiness.plan.sha256"),
        65,
        "7b878988972b5bc3c1a2ba24785b978cc26b973e1e44e8059ff8d3133227842e",
    ),
    InputPin(
        Path(
            ".omo/evidence/phase12-post-filter-v5-calibration-readiness/"
            "task-3-screening-stage-result.json"
        ),
        246,
        "583d1bd5a579af84b00ded45e67b66f491940237c4e708027d9da827b4bbb8f7",
    ),
    InputPin(
        Path(
            ".omo/evidence/phase12-post-filter-v5-calibration-readiness/"
            "task-5-bct-stage-result.json"
        ),
        240,
        "3d7b04540abb253583c345ba66a15163468e69ca5bdfc50ccdb4b68fa99d6792",
    ),
    InputPin(
        Path(
            ".omo/evidence/phase12-post-filter-v5-calibration-readiness/"
            "task-6-pilot-b-readiness-stage-result.json"
        ),
        254,
        "f595fc0a17e330e387e96b7506b65bd2631285e3506ec68afcef8a9294261fbe",
    ),
)


def _absolute(path: Path) -> Path:
    text = os.fspath(path)
    if not text.startswith("/") or text != os.path.normpath(text):
        raise SetupError("ROOTLESS_T1_INPUT_PATH_INVALID")
    return Path(text)


def _safe_directory(info: os.stat_result, *, current_uid_only: bool = False) -> None:
    owners = {os.getuid()} if current_uid_only else {0, os.getuid()}
    if not stat.S_ISDIR(info.st_mode) or info.st_uid not in owners or stat.S_IMODE(info.st_mode) & 0o022:
        raise SetupError("ROOTLESS_T1_INPUT_ANCESTOR_UNSAFE")


def _open_root(path: Path) -> int:
    target = _absolute(path)
    descriptor = os.open("/", DIRECTORY_FLAGS)
    try:
        root_info = os.fstat(descriptor)
        if root_info.st_uid != 0:
            raise SetupError("ROOTLESS_T1_INPUT_ANCESTOR_UNSAFE")
        _safe_directory(root_info)
        for component in target.parts[1:]:
            next_descriptor = os.open(component, DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
            _safe_directory(os.fstat(descriptor))
        if os.fstat(descriptor).st_uid != os.getuid():
            raise SetupError("ROOTLESS_T1_INPUT_ROOT_UNSAFE")
        return descriptor
    except (OSError, SetupError):
        os.close(descriptor)
        raise


def _read_descriptor(descriptor: int) -> bytes:
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise SetupError("ROOTLESS_T1_INPUT_FILE_UNSAFE")
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1_048_576):
        chunks.append(chunk)
    return b"".join(chunks)


def _read_relative(root: int, relative_path: Path) -> bytes:
    directory = os.dup(root)
    try:
        for component in relative_path.parts[:-1]:
            next_directory = os.open(component, DIRECTORY_FLAGS, dir_fd=directory)
            os.close(directory)
            directory = next_directory
            _safe_directory(os.fstat(directory))
        descriptor = os.open(relative_path.name, FILE_FLAGS, dir_fd=directory)
        try:
            return _read_descriptor(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def _validate_pin(pin: InputPin, raw: bytes) -> None:
    if len(raw) != pin.size_bytes or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), pin.sha256
    ):
        raise SetupError("ROOTLESS_T1_INPUT_PIN_MISMATCH")


def _destination_parent(root: int, relative_path: Path, *, create: bool) -> int | None:
    directory = os.dup(root)
    try:
        for component in relative_path.parts[:-1]:
            if create:
                try:
                    os.mkdir(component, 0o700, dir_fd=directory)
                except FileExistsError:
                    pass
                else:
                    os.fsync(directory)
            try:
                next_directory = os.open(component, DIRECTORY_FLAGS, dir_fd=directory)
            except FileNotFoundError:
                return None
            os.close(directory)
            directory = next_directory
            info = os.fstat(directory)
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700
            ):
                raise SetupError("ROOTLESS_T1_DESTINATION_UNSAFE")
        return os.dup(directory)
    finally:
        os.close(directory)


def _existing_destination(root: int, pin: InputPin, source_raw: bytes) -> bool:
    parent = _destination_parent(root, pin.relative_path, create=False)
    if parent is None:
        return False
    try:
        try:
            descriptor = os.open(pin.relative_path.name, FILE_FLAGS, dir_fd=parent)
        except FileNotFoundError:
            return False
        try:
            raw = _read_descriptor(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)
    _validate_pin(pin, raw)
    if not hmac.compare_digest(raw, source_raw):
        raise SetupError("ROOTLESS_T1_INPUT_DRIFT")
    return True


def _write_destination(root: int, pin: InputPin, raw: bytes) -> None:
    parent = _destination_parent(root, pin.relative_path, create=True)
    if parent is None:
        raise SetupError("ROOTLESS_T1_DESTINATION_UNSAFE")
    temporary = f".{pin.relative_path.name}.tmp"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=parent,
        )
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(descriptor, raw[offset:])
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(
            temporary,
            pin.relative_path.name,
            src_dir_fd=parent,
            dst_dir_fd=parent,
            follow_symlinks=False,
        )
        os.fsync(parent)
        os.unlink(temporary, dir_fd=parent)
        os.fsync(parent)
    finally:
        os.close(parent)
    observed = _read_relative(root, pin.relative_path)
    _validate_pin(pin, observed)
    if not hmac.compare_digest(observed, raw):
        raise SetupError("ROOTLESS_T1_INPUT_DRIFT")


def setup_inputs(repository_root: Path, source_root: Path) -> None:
    source_descriptor = -1
    repository_descriptor = -1
    try:
        source_descriptor = _open_root(source_root)
        repository_descriptor = _open_root(repository_root)
        sources: list[bytes] = []
        for pin in INPUT_PINS:
            raw = _read_relative(source_descriptor, pin.relative_path)
            _validate_pin(pin, raw)
            sources.append(raw)
        existing = tuple(
            _existing_destination(repository_descriptor, pin, raw)
            for pin, raw in zip(INPUT_PINS, sources, strict=True)
        )
        for pin, raw, present in zip(INPUT_PINS, sources, existing, strict=True):
            if not present:
                _write_destination(repository_descriptor, pin, raw)
    except OSError as error:
        raise SetupError("ROOTLESS_T1_INPUT_OPEN_FAILED") from error
    finally:
        if repository_descriptor >= 0:
            os.close(repository_descriptor)
        if source_descriptor >= 0:
            os.close(source_descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        setup_inputs(arguments.repo_root, arguments.source_root)
    except (OSError, SetupError) as error:
        print(error, file=sys.stderr)
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
