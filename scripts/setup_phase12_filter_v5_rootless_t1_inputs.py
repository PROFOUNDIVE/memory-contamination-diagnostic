from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path


INPUTS = (
    ".omo/plans/phase12-filter-v5-screening-bct-execution.md",
    ".omo/approvals/phase12-filter-v5-screening-bct-execution.plan.sha256",
    ".omo/approvals/phase12-post-filter-v5-calibration-readiness.plan.sha256",
    ".omo/evidence/phase12-post-filter-v5-calibration-readiness/task-3-screening-stage-result.json",
    ".omo/evidence/phase12-post-filter-v5-calibration-readiness/task-5-bct-stage-result.json",
    ".omo/evidence/phase12-post-filter-v5-calibration-readiness/task-6-pilot-b-readiness-stage-result.json",
)


def _read(path: Path) -> bytes:
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise RuntimeError("ROOTLESS_T1_INPUT_PATH_INVALID")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:-1]:
            next_descriptor = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        file_descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
        try:
            info = os.fstat(file_descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid not in {0, os.getuid()}:
                raise RuntimeError("ROOTLESS_T1_INPUT_UNSAFE")
            chunks: list[bytes] = []
            while chunk := os.read(file_descriptor, 1_048_576):
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(file_descriptor)
    finally:
        os.close(descriptor)


def _write(destination: Path, raw: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination.parent, 0o700)
    if destination.exists():
        if _read(destination) != raw:
            raise RuntimeError("ROOTLESS_T1_INPUT_DRIFT")
        return
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        file_descriptor = os.open(destination.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=descriptor)
        try:
            offset = 0
            while offset < len(raw):
                offset += os.write(file_descriptor, raw[offset:])
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        repository_root = arguments.repo_root.resolve(strict=True)
        source_root = arguments.source_root.resolve(strict=True)
        if not repository_root.is_dir() or not source_root.is_dir():
            raise RuntimeError("ROOTLESS_T1_INPUT_ROOT_INVALID")
        os.chmod(repository_root, stat.S_IMODE(repository_root.stat().st_mode) & ~0o022)
        for relative in INPUTS:
            _write(repository_root / relative, _read(source_root / relative))
    except (OSError, RuntimeError) as error:
        print(error, file=os.sys.stderr)
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
