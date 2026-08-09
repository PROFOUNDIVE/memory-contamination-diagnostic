from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
from typing import Final


PROFILE: Final = "local_rootless_non_authoritative"
_EMPTY_ENV: Final = {"LC_ALL": "C", "PATH": "/usr/bin:/bin"}
_SENTINEL_ROLES: Final = frozenset(
    {"f1-pytest", "f2-pytest", "f3-pytest", "f4-rootless-pytest", "f4-ruff", "f4-validate-config", "f4-replay-pytest"}
)
_BASETEMP_ROLES: Final = {
    "f1-pytest": "f1",
    "f2-pytest": "f2",
    "f3-pytest": "f3",
    "f4-rootless-pytest": "f4-rootless",
    "f4-ruff": "f4-ruff",
    "f4-validate-config": "f4-validate-config",
    "f4-replay-pytest": "f4-replay",
}
_FORBIDDEN_ENV: Final = frozenset(
    {"OPENAI_API_KEY", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"}
)


class OfflineQADenied(RuntimeError):
    pass


class ProcessPolicy(StrEnum):
    DENY_ALL = "deny_all"
    SCRUBBED_TEST_EXEC = "scrubbed_test_exec"
    FIXED_RUFF_EXEC = "fixed_ruff_exec"


@dataclass(slots=True)
class AuditPolicy:
    process_policy: ProcessPolicy
    ruff_executable: Path | None
    repository: Path | None = None
    executable_sha256: str | None = None
    process_count: int = 0

    def __call__(self, event: str, arguments: tuple[object, ...]) -> None:
        if event == "socket.__new__" and len(arguments) > 1 and arguments[1] in {socket.AF_INET, socket.AF_INET6}:
            raise OfflineQADenied("network denied")
        if event.startswith(("socket.getaddr", "socket.gethostby", "socket.getnameinfo")):
            raise OfflineQADenied("dns denied")
        if event in {"socket.connect", "socket.bind", "socket.listen"}:
            family = getattr(arguments[0], "family", None) if arguments else None
            if family in {socket.AF_INET, socket.AF_INET6}:
                raise OfflineQADenied("network denied")
        if event == "subprocess.Popen":
            if self.process_policy is ProcessPolicy.DENY_ALL:
                raise OfflineQADenied("process denied")
            if self.process_policy is ProcessPolicy.SCRUBBED_TEST_EXEC:
                cwd = self.repository if arguments[2] is None else Path(arguments[2]).resolve(strict=True)
                environment = arguments[3]
                if (
                    self.repository is None
                    or not cwd.is_relative_to(self.repository)
                    or environment is not None
                    and any(environment.get(name) for name in _FORBIDDEN_ENV)
                ):
                    raise OfflineQADenied("process denied")
                self.process_count += 1
                return
            if self.process_count or self.ruff_executable is None:
                raise OfflineQADenied("process denied")
            executable = Path(os.fsdecode(arguments[0])).resolve(strict=True)
            expected_argv = (
                os.fspath(self.ruff_executable), "check", "--no-cache", "src", "tests", "scripts"
            )
            if (
                executable != self.ruff_executable
                or len(arguments) != 4
                or tuple(arguments[1]) != expected_argv
                or Path(arguments[2]).resolve(strict=True) != self.repository
                or arguments[3] != _EMPTY_ENV
                or self.executable_sha256 is None
                or hashlib.sha256(executable.read_bytes()).hexdigest() != self.executable_sha256
            ):
                raise OfflineQADenied("process denied")
            self.process_count = 1
        if event == "os.system" or event.startswith("os.spawn") or event.startswith("os.posix_spawn"):
            raise OfflineQADenied("process denied")


def validate_environment(repository: Path) -> None:
    if repository.resolve(strict=True) != repository or any(os.environ.get(name) for name in _FORBIDDEN_ENV):
        raise OfflineQADenied("environment denied")


def _network_probes() -> None:
    probes = (
        lambda: socket.getaddrinfo("api.openai.com", 443),
        lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM),
        lambda: socket.socket(socket.AF_INET6, socket.SOCK_STREAM),
        lambda: socket.create_connection(("api.openai.com", 443)),
    )
    for probe in probes:
        try:
            probe()
        except OfflineQADenied:
            continue
        raise OfflineQADenied("network probe reached syscall")


def _seed_tokenizer_cache(temporary: Path) -> None:
    source = Path("/tmp/data-gym-cache")
    destination = temporary / "data-gym-cache"
    if not source.is_dir() or source.is_symlink():
        return
    destination.mkdir(mode=0o700)
    for path in source.iterdir():
        if path.is_file() and not path.is_symlink():
            target = destination / path.name
            target.write_bytes(path.read_bytes())
            target.chmod(0o600)


def _ruff_executable() -> tuple[Path, str]:
    distribution = metadata.distribution("ruff")
    record = distribution.locate_file(f"{distribution.metadata['Name'].replace('-', '_')}-{distribution.version}.dist-info/RECORD")
    raw = Path(record).read_bytes()
    candidates: list[tuple[Path, str]] = []
    for row in csv.reader(io.StringIO(raw.decode("utf-8")), strict=True):
        if len(row) != 3 or not row[0].replace("\\", "/").endswith("/bin/ruff") or not row[1].startswith("sha256="):
            continue
        path = distribution.locate_file(row[0]).resolve(strict=True)
        padding = "=" * (-len(row[1][7:]) % 4)
        expected = base64.urlsafe_b64decode(row[1][7:] + padding).hex()
        candidates.append((Path(path), expected))
    if len(candidates) != 1:
        raise OfflineQADenied("ruff RECORD denied")
    executable, expected = candidates[0]
    info = executable.lstat()
    prefix = Path(sys.prefix).resolve(strict=True)
    if (
        executable.is_symlink()
        or not executable.is_relative_to(prefix)
        or not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or not info.st_mode & 0o111
        or hashlib.sha256(executable.read_bytes()).hexdigest() != expected
    ):
        raise OfflineQADenied("ruff executable denied")
    return executable, expected


def _sentinel(repository: Path, role: str, module: str, policy: ProcessPolicy, digest: str | None) -> None:
    path = repository / "runs/phase12-filter-v5-rootless-qa/final/sentinels" / f"{role}.json"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = {
        "schema_version": "rootless_network_denial_sentinel_v1",
        "profile": PROFILE,
        "kind": "network_denial_sentinel",
        "role": role,
        "module": module,
        "process_policy": policy.value,
        "allowed_executable_sha256": digest,
        "dns_status": "denied_by_audit",
        "tcp_ipv4_status": "denied_by_audit",
        "tcp_ipv6_status": "denied_by_audit",
        "provider_client_status": "denied_by_audit",
        "syscalls_reached": 0,
        "created_at": datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_residue(repository: Path) -> None:
    directories = {"__pycache__", ".pytest_cache", ".ruff_cache"}
    for path in sorted(repository.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if ".git" in path.parts:
            continue
        if path.name in directories:
            if path.is_symlink():
                raise OfflineQADenied("cache residue unsafe")
            shutil.rmtree(path)
        elif path.suffix in {".pyc", ".pyo"}:
            if path.is_symlink() or not path.is_file():
                raise OfflineQADenied("bytecode residue unsafe")
            path.unlink()


def _run(arguments: argparse.Namespace) -> int:
    repository = arguments.repo_root.resolve(strict=True)
    if repository != arguments.repo_root or arguments.sentinel_role not in _SENTINEL_ROLES:
        raise OfflineQADenied("argument denied")
    validate_environment(repository)
    os.chdir(repository)
    role_root = repository / "runs/phase12-filter-v5-rootless-qa/basetemp" / _BASETEMP_ROLES[arguments.sentinel_role]
    pytest_root = role_root / "pytest"
    temporary = role_root / "tmp"
    previous_umask = os.umask(0o077)
    role_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    pytest_root.mkdir(mode=0o700)
    temporary.mkdir(mode=0o700)
    _seed_tokenizer_cache(temporary)
    digest: str | None = None
    policy = ProcessPolicy.DENY_ALL
    try:
        os.environ.clear()
        os.environ.update(_EMPTY_ENV | {"TMPDIR": os.fspath(temporary), "TMP": os.fspath(temporary), "TEMP": os.fspath(temporary)})
        executable: Path | None = None
        if arguments.module == "ruff":
            if arguments.sentinel_role != "f4-ruff" or arguments.target != ["check", "--no-cache", "src", "tests", "scripts"]:
                raise OfflineQADenied("ruff arguments denied")
            executable, digest = _ruff_executable()
            policy = ProcessPolicy.FIXED_RUFF_EXEC
        elif arguments.module == "pytest":
            policy = ProcessPolicy.SCRUBBED_TEST_EXEC
        audit = AuditPolicy(policy, executable, repository, digest)
        sys.addaudithook(audit)
        _network_probes()
        if arguments.module == "pytest":
            import pytest

            expected = ["-p", "no:cacheprovider", "--basetemp", os.fspath(pytest_root)]
            if arguments.target[:4] != expected:
                raise OfflineQADenied("pytest arguments denied")
            result = int(pytest.main(arguments.target))
        elif arguments.module == "memcontam.cli":
            from memcontam import cli

            previous = sys.argv
            try:
                sys.argv = ["memcontam", *arguments.target]
                cli.main()
            finally:
                sys.argv = previous
            result = 0
        elif executable is not None and digest is not None:
            if hashlib.sha256(executable.read_bytes()).hexdigest() != digest:
                raise OfflineQADenied("ruff changed")
            with Path("/dev/null").open("rb") as stdin:
                result = subprocess.run(
                    (os.fspath(executable), *arguments.target),
                    cwd=repository,
                    env=_EMPTY_ENV,
                    stdin=stdin,
                    close_fds=True,
                    check=False,
                ).returncode
        else:
            raise OfflineQADenied("module denied")
    finally:
        shutil.rmtree(role_root)
        os.umask(previous_umask)
    if result == 0:
        _remove_residue(repository)
        _sentinel(repository, arguments.sentinel_role, arguments.module, policy, digest)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--sentinel-role", required=True)
    parser.add_argument("--module", choices=("pytest", "ruff", "memcontam.cli"), required=True)
    parser.add_argument("target", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.target[:1] == ["--"]:
        arguments.target = arguments.target[1:]
    try:
        return _run(arguments)
    except (OfflineQADenied, OSError, ValueError):
        return 64


if __name__ == "__main__":
    raise SystemExit(main())
