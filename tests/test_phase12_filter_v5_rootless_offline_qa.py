from __future__ import annotations

import importlib.util
from pathlib import Path
import socket
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "run_phase12_rootless_offline_qa.py"
    spec = importlib.util.spec_from_file_location("phase12_offline_qa", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audit_policy_allows_unix_and_denies_inet_dns_and_processes() -> None:
    module = _module()
    policy = module.AuditPolicy(module.ProcessPolicy.DENY_ALL, None)

    policy("socket.__new__", (None, socket.AF_UNIX, socket.SOCK_STREAM, 0))
    with pytest.raises(module.OfflineQADenied):
        policy("socket.__new__", (None, socket.AF_INET, socket.SOCK_STREAM, 0))
    with pytest.raises(module.OfflineQADenied):
        policy("socket.getaddrinfo", ("example.invalid", 443, 0, 0, 0))
    with pytest.raises(module.OfflineQADenied):
        policy("subprocess.Popen", ("python", ("python",), None, None))


def test_environment_contract_rejects_provider_proxy_or_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setenv("OPENAI_API_KEY", "sentinel")

    with pytest.raises(module.OfflineQADenied):
        module.validate_environment(tmp_path)


def test_fixed_ruff_policy_allows_exactly_one_bound_executable(tmp_path: Path) -> None:
    module = _module()
    executable = tmp_path / "ruff"
    executable.write_bytes(b"native")
    digest = __import__("hashlib").sha256(executable.read_bytes()).hexdigest()
    policy = module.AuditPolicy(
        module.ProcessPolicy.FIXED_RUFF_EXEC, executable.resolve(), tmp_path.resolve(), digest
    )
    argv = (str(executable), "check", "--no-cache", "src", "tests", "scripts")

    policy("subprocess.Popen", (str(executable), argv, str(tmp_path), {"LC_ALL": "C", "PATH": "/usr/bin:/bin"}))

    with pytest.raises(module.OfflineQADenied):
        policy("subprocess.Popen", (str(executable), argv, str(tmp_path), {"LC_ALL": "C", "PATH": "/usr/bin:/bin"}))
