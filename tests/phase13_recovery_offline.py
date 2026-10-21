"""Opt-in pytest plugin; guarded child: python -B -m tests.phase13_recovery_offline SCRIPT."""

from __future__ import annotations

import os
import importlib
import runpy
import socket
import subprocess
import sys
from collections.abc import Iterator
from typing import Never

import pytest

from memcontam.clients.openai_compatible import OpenAICompatibleClient
from memcontam.clients.openai_responses import OpenAIResponsesClient


class OfflineTransportDenied(AssertionError):
    """A recovery test attempted an external transport or process launch."""


def _deny(*_args, **_kwargs) -> Never:
    raise OfflineTransportDenied("RECOVERY_OFFLINE_TRANSPORT_DENIED")


def install_offline_denial(monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    """Leave native producers intact; deny only external I/O and launch seams."""
    removed = tuple(sorted(name for name in os.environ if any(
        marker in name.upper() for marker in ("API_KEY", "ACCESS_TOKEN", "AUTH_TOKEN", "SECRET", "CREDENTIAL")
    )))
    for name in removed:
        monkeypatch.delenv(name)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE"):
        monkeypatch.setenv(name, "1")
    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, name, _deny)
    monkeypatch.setattr(socket, "create_connection", _deny)
    monkeypatch.setattr(socket, "getaddrinfo", _deny)
    for name in ("chat", "count_compiled_v3", "send_compiled_v3"):
        monkeypatch.setattr(OpenAIResponsesClient, name, _deny)
    monkeypatch.setattr(OpenAICompatibleClient, "chat", _deny)
    original_popen = subprocess.Popen

    def guarded_popen(args, *positional, **kwargs):
        if (not isinstance(args, (list, tuple)) or positional
            or kwargs.get("shell", False) or kwargs.get("executable") is not None):
            _deny()
        python_reopen = (len(args) == 5 and list(args[:4]) == [
            sys.executable, "-B", "-m", "tests.phase13_recovery_offline",
        ])
        cache_probe = (list(args) == ["/sbin/ldconfig", "-p"]
                       and os.path.realpath("/sbin/ldconfig") == "/usr/sbin/ldconfig")
        if not (python_reopen or cache_probe):
            _deny()
        return original_popen(args, *positional, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", guarded_popen)
    monkeypatch.setattr(os, "system", _deny)
    return removed


@pytest.fixture(autouse=True)
def offline_recovery() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        install_offline_denial(monkeypatch)
        yield


if __name__ == "__main__":
    plugin = importlib.import_module("tests.phase13_recovery_offline")
    with pytest.MonkeyPatch.context() as child_patch:
        plugin.install_offline_denial(child_patch)
        runpy.run_path(sys.argv[1], run_name="__main__")
