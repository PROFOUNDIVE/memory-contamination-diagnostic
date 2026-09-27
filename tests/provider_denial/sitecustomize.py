from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess


def _deny_external(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("PHASE13_EXTERNAL_NETWORK_FORBIDDEN")


socket.socket.connect = _deny_external
socket.create_connection = _deny_external
socket.getaddrinfo = _deny_external


_DENIAL_ROOT = str(Path(__file__).resolve().parent)
_SOURCE_PATH = os.pathsep.join(
    str(Path(path).resolve())
    for path in os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if path and str(Path(path).resolve()) != _DENIAL_ROOT
)
os.environ["PYTHONPATH"] = _SOURCE_PATH

_popen_init = subprocess.Popen.__init__


def _guarded_popen_init(self, *args, **kwargs):
    environment = dict(kwargs.get("env") or os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((_DENIAL_ROOT, _SOURCE_PATH))
    kwargs["env"] = environment
    _popen_init(self, *args, **kwargs)


subprocess.Popen.__init__ = _guarded_popen_init
