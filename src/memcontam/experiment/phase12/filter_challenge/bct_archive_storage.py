from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Final, Mapping

from memcontam.experiment.phase12.filter_challenge.bct_archive_models import LedgerError


_ZERO_HASH: Final = "0" * 64


class _LockedJsonl:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._stream = None

    def __enter__(self):
        self._stream = self._path.open("a+", encoding="utf-8")
        fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX)
        return self._stream

    def __exit__(self, exc_type, exc, traceback) -> None:
        assert self._stream is not None
        fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise LedgerError("LEDGER_CHAIN_INVALID")
    return value


def _string_value(value: object) -> str:
    if not isinstance(value, str):
        raise LedgerError("LEDGER_CHAIN_INVALID")
    return value


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
