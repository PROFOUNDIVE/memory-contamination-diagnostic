"""Authoritative V3 inventories; AST/import reachability is never an input."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from memcontam.readiness.phase13_authority_files import AuthorityFileError, authority_directory
from memcontam.readiness.phase13_cost_policy_models import Sha256
from memcontam.readiness.phase13_v3_authority_models import FrozenModel
from memcontam.readiness.phase13_v3_resource_files import (
    FIXED_GOVERNED, ClosureError as ClosureError, FileBinding, ValidatedResource,
    is_governed, normalized_paths, read_files,
)


class GovernedInventory(FrozenModel):
    governed_source_commit: str
    rows: tuple[FileBinding, ...]
    governed_tree_sha256: Sha256


class ResourceClosure(FrozenModel):
    rows: tuple[FileBinding, ...]
    resource_closure_sha256: Sha256


def _rows_hash(rows: tuple[FileBinding, ...]) -> str:
    raw = json.dumps([row.model_dump() for row in rows], ensure_ascii=False,
                     sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    return hashlib.sha256(raw).hexdigest()


def _git(directory: int, arguments: tuple[str, ...]) -> bytes:
    result = subprocess.run(
        ("git", "--no-replace-objects", "-C", f"/proc/self/fd/{directory}", *arguments),
        pass_fds=(directory,), check=False, capture_output=True,
        env={**{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
             "GIT_MASTER": "1", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"},
    )
    if result.returncode:
        raise ClosureError("MAIN_GOVERNED_SOURCE_DRIFT")
    return result.stdout


def _committed_rows(root: Path, commit: str) -> tuple[FileBinding, ...]:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ClosureError("MAIN_GOVERNED_SOURCE_DRIFT")
    try:
        with authority_directory(root) as directory:
            if _git(directory, ("cat-file", "-t", commit)) != b"commit\n":
                raise ClosureError("MAIN_GOVERNED_SOURCE_DRIFT")
            _git(directory, ("merge-base", "--is-ancestor", commit, "HEAD"))
            tree = _git(directory, ("ls-tree", "-rz", "--full-tree", commit))
            rows: list[FileBinding] = []
            for record in tree.split(b"\x00"):
                if not record:
                    continue
                metadata, raw_path = record.split(b"\t", 1)
                path = raw_path.decode("utf-8", errors="strict")
                if not is_governed(path):
                    continue
                mode, kind, oid = metadata.split()
                if mode not in {b"100644", b"100755"} or kind != b"blob":
                    raise ClosureError("MAIN_PATH_UNSAFE")
                raw = _git(directory, ("cat-file", "blob", oid.decode("ascii")))
                rows.append(FileBinding(path=path, size=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            names = normalized_paths(tuple(row.path for row in rows))
            if not set(FIXED_GOVERNED) <= set(names) or not any(
                path.startswith("src/memcontam/") for path in names
            ):
                raise ClosureError("MAIN_GOVERNED_SOURCE_DRIFT")
            return tuple(sorted(rows, key=lambda row: row.path.encode("utf-8")))
    except (UnicodeError, AuthorityFileError, OSError) as error:
        raise ClosureError("MAIN_PATH_UNSAFE") from error


def freeze_governed(root: Path, commit: str) -> GovernedInventory:
    rows = _committed_rows(root, commit)
    try:
        current = tuple(item.binding for item in read_files(root, FIXED_GOVERNED, governed=True))
    except ClosureError as error:
        if isinstance(error.__cause__, FileNotFoundError):
            raise ClosureError("MAIN_GOVERNED_SOURCE_DRIFT") from error
        raise
    if current != rows:
        raise ClosureError("MAIN_GOVERNED_SOURCE_DRIFT")
    return GovernedInventory(governed_source_commit=commit, rows=rows,
                             governed_tree_sha256=_rows_hash(rows))


def validate_governed(root: Path, inventory: GovernedInventory) -> GovernedInventory:
    normalized_paths(tuple(row.path for row in inventory.rows))
    actual = freeze_governed(root, inventory.governed_source_commit)
    if actual != inventory:
        raise ClosureError("MAIN_GOVERNED_SOURCE_DRIFT")
    return actual


def _resource_names(paths: tuple[str, ...]) -> tuple[str, ...]:
    names = normalized_paths(paths)
    if any(is_governed(path) or path in {
        "data/phase13/main/mr_p5/execution_package_v3.json",
        "data/phase13/main/mr_p6/authorized_execution_v3.json",
        "data/phase13/main/mr_p6/authorized_execution_v3.sha256",
    } for path in names):
        raise ClosureError("MAIN_GOVERNED_SOURCE_DRIFT")
    return names


def freeze_resources(root: Path, paths: tuple[str, ...]) -> ResourceClosure:
    rows = tuple(item.binding for item in read_files(root, _resource_names(paths)))
    return ResourceClosure(rows=rows, resource_closure_sha256=_rows_hash(rows))


def validate_resources(
    root: Path, closure: ResourceClosure, expected_paths: tuple[str, ...],
) -> tuple[ValidatedResource, ...]:
    """Return immutable verified bytes, not paths for downstream reopening.

    expected_paths comes from the consuming contract, never from closure.rows or imports.
    Unrelated artifacts elsewhere in the repository are not closure members.
    """
    names = _resource_names(tuple(row.path for row in closure.rows))
    if names != _resource_names(expected_paths) or _rows_hash(closure.rows) != closure.resource_closure_sha256:
        raise ClosureError("MAIN_GOVERNED_SOURCE_DRIFT")
    resources = read_files(root, names)
    if tuple(item.binding for item in resources) != closure.rows:
        raise ClosureError("MAIN_GOVERNED_SOURCE_DRIFT")
    return resources
