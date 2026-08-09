from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Final

from memcontam.experiment.phase12.filter_challenge.rootless_local_acknowledgement import (
    create_skip_receipt,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    canonical_json_file,
    parse_canonical_object,
    sign_object,
)


PROFILE: Final = "local_rootless_non_authoritative"
_GIT_SHA: Final = re.compile(r"[0-9a-f]{40}\Z")
_QA_RELATIVE: Final = Path("runs/phase12-filter-v5-rootless-qa")


def qa_root(repository: Path) -> Path:
    return repository / _QA_RELATIVE


def state_root(state_home: Path | None) -> Path:
    if state_home is None:
        raise RootlessContractError("ROOTLESS_STATE_PATH_INVALID")
    return state_home / "memcontam/phase12-filter-v5-rootless-local"


def read_canonical(path: Path) -> dict[str, JsonValue]:
    return parse_canonical_object(path.read_bytes())


def write_new_or_same(path: Path, value: dict[str, JsonValue]) -> str:
    raw = canonical_json_file(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    previous = os.umask(0o077)
    try:
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
            )
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != raw:
                raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT") from None
        else:
            try:
                offset = 0
                while offset < len(raw):
                    offset += os.write(descriptor, raw[offset:])
                os.fchmod(descriptor, 0o600)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
    finally:
        os.umask(previous)
    return hashlib.sha256(raw).hexdigest()


def write_anchor(repository: Path, execution_commit: str) -> str:
    if _GIT_SHA.fullmatch(execution_commit) is None:
        raise RootlessContractError("ROOTLESS_EXECUTION_COMMIT_INVALID")
    return write_new_or_same(
        qa_root(repository) / "pre-egress/execution-anchor.json",
        {
            "schema_version": "rootless_execution_anchor_v1",
            "profile": PROFILE,
            "execution_commit": execution_commit,
        },
    )


def finalize_preclaim(
    repository: Path,
    state_home: Path | None,
    attempt_id: str,
    execution_commit: str,
    failed_command: str,
    observed_exit: int,
    created_at: str,
) -> str:
    write_anchor(repository, execution_commit)
    root = state_root(state_home)
    if (root / f"claims/{attempt_id}.json").exists() or (root / "live-attempt-claim.json").exists():
        raise RootlessContractError("ROOTLESS_CLAIM_ALREADY_EXISTS")
    seed_path = root / "keys/ed25519-private.key"
    seed = seed_path.read_bytes() if seed_path.is_file() and seed_path.stat().st_mode & 0o777 == 0o600 else None
    plan_path = root / "plan-bind.md"
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest() if plan_path.is_file() else None
    value = create_skip_receipt(
        reason="ROOTLESS_PRECLAIM_ADMIN_FAILED",
        missing_input_role=None,
        attempt_id=attempt_id,
        reviewed_plan_sha256=plan_hash,
        created_at=created_at,
        failed_command=failed_command,
        observed_exit=observed_exit,
        seed=seed,
    )
    return write_new_or_same(qa_root(repository) / "pre-egress/zero-call-skip.json", value)


def record_t7(repository: Path, state_home: Path | None = None) -> str:
    root = qa_root(repository)
    skip = root / "pre-egress/zero-call-skip.json"
    publication = repository / "docs/evidence/phase12-filter-v5-rootless-local/rehearsal-publication.json"
    if skip.is_file() == publication.is_file():
        raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
    if skip.is_file():
        source = read_canonical(skip)
        source_hash = hashlib.sha256(skip.read_bytes()).hexdigest()
        attempt = source.get("attempt_id")
        locator: dict[str, JsonValue] = {
            "schema_version": "rootless_f3_state_locator_v1", "profile": PROFILE,
            "kind": "f3_state_locator", "outcome": "zero_call_skip", "attempt_id": attempt,
            "state_home": None, "publication_receipt_sha256": None,
            "state_inventory_sha256": None, "zero_call_skip_sha256": source_hash,
        }
        locator_hash = write_new_or_same(root / "f3-state-locator.json", locator)
        t7: dict[str, JsonValue] = {
            "schema_version": "rootless_t7_attempt_qa_v1", "profile": PROFILE,
            "outcome": "zero_call_skip", "attempt_id": attempt,
            "zero_call_skip_sha256": source_hash, "publication_receipt_sha256": None,
            "final_terminal_sha256": None, "state_inventory_sha256": None,
            "screening_terminal_sha256": None, "bct_terminal_sha256": None,
            "bct_result_manifest_sha256": None, "provider_calls_issued": 0,
            "screening_nanousd": 0, "bct_nanousd": 0, "cumulative_nanousd": 0,
            "f3_state_locator_sha256": locator_hash, "created_at": source["created_at"],
        }
    else:
        source = read_canonical(publication)
        publication_hash = hashlib.sha256(publication.read_bytes()).hexdigest()
        locator = {
            "schema_version": "rootless_f3_state_locator_v1", "profile": PROFILE,
            "kind": "f3_state_locator", "outcome": "paid_attempt",
            "attempt_id": source["attempt_id"], "state_home": os.fspath(state_home) if state_home else None,
            "publication_receipt_sha256": publication_hash,
            "state_inventory_sha256": source["state_inventory_sha256"],
            "zero_call_skip_sha256": None,
        }
        locator_hash = write_new_or_same(root / "f3-state-locator.json", locator)
        t7 = {
            "schema_version": "rootless_t7_attempt_qa_v1", "profile": PROFILE,
            "outcome": "paid_attempt", "attempt_id": source["attempt_id"],
            "zero_call_skip_sha256": None, "publication_receipt_sha256": publication_hash,
            "final_terminal_sha256": source["final_terminal_sha256"],
            "state_inventory_sha256": source["state_inventory_sha256"],
            "screening_terminal_sha256": source["screening_terminal_sha256"],
            "bct_terminal_sha256": source["bct_terminal_sha256"],
            "bct_result_manifest_sha256": source["bct_result_manifest_sha256"],
            "provider_calls_issued": source.get("provider_calls_issued", 0),
            "screening_nanousd": source.get("screening_nanousd", 0),
            "bct_nanousd": source.get("bct_nanousd", 0),
            "cumulative_nanousd": source.get("cumulative_nanousd", 0),
            "f3_state_locator_sha256": locator_hash, "created_at": source["published_at"],
        }
    return write_new_or_same(root / "t7-real-attempt.json", t7)


def publish(repository: Path, state_home: Path | None, attempt_id: str, published_at: str) -> str:
    root = state_root(state_home)
    final_path = root / f"terminals/{attempt_id}/final.json"
    screening_path = root / f"terminals/{attempt_id}/screening.json"
    bct_path = root / f"terminals/{attempt_id}/bct.json"
    inventory_path = root / f"attempts/{attempt_id}/state-inventory.json"
    final = read_canonical(final_path)
    read_canonical(inventory_path)
    read_canonical(screening_path)
    bct = read_canonical(bct_path) if bct_path.is_file() else None
    seed = (root / "keys/ed25519-private.key").read_bytes()
    anchor = read_canonical(qa_root(repository) / "pre-egress/execution-anchor.json")
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_publication_receipt_v1", "profile": PROFILE,
        "kind": "publication_receipt", "attempt_id": attempt_id,
        "plan_binding_sha256": hashlib.sha256((root / "plan-bind.md").read_bytes()).hexdigest(),
        "execution_commit": anchor["execution_commit"],
        "final_terminal_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        "screening_terminal_sha256": hashlib.sha256(screening_path.read_bytes()).hexdigest(),
        "bct_terminal_sha256": hashlib.sha256(bct_path.read_bytes()).hexdigest() if bct else None,
        "bct_result_manifest_sha256": bct.get("bct_result_manifest_sha256") if bct else None,
        "ledger_head_sha256": final["ledger_head_sha256"],
        "state_inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "published_at": published_at,
        "key_fingerprint": final["key_fingerprint"],
    }
    value = signed(seed, "publication-receipt-v1", payload)
    destination = repository / "docs/evidence/phase12-filter-v5-rootless-local/rehearsal-publication.json"
    return write_new_or_same(destination, value)


def seal_final_from_stage(
    root: Path,
    attempt_id: str,
    stage: str,
    status: str,
    reason: str,
    created_at: str,
) -> str:
    stage_path = root / f"terminals/{attempt_id}/{stage}.json"
    terminal = read_canonical(stage_path)
    seed = (root / "keys/ed25519-private.key").read_bytes()
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_attempt_terminal_v1", "profile": PROFILE,
        "kind": "attempt_terminal", "attempt_id": attempt_id, "status": status,
        "reason_code": reason,
        "predecessor_stage_terminal_sha256": hashlib.sha256(stage_path.read_bytes()).hexdigest(),
        "ledger_head_sha256": terminal["ledger_head_sha256"], "created_at": created_at,
        "key_fingerprint": terminal["key_fingerprint"],
    }
    final_path = root / f"terminals/{attempt_id}/final.json"
    digest = write_new_or_same(final_path, signed(seed, "attempt-terminal-v1", payload))
    files: list[JsonValue] = []
    inventory_path = root / f"attempts/{attempt_id}/state-inventory.json"
    for path in sorted(root.rglob("*"), key=lambda item: os.fsencode(item.relative_to(root))):
        if (
            not path.is_file()
            or path == inventory_path
            or path.name == "runtime.lock"
            or path.name.startswith(".")
            or path.name == "current-head.json"
        ):
            continue
        raw = path.read_bytes()
        files.append(
            {
                "state_relative_path": path.relative_to(root).as_posix(),
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    inventory_payload: dict[str, JsonValue] = {
        "schema_version": "rootless_state_inventory_v1", "profile": PROFILE,
        "kind": "state_inventory", "attempt_id": attempt_id, "final_terminal_sha256": digest,
        "ordered_files": files, "created_at": created_at,
        "key_fingerprint": terminal["key_fingerprint"],
    }
    write_new_or_same(inventory_path, signed(seed, "state-inventory-v1", inventory_payload))
    return digest


def acquire_attempt_lock(root: Path) -> int:
    descriptor = os.open(root / "runtime.lock", os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        os.close(descriptor)
        raise RootlessContractError("ROOTLESS_RUNTIME_LOCK_INVALID")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise SystemExit(68) from None
    return descriptor


def signed(seed: bytes, domain: str, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    value = dict(payload)
    value["signature"] = sign_object(seed, domain, payload)
    return value
