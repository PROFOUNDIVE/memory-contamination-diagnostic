from __future__ import annotations

# allow: SIZE_OK — existing finalization and publication remain one reporting authority.

import fcntl
import hashlib
import os
from pathlib import Path
import re
import stat
from collections.abc import Mapping
from typing import Final

from memcontam.experiment.phase12.filter_challenge.rootless_local_acknowledgement import (
    create_skip_receipt,
    parse_timestamp,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    canonical_json_file,
    parse_canonical_object,
    public_key_from_seed,
    sign_object,
    verify_object_signature,
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
    temporary = path.with_name(f".{path.name}.tmp")
    previous = os.umask(0o077)
    try:
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
            )
        except FileExistsError:
            if _read_exact_regular(temporary) != raw:
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
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if _read_exact_regular(path) != raw:
                raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT") from None
        os.unlink(temporary)
        parent = os.open(
            path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        os.umask(previous)
    return hashlib.sha256(raw).hexdigest()


def _read_exact_regular(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_nlink not in {1, 2}
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1_048_576):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


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
    *,
    missing_input_role: str | None = None,
    external_authority_diagnostic: str | None = None,
) -> str:
    write_anchor(repository, execution_commit)
    root = state_root(state_home)
    if (root / f"claims/{attempt_id}.json").exists() or (root / "live-attempt-claim.json").exists():
        raise RootlessContractError("ROOTLESS_CLAIM_ALREADY_EXISTS")
    seed_path = root / "keys/ed25519-private.key"
    seed = seed_path.read_bytes() if seed_path.is_file() and seed_path.stat().st_mode & 0o777 == 0o600 else None
    plan_path = root / "plan-bind.md"
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest() if plan_path.is_file() else None
    external_failure = missing_input_role is not None
    value = create_skip_receipt(
        reason="ROOTLESS_MISSING_EXTERNAL_INPUT" if external_failure else "ROOTLESS_PRECLAIM_ADMIN_FAILED",
        missing_input_role=missing_input_role,
        attempt_id=attempt_id,
        reviewed_plan_sha256=plan_hash,
        created_at=created_at,
        external_authority_diagnostic=external_authority_diagnostic,
        failed_command=None if external_failure else failed_command,
        observed_exit=None if external_failure else observed_exit,
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
        f3_state_locator_sha256 = write_new_or_same(root / "f3-state-locator.json", locator)
        t7: dict[str, JsonValue] = {
            "schema_version": "rootless_t7_attempt_qa_v1", "profile": PROFILE,
            "outcome": "zero_call_skip", "attempt_id": attempt,
            "f3_state_locator_sha256": f3_state_locator_sha256,
            "zero_call_skip_sha256": source_hash, "publication_receipt_sha256": None,
            "final_terminal_sha256": None, "state_inventory_sha256": None,
            "screening_terminal_sha256": None, "bct_terminal_sha256": None,
            "bct_result_manifest_sha256": None, "provider_calls_issued": 0,
            "screening_nanousd": 0, "bct_nanousd": 0, "cumulative_nanousd": 0,
            "created_at": source["created_at"],
        }
    else:
        if state_home is None:
            raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
        state = state_root(state_home)
        seed = (state / "keys/ed25519-private.key").read_bytes()
        source = _read_signed(publication, seed, "publication-receipt-v1")
        publication_hash = hashlib.sha256(publication.read_bytes()).hexdigest()
        attempt_id = source.get("attempt_id")
        if not isinstance(attempt_id, str):
            raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
        screening = _read_signed(
            state / f"terminals/{attempt_id}/screening.json", seed, "stage-terminal-v1"
        )
        bct_path = state / f"terminals/{attempt_id}/bct.json"
        bct = _read_signed(bct_path, seed, "stage-terminal-v1") if bct_path.is_file() else None
        head_hash = source.get("ledger_head_sha256")
        head_paths = tuple((state / "ledger/global/heads").glob(f"*-{head_hash}.json"))
        if len(head_paths) != 1:
            raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
        head = _read_signed(head_paths[0], seed, "ledger-head-v1")
        provider_calls = _integer(screening, "provider_calls_issued") + (
            _integer(bct, "provider_calls_issued") if bct is not None else 0
        )
        screening_nanousd = _integer(screening, "settled_nanousd")
        bct_nanousd = _integer(bct, "settled_nanousd") if bct is not None else 0
        if (
            provider_calls != head.get("cumulative_issued")
            or screening_nanousd != head.get("screening_settled_nanousd")
            or bct_nanousd != head.get("bct_settled_nanousd")
        ):
            raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
        locator = {
            "schema_version": "rootless_f3_state_locator_v1", "profile": PROFILE,
            "kind": "f3_state_locator", "outcome": "paid_attempt",
            "attempt_id": source["attempt_id"], "state_home": os.fspath(state_home) if state_home else None,
            "publication_receipt_sha256": publication_hash,
            "state_inventory_sha256": source["state_inventory_sha256"],
            "zero_call_skip_sha256": None,
        }
        f3_state_locator_sha256 = write_new_or_same(root / "f3-state-locator.json", locator)
        t7 = {
            "schema_version": "rootless_t7_attempt_qa_v1", "profile": PROFILE,
            "outcome": "paid_attempt", "attempt_id": source["attempt_id"],
            "f3_state_locator_sha256": f3_state_locator_sha256,
            "zero_call_skip_sha256": None, "publication_receipt_sha256": publication_hash,
            "final_terminal_sha256": source["final_terminal_sha256"],
            "state_inventory_sha256": source["state_inventory_sha256"],
            "screening_terminal_sha256": source["screening_terminal_sha256"],
            "bct_terminal_sha256": source["bct_terminal_sha256"],
            "bct_result_manifest_sha256": source["bct_result_manifest_sha256"],
            "provider_calls_issued": provider_calls,
            "screening_nanousd": screening_nanousd,
            "bct_nanousd": bct_nanousd,
            "cumulative_nanousd": _integer(head, "cumulative_settled_nanousd"),
            "created_at": source["published_at"],
        }
    return write_new_or_same(root / "t7-real-attempt.json", t7)


def _integer(value: Mapping[str, JsonValue], field: str) -> int:
    result = value.get(field)
    if not isinstance(result, int) or isinstance(result, bool):
        raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
    return result


def publish(repository: Path, state_home: Path | None, attempt_id: str, published_at: str) -> str:
    return publish_from_root(repository, state_root(state_home), attempt_id, published_at)


def publish_from_root(
    repository: Path, root: Path, attempt_id: str, published_at: str
) -> str:
    final_path = root / f"terminals/{attempt_id}/final.json"
    screening_path = root / f"terminals/{attempt_id}/screening.json"
    bct_path = root / f"terminals/{attempt_id}/bct.json"
    inventory_path = root / f"attempts/{attempt_id}/state-inventory.json"
    seed = (root / "keys/ed25519-private.key").read_bytes()
    final = _read_signed(final_path, seed, "attempt-terminal-v1")
    inventory = _read_signed(inventory_path, seed, "state-inventory-v1")
    screening = _read_signed(screening_path, seed, "stage-terminal-v1")
    bct = _read_signed(bct_path, seed, "stage-terminal-v1") if bct_path.is_file() else None
    predecessor = bct if bct is not None else screening
    stage = "bct" if bct is not None else "screening"
    binding_path = root / f"bindings/{attempt_id}/{stage}.json"
    binding = read_canonical(binding_path) if binding_path.is_file() else None
    if binding is None or binding.get("transport_mode") != "live":
        raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
    plan_binding_sha256 = hashlib.sha256((root / "plan-bind.md").read_bytes()).hexdigest()
    ledger_head_sha256 = final.get("ledger_head_sha256")
    ledger_heads = tuple((root / "ledger/global/heads").glob(f"*-{ledger_head_sha256}.json"))
    inventory_entries = _inventory_entries(root, inventory_path)
    if ledger_heads:
        _read_signed(ledger_heads[0], seed, "ledger-head-v1")
    if (
        final.get("attempt_id") != attempt_id
        or screening.get("attempt_id") != attempt_id
        or inventory.get("attempt_id") != attempt_id
        or inventory.get("final_terminal_sha256")
        != hashlib.sha256(final_path.read_bytes()).hexdigest()
        or inventory.get("ordered_files") != inventory_entries
        or len(ledger_heads) != 1
        or hashlib.sha256(ledger_heads[0].read_bytes()).hexdigest() != ledger_head_sha256
        or predecessor.get("ledger_head_sha256") != ledger_head_sha256
        or predecessor.get("receipt_manifest_sha256")
        != hashlib.sha256(
            (root / f"attempts/{attempt_id}/{stage}/receipt-manifest.json").read_bytes()
        ).hexdigest()
        or (
            binding is not None
            and (
                predecessor.get("stage_binding_sha256")
                != hashlib.sha256(binding_path.read_bytes()).hexdigest()
                or binding.get("attempt_id") != attempt_id
                or binding.get("plan_binding_sha256") != plan_binding_sha256
            )
        )
        or (
            bct is not None
            and (
                bct.get("attempt_id") != attempt_id
                or final.get("predecessor_stage_terminal_sha256")
                != hashlib.sha256(bct_path.read_bytes()).hexdigest()
                or final.get("ledger_head_sha256") != bct.get("ledger_head_sha256")
            )
        )
        or (
            bct is None
            and (
                final.get("predecessor_stage_terminal_sha256")
                != hashlib.sha256(screening_path.read_bytes()).hexdigest()
                or final.get("ledger_head_sha256") != screening.get("ledger_head_sha256")
                or final.get("status") not in {"not_estimable", "blocked", "interrupted"}
            )
        )
    ):
        raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
    anchor = read_canonical(qa_root(repository) / "pre-egress/execution-anchor.json")
    if binding is not None and anchor.get("execution_commit") != binding.get("execution_commit"):
        raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
    if binding is not None and binding.get("transport_mode") == "live":
        claim = _read_signed(root / "live-attempt-claim.json", seed, "live-attempt-claim-v1")
        _validate_runtime_clock_chain(
            root, seed, binding, claim, predecessor, attempt_id, stage
        )
    if bct is not None:
        if screening.get("status") != "completed_estimable" or bct.get("status") != "review_required":
            raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
        freeze_path = root / f"freeze/{attempt_id}/freeze_b.json"
        freeze = _read_signed(freeze_path, seed, "freeze-b-v1")
        result_path = root / f"attempts/{attempt_id}/bct/evidence/bct-result-manifest.json"
        result = _read_signed(result_path, seed, "bct-result-manifest-v1")
        projection_path = root / f"attempts/{attempt_id}/bct/evidence/projection-manifest.json"
        rows_path = root / f"attempts/{attempt_id}/bct/evidence/search-config-rows.json"
        projection = _read_signed(projection_path, seed, "bct-projection-manifest-v1")
        rows = _read_signed(rows_path, seed, "search-config-rows-v1")
        family_paths = {
            "family_certified_false_sha256": root / f"attempts/{attempt_id}/bct/evidence/families/BCT-FV5-01-CERTIFIED-FALSE.json",
            "family_correct_sha256": root / f"attempts/{attempt_id}/bct/evidence/families/BCT-FV5-02-CORRECT.json",
            "family_irrelevant_sha256": root / f"attempts/{attempt_id}/bct/evidence/families/BCT-FV5-03-IRRELEVANT.json",
            "family_ordinary_false_sha256": root / f"attempts/{attempt_id}/bct/evidence/families/BCT-FV5-04-ORDINARY-FALSE.json",
        }
        for path in family_paths.values():
            _read_signed(path, seed, "bct-family-evidence-v1")
        family_hashes = {
            field: hashlib.sha256(path.read_bytes()).hexdigest()
            for field, path in family_paths.items()
        }
        if (
            bct.get("bct_result_manifest_sha256")
            != hashlib.sha256(result_path.read_bytes()).hexdigest()
            or bct.get("freeze_b_sha256") != hashlib.sha256(freeze_path.read_bytes()).hexdigest()
            or freeze.get("screening_stage_terminal_sha256")
            != hashlib.sha256(screening_path.read_bytes()).hexdigest()
            or result.get("projection_manifest_sha256")
            != hashlib.sha256(projection_path.read_bytes()).hexdigest()
            or result.get("search_config_rows_sha256")
            != hashlib.sha256(rows_path.read_bytes()).hexdigest()
            or rows.get("projection_manifest_sha256")
            != hashlib.sha256(projection_path.read_bytes()).hexdigest()
            or any(result.get(field) != digest for field, digest in family_hashes.items())
            or any(projection.get(field) != digest for field, digest in family_hashes.items())
        ):
            raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_publication_receipt_v1", "profile": PROFILE,
        "kind": "publication_receipt", "attempt_id": attempt_id,
        "plan_binding_sha256": plan_binding_sha256,
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


def _read_signed(path: Path, seed: bytes, domain: str) -> dict[str, JsonValue]:
    value = parse_canonical_object(_read_exact_regular(path))
    signature = value.get("signature")
    if not isinstance(signature, str):
        raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
    unsigned = dict(value)
    del unsigned["signature"]
    try:
        verify_object_signature(public_key_from_seed(seed), domain, unsigned, signature)
    except RootlessContractError as error:
        raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT") from error
    return value


def _validate_runtime_clock_chain(
    root: Path,
    seed: bytes,
    binding: Mapping[str, JsonValue],
    claim: Mapping[str, JsonValue],
    predecessor: Mapping[str, JsonValue],
    attempt_id: str,
    stage: str,
) -> None:
    authority_path = root / f"authorities/{attempt_id}/{stage}.json"
    _read_signed(authority_path, seed, "stage-execution-authority-v1")
    claim_created_at = claim.get("created_at")
    fingerprint = hashlib.sha256(public_key_from_seed(seed)).hexdigest()
    stages = ("screening",) if stage == "screening" else ("screening", "bct")
    clocks = tuple(sorted((root / "runtime-clock").glob("*.json")))
    if (
        claim.get("attempt_id") != attempt_id
        or claim.get("execution_commit") != binding.get("execution_commit")
        or claim.get("plan_binding_sha256") != binding.get("plan_binding_sha256")
        or claim.get("key_fingerprint") != fingerprint
        or not isinstance(claim_created_at, str)
        or predecessor.get("execution_authority_sha256")
        != hashlib.sha256(_read_exact_regular(authority_path)).hexdigest()
        or len(clocks) != len(stages) + 1
    ):
        raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
    try:
        claim_time = parse_timestamp(claim_created_at)
    except RootlessContractError as error:
        raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT") from error
    previous_hash: str | None = None
    previous_monotonic: int | None = None
    previous_realtime = None
    boot_id: str | None = None
    for sequence, clock_path in enumerate(clocks):
        raw = _read_exact_regular(clock_path)
        checkpoint = _read_signed(clock_path, seed, "runtime-clock-checkpoint-v1")
        monotonic = checkpoint.get("checkpoint_monotonic_ns")
        realtime = checkpoint.get("checkpoint_realtime")
        checkpoint_boot_id = checkpoint.get("boot_id_sha256")
        if (
            checkpoint.get("sequence") != sequence
            or checkpoint.get("attempt_id") != attempt_id
            or checkpoint.get("previous_checkpoint_sha256") != previous_hash
            or checkpoint.get("key_fingerprint") != fingerprint
            or clock_path.name != f"{sequence:08d}-{hashlib.sha256(raw).hexdigest()}.json"
        ):
            raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
        if (
            not isinstance(monotonic, int)
            or isinstance(monotonic, bool)
            or not isinstance(realtime, str)
            or not isinstance(checkpoint_boot_id, str)
            or (boot_id is not None and checkpoint_boot_id != boot_id)
        ):
            raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
        try:
            realtime_value = parse_timestamp(realtime)
        except RootlessContractError as error:
            raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT") from error
        if previous_monotonic is not None and (
            monotonic < previous_monotonic
            or previous_realtime is None
            or realtime_value < previous_realtime
        ):
            raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
        if sequence == 0:
            if (
                checkpoint.get("stage") is not None
                or checkpoint.get("stage_started_at") is not None
                or checkpoint.get("stage_monotonic_ns") is not None
                or checkpoint.get("realtime_at_claim") != claim_created_at
                or checkpoint.get("monotonic_ns_at_claim") != monotonic
                or realtime_value != claim_time
            ):
                raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
            boot_id = checkpoint_boot_id
        else:
            started_at = checkpoint.get("stage_started_at")
            started_monotonic = checkpoint.get("stage_monotonic_ns")
            if (
                checkpoint.get("stage") != stages[sequence - 1]
                or not isinstance(started_at, str)
                or not isinstance(started_monotonic, int)
                or isinstance(started_monotonic, bool)
                or checkpoint.get("realtime_at_claim") != claim_created_at
                or checkpoint.get("checkpoint_realtime") != started_at
                or checkpoint.get("checkpoint_monotonic_ns") != started_monotonic
            ):
                raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
            try:
                parse_timestamp(started_at)
            except RootlessContractError as error:
                raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT") from error
        previous_hash = hashlib.sha256(raw).hexdigest()
        previous_monotonic = monotonic
        previous_realtime = realtime_value


def seal_final_from_stage(
    root: Path,
    attempt_id: str,
    stage: str,
    status: str,
    reason: str,
    created_at: str,
) -> str:
    stage_path = root / f"terminals/{attempt_id}/{stage}.json"
    seed = (root / "keys/ed25519-private.key").read_bytes()
    terminal = _read_signed(stage_path, seed, "stage-terminal-v1")
    if (
        terminal.get("attempt_id") != attempt_id
        or terminal.get("stage") != stage
        or terminal.get("status") != status
        or terminal.get("reason_code") != reason
        or terminal.get("created_at") != created_at
    ):
        raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
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
    inventory_path = root / f"attempts/{attempt_id}/state-inventory.json"
    files = _inventory_entries(root, inventory_path)
    ordered_files: list[JsonValue] = [item for item in files]
    inventory_payload: dict[str, JsonValue] = {
        "schema_version": "rootless_state_inventory_v1", "profile": PROFILE,
        "kind": "state_inventory", "attempt_id": attempt_id, "final_terminal_sha256": digest,
        "ordered_files": ordered_files, "created_at": created_at,
        "key_fingerprint": terminal["key_fingerprint"],
    }
    write_new_or_same(inventory_path, signed(seed, "state-inventory-v1", inventory_payload))
    return digest


def seal_post_screening_setup_failure(
    root: Path, attempt_id: str, created_at: str
) -> str:
    """Close an estimable screening lineage when BCT cannot be set up."""
    screening_path = root / f"terminals/{attempt_id}/screening.json"
    seed = (root / "keys/ed25519-private.key").read_bytes()
    screening = _read_signed(screening_path, seed, "stage-terminal-v1")
    if (
        screening.get("attempt_id") != attempt_id
        or screening.get("stage") != "screening"
        or screening.get("status") != "completed_estimable"
        or screening.get("reason_code") != "SCREENING_ESTIMABLE"
    ):
        raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
    payload: dict[str, JsonValue] = {
        "schema_version": "rootless_attempt_terminal_v1",
        "profile": PROFILE,
        "kind": "attempt_terminal",
        "attempt_id": attempt_id,
        "status": "blocked",
        "reason_code": "ROOTLESS_BCT_SETUP_FAILED",
        "predecessor_stage_terminal_sha256": hashlib.sha256(screening_path.read_bytes()).hexdigest(),
        "ledger_head_sha256": screening["ledger_head_sha256"],
        "created_at": created_at,
        "key_fingerprint": screening["key_fingerprint"],
    }
    final_path = root / f"terminals/{attempt_id}/final.json"
    digest = write_new_or_same(final_path, signed(seed, "attempt-terminal-v1", payload))
    inventory_path = root / f"attempts/{attempt_id}/state-inventory.json"
    ordered_files: list[JsonValue] = [item for item in _inventory_entries(root, inventory_path)]
    inventory_payload: dict[str, JsonValue] = {
        "schema_version": "rootless_state_inventory_v1",
        "profile": PROFILE,
        "kind": "state_inventory",
        "attempt_id": attempt_id,
        "final_terminal_sha256": digest,
        "ordered_files": ordered_files,
        "created_at": created_at,
        "key_fingerprint": screening["key_fingerprint"],
    }
    write_new_or_same(inventory_path, signed(seed, "state-inventory-v1", inventory_payload))
    return digest


def _inventory_entries(
    root: Path, inventory_path: Path
) -> list[dict[str, JsonValue]]:
    files: list[dict[str, JsonValue]] = []
    total = 0
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        for name in (*names, *filenames):
            path = base / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
        for name in filenames:
            path = base / name
            if (
                path == inventory_path
                or name == "runtime.lock"
                or name.startswith(".")
                or name == "current-head.json"
            ):
                continue
            info = path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
            raw = _read_exact_regular(path)
            total += len(raw)
            if total > 1_661_992_960:
                raise RootlessContractError("ROOTLESS_REPORTING_CONFLICT")
            files.append(
                {
                    "state_relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    return sorted(files, key=lambda item: os.fsencode(str(item["state_relative_path"])))


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
