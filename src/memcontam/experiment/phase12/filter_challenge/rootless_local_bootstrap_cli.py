from __future__ import annotations

# allow: SIZE_OK — Task 3 freezes one exhaustive administrative command grammar.

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from memcontam.experiment.phase12.filter_challenge.rootless_local_acknowledgement import (
    AcknowledgementClock,
    create_plan_acknowledgement,
    create_rate_acknowledgement,
    create_stage_acknowledgement,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_binding import (
    ExternalAuthorityObservationError,
    build_execution_authority,
    build_stage_binding,
    observe_external_authorities,
    validate_rootless_configs,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    canonical_json_file,
    canonical_json_value,
    parse_canonical_object,
    public_key_from_seed,
    verify_object_signature,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_closure import (
    derive_freeze_b,
    seal_bct_setup_failure,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_state import (
    InitStateRequest,
    _validate_reviewed_plan,
    cache_tokenizer_source,
    initialize_state,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_execution import (
    write_request_goldens,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_broker import build_live_broker
from memcontam.experiment.phase12.filter_challenge.rootless_local_state import read_private_file
from memcontam.experiment.phase12.filter_challenge.rootless_local_operator import (
    acquire_attempt_lock,
    finalize_preclaim,
    publish,
    qa_root,
    read_canonical,
    record_t7,
    seal_final_from_stage,
    state_root,
    write_anchor,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_pre_egress import (
    run_pre_egress,
    verify_pre_egress,
)

PROFILE: Final = "local_rootless_non_authoritative"
TRUSTED_BASE_COMMIT: Final = "c057fb1adf9571ef21cd19fa2733c5ac47b40798"
_PREFLIGHT_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_TOKENIZER_SOURCE_SHA256: Final = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
_PREFLIGHT_FILES: Final = (
    "plan_source",
    "plan_descriptor",
    "review_metadata",
    "historical_screening_plan",
    "historical_screening_descriptor",
    "historical_post_descriptor",
    "tokenizer_source",
)


def _attempt(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--attempt-id", required=True)


def _operator(parser: argparse.ArgumentParser) -> None:
    _attempt(parser)
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--operator-index", type=int, choices=(1, 2), required=True)
    parser.add_argument("--operator-label", required=True)


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    rootless = commands.add_parser("filter-v5-rootless")
    rootless.add_argument("--repo-root", type=Path, required=True)
    rootless.add_argument("--state-home", type=Path)
    subcommands = rootless.add_subparsers(dest="rootless_command", required=True)

    verify = subcommands.add_parser("verify-bootstrap-runtime")
    verify.add_argument("--python", type=Path, required=True)

    initialize = subcommands.add_parser("init-state")
    _attempt(initialize)
    initialize.add_argument("--plan-source", type=Path, required=True)
    initialize.add_argument("--plan-descriptor", type=Path, required=True)
    initialize.add_argument("--review-metadata", type=Path, required=True)
    initialize.add_argument("--historical-screening-plan", type=Path)
    initialize.add_argument("--historical-screening-descriptor", type=Path)
    initialize.add_argument("--historical-post-descriptor", type=Path)
    initialize.add_argument("--tokenizer-source", type=Path)
    initialize.add_argument("--execution-commit")

    plan = subcommands.add_parser("acknowledge-plan")
    _operator(plan)
    stage = subcommands.add_parser("acknowledge-stage")
    _operator(stage)
    stage.add_argument("--stage", choices=("screening", "bct"), required=True)
    rate = subcommands.add_parser("acknowledge-rate-capability")
    _attempt(rate)
    rate.add_argument("--provider-account-label", required=True)
    rate.add_argument("--rpm-limit", type=int, required=True)
    rate.add_argument("--tpm-limit", type=int, required=True)
    for name in ("bind-screening", "bind-bct"):
        parser = subcommands.add_parser(name)
        _attempt(parser)
    authority = subcommands.add_parser("build-execution-authority")
    _attempt(authority)
    authority.add_argument("--stage", choices=("screening", "bct"), required=True)
    authority.add_argument("--plan-set-id", required=True)
    authority.add_argument("--stage-set-id", required=True)

    preflight = subcommands.add_parser("preflight")
    _attempt(preflight)
    for name in (
        "plan-source",
        "plan-descriptor",
        "review-metadata",
        "historical-screening-plan",
        "historical-screening-descriptor",
        "historical-post-descriptor",
        "tokenizer-source",
        "operator-1-label",
        "operator-2-label",
        "provider-account-label",
        "rpm-limit",
        "tpm-limit",
        "paid-egress-ack",
    ):
        preflight.add_argument(f"--{name}")

    finalize = subcommands.add_parser("finalize-zero-call-preclaim")
    _attempt(finalize)
    finalize.add_argument("--execution-commit", required=True)
    finalize.add_argument("--failed-command", required=True)
    finalize.add_argument("--observed-exit", type=int, required=True)
    finalize.add_argument("--missing-input-role")
    finalize.add_argument("--external-authority-diagnostic")
    subcommands.add_parser("record-t7-qa")
    publish = subcommands.add_parser("publish-receipt")
    _attempt(publish)
    reconcile = subcommands.add_parser("reconcile-attempt")
    _attempt(reconcile)
    reconcile.add_argument("--execution-commit", required=True)
    anchor = subcommands.add_parser("write-execution-anchor")
    anchor.add_argument("--execution-commit", required=True)
    continuation = subcommands.add_parser("continue-after-screening")
    _attempt(continuation)
    freeze = subcommands.add_parser("derive-freeze-b")
    _attempt(freeze)
    block = subcommands.add_parser("seal-post-screening-block")
    _attempt(block)
    block.add_argument("--reason", choices=("ROOTLESS_BCT_SETUP_FAILED",), required=True)
    stage_exit = subcommands.add_parser("finalize-after-stage-exit")
    _attempt(stage_exit)
    stage_exit.add_argument("--stage", choices=("screening", "bct"), required=True)
    stage_exit.add_argument("--observed-exit", type=int, required=True)
    qa = subcommands.add_parser("run-pre-egress-qa")
    qa.add_argument("--role", choices=("pre_f1", "pre_f2", "pre_f3"), required=True)
    qa.add_argument("--execution-commit", required=True)
    verify_qa = subcommands.add_parser("verify-pre-egress-qa")
    verify_qa.add_argument("--execution-commit", required=True)
    broker = subcommands.add_parser("broker-runtime")
    _attempt(broker)
    broker.add_argument("--stage", choices=("screening", "bct"), required=True)
    broker.add_argument("--authority", type=Path, required=True)
    broker.add_argument("--worker-fd", type=int, choices=(3,), required=True)
    subcommands.add_parser("materialize-request-goldens")


def _root(arguments: argparse.Namespace) -> Path:
    state_home = arguments.state_home
    if state_home is None:
        raise RootlessContractError("ROOTLESS_STATE_PATH_INVALID")
    raw = os.fspath(state_home)
    if not raw.startswith("/") or raw != os.path.normpath(raw):
        raise RootlessContractError("ROOTLESS_STATE_PATH_INVALID")
    return state_home / "memcontam" / "phase12-filter-v5-rootless-local"


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _status(arguments: argparse.Namespace, role: str | None, digest: str | None) -> None:
    payload = {
        "schema_version": "rootless_cli_status_v1",
        "profile": PROFILE,
        "command": arguments.rootless_command,
        "outcome": "ok",
        "next_action": "continue",
        "reason_code": None,
        "attempt_id": getattr(arguments, "attempt_id", None),
        "artifact_role": role,
        "artifact_sha256": digest,
        "provider_calls_issued": 0,
        "exit_code": 0,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _exit_status(
    arguments: argparse.Namespace,
    exit_code: int,
    reason: str | None,
    role: str | None = None,
    digest: str | None = None,
    missing_input_role: str | None = None,
    external_authority_diagnostic: str | None = None,
) -> None:
    payload = {
        "schema_version": "rootless_cli_status_v1",
        "profile": PROFILE,
        "command": arguments.rootless_command,
        "outcome": "ok" if exit_code == 0 else "blocked",
        "next_action": "continue" if exit_code == 0 else "publish_and_stop" if exit_code == 69 else "stop",
        "reason_code": reason,
        "attempt_id": getattr(arguments, "attempt_id", None),
        "artifact_role": role,
        "artifact_sha256": digest,
        "provider_calls_issued": 0,
        "exit_code": exit_code,
    }
    if missing_input_role is not None:
        payload["missing_input_role"] = missing_input_role
        payload["external_authority_diagnostic"] = external_authority_diagnostic
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if exit_code:
        raise SystemExit(exit_code)


def _administrative(arguments: argparse.Namespace) -> None:
    command = arguments.rootless_command
    repository = arguments.repo_root
    if command == "write-execution-anchor":
        _status(arguments, "execution_anchor", write_anchor(repository, arguments.execution_commit))
        return
    if command == "finalize-zero-call-preclaim":
        digest = finalize_preclaim(
            repository, arguments.state_home, arguments.attempt_id, arguments.execution_commit,
            arguments.failed_command, arguments.observed_exit, _timestamp(),
            missing_input_role=arguments.missing_input_role,
            external_authority_diagnostic=arguments.external_authority_diagnostic,
        )
        _status(arguments, "zero_call_skip", digest)
        return
    if command == "record-t7-qa":
        _status(arguments, "t7_attempt_qa", record_t7(repository, arguments.state_home))
        return
    if command == "publish-receipt":
        _status(arguments, "publication_receipt", publish(repository, arguments.state_home, arguments.attempt_id, _timestamp()))
        return
    if command == "run-pre-egress-qa":
        digest = run_pre_egress(repository, arguments.role, arguments.execution_commit, _timestamp())
        _status(arguments, "pre_egress_qa", digest)
        return
    if command == "verify-pre-egress-qa":
        verify_pre_egress(repository, arguments.execution_commit)
        _status(arguments, None, None)
        return
    root = state_root(arguments.state_home)
    if command == "reconcile-attempt":
        skip = qa_root(repository) / "pre-egress/zero-call-skip.json"
        publication = repository / "docs/evidence/phase12-filter-v5-rootless-local/rehearsal-publication.json"
        if skip.is_file():
            record_t7(repository, arguments.state_home)
            _exit_status(arguments, 65, str(read_canonical(skip)["reason"]), "zero_call_skip", hashlib.sha256(skip.read_bytes()).hexdigest())
        if publication.is_file():
            record_t7(repository, arguments.state_home)
            _exit_status(arguments, 69, "BCT_COMPLETED_REVIEW_REQUIRED", "publication_receipt", hashlib.sha256(publication.read_bytes()).hexdigest())
        if not root.exists():
            _status(arguments, None, None)
            return
        descriptor: int | None = None
        try:
            descriptor = acquire_attempt_lock(root)
        except (FileNotFoundError, RootlessContractError):
            digest = finalize_preclaim(repository, arguments.state_home, arguments.attempt_id, arguments.execution_commit, "orchestration_interrupted", 70, _timestamp())
            record_t7(repository, arguments.state_home)
            _exit_status(arguments, 65, "ROOTLESS_PRECLAIM_ADMIN_FAILED", "zero_call_skip", digest)
        try:
            if _has_valid_ledger_fork(root, arguments.attempt_id):
                _exit_status(arguments, 67, "ROOTLESS_ARCHIVE_INVALID")
            claim = (root / "live-attempt-claim.json").is_file() or (root / f"claims/{arguments.attempt_id}.json").is_file()
            attempt_markers = (
                root / f"bindings/{arguments.attempt_id}",
                root / f"authorities/{arguments.attempt_id}",
                root / f"acknowledgements/plan/{arguments.attempt_id}",
                root / f"acknowledgements/rate/{arguments.attempt_id}.json",
                root / f"terminals/{arguments.attempt_id}",
                root / f"attempts/{arguments.attempt_id}",
            )
            if not claim and any(path.exists() for path in attempt_markers):
                digest = finalize_preclaim(repository, arguments.state_home, arguments.attempt_id, arguments.execution_commit, "orchestration_interrupted", 70, _timestamp())
                record_t7(repository, arguments.state_home)
                _exit_status(arguments, 65, "ROOTLESS_PRECLAIM_ADMIN_FAILED", "zero_call_skip", digest)
            _status(arguments, None, None)
            return
        finally:
            if descriptor is not None:
                os.close(descriptor)
    descriptor = acquire_attempt_lock(root)
    try:
        if _has_valid_ledger_fork(root, arguments.attempt_id):
            _exit_status(arguments, 67, "ROOTLESS_ARCHIVE_INVALID")
        claim = (root / "live-attempt-claim.json").is_file() or (root / f"claims/{arguments.attempt_id}.json").is_file()
        if command == "continue-after-screening":
            terminal = read_canonical(root / f"terminals/{arguments.attempt_id}/screening.json")
            status = terminal.get("status")
            if status == "completed_estimable":
                _status(arguments, "stage_terminal", hashlib.sha256(canonical_json_file(terminal)).hexdigest())
                return
            if status in {"not_estimable", "blocked", "interrupted"}:
                final_status = "not_estimable" if status == "not_estimable" else str(status)
                seal_final_from_stage(root, arguments.attempt_id, "screening", final_status, str(terminal.get("reason_code")), str(terminal["created_at"]))
                publish(repository, arguments.state_home, arguments.attempt_id, _timestamp())
                record_t7(repository, arguments.state_home)
                _exit_status(arguments, 69, str(terminal.get("reason_code")))
            _exit_status(arguments, 67, "ROOTLESS_ARCHIVE_INVALID")
        if command == "seal-post-screening-block":
            seal_final_from_stage(root, arguments.attempt_id, "screening", "blocked", arguments.reason, _timestamp())
            _status(arguments, "attempt_terminal", hashlib.sha256((root / f"terminals/{arguments.attempt_id}/final.json").read_bytes()).hexdigest())
            return
        if command == "finalize-after-stage-exit":
            if not claim:
                _exit_status(arguments, 65, "ROOTLESS_PRECLAIM_ADMIN_FAILED")
            stage_path = root / f"terminals/{arguments.attempt_id}/{arguments.stage}.json"
            if not stage_path.is_file():
                raise RootlessContractError("ROOTLESS_STAGE_PREFIX_INVALID")
            terminal = read_canonical(stage_path)
            status = str(terminal.get("status"))
            final_status = "not_estimable" if status == "not_estimable" else "review_required" if status == "review_required" else "interrupted" if status == "interrupted" else "blocked"
            seal_final_from_stage(root, arguments.attempt_id, arguments.stage, final_status, str(terminal.get("reason_code")), str(terminal["created_at"]))
            _status(arguments, "attempt_terminal", hashlib.sha256((root / f"terminals/{arguments.attempt_id}/final.json").read_bytes()).hexdigest())
            return
    finally:
        os.close(descriptor)
    raise RootlessContractError("ROOTLESS_ADMIN_COMMAND_INVALID")


def _has_valid_ledger_fork(root: Path, attempt_id: str) -> bool:
    heads_root = root / "ledger/global/heads"
    records_root = root / "ledger/global/records"
    seed_path = root / "keys/ed25519-private.key"
    if not heads_root.is_dir() or not records_root.is_dir() or not seed_path.is_file():
        return False
    try:
        seed = read_private_file(seed_path)
    except (OSError, RootlessContractError):
        return False
    public_key = public_key_from_seed(seed)
    records: dict[str, dict[str, JsonValue]] = {}
    for path in records_root.glob("*.json"):
        try:
            raw = read_private_file(path)
            digest = hashlib.sha256(raw).hexdigest()
            value = parse_canonical_object(raw)
            signature = value.get("signature")
            unsigned = dict(value)
            del unsigned["signature"]
            if (
                not isinstance(signature, str)
                or path.name != f"{value.get('sequence'):06d}-{digest}.json"
                or value.get("attempt_id") != attempt_id
            ):
                continue
            verify_object_signature(public_key, "ledger-record-v1", unsigned, signature)
        except (OSError, KeyError, TypeError, ValueError, RootlessContractError):
            continue
        records[digest] = value
    sequences: dict[int, str] = {}
    for path in heads_root.glob("*.json"):
        try:
            raw = read_private_file(path)
            digest = hashlib.sha256(raw).hexdigest()
            value = parse_canonical_object(raw)
            signature = value.get("signature")
            sequence = value.get("sequence")
            record_sha256 = value.get("record_sha256")
            unsigned = dict(value)
            del unsigned["signature"]
            record = records.get(record_sha256) if isinstance(record_sha256, str) else None
            if (
                not isinstance(signature, str)
                or not isinstance(sequence, int)
                or isinstance(sequence, bool)
                or path.name != f"{sequence:06d}-{digest}.json"
                or value.get("attempt_id") != attempt_id
                or record is None
                or record.get("sequence") != sequence
            ):
                continue
            verify_object_signature(public_key, "ledger-head-v1", unsigned, signature)
        except (OSError, KeyError, TypeError, ValueError, RootlessContractError):
            continue
        previous = sequences.setdefault(sequence, digest)
        if previous != digest:
            return True
    return False


def _read(path: Path) -> dict[str, JsonValue]:
    return parse_canonical_object(path.read_bytes())


def _write(path: Path, value: dict[str, JsonValue]) -> str:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.parent.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise RootlessContractError("ROOTLESS_STATE_PATH_UNSAFE")
    raw = canonical_json_file(value)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def _seed(root: Path) -> bytes:
    raw = (root / "keys" / "ed25519-private.key").read_bytes()
    if len(raw) != 32:
        raise RootlessContractError("ROOTLESS_PRIVATE_KEY_INVALID")
    return raw


def _preflight_file(arguments: argparse.Namespace, field: str) -> tuple[Path, bytes]:
    value = getattr(arguments, field)
    if not isinstance(value, str) or not value:
        raise RootlessContractError("ROOTLESS_MISSING_EXTERNAL_INPUT")
    path = Path(value)
    if not path.is_absolute() or os.fspath(path) != os.path.normpath(os.fspath(path)):
        raise RootlessContractError("ROOTLESS_MISSING_EXTERNAL_INPUT")
    try:
        raw = read_private_file(path)
    except (OSError, RootlessContractError) as error:
        raise RootlessContractError("ROOTLESS_MISSING_EXTERNAL_INPUT") from error
    if not raw:
        raise RootlessContractError("ROOTLESS_MISSING_EXTERNAL_INPUT")
    return path, raw


def _preflight(arguments: argparse.Namespace) -> None:
    state_home = arguments.state_home
    if state_home is None:
        raise RootlessContractError("ROOTLESS_MISSING_EXTERNAL_INPUT")
    try:
        info = os.lstat(state_home)
    except OSError as error:
        raise RootlessContractError("ROOTLESS_MISSING_EXTERNAL_INPUT") from error
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise RootlessContractError("ROOTLESS_MISSING_EXTERNAL_INPUT")
    files = {field: _preflight_file(arguments, field) for field in _PREFLIGHT_FILES}
    plan_source, plan_raw = files["plan_source"]
    _validate_reviewed_plan(
        InitStateRequest(
            state_home,
            plan_source,
            files["plan_descriptor"][0],
            files["review_metadata"][0],
            arguments.attempt_id,
        ),
        hashlib.sha256(plan_raw).hexdigest(),
    )
    if hashlib.sha256(files["tokenizer_source"][1]).hexdigest() != _TOKENIZER_SOURCE_SHA256:
        raise RootlessContractError("ROOTLESS_MISSING_EXTERNAL_INPUT")
    labels = (arguments.operator_1_label, arguments.operator_2_label, arguments.provider_account_label)
    if (
        not all(isinstance(label, str) and _PREFLIGHT_ID.fullmatch(label) for label in labels)
        or arguments.operator_1_label == arguments.operator_2_label
    ):
        raise RootlessContractError("ROOTLESS_MISSING_EXTERNAL_INPUT")
    rates = (arguments.rpm_limit, arguments.tpm_limit)
    if (
        not all(isinstance(rate, str) and rate.isdigit() for rate in rates)
        or int(arguments.rpm_limit) < 6
        or int(arguments.tpm_limit) < 30_000
    ):
        raise RootlessContractError("ROOTLESS_RATE_CAPABILITY_MISSING")
    if not isinstance(arguments.paid_egress_ack, str) or not arguments.paid_egress_ack:
        raise RootlessContractError("ROOTLESS_PAID_EGRESS_NOT_ENABLED")
    validate_rootless_configs(arguments.repo_root)
    authority_path = arguments.repo_root / "configs/phase12/filter_v5_rootless_local/decoding_authority.json"
    observe_external_authorities(parse_canonical_object(authority_path.read_bytes()))


def _verify_runtime(arguments: argparse.Namespace) -> None:
    repository = arguments.repo_root.resolve(strict=True)
    python = arguments.python.resolve(strict=True)
    if repository != arguments.repo_root or python != arguments.python or not os.path.samefile(python, sys.executable):
        raise RootlessContractError("ROOTLESS_BOOTSTRAP_RUNTIME_INVALID")
    if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 11):
        raise RootlessContractError("ROOTLESS_BOOTSTRAP_RUNTIME_INVALID")
    import memcontam

    imported = Path(memcontam.__file__).resolve(strict=True)
    if not imported.is_relative_to(repository / "src" / "memcontam"):
        raise RootlessContractError("ROOTLESS_BOOTSTRAP_RUNTIME_INVALID")
    validate_rootless_configs(repository)
    _status(arguments, None, None)


def _acknowledge(arguments: argparse.Namespace) -> None:
    root = _root(arguments)
    seed = _seed(root)
    clock = AcknowledgementClock(datetime.now(UTC).replace(microsecond=0))
    plan_hash = hashlib.sha256((root / "plan-bind.md").read_bytes()).hexdigest()
    nonce = hashlib.sha256(os.urandom(32)).hexdigest()
    if arguments.rootless_command == "acknowledge-plan":
        value = create_plan_acknowledgement(
            attempt_id=arguments.attempt_id,
            set_id=arguments.set_id,
            operator_label=arguments.operator_label,
            operator_index=arguments.operator_index,
            plan_binding_sha256=plan_hash,
            nonce=nonce,
            seed=seed,
            clock=clock,
        )
        path = root / "acknowledgements" / "plan" / arguments.attempt_id / arguments.set_id / f"operator-{arguments.operator_index}.json"
    elif arguments.rootless_command == "acknowledge-stage":
        binding_path = root / "bindings" / arguments.attempt_id / f"{arguments.stage}.json"
        value = create_stage_acknowledgement(
            attempt_id=arguments.attempt_id,
            stage=arguments.stage,
            set_id=arguments.set_id,
            operator_label=arguments.operator_label,
            operator_index=arguments.operator_index,
            plan_binding_sha256=plan_hash,
            stage_binding_sha256=hashlib.sha256(
                canonical_json_value(_read(binding_path))
            ).hexdigest(),
            nonce=nonce,
            seed=seed,
            clock=clock,
        )
        path = root / "acknowledgements" / "stage" / arguments.attempt_id / arguments.stage / arguments.set_id / f"operator-{arguments.operator_index}.json"
    else:
        value = create_rate_acknowledgement(
            attempt_id=arguments.attempt_id,
            provider_account_label=arguments.provider_account_label,
            rpm_limit=arguments.rpm_limit,
            tpm_limit=arguments.tpm_limit,
            observed_at=_timestamp(),
            nonce=nonce,
            seed=seed,
            clock=clock,
        )
        path = root / "acknowledgements" / "rate" / f"{arguments.attempt_id}.json"
    digest = _write(path, value)
    _status(arguments, "acknowledgement", digest)


def _bind(arguments: argparse.Namespace) -> None:
    root = _root(arguments)
    stage = "screening" if arguments.rootless_command == "bind-screening" else "bct"
    attempt = arguments.attempt_id
    if stage == "bct":
        digest = seal_bct_setup_failure(root, attempt, _seed(root))
        _exit_status(
            arguments,
            69,
            "ROOTLESS_BCT_SETUP_FAILED",
            "attempt_terminal",
            digest,
        )
    manifests = root / "manifests" / attempt
    configs = validate_rootless_configs(arguments.repo_root)
    names = {
        "source": "source.json",
        "runtime": "runtime.json",
        "input": "input.json",
        "compiler": "compiler.json",
        "schedule": f"{stage}-schedule.json",
    }
    hashes = {name: hashlib.sha256((manifests / filename).read_bytes()).hexdigest() for name, filename in names.items()}
    source = _read(manifests / "source.json")
    signature = source.get("signature")
    if not isinstance(signature, str):
        raise RootlessContractError("ROOTLESS_SOURCE_MANIFEST_INVALID")
    unsigned = dict(source)
    del unsigned["signature"]
    verify_object_signature(
        public_key_from_seed(_seed(root)), "source-manifest-v1", unsigned, signature
    )
    execution_commit = source.get("execution_commit")
    anchor = read_canonical(qa_root(arguments.repo_root) / "pre-egress/execution-anchor.json")
    if not isinstance(execution_commit, str) or anchor.get("execution_commit") != execution_commit:
        raise RootlessContractError("ROOTLESS_EXECUTION_COMMIT_INVALID")
    value = build_stage_binding(
        attempt_id=attempt,
        stage=stage,
        plan_binding_sha256=hashlib.sha256((root / "plan-bind.md").read_bytes()).hexdigest(),
        trusted_base_commit=TRUSTED_BASE_COMMIT,
        execution_commit=execution_commit,
        decoding_authority_sha256=configs["decoding_authority"],
        rate_card_sha256=configs["rate_card"],
        source_manifest_sha256=hashes["source"],
        runtime_manifest_sha256=hashes["runtime"],
        input_manifest_sha256=hashes["input"],
        compiler_sha256=hashes["compiler"],
        schedule_sha256=hashes["schedule"],
        predecessor_terminal_sha256=None if stage == "screening" else hashlib.sha256((root / "terminals" / attempt / "screening.json").read_bytes()).hexdigest(),
        freeze_b_sha256=None if stage == "screening" else hashlib.sha256((root / "freeze" / attempt / "freeze_b.json").read_bytes()).hexdigest(),
        registered_slots=90 if stage == "screening" else 480,
        stage_cap_nanousd=2_000_000_000 if stage == "screening" else 8_000_000_000,
        created_at=_timestamp(),
    )
    digest = _write(root / "bindings" / attempt / f"{stage}.json", value)
    _status(arguments, "stage_binding", digest)


def _authority(arguments: argparse.Namespace) -> None:
    root = _root(arguments)
    attempt, stage = arguments.attempt_id, arguments.stage
    binding = _read(root / "bindings" / attempt / f"{stage}.json")
    plan_root = root / "acknowledgements" / "plan" / attempt / arguments.plan_set_id
    stage_root = root / "acknowledgements" / "stage" / attempt / stage / arguments.stage_set_id
    plans = [_read(plan_root / f"operator-{index}.json") for index in (1, 2)]
    stages = [_read(stage_root / f"operator-{index}.json") for index in (1, 2)]
    rate = _read(root / "acknowledgements" / "rate" / f"{attempt}.json")
    value = build_execution_authority(binding, plans, stages, rate, seed=_seed(root), issued_at=_timestamp())
    digest = _write(root / "authorities" / attempt / f"{stage}.json", value)
    _status(arguments, "execution_authority", digest)


def _broker_runtime(arguments: argparse.Namespace) -> None:
    runtime = importlib.import_module(
        "memcontam.experiment.phase12.filter_challenge.rootless_local_runtime"
    )
    compilation_loader = importlib.import_module(
        "memcontam.experiment.phase12.filter_challenge.rootless_local_compilation"
    )
    authority_validator = importlib.import_module(
        "memcontam.experiment.phase12.filter_challenge.rootless_local_authority"
    )
    if arguments.worker_fd != 3:
        raise RootlessContractError("ROOTLESS_BROKER_FD_INVALID")
    root = _root(arguments)
    authority_path = root / "authorities" / arguments.attempt_id / f"{arguments.stage}.json"
    if arguments.authority != authority_path:
        raise RootlessContractError("ROOTLESS_EXECUTION_AUTHORITY_INVALID")
    authority = parse_canonical_object(read_private_file(authority_path))
    binding = parse_canonical_object(
        read_private_file(root / "bindings" / arguments.attempt_id / f"{arguments.stage}.json")
    )
    if (
        authority.get("attempt_id") != arguments.attempt_id
        or authority.get("stage") != arguments.stage
    ):
        raise RootlessContractError("ROOTLESS_EXECUTION_AUTHORITY_INVALID")
    public_key = authority_validator.load_public_key(root)
    authority_validator.validate_execution_authority(
        authority,
        binding,
        public_key,
    )
    compilation = compilation_loader.load_live_stage_compilation(
        binding, root, arguments.repo_root, public_key
    )
    exit_code = runtime.run_stage_process(
        compilation.slots,
        lambda: build_live_broker(binding, root, arguments.repo_root),
    )
    if exit_code:
        raise SystemExit(exit_code)


def _derive_freeze(arguments: argparse.Namespace) -> None:
    root = _root(arguments)
    binding = parse_canonical_object(
        read_private_file(root / "bindings" / arguments.attempt_id / "screening.json")
    )
    authority_validator = importlib.import_module(
        "memcontam.experiment.phase12.filter_challenge.rootless_local_authority"
    )
    compilation_loader = importlib.import_module(
        "memcontam.experiment.phase12.filter_challenge.rootless_local_compilation"
    )
    public_key = authority_validator.load_public_key(root)
    compilation = compilation_loader.load_live_stage_compilation(
        binding, root, arguments.repo_root, public_key
    )
    digest = derive_freeze_b(root, _seed(root), compilation.slots)
    _status(arguments, "freeze_b", digest)


def run(arguments: argparse.Namespace) -> None:
    command = arguments.rootless_command
    if command == "verify-bootstrap-runtime":
        _verify_runtime(arguments)
        return
    if command == "init-state":
        if arguments.state_home is None:
            raise RootlessContractError("ROOTLESS_STATE_PATH_INVALID")
        result = initialize_state(
            InitStateRequest(
                arguments.state_home,
                arguments.plan_source,
                arguments.plan_descriptor,
                arguments.review_metadata,
                arguments.attempt_id,
            )
        )
        if arguments.tokenizer_source is not None:
            cache_tokenizer_source(
                arguments.tokenizer_source,
                result.tokenizer_cache,
                expected_rank_count=199_998,
            )
        _status(arguments, "plan_binding", hashlib.sha256(result.plan_binding.read_bytes()).hexdigest())
        return
    if command in {"acknowledge-plan", "acknowledge-stage", "acknowledge-rate-capability"}:
        _acknowledge(arguments)
        return
    if command in {"bind-screening", "bind-bct"}:
        _bind(arguments)
        return
    if command == "build-execution-authority":
        _authority(arguments)
        return
    if command == "broker-runtime":
        _broker_runtime(arguments)
        return
    if command == "derive-freeze-b":
        _derive_freeze(arguments)
        return
    if command == "materialize-request-goldens":
        digest = write_request_goldens(arguments.repo_root)
        _status(arguments, "request_goldens", digest)
        return
    if command == "preflight":
        try:
            _preflight(arguments)
        except ExternalAuthorityObservationError as error:
            _exit_status(
                arguments,
                65,
                "ROOTLESS_MISSING_EXTERNAL_INPUT",
                missing_input_role=error.missing_input_role,
                external_authority_diagnostic=error.code,
            )
        except RootlessContractError as error:
            _exit_status(arguments, 65, error.code)
        _status(arguments, None, None)
        return
    if command in {
        "finalize-zero-call-preclaim", "record-t7-qa", "publish-receipt", "reconcile-attempt",
        "write-execution-anchor", "continue-after-screening", "seal-post-screening-block",
        "finalize-after-stage-exit", "run-pre-egress-qa", "verify-pre-egress-qa",
    }:
        _administrative(arguments)
        return
    raise RootlessContractError("ROOTLESS_ADMIN_COMMAND_INVALID")
