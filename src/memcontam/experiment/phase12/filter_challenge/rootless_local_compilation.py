from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypeGuard, assert_never

from memcontam.experiment.phase12.filter_challenge.registry_calibration import TASKS
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    parse_canonical_object,
    verify_object_signature,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_execution import (
    CompileContext,
    StageCompilation,
    build_bct_compilation,
    build_screening_compilation,
    load_probe_ids,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_ledger import Stage
from memcontam.experiment.phase12.filter_challenge.rootless_local_state import read_private_file

PROFILE = "local_rootless_non_authoritative"


def _is_stage(value: JsonValue) -> TypeGuard[Stage]:
    return value in {"screening", "bct"}


def load_live_stage_compilation(
    binding: dict[str, JsonValue],
    state_root: Path,
    repository: Path,
    public_key: bytes,
) -> StageCompilation:
    attempt_id = binding.get("attempt_id")
    stage = binding.get("stage")
    source_hash = binding.get("source_manifest_sha256")
    input_hash = binding.get("input_manifest_sha256")
    compiler_hash = binding.get("compiler_sha256")
    schedule_hash = binding.get("schedule_sha256")
    if (
        not isinstance(attempt_id, str)
        or not isinstance(source_hash, str)
        or not isinstance(input_hash, str)
        or not isinstance(compiler_hash, str)
        or not isinstance(schedule_hash, str)
    ):
        raise RootlessContractError("ROOTLESS_BINDING_INVALID")
    if not _is_stage(stage):
        raise RootlessContractError("ROOTLESS_BINDING_INVALID")
    typed_stage = stage
    manifest_root = state_root / "manifests" / attempt_id
    names = ("source", "runtime", "input", "compiler", f"{typed_stage}-schedule")
    expected = (
        source_hash,
        binding.get("runtime_manifest_sha256"),
        input_hash,
        compiler_hash,
        schedule_hash,
    )
    domains = (
        "source-manifest-v1",
        "runtime-manifest-v1",
        "input-manifest-v1",
        "request-compiler-manifest-v1",
        "schedule-manifest-v1",
    )
    manifests: list[dict[str, JsonValue]] = []
    for name, digest, domain in zip(names, expected, domains, strict=True):
        if not isinstance(digest, str):
            raise RootlessContractError("ROOTLESS_BINDING_INVALID")
        raw = read_private_file(manifest_root / f"{name}.json")
        if hashlib.sha256(raw).hexdigest() != digest:
            raise RootlessContractError("ROOTLESS_BINDING_INVALID")
        manifest = parse_canonical_object(raw)
        signature = manifest.get("signature")
        if not isinstance(signature, str):
            raise RootlessContractError("ROOTLESS_BINDING_INVALID")
        unsigned = dict(manifest)
        del unsigned["signature"]
        verify_object_signature(public_key, domain, unsigned, signature)
        manifests.append(manifest)
    for name in ("decoding-authority", "rate-card"):
        raw = read_private_file(manifest_root / f"{name}.json")
        if hashlib.sha256(raw).hexdigest() != binding.get(f"{name.replace('-', '_')}_sha256"):
            raise RootlessContractError("ROOTLESS_BINDING_INVALID")
    input_manifest = manifests[2]
    if (
        input_manifest.get("decoding_authority_sha256")
        != binding.get("decoding_authority_sha256")
        or input_manifest.get("rate_card_sha256") != binding.get("rate_card_sha256")
    ):
        raise RootlessContractError("ROOTLESS_BINDING_INVALID")
    context = CompileContext(attempt_id, typed_stage, source_hash, input_hash, compiler_hash)
    match typed_stage:
        case "screening":
            if (
                binding.get("predecessor_terminal_sha256") is not None
                or binding.get("freeze_b_sha256") is not None
            ):
                raise RootlessContractError("ROOTLESS_BINDING_INVALID")
            compilation = build_screening_compilation(context, load_probe_ids(repository))
        case "bct":
            terminal_raw = read_private_file(
                state_root / "terminals" / attempt_id / "screening.json"
            )
            freeze_raw = read_private_file(state_root / "freeze" / attempt_id / "freeze_b.json")
            if (
                hashlib.sha256(terminal_raw).hexdigest()
                != binding.get("predecessor_terminal_sha256")
                or hashlib.sha256(freeze_raw).hexdigest() != binding.get("freeze_b_sha256")
            ):
                raise RootlessContractError("ROOTLESS_BINDING_INVALID")
            freeze = parse_canonical_object(freeze_raw)
            selected: dict[str, tuple[str, ...]] = {}
            for task in TASKS:
                values = freeze.get(f"selected_{task}_probe_ids")
                if not isinstance(values, list) or len(values) != 2 or not all(
                    isinstance(value, str) for value in values
                ):
                    raise RootlessContractError("ROOTLESS_BINDING_INVALID")
                selected[task] = tuple(value for value in values if isinstance(value, str))
            compilation = build_bct_compilation(context, selected)
        case unreachable:
            assert_never(unreachable)
    schedule = manifests[-1]
    if (
        schedule.get("schema_version") != "rootless_schedule_manifest_v1"
        or schedule.get("profile") != PROFILE
        or schedule.get("attempt_id") != attempt_id
        or schedule.get("stage") != typed_stage
        or schedule.get("slot_count") != len(compilation.slots)
        or schedule.get("ordered_slot_root_sha256") != compilation.ordered_slot_root_sha256
        or binding.get("registered_slots") != len(compilation.slots)
    ):
        raise RootlessContractError("ROOTLESS_BINDING_INVALID")
    return compilation


__all__ = ("load_live_stage_compilation",)
