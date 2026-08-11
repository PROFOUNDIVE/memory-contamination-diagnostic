from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal, TypeAlias, assert_never

from memcontam.contamination.phase12.models import CandidateTriplet
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.memory.cards_v3 import validate_memory_envelope
from memcontam.memory.checkpoint_v3 import (
    NativeState,
    Phase12Checkpoint,
    serialize_checkpoint,
)
from memcontam.memory.writer_registry import WriterRegistry
from memcontam.tasks.base import TaskInstance
from memcontam.experiment.phase12.filter_challenge.ordinary_replay import (
    JsonValue,
    OrdinaryAuthorityError,
    _bot,
    _full_history,
    _json_mapping,
    _rag,
    _reflexion,
)

Baseline: TypeAlias = Literal["full_history", "rag_frozen", "bot_style", "reflexion_style"]


def realize_ordinary_false(
    triplet: CandidateTriplet, baseline: Baseline, checkpoint: Phase12Checkpoint
) -> dict[str, JsonValue]:
    task = TaskInstance(
        sample_id=f"fv5-ordinary-{triplet.task}-position-9",
        task_name=triplet.task,
        input={"position": 9},
    )
    trial_id = f"fv5-ordinary-{triplet.task}-{baseline}-position-9"
    match baseline:
        case "full_history":
            native, envelope, event, interaction = _full_history(
                task, trial_id, triplet.false_candidate.content, checkpoint
            )
        case "rag_frozen":
            native, envelope, event, interaction = _rag(
                task, trial_id, triplet.false_candidate.content, checkpoint
            )
        case "bot_style":
            native, envelope, event, interaction = _bot(
                task, trial_id, triplet.false_candidate.content, checkpoint
            )
        case "reflexion_style":
            native, envelope, event, interaction = _reflexion(
                task, trial_id, triplet.false_candidate.content, checkpoint
            )
        case unreachable:
            assert_never(unreachable)
    validate_memory_envelope(envelope, WriterRegistry.native())
    if (
        event.get("event_id") != envelope.writer_event_id
        or event.get("entry_id") != native.entry_id
        or event.get("writer_stage") != envelope.writer_stage
        or native.content != envelope.content
        or native.content_hash != envelope.content_hash
    ):
        raise OrdinaryAuthorityError("ORDINARY_NATIVE_WRITER_AUTHORITY_INVALID")
    post = serialize_checkpoint(
        NativeState(
            baseline=checkpoint.state.baseline,
            entries=(*checkpoint.state.entries, native),
            native_state=checkpoint.state.native_state,
            schema_version=checkpoint.state.schema_version,
        )
    )
    suite_key = hashlib.sha256(
        f"phase13-fv5-family\0{triplet.triplet_id}".encode()
    ).hexdigest()[:24]
    return {
        "task": triplet.task,
        "baseline": baseline,
        "eligible_position": 9,
        "candidate_family_id": triplet.triplet_id,
        "certified_false_candidate_id": triplet.false_candidate.candidate_id,
        "certified_false_content_hash": triplet.false_candidate.content_hash,
        "challenge_suite_key": suite_key,
        "source_interaction": interaction,
        "writer_event": event,
        "writer_envelope": _json_mapping(asdict(envelope)),
        "native_entry": _json_mapping(native.to_mapping()),
        "pre_native_state_sha256": checkpoint.canonical_sha256,
        "post_native_state_sha256": post.canonical_sha256,
        "post_native_state": _json_mapping(post.state.to_mapping()),
    }


def validate_ordinary_authority(root: Path) -> tuple[dict[str, JsonValue], ...]:
    base = root / "data/phase12/filter_v5_bct_v1"
    try:
        manifest = json.loads(
            (base / "ordinary_route_false_manifest_v1.json").read_text(encoding="utf-8")
        )
        checkpoints = json.loads(
            (base / "checkpoint_manifest_v1.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OrdinaryAuthorityError("ORDINARY_NATIVE_WRITER_AUTHORITY_INVALID") from error
    rows = checkpoints.get("checkpoints") if isinstance(checkpoints, dict) else None
    if not isinstance(rows, list):
        raise OrdinaryAuthorityError("ORDINARY_NATIVE_WRITER_AUTHORITY_INVALID")
    by_key: dict[tuple[str, str], Phase12Checkpoint] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("state"), dict):
            raise OrdinaryAuthorityError("ORDINARY_NATIVE_WRITER_AUTHORITY_INVALID")
        state = NativeState.from_mapping(row["state"])
        checkpoint = serialize_checkpoint(state)
        task = _checkpoint_task(row.get("checkpoint_id"), state.baseline)
        if row.get("sha256") != checkpoint.canonical_sha256 or (task, state.baseline) in by_key:
            raise OrdinaryAuthorityError("ORDINARY_NATIVE_WRITER_AUTHORITY_INVALID")
        by_key[task, state.baseline] = checkpoint
    registry = load_candidate_registry(root / "data/phase12/registries/candidate_registry_v1.json")
    expected = tuple(
        realize_ordinary_false(triplet, baseline, by_key[triplet.task, baseline])
        for triplet in registry.triplets
        for baseline in ("full_history", "rag_frozen", "bot_style", "reflexion_style")
    )
    expected_manifest = {
        "schema_version": "phase13_fv5_ordinary_route_false_manifest_v2",
        "realization_count": 12,
        "realizations": list(expected),
    }
    if manifest != expected_manifest:
        raise OrdinaryAuthorityError("ORDINARY_NATIVE_WRITER_AUTHORITY_INVALID")
    return expected


def _checkpoint_task(value: object, baseline: str) -> str:
    if not isinstance(value, str):
        raise OrdinaryAuthorityError("ORDINARY_NATIVE_WRITER_AUTHORITY_INVALID")
    prefix = "fv5-cal-checkpoint-"
    suffix = f"-{baseline}-v1"
    if not value.startswith(prefix) or not value.endswith(suffix):
        raise OrdinaryAuthorityError("ORDINARY_NATIVE_WRITER_AUTHORITY_INVALID")
    return value[len(prefix) : -len(suffix)]
