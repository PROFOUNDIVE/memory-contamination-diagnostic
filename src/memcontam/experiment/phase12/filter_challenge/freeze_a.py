from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from itertools import combinations, combinations_with_replacement, product
from pathlib import Path
from typing import Final

from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.contamination.phase12.renderers import render_false
from memcontam.experiment.phase12.filter_challenge.calibration_laws import (
    game24_certificate,
    meb_certificate,
    word_sorting_certificate,
)
from memcontam.experiment.phase12.filter_challenge.registry_calibration import screening_schedule
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, serialize_checkpoint
from memcontam.memory.serializer_registry import SerializerRegistry
from memcontam.experiment.phase12.filter_challenge.ordinary_authority import (
    Baseline,
    OrdinaryAuthorityError,
    realize_ordinary_false,
)


TASKS: Final = ("game24", "math_equation_balancer", "word_sorting")
BASELINES: Final[tuple[Baseline, ...]] = (
    "full_history",
    "rag_frozen",
    "bot_style",
    "reflexion_style",
)
_EXCLUDED: Final = {
    "game24": {"3,3,8,8"},
    "math_equation_balancer": {"1,2,3,7"},
    "word_sorting": {"ayz|aza"},
}


class FreezeAError(ValueError):
    pass


def _canonical(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(payload))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repository_root(config: Path) -> Path:
    return config.resolve().parents[2]


def _validate_source_universe(path: Path, root: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "phase12_fv5_source_universe_v1":
        raise FreezeAError("SOURCE_UNIVERSE_DIGEST_MISMATCH")
    files = payload.get("source_files")
    if not isinstance(files, dict):
        raise FreezeAError("SOURCE_UNIVERSE_DIGEST_MISMATCH")
    for relative, digest in files.items():
        if not isinstance(relative, str) or not isinstance(digest, str) or _sha(root / relative) != digest:
            raise FreezeAError("SOURCE_UNIVERSE_DIGEST_MISMATCH")
    return payload


def _game24() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for numbers in combinations_with_replacement(range(1, 10), 4):
        certificate = game24_certificate(numbers)
        signature = ",".join(map(str, numbers))
        if certificate is not None and signature not in _EXCLUDED["game24"]:
            records.append({"probe_id": f"fv5-cal-game24-{len(records) + 1:03d}", "certificate": certificate})
        if len(records) == 6:
            return records
    raise FreezeAError("GAME24_POOL_EXHAUSTED")


def _meb() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for a, b, c, target in product(range(1, 10), range(1, 10), range(1, 10), range(-24, 25)):
        certificate = meb_certificate(a, b, c, target)
        signature = f"{a},{b},{c},{target}"
        if certificate is not None and signature not in _EXCLUDED["math_equation_balancer"]:
            records.append({"probe_id": f"fv5-cal-meb-{len(records) + 1:03d}", "certificate": certificate})
        if len(records) == 6:
            return records
    raise FreezeAError("MEB_POOL_EXHAUSTED")


def _words() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    words = tuple("".join(value) for value in product("abcdef", repeat=3))
    for combination in combinations(words, 3):
        certificate = word_sorting_certificate(combination)
        witnesses = "|".join(sorted((combination[0], combination[1])))
        if certificate is not None and witnesses not in _EXCLUDED["word_sorting"]:
            records.append({"probe_id": f"fv5-cal-words-{len(records) + 1:03d}", "certificate": certificate})
        if len(records) == 6:
            return records
    raise FreezeAError("WORD_POOL_EXHAUSTED")


def _state(task: str, baseline: str) -> NativeState:
    schema = SerializerRegistry.native().schema_for(baseline)
    entries = tuple(
        NativeEntry(
            entry_id=f"fv5-cal-construction-{task}-{baseline}-{position}",
            semantic_kind=schema.semantic_kind,
            schema_version="phase12_native_entry_v1",
            native_component=schema.native_component,
            content=f"calibration construction {task} position {position}",
            content_hash=hashlib.sha256(f"{task}:{baseline}:{position}".encode()).hexdigest(),
        )
        for position in (7, 8)
    )
    native_state: dict[str, list[object]] = {"records": []} if baseline == "full_history" else {"entries": []}
    return NativeState(baseline=baseline, entries=entries, native_state=native_state)


def _construction(
    root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[Mapping[str, object]]]:
    registry = load_candidate_registry(root / "data/phase12/registries/candidate_registry_v1.json")
    renders: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    ordinary: list[Mapping[str, object]] = []
    for triplet in registry.triplets:
        for baseline in BASELINES:
            checkpoint = serialize_checkpoint(_state(triplet.task, baseline))
            checkpoints.append({"checkpoint_id": f"fv5-cal-checkpoint-{triplet.task}-{baseline}-v1", "sha256": checkpoint.canonical_sha256, "state": checkpoint.state.to_mapping()})
            for candidate in (triplet.false_candidate, triplet.correct_twin, triplet.irrelevant_control):
                entry = render_false(baseline, triplet, checkpoint) if candidate.role == "false" else NativeEntry(
                    entry_id=candidate.candidate_id, semantic_kind=SerializerRegistry.native().schema_for(baseline).semantic_kind,
                    schema_version="phase12_native_entry_v1", native_component=SerializerRegistry.native().schema_for(baseline).native_component,
                    content=candidate.content, content_hash=candidate.content_hash, render_id=candidate.render_id,
                )
                renders.append({"task": triplet.task, "baseline": baseline, "entry": entry.to_mapping()})
            ordinary.append(realize_ordinary_false(triplet, baseline, checkpoint))
    return renders, checkpoints, ordinary


def build_freeze_a(config: Path, source_universe: Path, output_root: Path) -> Path:
    root = _repository_root(config)
    source = _validate_source_universe(source_universe, root)
    probes = {"game24": _game24(), "math_equation_balancer": _meb(), "word_sorting": _words()}
    probe_ids = {task: tuple(str(record["probe_id"]) for record in probes[task]) for task in TASKS}
    renders, checkpoints, ordinary = _construction(root)
    manifests = {
        "split_reservation_v1.json": {"schema_version": "phase12_fv5_split_reservation_v1", "reserved_extension_signatures": [], "excluded_signatures": {key: sorted(value) for key, value in _EXCLUDED.items()}},
        "probe_construction_manifest_v1.json": {"schema_version": "phase12_fv5_probe_construction_manifest_v1", "probes": probes},
        "candidate_triplets_v1.json": {"schema_version": "phase12_fv5_candidate_triplets_v1", "render_count": len(renders), "renders": renders},
        "checkpoint_manifest_v1.json": {"schema_version": "phase12_fv5_checkpoint_manifest_v1", "checkpoint_count": len(checkpoints), "checkpoints": checkpoints},
        "ordinary_route_false_manifest_v1.json": {"schema_version": "phase13_fv5_ordinary_route_false_manifest_v2", "realization_count": len(ordinary), "realizations": ordinary},
        "leakage_disjointness_report.json": {"schema_version": "phase12_fv5_leakage_disjointness_report_v1", "excluded_signatures": {key: sorted(value) for key, value in _EXCLUDED.items()}, "candidate_conditioned_outcomes": False},
    }
    for name, payload in manifests.items():
        _write(output_root / name, payload)
    control_schedule = [{"task": task, "baseline": baseline, "probe_id": probe} for task in TASKS for baseline in BASELINES for probe in probe_ids[task]]
    method_calls = [item.model_dump(mode="json") for item in screening_schedule(probe_ids)]
    freeze = {
        "schema_version": "phase12_fv5_freeze_a_v1", "approved_plan_sha256": source["approved_plan_sha256"],
        "probes": probe_ids, "control_schedule": control_schedule, "method_call_schedule": method_calls,
        "manifest_sha256": {name: _sha(output_root / name) for name in manifests}, "provider_calls_issued": 0,
        "provider": "openai_responses", "model_id": "gpt-4o-2024-11-20", "tool_mode": "text_only",
        "call_capacity": {"maximum_calls": 90, "input_tokens": 368640, "output_tokens": 57600, "wall_seconds": 3600, "hard_ceiling_usd": 2},
    }
    path = output_root / "freeze_a.json"
    _write(path, freeze)
    return path


def validate_freeze_a(config: Path, source_universe: Path, output_root: Path) -> dict[str, object]:
    root = _repository_root(config)
    _validate_source_universe(source_universe, root)
    try:
        freeze = json.loads((output_root / "freeze_a.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreezeAError("FREEZE_A_INVALID") from error
    if not isinstance(freeze, dict) or freeze.get("schema_version") != "phase12_fv5_freeze_a_v1":
        raise FreezeAError("FREEZE_A_INVALID")
    manifests = freeze.get("manifest_sha256")
    if not isinstance(manifests, dict):
        raise FreezeAError("FREEZE_A_INVALID")
    for name, digest in manifests.items():
        if not isinstance(name, str) or not isinstance(digest, str) or _sha(output_root / name) != digest:
            raise FreezeAError("FREEZE_A_MANIFEST_HASH_MISMATCH")
    probes = freeze.get("probes")
    if not isinstance(probes, dict) or tuple(probes) != TASKS or any(not isinstance(probes[task], list | tuple) or len(probes[task]) != 6 for task in TASKS):
        raise FreezeAError("CALIBRATION_PROBE_SCHEDULE_INVALID")
    construction = json.loads((output_root / "probe_construction_manifest_v1.json").read_text(encoding="utf-8"))
    if not isinstance(construction, dict) or not isinstance(construction.get("probes"), dict):
        raise FreezeAError("FREEZE_A_INVALID")
    for task, records in construction["probes"].items():
        if not isinstance(task, str) or not isinstance(records, list):
            raise FreezeAError("FREEZE_A_INVALID")
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("certificate"), dict):
                raise FreezeAError("FREEZE_A_INVALID")
            certificate = record["certificate"]
            signature = str(certificate.get("input_canonical"))
            if signature in _EXCLUDED.get(task, set()):
                raise FreezeAError("LEAKAGE_PILOT_INSTANCE" if task == "game24" else "LEAKAGE_CANDIDATE_EXAMPLE")
    try:
        expected_ordinary = _construction(root)[2]
    except OrdinaryAuthorityError as error:
        raise FreezeAError(error.code) from error
    ordinary = json.loads(
        (output_root / "ordinary_route_false_manifest_v1.json").read_text(encoding="utf-8")
    )
    if ordinary != {
        "schema_version": "phase13_fv5_ordinary_route_false_manifest_v2",
        "realization_count": 12,
        "realizations": expected_ordinary,
    }:
        raise FreezeAError("ORDINARY_NATIVE_WRITER_AUTHORITY_INVALID")
    if len(freeze.get("control_schedule", [])) != 72 or len(freeze.get("method_call_schedule", [])) != 90:
        raise FreezeAError("CALL_SCHEDULE_MISMATCH")
    return freeze
