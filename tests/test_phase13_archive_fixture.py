from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "data/phase13/authority"
STREAM_HASH = "9f74ad462d286796e671544745f55cae323eb48aed22e151638ed99345230bb8"
CHECKPOINT = "checkpoint-3de74961a1870cb9"
CHECKPOINT_SHA256 = "3de74961a1870cb9bbbca13d9bf7b2148eb3c3409ee84a6d1679e44bdb67bb9f"
FAMILY = "game24-h5-primary-holm-v1"
HASHES = {
    "execution": "7c92189f645b74381f3fedf2d3ffbc8c4768a7019eaeb621edb8c22fb4a34970",
    "analysis": "b58e6aec8acc040fb934e9b25842eb68c702d098a08b41ba0eab9502a198a0f3",
    "historical": "446e5634d7be2bd049ffd3af733262e72a076d22ec24a0e9c11d7259b60264d4",
    "checkpoint": "c2173d1fb5557611050a7e281fcf0613671bda06ffad6e7cc370de568f37ecff",
}


def _event(event_time: int, score: int) -> dict[str, Any]:
    previous = (
        CHECKPOINT_SHA256
        if event_time == 0
        else hashlib.sha256(f"state-{event_time}".encode()).hexdigest()
    )
    following = hashlib.sha256(f"state-{event_time + 1}".encode()).hexdigest()
    return {
        "event_id": f"event-{event_time}",
        "event_time": event_time,
        "absolute_trial_index": event_time + 2,
        "source_checkpoint_id": CHECKPOINT,
        "baseline": "fh_bounded",
        "arm": "clean",
        "task": "game24",
        "model": "gpt-4o-2024-11-20",
        "session_id": "session-10000",
        "native_state_id": "phase13-native-capacity-v1",
        "intervention_id": None,
        "state_before_sha256": previous,
        "state_after_sha256": following,
        "write_event_ids": [f"write-{event_time}"],
        "retention_event_ids": [f"retain-{event_time}"],
        "eviction_event_ids": [],
        "lineage_parent_ids": [] if event_time == 0 else [f"event-{event_time - 1}"],
        "semantic_call_id": f"call-{event_time}",
        "call_owner_id": "phase13-h10-execution-owner-v1",
        "verified_score": score,
        "status": "succeeded",
    }


def complete_archive(root: Path) -> dict[str, Any]:
    events = [_event(index, index % 2) for index in range(10)]
    raw = root / "source.jsonl"
    raw.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in events), encoding="utf-8")
    source_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    invalidated_hash = hashlib.sha256(raw.read_bytes().splitlines(keepends=True)[0]).hexdigest()
    attempts = [
        {
            "attempt_id": "attempt-invalid",
            "source_run_id": "game24-seed-10000-invalid",
            "status": "invalidated",
            "invalidated_reason": "PROVIDER_PARTIAL_RESPONSE",
            "raw_evidence_sha256": invalidated_hash,
            "rerun_parent_id": None,
            "source_raw_path": str(raw),
            "source_raw_sha256": source_hash,
            "raw_record_range": [0, 0],
            "events": [deepcopy(events[0])],
        },
        {
            "attempt_id": "attempt-complete",
            "source_run_id": "game24-seed-10000",
            "source_manifest_id": "game24-seed-10000",
            "source_ordered_stream_sha256": STREAM_HASH,
            "execution_contract_id": "phase13-main-a-h10-execution-v1",
            "status": "completed",
            "invalidated_reason": None,
            "raw_evidence_sha256": source_hash,
            "rerun_parent_id": "attempt-invalid",
            "source_raw_path": str(raw),
            "source_raw_sha256": source_hash,
            "raw_record_range": [0, 9],
            "events": events,
        },
    ]
    provider_calls = [
        {
            "semantic_call_id": f"call-{index}",
            "execution_owner_id": "phase13-h10-execution-owner-v1",
            "transport_attempt_ids": [f"transport-{index}"],
        }
        for index in range(10)
    ]
    payload = {
        "schema_version": "phase13_archive_v2",
        "authorities": {
            name: {"path": str(AUTHORITY / f"{filename}.json"), "sha256": HASHES[name]}
            for name, filename in {
                "execution": "execution_registry_v1",
                "analysis": "analysis_registry_v1",
                "historical": "historical_compatibility_v1",
                "checkpoint": "structural_checkpoint_registry_v1",
            }.items()
        },
        "source_attempts": attempts,
        "derived_windows": [
            {
                "window_id": f"game24-seed-10000-{window_id}",
                "analysis_window_id": window_id,
                "source_run_id": "game24-seed-10000",
                "source_raw_sha256": source_hash,
                "source_event_range": [0, length - 1],
                "event_ids": [f"event-{index}" for index in range(length)],
                "window_length": length,
                "evidence_status": evidence,
                "multiplicity_status": multiplicity,
                "provider_calls": 0,
                "owner_id": "phase13-offline-compute-owner-v1",
            }
            for window_id, length, evidence, multiplicity in (
                ("accuracy-h2-sensitivity", 2, "prespecified_sensitivity", "descriptive_no_inferential_family"),
                ("recurrence-h2-descriptive", 2, "descriptive", "estimation_only"),
                ("accuracy-h5-primary", 5, "confirmatory_primary", "primary_holm_family"),
                ("recurrence-h5-secondary", 5, "confirmatory_secondary", "estimation_only"),
                ("persistence-h5-secondary", 5, "confirmatory_secondary", "estimation_only"),
                ("propagation-h5-conditional", 5, "descriptive", "descriptive_no_inferential_family"),
                ("collapse-h5-exploratory", 5, "exploratory", "descriptive_no_inferential_family"),
            )
        ],
        "provider_ledger": provider_calls,
        "offline_ledger": [
            {
                "operation": operation,
                "owner_id": "phase13-offline-compute-owner-v1",
                "provider_calls": 0,
                "cost_microusd": 0,
            }
            for operation in ("prefix_derivation", "paired_seed_bootstrap", "report_rendering")
        ],
        "aggregates": [{
            "aggregate_id": "game24-h5-score",
            "source_ids": ["game24-seed-10000-accuracy-h5-primary"],
            "status": "ESTIMABLE",
            "family_id": FAMILY,
            "original_weights": {"game24-seed-10000": 1.0},
            "weights": {"game24-seed-10000": 1.0},
            "estimate": 0.4,
        }],
        "claims": [{
            "claim_id": "game24-h5-score-claim",
            "aggregate_id": "game24-h5-score",
            "status": "supported",
            "family_id": FAMILY,
            "estimate": 0.4,
        }],
        "historical_reference": {
            "run_id": "phase13-pre-main-calibration-15usd-rerun1",
            "availability": "external_reference_unavailable",
            "imported": False,
        },
    }
    write_archive(root, payload)
    return payload


def write_archive(root: Path, payload: dict[str, Any]) -> None:
    (root / "phase13_archive.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def resign_source(root: Path, payload: dict[str, Any], attempt_index: int = 1) -> None:
    attempt = payload["source_attempts"][attempt_index]
    raw = root / f"source-{attempt_index}.jsonl"
    raw.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in attempt["events"]),
        encoding="utf-8",
    )
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    attempt["source_raw_path"] = str(raw)
    attempt["source_raw_sha256"] = digest
    attempt["raw_record_range"] = [0, len(attempt["events"]) - 1]
    if attempt["status"] == "completed":
        attempt["raw_evidence_sha256"] = digest
        for window in payload["derived_windows"]:
            if window["source_run_id"] == attempt["source_run_id"]:
                window["source_raw_sha256"] = digest


def mutate(payload: dict[str, Any], change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    changed = deepcopy(payload)
    change(changed)
    return changed
