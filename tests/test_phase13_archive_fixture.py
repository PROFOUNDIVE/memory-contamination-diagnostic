from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "data/phase13/authority"
HASHES = {
    "execution": "7c92189f645b74381f3fedf2d3ffbc8c4768a7019eaeb621edb8c22fb4a34970",
    "analysis": "b58e6aec8acc040fb934e9b25842eb68c702d098a08b41ba0eab9502a198a0f3",
    "historical": "446e5634d7be2bd049ffd3af733262e72a076d22ec24a0e9c11d7259b60264d4",
    "checkpoint": "c2173d1fb5557611050a7e281fcf0613671bda06ffad6e7cc370de568f37ecff",
}


def _event(event_time: int, score: int) -> dict[str, Any]:
    previous = hashlib.sha256(f"state-{event_time}".encode()).hexdigest()
    following = hashlib.sha256(f"state-{event_time + 1}".encode()).hexdigest()
    return {
        "event_id": f"event-{event_time}",
        "event_time": event_time,
        "absolute_trial_index": event_time + 2,
        "source_checkpoint_id": "checkpoint-game24-seed-10000-fh-bounded-clean",
        "task": "game24",
        "model": "gpt-4o-2024-11-20",
        "session_id": "session-10000",
        "native_state_id": "phase13-native-capacity-registry-v1",
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
    attempts = [
        {
            "attempt_id": "attempt-invalid",
            "source_run_id": "game24-seed-10000-invalid",
            "status": "invalidated",
            "invalidated_reason": "PROVIDER_PARTIAL_RESPONSE",
            "raw_evidence_sha256": "1" * 64,
            "rerun_parent_id": None,
            "source_raw_path": str(raw),
            "source_raw_sha256": source_hash,
            "raw_record_range": [0, 0],
            "events": [events[0]],
        },
        {
            "attempt_id": "attempt-complete",
            "source_run_id": "game24-seed-10000",
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
        "derived_windows": [{
            "window_id": "game24-seed-10000-h5",
            "source_run_id": "game24-seed-10000",
            "source_raw_sha256": source_hash,
            "source_event_range": [0, 4],
            "event_ids": [f"event-{index}" for index in range(5)],
            "window_length": 5,
            "status": "ESTIMABLE",
            "family_id": "phase13-primary-seven-slot-family-game24-v1",
            "provider_calls": 0,
            "owner_id": "phase13-offline-compute-owner-v1",
        }],
        "provider_ledger": provider_calls,
        "offline_ledger": [
            {"operation": operation, "owner_id": "phase13-offline-compute-owner-v1", "provider_calls": 0}
            for operation in ("prefix_derivation", "paired_seed_bootstrap", "report_rendering")
        ],
        "aggregates": [{
            "aggregate_id": "game24-h5-score",
            "source_ids": ["game24-seed-10000-h5"],
            "status": "ESTIMABLE",
            "family_id": "phase13-primary-seven-slot-family-game24-v1",
            "original_weights": {"game24-seed-10000": 1.0},
            "weights": {"game24-seed-10000": 1.0},
            "estimate": 0.4,
        }],
        "claims": [{
            "claim_id": "game24-h5-score-claim",
            "aggregate_id": "game24-h5-score",
            "status": "supported",
            "family_id": "phase13-primary-seven-slot-family-game24-v1",
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


def mutate(payload: dict[str, Any], change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    changed = deepcopy(payload)
    change(changed)
    return changed
