from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from memcontam.experiment.phase12.filter_challenge.bct_live_authorization import runtime_decoding_sha256
from memcontam.experiment.phase12.filter_challenge.freeze_a import validate_freeze_a
from memcontam.experiment.phase12.filter_challenge.registry_calibration import (
    ARTIFACT_ROOT,
    LEDGER_ID,
)


def build_cost_preview(
    args: argparse.Namespace,
    stage: Literal["screening", "bct"],
    validate_config: Callable[[Path], None],
) -> dict[str, object]:
    validate_config(args.config)
    calls, wall_seconds, hard_ceiling = (90, 3600, 2) if stage == "screening" else (480, 7200, 8)
    freeze_path = args.freeze_a if stage == "screening" else args.freeze_b
    if stage == "screening":
        validate_freeze_a(
            args.config,
            args.config.resolve().parents[2] / "data/phase12/filter_v5_bct_v1/source_universe_v1.json",
            freeze_path.parent,
        )
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (AttributeError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("CALIBRATION_FREEZE_INVALID") from error
    if not isinstance(freeze, dict):
        raise ValueError("CALIBRATION_FREEZE_INVALID")
    schedule = freeze.get("method_call_schedule")
    if stage == "screening" and (not isinstance(schedule, list) or len(schedule) != calls):
        raise ValueError("CALL_SCHEDULE_MISMATCH")
    return {
        "schema_version": "phase12_fv5_authorization_request_v1",
        "stage": stage,
        "provider": "openai_responses",
        "model_id": "gpt-4o-2024-11-20",
        "decoding_sha256": runtime_decoding_sha256(args.config, freeze_path),
        "maximum_calls": calls,
        "maximum_input_tokens": 368640 if stage == "screening" else 1966080,
        "maximum_output_tokens": 57600 if stage == "screening" else 307200,
        "wall_seconds": wall_seconds,
        "hard_ceiling_usd": hard_ceiling,
        "ledger_id": LEDGER_ID,
        "artifact_root": str(ARTIFACT_ROOT),
        "freeze_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
        "schedule_sha256": hashlib.sha256(
            json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "approved_plan_sha256": freeze.get("approved_plan_sha256"),
        "provider_calls_issued": 0,
    }
