from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from memcontam.clients.config import ProviderConfig
from memcontam.clients.provider_profile import normalize_provider_profile
from memcontam.cli import load_config
from memcontam.config.resolution import resolve_run_config
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    read_regular_nofollow,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-config", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()
    manifest = args.run_root / "run.json"
    trials = args.run_root / "trials.jsonl"
    resolved_config = args.run_root / "resolved_config.json"
    try:
        payload = json.loads(read_regular_nofollow(manifest, "REPLAY_GATE_EVIDENCE_INVALID"))
        resolved = json.loads(read_regular_nofollow(resolved_config, "REPLAY_GATE_EVIDENCE_INVALID"))
        rows = [line for line in read_regular_nofollow(trials, "REPLAY_GATE_EVIDENCE_INVALID").splitlines() if line]
        expected = _expected_config(Path(args.expected_config))
        config_hash = _config_hash(expected)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, SystemExit):
        print("REPLAY_GATE_EVIDENCE_INVALID")
        return 2
    if (
        args.run_root.name != args.expected_run_id
        or payload.get("run_metadata", {}).get("run_id") != args.expected_run_id
        or len(rows) != args.expected_rows
        or payload.get("run_metadata", {}).get("config_hash") != config_hash
        or _config_hash(resolved) != config_hash
    ):
        print("REPLAY_GATE_EVIDENCE_INVALID")
        return 2
    print("REPLAY_GATE_VALID")
    return 0


def _config_hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _expected_config(path: Path) -> object:
    config = load_config(path)
    provider = ProviderConfig.from_run_config(config)
    profile = normalize_provider_profile(
        provider,
        served_models=config["models"],
        model_snapshots=config.get("run", {}).get("model_snapshots", {}),
    )
    return resolve_run_config(config, provider_profile=profile)


if __name__ == "__main__":
    raise SystemExit(main())
