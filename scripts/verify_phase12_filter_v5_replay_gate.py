from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-config", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()
    manifest = args.run_root / "run.json"
    trials = args.run_root / "trials.jsonl"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        rows = [line for line in trials.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        print("REPLAY_GATE_EVIDENCE_INVALID")
        return 2
    if (
        args.run_root.name != args.expected_run_id
        or payload.get("run_metadata", {}).get("run_id") != args.expected_run_id
        or len(rows) != args.expected_rows
    ):
        print("REPLAY_GATE_EVIDENCE_INVALID")
        return 2
    print("REPLAY_GATE_VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
