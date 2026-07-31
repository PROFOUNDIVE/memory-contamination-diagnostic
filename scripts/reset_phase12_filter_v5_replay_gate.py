from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-run-id", required=True)
    args = parser.parse_args()
    if args.run_root.name != args.expected_run_id:
        print("REPLAY_GATE_RUN_ID_INVALID")
        return 2
    if not args.run_root.exists():
        return 0
    manifest = args.run_root / "run.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        print("REPLAY_GATE_OWNERSHIP_INVALID")
        return 2
    if payload.get("run_metadata", {}).get("run_id") != args.expected_run_id:
        print("REPLAY_GATE_OWNERSHIP_INVALID")
        return 2
    shutil.rmtree(args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
