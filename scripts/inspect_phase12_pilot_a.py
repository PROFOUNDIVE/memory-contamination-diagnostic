from __future__ import annotations

import argparse
import json
from pathlib import Path

from memcontam.readiness.pilot_a_invariants import inspect_run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_run(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
