from __future__ import annotations

import argparse
import json
from pathlib import Path

from memcontam.readiness.pilot_a_invariants import inspect_run
from memcontam.readiness.pilot_a_scientific_archive import validate_scientific_archive


def inspect_pilot_a(run_dir: Path) -> dict[str, object]:
    if (run_dir / "decision_ledger.json").is_file():
        return validate_scientific_archive(run_dir)
    return inspect_run(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_pilot_a(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
