#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# ///
# How to run: python scripts/build_phase12_filter_mft.py --output /tmp/filter-v4-mft.json
from __future__ import annotations

import argparse
import json
from pathlib import Path

from memcontam.experiment.phase12.filter_mft import write_filter_mft_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = write_filter_mft_report(args.output)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
