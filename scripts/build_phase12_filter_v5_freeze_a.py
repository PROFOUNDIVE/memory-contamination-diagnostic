from __future__ import annotations

import argparse
from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.freeze_a import build_freeze_a


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-universe", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    build_freeze_a(arguments.config, arguments.source_universe, arguments.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
