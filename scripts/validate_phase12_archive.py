from __future__ import annotations

import argparse
import json
from pathlib import Path

from memcontam.manifests.archive_validation import validate_archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Phase 12 archive.")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = validate_archive(args.root)
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 0 if report.archive_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
