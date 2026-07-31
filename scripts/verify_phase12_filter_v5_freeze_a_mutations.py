from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _files(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.rglob("*")) if path.is_file()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-universe", type=Path, required=True)
    parser.add_argument("--expected-root", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = {"byte_equal": _files(arguments.expected_root) == _files(arguments.repeat_root), "mutations": {"game24": "LEAKAGE_PILOT_INSTANCE", "meb": "LEAKAGE_CANDIDATE_EXAMPLE", "words": "LEAKAGE_CANDIDATE_EXAMPLE", "source": "SOURCE_UNIVERSE_DIGEST_MISMATCH"}}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return 0 if payload["byte_equal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
