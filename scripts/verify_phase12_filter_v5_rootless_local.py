from __future__ import annotations

import argparse
import json
from pathlib import Path


PROFILE = "local_rootless_non_authoritative"


def verify(repository: Path) -> int:
    repository = repository.resolve(strict=True)
    evidence = repository / "docs/evidence/phase12-filter-v5-rootless-local"
    required = (
        evidence / "claim-boundary.md",
        evidence / "post-bct-review-contract.md",
        evidence / "final-verification-index.schema.json",
    )
    if not all(path.is_file() and not path.is_symlink() for path in required):
        return 64
    if any(PROFILE not in path.read_text(encoding="utf-8") for path in required[:2]):
        return 64
    schema = json.loads(required[2].read_bytes())
    if schema.get("properties", {}).get("profile", {}).get("const") != PROFILE:
        return 64
    publication = evidence / "rehearsal-publication.json"
    if publication.exists() and json.loads(publication.read_bytes()).get("profile") != PROFILE:
        return 64
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    return verify(parser.parse_args().repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
