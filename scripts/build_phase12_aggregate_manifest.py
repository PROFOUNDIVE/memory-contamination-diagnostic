from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memcontam.manifests.aggregate_manifest import (
    build_aggregate_manifest,
    read_aggregate_manifest,
    validate_aggregate_manifest,
    write_aggregate_manifest,
)
from memcontam.manifests.claim_scope import (
    build_claim_scope,
    read_claim_scope,
    validate_claim_scope,
    write_claim_scope,
)
from memcontam.manifests.run_manifest import read_run_manifest


def _records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("rows", payload.get("aggregates", payload.get("claims")))
    if not isinstance(payload, list) or any(not isinstance(record, dict) for record in payload):
        raise ValueError("MANIFEST_INPUT_INVALID")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or validate Phase 12 aggregate manifests.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--claims", type=Path)
    parser.add_argument("--claim-output", type=Path)
    parser.add_argument("--claim-scope", type=Path)
    args = parser.parse_args()

    run_manifest = read_run_manifest(args.run_manifest)
    if args.input is not None:
        if args.output is None:
            parser.error("--output is required with --input")
        aggregate_manifest = build_aggregate_manifest(_records(args.input), run_manifest)
        result: dict[str, Any] = {
            "aggregate_manifest_hash": write_aggregate_manifest(aggregate_manifest, args.output)
        }
        if args.claims is not None:
            if args.claim_output is None:
                parser.error("--claim-output is required with --claims")
            ledger = build_claim_scope(_records(args.claims), aggregate_manifest)
            result["claim_scope_hash"] = write_claim_scope(ledger, args.claim_output)
        print(json.dumps(result, sort_keys=True))
        return 0

    aggregate_manifest = read_aggregate_manifest(args.manifest)
    validate_aggregate_manifest(aggregate_manifest, run_manifest)
    if args.claim_scope is not None:
        validate_claim_scope(read_claim_scope(args.claim_scope), aggregate_manifest)
    print(json.dumps({"valid": True, "rows": len(aggregate_manifest.rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
