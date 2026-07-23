from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memcontam.experiment.phase12.contracts import (
    ExploratoryActivationManifest,
    RouteSelectionManifest,
    SeedAllocationManifest,
)
from memcontam.manifests.run_manifest import (
    build_run_manifest,
    load_run_artifact_refs,
    read_run_manifest,
    validate_run_manifest,
    write_run_manifest,
)


def _models(paths: list[Path], model: Any) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload if isinstance(payload, list) else [payload])
    return {record.manifest_id: record for record in (model.model_validate(item) for item in records)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or validate a canonical Phase 12 run manifest.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--route-selection", type=Path, action="append", default=[])
    parser.add_argument("--seed-allocation", type=Path, action="append", default=[])
    parser.add_argument("--activation", type=Path, action="append", default=[])
    args = parser.parse_args()

    if args.input is not None:
        if args.output is None:
            parser.error("--output is required with --input")
        manifest_hash = write_run_manifest(build_run_manifest(load_run_artifact_refs(args.input)), args.output)
        print(manifest_hash)
        return 0

    manifest = read_run_manifest(args.manifest)
    validate_run_manifest(
        manifest,
        _models(args.route_selection, RouteSelectionManifest),
        _models(args.seed_allocation, SeedAllocationManifest),
        _models(args.activation, ExploratoryActivationManifest),
    )
    print(json.dumps({"valid": True, "rows": len(manifest.rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
