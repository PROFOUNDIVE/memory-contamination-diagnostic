from __future__ import annotations

import argparse
import json
from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.evidence import (
    EvidenceBuildRequest,
    build_evidence_bundle,
)
from memcontam.experiment.phase12.filter_challenge.evidence_contract import EvidenceBuildError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        bundle = build_evidence_bundle(
            EvidenceBuildRequest(
                repository_root=arguments.repository_root,
                plan=arguments.plan,
                expected_plan_sha256=arguments.expected_plan_sha256,
                implementation_commit=arguments.implementation_commit,
                search_config=arguments.search_config,
                fixture_root=arguments.fixture_root,
                validation_summary=arguments.validation_summary,
                output_root=arguments.output_root,
            )
        )
    except EvidenceBuildError as error:
        print(error.code)
        return 2
    print(
        json.dumps(
            {
                "files": sorted(path.name for path in bundle.root.iterdir()),
                "implementation_manifest_sha256": bundle.implementation_manifest_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
