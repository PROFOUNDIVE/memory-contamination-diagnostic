from __future__ import annotations

import argparse
import json
from pathlib import Path

from memcontam.experiment.phase12.filter_challenge.final_verifier import (
    FinalVerifierError,
    FinalVerifierRequest,
    verify_final_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("plan-compliance", "code-quality", "integration", "scope", "terminal"))
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-repository-root", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--validation-summary", type=Path, required=True)
    parser.add_argument("--base-commit")
    parser.add_argument("--search-config", type=Path)
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--execution-prerequisites", type=Path)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--f1", type=Path)
    parser.add_argument("--f2", type=Path)
    parser.add_argument("--f3", type=Path)
    parser.add_argument("--f4", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    approvals = tuple(path for path in (arguments.f1, arguments.f2, arguments.f3, arguments.f4) if path)
    try:
        report = verify_final_report(
            FinalVerifierRequest(
                mode=arguments.mode,
                repository_root=arguments.repository_root,
                source_repository_root=arguments.source_repository_root,
                plan=arguments.plan,
                expected_plan_sha256=arguments.expected_plan_sha256,
                evidence_root=arguments.evidence_root,
                validation_summary=arguments.validation_summary,
                output=arguments.output,
                approval_paths=approvals,
                base_commit=arguments.base_commit,
                execution_prerequisites=arguments.execution_prerequisites,
                fixture_root=arguments.fixture_root,
                scratch_root=arguments.scratch_root,
                search_config=arguments.search_config,
            )
        )
    except FinalVerifierError as error:
        print(error.code)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if arguments.mode == "terminal":
        print("FILTER_V5_BUILD_AND_MFT_COMPLETE")
    else:
        print("APPROVE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
