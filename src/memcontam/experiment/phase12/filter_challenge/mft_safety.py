from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

from memcontam.experiment.phase12.filter_challenge.mft_safety_assessment import (
    gate_coverage,
    gate_eligibility,
    gate_parser_boundary,
    gate_probe_invariance,
)
from memcontam.experiment.phase12.filter_challenge.mft_safety_executor import (
    gate_activation,
    gate_control_cache,
    gate_shadow_share,
)
from memcontam.experiment.phase12.filter_challenge.mft_safety_provenance import (
    gate_provenance,
)
from memcontam.experiment.phase12.filter_challenge.mft_safety_types import (
    MFT_SAFETY_FAILURE_REASONS,
    MFT_SAFETY_IDS,
    Gate,
    MftExecutionCount,
    MftSafetyCase,
    MftSafetyError,
    MftSafetyReport,
)


_GATES: Final[tuple[Gate, ...]] = (
    gate_shadow_share,
    gate_parser_boundary,
    gate_control_cache,
    gate_probe_invariance,
    gate_provenance,
    gate_activation,
    gate_eligibility,
    gate_coverage,
)


def build_mft_safety_report(mutations: tuple[str, ...] = ()) -> MftSafetyReport:
    if len(set(mutations)) != len(mutations) or not set(mutations).issubset(MFT_SAFETY_IDS):
        raise MftSafetyError("INVALID_MFT_SAFETY_MUTATION")
    counts = dict.fromkeys(MFT_SAFETY_IDS, 0)
    cases: list[MftSafetyCase] = []
    for test_id, failure_reason, gate in zip(
        MFT_SAFETY_IDS, MFT_SAFETY_FAILURE_REASONS, _GATES, strict=True
    ):
        counts[test_id] += 1
        evidence = gate(test_id in mutations)
        passed = all(assertion.matched for assertion in evidence.assertions)
        case = MftSafetyCase(
            test_id=test_id,
            input_identities=evidence.identities,
            assertions=evidence.assertions,
            status="pass" if passed else "implementation_failure",
            reason_code=None if passed else failure_reason,
            evidence_hash="",
        )
        canonical = json.dumps(
            case.model_dump(mode="json", exclude={"evidence_hash"}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        cases.append(
            case.model_copy(
                update={"evidence_hash": hashlib.sha256(canonical.encode()).hexdigest()}
            )
        )
    frozen_cases = tuple(cases)
    return MftSafetyReport(
        test_ids=MFT_SAFETY_IDS,
        cases=frozen_cases,
        execution_counts=tuple(
            MftExecutionCount(test_id=test_id, count=counts[test_id])
            for test_id in MFT_SAFETY_IDS
        ),
        all_passed=all(case.status == "pass" for case in frozen_cases),
    )


def write_mft_safety_report(
    output: Path, mutations: tuple[str, ...] = ()
) -> MftSafetyReport:
    report = build_mft_safety_report(mutations)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            report.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return report


__all__ = (
    "MFT_SAFETY_FAILURE_REASONS",
    "MFT_SAFETY_IDS",
    "MftSafetyCase",
    "MftSafetyError",
    "MftSafetyReport",
    "build_mft_safety_report",
    "write_mft_safety_report",
)
