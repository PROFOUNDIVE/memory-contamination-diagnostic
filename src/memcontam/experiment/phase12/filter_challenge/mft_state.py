from __future__ import annotations

import json
from pathlib import Path
from typing import assert_never

from memcontam.experiment.phase12.filter_challenge.mft_state_models import (
    MFT_STATE_IDS,
    MFT_STATE_SCHEMA_VERSION,
    GateEvaluation,
    MftGateInputs,
    MftGateResult,
    MftMachineObservation,
    MftStateContext,
    MftStateId,
    MftStateMutation,
    MftStateReport,
    mft_state_evidence_hash,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_runtime_gates import (
    no_writeback_gate,
    pair_gate,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_scripted import (
    _FakeClient as _FakeClient,
    answer as _answer,
    build_scripts as _scripts,
    combine,
    failure_attempt_counts,
    persistent_summary,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_scripted_gates import (
    exposure_gate,
    nonharm_gate,
    route_invariance_gate,
    tristate_gate,
)


def run_mft_state_gates(
    context: MftStateContext, *, mutation: MftStateMutation = "none"
) -> MftStateReport:
    evaluations = tuple(
        _evaluate(test_id, context, mutation, "") for test_id in MFT_STATE_IDS
    )
    source_hash = evaluations[0].actual.source_state_before_hash
    if source_hash is None:
        raise ValueError("MFT_RUNTIME_SOURCE_REQUIRED")
    results = []
    for index, (test_id, evaluation) in enumerate(
        zip(MFT_STATE_IDS, evaluations, strict=True), 1
    ):
        inputs = MftGateInputs(
            registry_context=context,
            candidate_entry_id="synthetic-build-candidate",
            source_checkpoint_id="synthetic-build-checkpoint",
            source_state_hash=source_hash,
            scenario_ids=evaluation.scenarios,
        )
        passed = evaluation.expected == evaluation.actual
        draft = MftGateResult(
            test_id=test_id,
            execution_index=index,
            inputs=inputs,
            expected=evaluation.expected,
            actual=evaluation.actual,
            reason="MFT_GATE_PASSED" if passed else evaluation.failure_reason,
            status="pass" if passed else "fail",
            evidence_hash="0" * 64,
        )
        results.append(
            draft.model_copy(update={"evidence_hash": mft_state_evidence_hash(draft)})
        )
    return MftStateReport(ordered_test_ids=MFT_STATE_IDS, results=tuple(results))


def write_mft_state_report(
    output: Path, context: MftStateContext, *, mutation: MftStateMutation = "none"
) -> MftStateReport:
    report = run_mft_state_gates(context, mutation=mutation)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    return report


def _evaluate(
    test_id: MftStateId,
    context: MftStateContext,
    mutation: MftStateMutation,
    _source_hash: str,
) -> GateEvaluation:
    match test_id:
        case "MFT-FV5-01-PAIR-MATCH":
            return pair_gate(mutation)
        case "MFT-FV5-02-EXPOSURE-REQUIRED":
            return exposure_gate(context, mutation == "exposure")
        case "MFT-FV5-03-TRISTATE":
            return tristate_gate(context, mutation == "route")
        case "MFT-FV5-04-FAIL-OPEN":
            return _fail_open_gate(context)
        case "MFT-FV5-05-ROUTE-INVARIANCE":
            return route_invariance_gate(context)
        case "MFT-FV5-06-SCRIPTED-CORRECT":
            return nonharm_gate(context, False, "correct-candidate")
        case "MFT-FV5-07-SCRIPTED-IRRELEVANT":
            return nonharm_gate(context, True, "irrelevant-candidate")
        case "MFT-FV5-08-NO-WRITEBACK":
            return no_writeback_gate(mutation)
        case unreachable:
            assert_never(unreachable)


def _fail_open_gate(context: MftStateContext) -> GateEvaluation:
    failures = (
        ("control-provider", _answer(provider="failed"), _answer(correct=False)),
        ("control-parser", _answer(parse="parse_failed"), _answer(correct=False)),
        ("control-verifier", _answer(verifier="failed"), _answer(correct=False)),
        ("challenge-provider", _answer(), _answer(provider="failed")),
        ("challenge-parser", _answer(), _answer(parse="parse_failed")),
        ("challenge-verifier", _answer(), _answer(verifier="failed")),
    )
    scenario_scripts = tuple(
        _scripts(context, control=control, challenge=challenge, attempts=2)
        for _, control, challenge in failures
    )
    summaries = tuple(
        persistent_summary(context, scripts) for scripts in scenario_scripts
    )
    attempt_counts = tuple(
        _uniform_attempt_count(failure_attempt_counts(scripts))
        for scripts in scenario_scripts
    )
    reasons = (
        "CONTROL_PROVIDER_FAILURE",
        "CONTROL_PARSE_FAILURE",
        "CONTROL_VERIFIER_FAILURE",
        "CHALLENGE_PROVIDER_FAILURE",
        "CHALLENGE_PARSE_FAILURE",
        "CHALLENGE_VERIFIER_FAILURE",
    )
    expected = MftMachineObservation(
        assessment_states=("not_evaluable",) * 6,
        route_targets=("active",) * 6,
        audit_flags=(True,) * 6,
        probe_reason_codes=reasons,
        routing_reason_codes=("FAIL_OPEN_NOT_EVALUABLE",) * 6,
        scripted_attempt_counts=(2,) * 6,
    )
    actual = combine(summaries).model_copy(
        update={
            "probe_reason_codes": tuple(
                summary.probe_reason_codes[0] for summary in summaries
            ),
            "scripted_attempt_counts": attempt_counts,
        }
    )
    return GateEvaluation(
        tuple(item[0] for item in failures),
        expected,
        actual,
        "FAIL_OPEN_ASSERTION_FAILED",
    )


def _uniform_attempt_count(counts: tuple[int, ...]) -> int:
    return counts[0] if len(set(counts)) == 1 else -1


__all__ = (
    "MFT_STATE_IDS",
    "MFT_STATE_SCHEMA_VERSION",
    "MftGateInputs",
    "MftGateResult",
    "MftMachineObservation",
    "MftStateContext",
    "MftStateMutation",
    "MftStateReport",
    "mft_state_evidence_hash",
    "run_mft_state_gates",
    "write_mft_state_report",
)
