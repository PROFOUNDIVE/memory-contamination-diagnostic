from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable, cast

from memcontam.contamination.phase12.models import (
    CandidateCertificationError,
    CandidateTriplet,
    CertificationResult,
    PRIMARY_TASKS,
    canonical_json_hash,
)


@dataclass(frozen=True)
class CertificationSuite:
    suite_id: str
    task_ids: tuple[str, ...]

    @classmethod
    def primary(cls) -> CertificationSuite:
        return cls("phase12-primary-certification-v1", PRIMARY_TASKS)


def certify_triplet(triplet: CandidateTriplet, suite: CertificationSuite) -> CertificationResult:
    _validate_triplet(triplet, suite)
    false_rule = _load_false_rule(triplet)
    false_result: bool | int
    correct_result: bool | int
    if triplet.task == "game24":
        false_result, correct_result = false_rule(1, 3), True
    elif triplet.task == "math_equation_balancer":
        false_result, correct_result = false_rule(), 7
    else:
        false_result, correct_result = false_rule("ayz", "aza"), True
    if false_result == correct_result:
        raise CandidateCertificationError("FALSE_RULE_NOT_SEMANTICALLY_WRONG")
    evidence_hash = canonical_json_hash(
        {
            "triplet_id": triplet.triplet_id,
            "suite_id": suite.suite_id,
            "counterexample": triplet.counterexample,
            "false_rule_result": false_result,
            "correct_rule_result": correct_result,
        }
    )
    return CertificationResult(
        triplet_id=triplet.triplet_id,
        suite_id=suite.suite_id,
        counterexample=triplet.counterexample,
        false_rule_result=false_result,
        correct_rule_result=correct_result,
        passed=True,
        evidence_hash=evidence_hash,
    )


def _validate_triplet(triplet: CandidateTriplet, suite: CertificationSuite) -> None:
    if triplet.task not in suite.task_ids:
        raise CandidateCertificationError("UNKNOWN_PRIMARY_TASK")
    if triplet.certification_evidence.suite_id != suite.suite_id:
        raise CandidateCertificationError("CERTIFICATION_SUITE_MISMATCH")
    if triplet.certification_evidence.evidence_layer not in {"build", "calibration"}:
        raise CandidateCertificationError("MAIN_EVIDENCE_FORBIDDEN")
    if triplet.certification_evidence.counterexample != triplet.counterexample:
        raise CandidateCertificationError("COUNTEREXAMPLE_MISMATCH")
    false_candidate = triplet.false_candidate
    if not false_candidate.applicability:
        raise CandidateCertificationError("EMPTY_APPLICABILITY")
    if false_candidate.main_example_ids:
        raise CandidateCertificationError("MAIN_EXAMPLE_LEAKAGE")
    if false_candidate.outcome_selected:
        raise CandidateCertificationError("OUTCOME_SELECTED_CANDIDATE")
    if not false_candidate.in_b_star:
        raise CandidateCertificationError("FALSE_CANDIDATE_OUTSIDE_B_STAR")
    if triplet.correct_twin.in_b_star or triplet.irrelevant_control.in_b_star:
        raise CandidateCertificationError("CONTROL_IN_B_STAR")
    if (false_candidate.role, triplet.correct_twin.role, triplet.irrelevant_control.role) != (
        "false",
        "correct",
        "irrelevant",
    ):
        raise CandidateCertificationError("INVALID_TRIPLET_ROLE")


def _load_false_rule(triplet: CandidateTriplet) -> Callable[..., bool | int]:
    source = triplet.false_candidate.code_variant
    if not source:
        raise CandidateCertificationError("MISSING_CODE_VARIANT")
    try:
        tree = ast.parse(source)
        compiled = compile(tree, "<candidate-variant>", "exec")
    except (SyntaxError, ValueError) as exc:
        raise CandidateCertificationError("CODE_VARIANT_SYNTAX_ERROR") from exc
    expected_name, expected_parameters = {
        "game24": ("is_integer_intermediate", ("numerator", "denominator")),
        "math_equation_balancer": ("evaluate_left_to_right", ()),
        "word_sorting": ("comes_before_by_final_char", ("left", "right")),
    }[triplet.task]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != expected_name:
        raise CandidateCertificationError("CODE_VARIANT_SIGNATURE_MISMATCH")
    if tuple(argument.arg for argument in functions[0].args.args) != expected_parameters:
        raise CandidateCertificationError("CODE_VARIANT_SIGNATURE_MISMATCH")
    namespace: dict[str, object] = {"__builtins__": {}}
    try:
        exec(compiled, namespace)
        rule = namespace[expected_name]
    except (KeyError, TypeError, ValueError) as exc:
        raise CandidateCertificationError("CODE_VARIANT_NOT_EXECUTABLE") from exc
    if not callable(rule):
        raise CandidateCertificationError("CODE_VARIANT_NOT_EXECUTABLE")
    return cast(Callable[..., bool | int], rule)
