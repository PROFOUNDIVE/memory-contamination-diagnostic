from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from memcontam.contamination.phase13_v2_applicability import (
    game24_false_rule_applicable,
    word_sorting_false_rule_applicable,
)
from memcontam.evaluation.phase13_observability_models import (
    MetricValue,
    Phase13ObservabilityError,
)
from memcontam.logging.schema import VerifierResult
from memcontam.readiness.phase13_unicode_15_1 import mcq_normalize, mcq_tokens
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.multiple_choice import verify_answer as verify_mcq_answer
from memcontam.verifiers.game24 import verify_expression
from memcontam.verifiers.math_equation_balancer import _evaluate, _parse_input
from memcontam.verifiers.math_equation_balancer import verify_answer as verify_meb_answer
from memcontam.verifiers.word_sorting import verify_words

REGISTERED_FAILURE_CLASSES: Final = {
    "game24": "G24_CANONICAL_FALSE_RULE_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V2",
    "math_equation_balancer": "MEB_CANONICAL_FALSE_RULE_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V1",
    "word_sorting": "WS_CANONICAL_FALSE_RULE_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V2",
    "mmlu_pro_engineering": "MMLUENG_SURFACE_CUE_HEURISTIC_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V1",
    "mmlu_pro_physics": "MMLUPHY_SURFACE_CUE_HEURISTIC_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V1",
}
AUTHORITY_HASHES: Final = {
    "experiment_design_revised_v14": "293b71dcd0907d6eb5df4a4bb5a80e4b37c0a62a851b851f42ed3deeaaf0865f",
    "protocol_revised_v9": "618320f7129a6b4eccfd66246e54ce8023ca14ede2ae6355677651c70bbbd08e",
}
_HISTORICAL_AUTHORITY_HASHES: Final = {
    "experiment_design_revised_v10": "5597f27d688c19efbcf47dc7369de02a947eac55a5493a69a3aa9098dfe25616",
    "protocol_revised_v8": "022879f559b145e30e645b6ccbd139e9927899d370f1956d27a0562580acf85f",
}
_HISTORICAL_FAILURE_CLASSES: Final = {
    **REGISTERED_FAILURE_CLASSES,
    "game24": "G24_CANONICAL_FALSE_RULE_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V1",
    "word_sorting": "WS_CANONICAL_FALSE_RULE_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V1",
}
VERIFIER_PATHS: Final = {
    "game24": "src/memcontam/verifiers/game24.py",
    "math_equation_balancer": "src/memcontam/verifiers/math_equation_balancer.py",
    "word_sorting": "src/memcontam/verifiers/word_sorting.py",
    "mmlu_pro_engineering": "src/memcontam/tasks/multiple_choice.py",
    "mmlu_pro_physics": "src/memcontam/tasks/multiple_choice.py",
}
APPLICABILITY_PATHS: Final = {
    "game24": "data/phase12/registries/candidate_registry_v2.json",
    "math_equation_balancer": "data/phase12/registries/candidate_registry_v2.json",
    "word_sorting": "data/phase12/registries/candidate_registry_v2.json",
    "mmlu_pro_engineering": "src/memcontam/readiness/phase13_new_mcq_candidate.py",
    "mmlu_pro_physics": "src/memcontam/readiness/phase13_new_mcq_candidate.py",
}


class BoundIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ObservabilityRegistrationPacket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["phase13_observability_registration_packet_v1"]
    packet_id: Literal["OBSERVABILITY_REGISTRATION_PACKET_V1"]
    authority_hashes: dict[str, str]
    failure_classes: dict[str, str]
    recurrence_lookback_h: Literal[10]
    exposure_conditioning: Literal["CURRENT_Z_T_EQUALS_1_PRIOR_MATCH_NEED_NOT_BE_EXPOSED"]
    exact_lineage_recurrence: Literal["SAME_EXACT_ROOT_EXPOSED_AT_BOTH_OCCURRENCES"]
    post_eviction: Literal[
        "EXACT_ROOT_PRESENT_EXPLICITLY_REMOVED_ABSENT_AFTER_NEXT_ORDINARY_ROW_FIRST_RISK"
    ]
    retention: Literal["FIRST_CONTINUOUS_FINITE_WINDOW_EPISODE_NO_GAP_BRIDGING"]
    censoring: Literal["RIGHT_CENSORED_AT_REGISTERED_OR_FIXTURE_ENDPOINT"]
    u_t_status: Literal["NOT_REGISTERED_FOR_CURRENT_MAIN"]
    implementation_identities: dict[str, BoundIdentity]
    verifier_identities: dict[str, BoundIdentity]
    applicability_identities: dict[str, BoundIdentity]

    @model_validator(mode="after")
    def _exact_registry(self) -> ObservabilityRegistrationPacket:
        expected_tasks = set(REGISTERED_FAILURE_CLASSES)
        historical = self.authority_hashes == _HISTORICAL_AUTHORITY_HASHES
        expected_paths = APPLICABILITY_PATHS if not historical else {
            **APPLICABILITY_PATHS,
            **{task: "data/phase12/registries/candidate_registry_v1.json"
               for task in ("game24", "math_equation_balancer", "word_sorting")},
        }
        if (
            self.failure_classes != (_HISTORICAL_FAILURE_CLASSES if historical else REGISTERED_FAILURE_CLASSES)
            or set(self.verifier_identities) != expected_tasks
            or set(self.applicability_identities) != expected_tasks
            or set(self.implementation_identities)
            != {"registration", "sequence", "authority_state"}
            or self.authority_hashes not in (AUTHORITY_HASHES, _HISTORICAL_AUTHORITY_HASHES)
            or {
                task: identity.path for task, identity in self.verifier_identities.items()
            }
            != VERIFIER_PATHS
            or {
                task: identity.path for task, identity in self.applicability_identities.items()
            }
            != expected_paths
        ):
            raise Phase13ObservabilityError("OBSERVABILITY_REGISTRATION_PACKET_STALE")
        return self


def load_registration_packet(path: Path) -> ObservabilityRegistrationPacket:
    return ObservabilityRegistrationPacket.model_validate_json(path.read_bytes())


def registered_verifier_result(task: TaskInstance, answer: str) -> VerifierResult:
    match task.task_name:
        case "game24":
            return verify_expression(answer, task.input["numbers"], task.input.get("target", 24))
        case "math_equation_balancer":
            return verify_meb_answer(answer, task)
        case "word_sorting":
            return verify_words(answer.split(), task.verifier_spec["sorted_words"])
        case "mmlu_pro_engineering" | "mmlu_pro_physics":
            return verify_mcq_answer(answer, task)
        case _:
            raise Phase13ObservabilityError("UNKNOWN_TASK_FAILURE_CLASS")


def registered_failure_class(task: TaskInstance, answer: str, verified_outcome: int) -> str | None:
    verifier = registered_verifier_result(task, answer)
    if int(verifier.is_correct) != verified_outcome:
        raise Phase13ObservabilityError("PRODUCTION_CLASSIFIER_JOIN_MISMATCH")
    match task.task_name:
        case "game24":
            applicable = game24_false_rule_applicable(tuple(task.input["numbers"]))
            eligible = verifier.reason in {"numbers_used_do_not_match", "value_does_not_match_target"}
        case "word_sorting":
            applicable = word_sorting_false_rule_applicable(tuple(task.input["words"]))
            eligible = verifier.reason == "wrong_order"
        case "math_equation_balancer":
            parsed = _parse_input(task.input["input"])
            if parsed is None:
                raise Phase13ObservabilityError("PRODUCTION_CLASSIFIER_JOIN_MISMATCH")
            numbers, target = parsed
            assignments = tuple(product(("+", "-", "*", "/"), repeat=len(numbers) - 1))
            standard = any(_evaluate(numbers, operators) == target for operators in assignments)
            left_to_right = any(_evaluate_left_to_right(numbers, operators) == target for operators in assignments)
            applicable = standard and not left_to_right
            eligible = verifier.reason == "wrong_answer"
        case "mmlu_pro_engineering" | "mmlu_pro_physics":
            scores = tuple((len(mcq_tokens(option)), sum(char != " " for char in mcq_normalize(option)))
                           for option in task.input["options"])
            applicable = scores.count(max(scores)) == 1
            eligible = verifier.parsed_answer in tuple(chr(65 + index) for index in range(len(scores)))
        case _:
            raise Phase13ObservabilityError("UNKNOWN_TASK_FAILURE_CLASS")
    return REGISTERED_FAILURE_CLASSES[task.task_name] if not verified_outcome and applicable and eligible else None


def _evaluate_left_to_right(numbers: tuple[int, ...], operators: tuple[str, ...]) -> Fraction | None:
    value = Fraction(numbers[0])
    for operator, number in zip(operators, numbers[1:], strict=True):
        operand = Fraction(number)
        match operator:
            case "+":
                value += operand
            case "-":
                value -= operand
            case "*":
                value *= operand
            case "/":
                if not operand:
                    return None
                value /= operand
            case _:
                raise Phase13ObservabilityError("PRODUCTION_CLASSIFIER_JOIN_MISMATCH")
    return value


def classify_registered_failure(
    task: str,
    verified_outcome: int,
    precomputed_failure_class: str | None,
    registered_classes: Mapping[str, str] = REGISTERED_FAILURE_CLASSES,
) -> MetricValue:
    expected = registered_classes.get(task)
    if expected is None or precomputed_failure_class not in {None, expected}:
        raise Phase13ObservabilityError("UNKNOWN_TASK_FAILURE_CLASS")
    if verified_outcome == 1:
        if precomputed_failure_class is not None:
            raise Phase13ObservabilityError("CORRECT_RESPONSE_HAS_FAILURE_CLASS")
        return MetricValue(status="supported", reason="NO_REGISTERED_SUBSTANTIVE_FAILURE")
    return MetricValue(
        status="supported",
        value=precomputed_failure_class,
        reason=(
            "PACKET_BOUND_PRECOMPUTED_KAPPA"
            if precomputed_failure_class is not None
            else "NO_REGISTERED_SUBSTANTIVE_FAILURE"
        ),
    )


__all__ = [
    "APPLICABILITY_PATHS",
    "AUTHORITY_HASHES",
    "REGISTERED_FAILURE_CLASSES",
    "VERIFIER_PATHS",
    "BoundIdentity",
    "ObservabilityRegistrationPacket",
    "classify_registered_failure",
    "load_registration_packet",
]
