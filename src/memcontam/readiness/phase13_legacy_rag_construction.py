from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias, assert_never

from pydantic import BaseModel, ConfigDict

from memcontam.tasks.base import TaskInstance
from memcontam.verifiers.game24 import verify_expression
from memcontam.verifiers.math_equation_balancer import verify_answer
from memcontam.verifiers.word_sorting import verify_words

from .phase13_legacy_rag_bytes import JsonValue, canonical_json_bytes
from .phase13_legacy_rag_generators import (
    Game24Candidate,
    MebCandidate,
    WordSortingCandidate,
    game24_candidates,
    meb_candidates,
    word_sorting_candidates,
)
from .phase13_legacy_rag_models import (
    ArtifactReference,
    BuildCandidate,
    BuildRegistry,
    CandidateAuditRecord,
    FeasibleTaskName,
    GeneratorIdentity,
)


Candidate = Game24Candidate | MebCandidate | WordSortingCandidate
FileCalibrationTask: TypeAlias = Literal["game24", "word_sorting"]
_ADDENDUM_PATH: Final = (
    "References/Theoretical Artifacts/"
    "2026-08-24_Phase13_MainA_PostCutoff_Acceleration_Addendum.md"
)
_ADDENDUM_SHA256: Final = "d971c24439cc551655e9e1f5dbba6efa5a27242802f1db66a32749ec61350edc"
_GENERATOR_PATH: Final = "src/memcontam/readiness/phase13_legacy_rag_generators.py"
CALIBRATION_PATHS: Final = {
    "game24": "data/tasks/game24_pilot.jsonl",
    "word_sorting": "data/tasks/word_sorting_pilot.jsonl",
}
_DOMAINS: Final = {
    "game24": "1<=a<=b<=c<=d<=13;target=24;exact-rational-Game24-law",
    "math_equation_balancer": "m-in-{3,4};ordered-operands-in-{1,...,9};exact-rational",
    "word_sorting": "all-five-token-subsets-of-frozen-32-token-vocabulary",
}
_KEY_SCHEMAS: Final = {
    "game24": "{generator,numbers[4],target}",
    "math_equation_balancer": (
        "{generator,ordered_operands[m],target_value,canonical_operator_tuple[m-1]}"
    ),
    "word_sorting": "{generator,input_words[5],sorted_words[5]}",
}
_GENERATOR_IDS: Final = {
    "game24": "legacy_game24_build_generator_v1",
    "math_equation_balancer": "legacy_meb_build_generator_v1",
    "word_sorting": "legacy_word_sorting_build_generator_v1",
}


class _Game24Calibration(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    numbers: tuple[int, ...]
    target: int


class _WordSortingCalibration(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    words: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BuildRegistrySource:
    repository_root: Path
    task: FeasibleTaskName
    calibration_hashes: tuple[str, ...]
    evaluation_exclusion_hashes: tuple[str, ...]
    calibration_registry_path: str
    calibration_registry_id: str
    calibration_registry_sha256: str
    calibration_selection_law: str
    build_partition_law: str
    historical_pilot_status: Literal["HISTORICAL_EVIDENCE_ONLY"] | None
    leakage_calibration_artifact: ArtifactReference | None
    opaque_hash: str
    candidates: tuple[Candidate, ...]


def calibration_path(repository_root: Path, task: FileCalibrationTask) -> Path:
    return repository_root / CALIBRATION_PATHS[task]


def calibration_signatures(task: FileCalibrationTask, path: Path) -> tuple[str, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    match task:
        case "game24":
            rows = (_Game24Calibration.model_validate_json(line) for line in lines)
            values: tuple[JsonValue, ...] = tuple(
                {"numbers": list(sorted(row.numbers)), "target": row.target} for row in rows
            )
        case "word_sorting":
            word_rows = (_WordSortingCalibration.model_validate_json(line) for line in lines)
            values = tuple(
                {
                    "tokens": [
                        unicodedata.normalize("NFC", word) for word in sorted(row.words)
                    ]
                }
                for row in word_rows
            )
        case unreachable:
            assert_never(unreachable)
    return tuple(sorted(hashlib.sha256(canonical_json_bytes(value)).hexdigest() for value in values))


def generated_candidates(
    task: FeasibleTaskName, exclusions: frozenset[str]
) -> tuple[Candidate, ...]:
    match task:
        case "game24":
            return game24_candidates(exclusions, limit=64)
        case "math_equation_balancer":
            return meb_candidates(exclusions, limit=64)
        case "word_sorting":
            return word_sorting_candidates(exclusions, limit=64)
        case unreachable:
            assert_never(unreachable)


def build_registry(source: BuildRegistrySource) -> BuildRegistry:
    repository_root = source.repository_root
    task = source.task
    generator_id = _GENERATOR_IDS[task]
    generator_path = repository_root / _GENERATOR_PATH
    candidate_signatures = {
        candidate.canonical_signature for candidate in source.candidates
    }
    if candidate_signatures & set(source.calibration_hashes):
        message = "legacy RAG build/calibration partition overlap"
        raise ValueError(message)
    if candidate_signatures & set(source.evaluation_exclusion_hashes):
        message = "legacy RAG build/evaluation exclusion overlap"
        raise ValueError(message)
    audits = tuple(_audit_candidate(candidate) for candidate in source.candidates)
    return BuildRegistry(
        schema_version="phase13_legacy_rag_build_registry_v1",
        task_id=task,
        build_source_contract_id="legacy_rag_build_source_contract_v2",
        canonical_byte_contract_id="legacy_rag_canonical_bytes_v1",
        generator=GeneratorIdentity(
            generator_id=generator_id,
            implementation=ArtifactReference(
                path=_GENERATOR_PATH,
                sha256=hashlib.sha256(generator_path.read_bytes()).hexdigest(),
            ),
            authority=ArtifactReference(path=_ADDENDUM_PATH, sha256=_ADDENDUM_SHA256),
            candidate_domain=_DOMAINS[task],
            candidate_key_schema=_KEY_SCHEMAS[task],
            candidate_ordering_contract="sha256_raw_bytes_then_canonical_json_bytes_v1",
        ),
        calibration_registry_path=source.calibration_registry_path,
        calibration_registry_id=source.calibration_registry_id,
        calibration_registry_sha256=source.calibration_registry_sha256,
        calibration_signature_hashes=source.calibration_hashes,
        calibration_selection_law=source.calibration_selection_law,
        build_partition_law=source.build_partition_law,
        historical_pilot_status=source.historical_pilot_status,
        leakage_calibration_artifact=source.leakage_calibration_artifact,
        opaque_exclusion_registry_sha256=source.opaque_hash,
        eligible_candidate_count=64,
        candidates=tuple(
            BuildCandidate(
                candidate_id=candidate.digest,
                canonical_signature=candidate.canonical_signature,
                candidate_bytes=candidate.candidate_bytes.decode("utf-8"),
                response=candidate.response,
            )
            for candidate in source.candidates
        ),
        candidate_audits=audits,
        selected_worked_example_ids=tuple(
            candidate.digest for candidate in source.candidates[:6]
        ),
        partition_disjointness="PASS",
    )


def _audit_candidate(candidate: Candidate) -> CandidateAuditRecord:
    match candidate:
        case Game24Candidate(numbers=numbers, target=target, response=response):
            is_correct = verify_expression(response, list(numbers), target).is_correct
        case MebCandidate(
            ordered_operands=operands,
            target_value=target,
            response=response,
        ):
            task = TaskInstance(
                sample_id=candidate.digest,
                task_name="math_equation_balancer",
                input={"input": f"{' ? '.join(map(str, operands))} = {target}"},
                verifier_spec={"target": response, "target_value": target},
            )
            is_correct = verify_answer(response, task).is_correct
        case WordSortingCandidate(sorted_words=words, response=response):
            is_correct = verify_words(response.split(), list(words)).is_correct
        case unreachable:
            assert_never(unreachable)
    if not is_correct:
        message = f"legacy RAG semantic validation failed for {candidate.digest}"
        raise ValueError(message)
    return CandidateAuditRecord(
        candidate_id=candidate.digest,
        semantic_validator_status="PASS",
        leakage_audit_status="PASS",
        leakage_reason_code="NO_REGISTERED_COLLISION",
    )


__all__ = [
    "CALIBRATION_PATHS",
    "BuildRegistrySource",
    "Candidate",
    "build_registry",
    "calibration_path",
    "calibration_signatures",
    "generated_candidates",
]
