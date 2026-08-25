from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from .phase13_legacy_rag_bytes import canonical_json_bytes
from .phase13_legacy_rag_calibration import word_sorting_similarities
from .phase13_legacy_rag_generators import WORD_SORTING_VOCABULARY, iter_meb_candidates
from .phase13_legacy_rag_meb_threshold import (
    MebStructuralEndpoint,
    MebStructuralThreshold,
    build_meb_structural_threshold,
    near_duplicate_exclusions,
)
from .phase13_legacy_rag_models import ArtifactReference


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
TaskName = Literal["game24", "math_equation_balancer", "word_sorting"]
AuditedTask = TaskName

_GAME24_FILE: Final = "game24_main_v1.jsonl"
_MEB_FILE: Final = "math_equation_balancer_main_v1.jsonl"
_WORD_SORTING_FILE: Final = "word_sorting_main_v1.jsonl"
_MANIFEST_FILE: Final = "main_registry_manifest_v1.json"
_MANIFEST_SHA256: Final = "8b84820aa7e62cbcb71b035d002e5466ed6d5f32fbc3e4cd66ad636ca4303efb"
_SOURCE_IDENTITIES: Final = {
    "game24": (_GAME24_FILE, "ae682f138d8035fc1de9382eb8903730d392851def720351a78846df160b615f", 95),
    "math_equation_balancer": (
        _MEB_FILE,
        "dfa07c8c3ada1b0030a735cca97022f98dfb8da30d8ce86f82013eb51b4a7037",
        250,
    ),
    "word_sorting": (
        _WORD_SORTING_FILE,
        "e7ff0507512af4e71ae027a5226984b175d9b75dca898df79ca88535326c9c54",
        250,
    ),
}
_SIGNATURE_IDENTITIES: Final[dict[AuditedTask, tuple[int, Sha256]]] = {
    "game24": (95, "2fca0e90c38729a9ff9df8987aa2456ab73a1577b5df81338f93eb6880ac91d0"),
    "math_equation_balancer": (
        255,
        "dfab5883795f7a097dfbfd374b57dbd627d6fc1ee4efa169f0ede7964c35e76c",
    ),
    "word_sorting": (0, "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"),
}
_MEB_THRESHOLD_SHA256: Final = (
    "e113ecef6d43190bec136f4b32375172e58a1f4113a1e9f3bc7467ab65615921"
)


class LegacyRagAuditError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)

    def __str__(self) -> str:
        return self.code


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _Game24Row(_FrozenModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    numbers: tuple[int, ...]
    target: int


class _WordSortingRow(_FrozenModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    words: tuple[str, ...]


class _MebVerifierSpec(_FrozenModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    target_value: int


class _MebRow(_FrozenModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    input: str
    verifier_spec: _MebVerifierSpec


class SourceFile(_FrozenModel):
    path: str
    sha256: Sha256
    row_count: Annotated[int, Field(ge=0)]


class WordSortingAuditContract(_FrozenModel):
    thresholds: dict[Literal["token_overlap", "lexical_signature"], str]
    boundary_rule: Literal["similarity_greater_than_or_equal_to_threshold_rejects"]


class OpaqueExclusionRegistry(_FrozenModel):
    schema_version: Literal["phase13_legacy_rag_opaque_exclusion_registry_v5"]
    status: Literal["PASS"]
    task_statuses: dict[TaskName, Literal["PASS"]]
    task_reason_codes: dict[TaskName, str | None]
    main_registry_manifest: SourceFile
    signature_hashes: dict[AuditedTask, tuple[Sha256, ...]]
    source_files: dict[TaskName, SourceFile]
    audit_contracts: dict[Literal["word_sorting"], WordSortingAuditContract]
    meb_structural_threshold: MebStructuralThreshold


def _canonical_signature_bytes(value: dict[str, int | list[int] | list[str]]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _verified_source(root: Path, task: TaskName) -> tuple[bytes, SourceFile]:
    file_name, expected_hash, expected_count = _SOURCE_IDENTITIES[task]
    path = root / file_name
    raw = path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    row_count = len(raw.splitlines())
    if actual_hash != expected_hash or row_count != expected_count:
        raise LegacyRagAuditError("LEGACY_RAG_MAIN_SOURCE_IDENTITY_MISMATCH")
    return raw, SourceFile(path=file_name, sha256=actual_hash, row_count=row_count)


def _read_game24(raw: bytes) -> tuple[Sha256, ...]:
    rows = tuple(_Game24Row.model_validate_json(line) for line in raw.splitlines())
    return tuple(
        sorted(
            {
                hashlib.sha256(
                    _canonical_signature_bytes(
                        {"numbers": sorted(row.numbers), "target": row.target}
                    )
                ).hexdigest()
                for row in rows
            }
        )
    )


def _read_word_sorting(raw: bytes) -> tuple[Sha256, ...]:
    rows = tuple(_WordSortingRow.model_validate_json(line) for line in raw.splitlines())
    normalized = tuple(
        tuple(unicodedata.normalize("NFC", word) for word in row.words) for row in rows
    )
    relevant = tuple(
        row for row in normalized if set(row) & set(WORD_SORTING_VOCABULARY)
    )
    return tuple(
        sorted(
            {
                hashlib.sha256(
                    _canonical_signature_bytes({"tokens": list(candidate)})
                ).hexdigest()
                for candidate in combinations(WORD_SORTING_VOCABULARY, 5)
                if any(
                    (similarities := word_sorting_similarities(candidate, row))[0]
                    >= Fraction(1, 4)
                    or similarities[1] >= Fraction(1, 6)
                    for row in relevant
                )
            }
        )
    )


def _read_meb(raw: bytes) -> tuple[tuple[Sha256, ...], tuple[MebStructuralEndpoint, ...]]:
    rows = tuple(_MebRow.model_validate_json(line) for line in raw.splitlines())
    endpoints = tuple(
        MebStructuralEndpoint(
            ordered_operands=tuple(
                int(value) for value in re.findall(r"-?\d+", row.input.split("=", 1)[0])
            ),
            target_value=row.verifier_spec.target_value,
            signature=hashlib.sha256(
                _canonical_signature_bytes(
                    {
                        "ordered_operands": [
                            int(value)
                            for value in re.findall(r"-?\d+", row.input.split("=", 1)[0])
                        ],
                        "target_value": row.verifier_spec.target_value,
                    }
                )
            ).hexdigest(),
        )
        for row in rows
    )
    return tuple(sorted({endpoint.signature for endpoint in endpoints})), endpoints


def build_opaque_exclusion_registry(
    evaluation_root: Path,
    output: Path,
) -> OpaqueExclusionRegistry:
    try:
        manifest_raw = (evaluation_root / _MANIFEST_FILE).read_bytes()
    except OSError as error:
        raise LegacyRagAuditError("LEGACY_RAG_MAIN_SOURCE_IDENTITY_MISMATCH") from error
    if hashlib.sha256(manifest_raw).hexdigest() != _MANIFEST_SHA256:
        raise LegacyRagAuditError("LEGACY_RAG_MAIN_SOURCE_IDENTITY_MISMATCH")
    try:
        game24_raw, game24_source = _verified_source(evaluation_root, "game24")
        meb_raw, meb_source = _verified_source(evaluation_root, "math_equation_balancer")
        word_raw, word_source = _verified_source(evaluation_root, "word_sorting")
    except OSError as error:
        raise LegacyRagAuditError("LEGACY_RAG_MAIN_SOURCE_IDENTITY_MISMATCH") from error
    meb_exact_hashes, meb_endpoints = _read_meb(meb_raw)
    meb_candidates = tuple(iter_meb_candidates(frozenset(meb_exact_hashes)))
    meb_threshold = build_meb_structural_threshold(
        meb_candidates,
        ArtifactReference(path=_MEB_FILE, sha256=meb_source.sha256, row_count=meb_source.row_count),
    )
    meb_near_hashes = near_duplicate_exclusions(
        meb_candidates, meb_endpoints, Fraction(meb_threshold.tau_meb)
    )
    artifact = OpaqueExclusionRegistry(
        schema_version="phase13_legacy_rag_opaque_exclusion_registry_v5",
        status="PASS",
        task_statuses={
            "game24": "PASS",
            "math_equation_balancer": "PASS",
            "word_sorting": "PASS",
        },
        task_reason_codes={
            "game24": None,
            "math_equation_balancer": None,
            "word_sorting": None,
        },
        main_registry_manifest=SourceFile(
            path=_MANIFEST_FILE,
            sha256=_MANIFEST_SHA256,
            row_count=len(manifest_raw.splitlines()),
        ),
        signature_hashes={
            "game24": _read_game24(game24_raw),
            "math_equation_balancer": tuple(sorted({*meb_exact_hashes, *meb_near_hashes})),
            "word_sorting": _read_word_sorting(word_raw),
        },
        source_files={
            "game24": game24_source,
            "math_equation_balancer": meb_source,
            "word_sorting": word_source,
        },
        audit_contracts={
            "word_sorting": WordSortingAuditContract(
                thresholds={"token_overlap": "1/4", "lexical_signature": "1/6"},
                boundary_rule="similarity_greater_than_or_equal_to_threshold_rejects",
            )
        },
        meb_structural_threshold=meb_threshold,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(artifact.model_dump(mode="json")))
    return artifact


def validate_opaque_registry_identity(artifact: OpaqueExclusionRegistry) -> None:
    expected_sources = {
        task: SourceFile(path=file_name, sha256=sha256, row_count=row_count)
        for task, (file_name, sha256, row_count) in _SOURCE_IDENTITIES.items()
    }
    if (
        artifact.main_registry_manifest.path != _MANIFEST_FILE
        or artifact.main_registry_manifest.sha256 != _MANIFEST_SHA256
        or artifact.status != "PASS"
        or artifact.task_statuses != {
            "game24": "PASS",
            "math_equation_balancer": "PASS",
            "word_sorting": "PASS",
        }
        or artifact.task_reason_codes != {
            "game24": None,
            "math_equation_balancer": None,
            "word_sorting": None,
        }
        or artifact.audit_contracts != {
            "word_sorting": WordSortingAuditContract(
                thresholds={"token_overlap": "1/4", "lexical_signature": "1/6"},
                boundary_rule="similarity_greater_than_or_equal_to_threshold_rejects",
            )
        }
        or artifact.source_files != expected_sources
        or hashlib.sha256(
            canonical_json_bytes(artifact.meb_structural_threshold.model_dump(mode="json"))
        ).hexdigest()
        != _MEB_THRESHOLD_SHA256
        or any(
            len(artifact.signature_hashes[task]) != expected_count
            or hashlib.sha256(
                canonical_json_bytes(list(artifact.signature_hashes[task]))
            ).hexdigest()
            != expected_hash
            for task, (expected_count, expected_hash) in _SIGNATURE_IDENTITIES.items()
        )
    ):
        raise LegacyRagAuditError("LEGACY_RAG_MAIN_SOURCE_IDENTITY_MISMATCH")


__all__ = [
    "LegacyRagAuditError",
    "OpaqueExclusionRegistry",
    "WordSortingAuditContract",
    "build_opaque_exclusion_registry",
    "validate_opaque_registry_identity",
]
