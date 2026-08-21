from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from memcontam.readiness.phase13_new_mcq_leakage import (
    AuditDocument,
    EvaluationItem,
    LeakageArtifactError,
    MetricEvidence,
    McqContent,
    audit_documents,
    compare_document_to_item,
    failed_thresholds,
    structural_representation,
    validate_leakage_artifact,
)
from memcontam.readiness.phase13_new_mcq_leakage_io import (
    load_leakage_artifact,
    write_leakage_artifact,
)


class _BgeProvider:
    @property
    def metadata(self) -> dict[str, str | int | bool]:
        return {
            "model_id": "BAAI/bge-m3",
            "revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "vector_dimension": 1024,
            "normalize_embeddings": True,
        }

    def encode_document(self, text: str) -> list[float]:
        if text.startswith("orthogonal"):
            return [0.0, 1.0, *([0.0] * 1022)]
        return [1.0, 0.0, *([0.0] * 1022)]


class _RecordingBgeProvider(_BgeProvider):
    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode_document(self, text: str) -> list[float]:
        self.seen.append(text)
        return super().encode_document(text)


def _item(
    *,
    task_id: str = "mmlu_pro_engineering",
    evaluation_id: str = "mmlu_pro_engineering:7",
    stem: str = "Choose the valid result.",
    options: tuple[str, ...] = ("First", "Second"),
    source_span_ids: tuple[str, ...] = ("evaluation:mmlu_pro_engineering:7",),
) -> EvaluationItem:
    return EvaluationItem(
        task_id=task_id,
        evaluation_id=evaluation_id,
        stem=stem,
        options=options,
        source_span_ids=source_span_ids,
        identity_keys=(evaluation_id,),
    )


def _document(
    text: str = "orthogonal reusable procedure",
    *,
    source_span_ids: tuple[str, ...] = ("public-spec",),
    mcq: McqContent | None = None,
) -> AuditDocument:
    return AuditDocument(
        document_id="doc-1",
        task_id="mmlu_pro_engineering",
        text=text,
        source_span_ids=source_span_ids,
        identity_keys=(),
        mcq=mcq,
    )


@pytest.mark.parametrize(
    ("semantic", "lexical_common", "lexical_total", "distance", "length", "failed"),
    [
        (0.90, 0, 1, 2, 20, ("semantic", "structural")),
        (0.0, 1, 2, 20, 20, ("lexical",)),
        (0.899999, 49, 100, 3, 20, ()),
    ],
)
def test_fixed_thresholds_fail_on_equality_and_pass_below(
    semantic: float,
    lexical_common: int,
    lexical_total: int,
    distance: int,
    length: int,
    failed: tuple[str, ...],
) -> None:
    metrics = MetricEvidence(
        semantic_cosine=sum(
            left * right
            for left, right in zip(
                (1.0, 0.0),
                (semantic, math.sqrt(1.0 - semantic**2)),
                strict=True,
            )
        ),
        lexical_intersection=lexical_common,
        lexical_union=lexical_total,
        structural_distance=distance,
        structural_length=length,
    )

    assert failed_thresholds(metrics) == failed


@pytest.mark.parametrize(
    ("document", "component"),
    [
        (_document("Choose the valid result."), "exact"),
        (_document("  CHOOSE\u00a0THE valid result.  "), "canonical"),
    ],
)
def test_exact_and_canonical_identity_fail_closed(
    document: AuditDocument,
    component: str,
) -> None:
    evidence = compare_document_to_item(
        document,
        _item(),
        document_vector=(0.0, 1.0),
        item_vector=(1.0, 0.0),
    )

    assert component in evidence.failed_components
    assert evidence.failed is True


def test_gpqa_permutation_is_one_leakage_identity() -> None:
    item = _item(
        task_id="gpqa_diamond",
        evaluation_id="gpqa_diamond:opaque-7",
        stem="Which claim follows?",
        options=("Alpha", "Beta", "Gamma", "Delta"),
    )
    document = AuditDocument(
        document_id="gpqa-copy",
        task_id="gpqa_diamond",
        text="A displayed permutation",
        source_span_ids=("public-spec",),
        identity_keys=(),
        mcq=McqContent(stem=item.stem, options=("Gamma", "Alpha", "Delta", "Beta")),
    )

    evidence = compare_document_to_item(
        document,
        item,
        document_vector=(0.0, 1.0),
        item_vector=(1.0, 0.0),
    )

    assert evidence.permutation_identity is True
    assert "permutation" in evidence.failed_components


def test_structural_mask_preserves_canonical_whitespace_punctuation_and_boundaries() -> None:
    represented = structural_representation(
        McqContent("  Alpha,\t beta  ", ("Gamma gamma!", "Delta"))
    )

    assert represented == "#, #\n␞\n# #!\n␟\n#"


def test_source_span_exclusion_fails_closed() -> None:
    evidence = compare_document_to_item(
        _document(source_span_ids=("evaluation:mmlu_pro_engineering:7",)),
        _item(),
        document_vector=(0.0, 1.0),
        item_vector=(1.0, 0.0),
    )

    assert evidence.excluded_source_span_ids == ("evaluation:mmlu_pro_engineering:7",)
    assert evidence.failed_components == ("source_span",)


def test_any_pair_component_fails_the_document_and_artifact() -> None:
    document = _document("Choose the valid result.")
    item = _item()
    artifact = audit_documents(
        documents=(document,),
        evaluation_items=(item,),
        provider=_BgeProvider(),
        input_hashes={"documents": "1" * 64, "evaluation": "2" * 64},
    )
    pair = compare_document_to_item(
        document,
        item,
        document_vector=(1.0, 0.0, *([0.0] * 1022)),
        item_vector=(1.0, 0.0, *([0.0] * 1022)),
    )

    assert artifact.status == "FAIL"
    assert artifact.document_evidence[0].failed is True
    assert artifact.document_evidence[0].offending_evaluation_ids == (
        "mmlu_pro_engineering:7",
    )
    assert artifact.document_evidence[0].maximum_lexical_intersection == (
        pair.metrics.lexical_intersection
    )
    assert artifact.document_evidence[0].maximum_lexical_union == pair.metrics.lexical_union
    assert artifact.document_evidence[0].maximum_structural_distance == (
        pair.metrics.structural_distance
    )
    assert artifact.document_evidence[0].maximum_structural_length == (
        pair.metrics.structural_length
    )


def test_artifact_tamper_is_rejected() -> None:
    artifact = audit_documents(
        documents=(_document(),),
        evaluation_items=(_item(),),
        provider=_BgeProvider(),
        input_hashes={"documents": "1" * 64, "evaluation": "2" * 64},
    )
    tampered = replace(artifact, status="FAIL")

    with pytest.raises(LeakageArtifactError, match="NEW_MCQ_LEAKAGE_ARTIFACT_HASH_MISMATCH"):
        validate_leakage_artifact(tampered)


def test_semantic_embedding_uses_canonical_document_text() -> None:
    provider = _RecordingBgeProvider()

    audit_documents(
        documents=(_document("  ORTHOGONAL\u00a0reusable  procedure "),),
        evaluation_items=(_item(),),
        provider=provider,
        input_hashes={"documents": "1" * 64, "evaluation": "2" * 64},
    )

    assert "orthogonal reusable procedure" in provider.seen


def test_persisted_artifact_rejects_unknown_fields(tmp_path: Path) -> None:
    artifact = audit_documents(
        documents=(_document(),),
        evaluation_items=(_item(),),
        provider=_BgeProvider(),
        input_hashes={"documents": "1" * 64, "evaluation": "2" * 64},
    )
    path = tmp_path / "leakage.json"
    write_leakage_artifact(path, artifact)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unknown"] = "tamper"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LeakageArtifactError, match="NEW_MCQ_LEAKAGE_ARTIFACT_INVALID"):
        load_leakage_artifact(path)

