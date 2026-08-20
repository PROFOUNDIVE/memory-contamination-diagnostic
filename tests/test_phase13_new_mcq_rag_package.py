from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from memcontam.readiness import phase13_new_mcq_rag
from memcontam.readiness import phase13_new_mcq_rag_artifacts
from memcontam.readiness import phase13_new_mcq_rag_frozen
from memcontam.contamination.phase12.models import canonical_json_hash
from memcontam.memory.embeddings import BgeM3EmbeddingProvider


PACKAGE_ROOT = Path("data/phase13/rag/new_mcq")
EVALUATION_ROOT = Path("data/phase13/core/materialized")
STATUS_PATH = PACKAGE_ROOT.parent / "new_mcq_rag_status_v1.json"


def _copy_package(tmp_path: Path) -> Path:
    package = tmp_path / "new_mcq"
    shutil.copytree(PACKAGE_ROOT, package)
    shutil.copy2(STATUS_PATH, tmp_path / STATUS_PATH.name)
    return package


def _update_status_manifest_hash(package: Path) -> None:
    status_path = package.parent / STATUS_PATH.name
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["candidate_package"]["sha256"] = hashlib.sha256(
        (package / "package_manifest_v1.json").read_bytes()
    ).hexdigest()
    status_path.write_text(json.dumps(status), encoding="utf-8")


def test_clean_package_remains_blocked_by_unfrozen_required_artifacts() -> None:
    report = phase13_new_mcq_rag.validate_new_mcq_rag_package(PACKAGE_ROOT, EVALUATION_ROOT)
    payload = asdict(report)

    assert report.status == "NOT_READY"
    assert report.reason == "NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN"
    assert payload["reviewed_candidates"] == {
        "mmlu_pro_engineering": 24,
        "mmlu_pro_physics": 24,
        "gpqa_diamond": 24,
    }
    assert report.source_classes == {
        "mmlu_pro_engineering": "public_task_specification",
        "mmlu_pro_physics": "public_task_specification",
        "gpqa_diamond": "public_task_specification",
    }
    assert payload["candidate_corpus_hashes"].keys() == payload["reviewed_candidates"].keys()
    assert payload["clean_index_hashes"].keys() == payload["reviewed_candidates"].keys()
    assert report.remaining_objects == (
        "authority_required_leakage_gate_artifacts",
        "verified_bge_m3_snapshot_tree_and_runtime_binding",
        "task_local_candidate_selection_and_certification",
        "task_local_intervention_relevance",
        "clean_correct_irrelevant_contam_branch_indices",
    )
    assert report.promotion_ready is False


def test_candidate_corpus_rejects_exact_evaluation_text(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    evaluation = json.loads(
        Path("data/phase13/core/materialized/mmlu_pro_engineering.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    candidate_path = package / "candidates" / "mmlu_pro_engineering.jsonl"
    candidates = candidate_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(candidates[0])
    first["text"] = evaluation["input"]["question"]
    candidates[0] = json.dumps(first, separators=(",", ":"))
    candidate_path.write_text("\n".join(candidates) + "\n", encoding="utf-8")

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_EVALUATION_OVERLAP",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


def test_candidate_package_rejects_stale_manifest_hash(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    candidate_path = package / "candidates" / "mmlu_pro_engineering.jsonl"
    candidates = candidate_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(candidates[0])
    first["text"] = f"{first['text']} Recheck the extracted constraints."
    candidates[0] = json.dumps(first, separators=(",", ":"))
    candidate_path.write_text("\n".join(candidates) + "\n", encoding="utf-8")

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


def test_candidate_package_rejects_incomplete_remaining_objects(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    manifest_path = package / "package_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["promotion"]["remaining_objects"].pop()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


@pytest.mark.parametrize(
    "relative_path",
    (
        "source_eligibility_registry_v1.json",
        "accepted/mmlu_pro_engineering.jsonl",
        "embedding_runtime_v1.json",
        "indices/mmlu_pro_physics.json",
        "leakage_report_v1.json",
    ),
)
def test_package_rejects_tampering_in_every_required_object_class(
    tmp_path: Path,
    relative_path: str,
) -> None:
    package = _copy_package(tmp_path)
    artifact = package / relative_path
    artifact.write_bytes(artifact.read_bytes() + b"\n")

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


def test_clean_index_rejects_text_not_bound_to_accepted_registry(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    index_path = package / "indices" / "gpqa_diamond.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["documents"][0]["text"] += " altered"
    index["corpus_content_hash"] = canonical_json_hash(index["documents"])
    index["index_artifact_hash"] = canonical_json_hash(
        {
            "documents": index["documents"],
            "embedding_contract": index["embedding_contract"],
            "vectors": index["vectors"],
        }
    )
    index_path.write_text(json.dumps(index), encoding="utf-8")
    report = phase13_new_mcq_rag.validate_new_mcq_rag_package(
        PACKAGE_ROOT,
        EVALUATION_ROOT,
    )

    with pytest.raises(
        phase13_new_mcq_rag_frozen.FrozenArtifactError,
        match="NEW_MCQ_RAG_SERIALIZED_INDEX_INVALID",
    ):
        phase13_new_mcq_rag_frozen.validate_frozen_artifacts(
            package,
            EVALUATION_ROOT,
            report.candidate_corpus_hashes,
        )


def test_reconstruction_identity_binds_task_review(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    review_path = package / "reviews" / "gpqa_diamond.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    manifest_path = package / "package_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tasks"]["gpqa_diamond"]["review"]["sha256"] = hashlib.sha256(
        review_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _update_status_manifest_hash(package)

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


def test_package_rejects_status_manifest_hash_mismatch(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    status_path = package.parent / STATUS_PATH.name
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["candidate_package"]["sha256"] = "0" * 64
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


def test_package_rejects_unrecognized_status_fields(tmp_path: Path) -> None:
    package = _copy_package(tmp_path)
    status_path = package.parent / STATUS_PATH.name
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["unregistered_override"] = True
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(
        phase13_new_mcq_rag.NewMcqRagError,
        match="NEW_MCQ_RAG_PACKAGE_MANIFEST_STALE",
    ):
        phase13_new_mcq_rag.validate_new_mcq_rag_package(package, EVALUATION_ROOT)


def test_package_validation_is_independent_of_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    report = phase13_new_mcq_rag.validate_new_mcq_rag_package(
        repository_root / PACKAGE_ROOT,
        repository_root / EVALUATION_ROOT,
    )

    assert report.reason == "NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN"


class _BgeIdentityProvider:
    metadata = {
        "model_id": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "vector_dimension": 1024,
        "normalize_embeddings": True,
    }

    def encode_query(self, text: str) -> list[float]:
        del text
        return [1.0, *([0.0] * 1023)]

    def encode_document(self, text: str) -> list[float]:
        del text
        raise AssertionError("frozen vectors must not be recomputed")


class _RuntimeProvider(BgeM3EmbeddingProvider):
    def __init__(self) -> None:
        self.batch_size = 32
        self.model = SimpleNamespace(
            tokenizer=SimpleNamespace(truncation_side="right", padding_side="right"),
            max_seq_length=8192,
            device="cpu",
        )

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model_id": self.MODEL_ID,
            "revision": self.REVISION,
            "vector_dimension": self.VECTOR_DIMENSION,
            "normalize_embeddings": self.NORMALIZE_EMBEDDINGS,
        }


def test_frozen_clean_index_reconstructs_without_intervention_state() -> None:
    frozen = phase13_new_mcq_rag.load_new_mcq_clean_rag_state(
        PACKAGE_ROOT,
        EVALUATION_ROOT,
        "gpqa_diamond",
        _BgeIdentityProvider(),
        allow_test_embedder=True,
        allow_unverified_snapshot=True,
    )

    assert frozen.state.branch == "clean"
    assert frozen.state.corpus is not None
    assert frozen.state.index is not None
    assert len(frozen.state.corpus.active_documents) == 24
    assert frozen.state.index.artifact_hash == frozen.index_artifact_hash
    assert frozen.reconstruction_identity


def test_frozen_clean_index_rejects_test_embedder_without_explicit_override() -> None:
    with pytest.raises(
        phase13_new_mcq_rag_frozen.FrozenArtifactError,
        match="NEW_MCQ_RAG_RUNTIME_IDENTITY_INVALID",
    ):
        phase13_new_mcq_rag.load_new_mcq_clean_rag_state(
            PACKAGE_ROOT,
            EVALUATION_ROOT,
            "gpqa_diamond",
            _BgeIdentityProvider(),
        )


def test_frozen_clean_index_blocks_unverified_snapshot_without_test_override() -> None:
    with pytest.raises(
        phase13_new_mcq_rag_frozen.FrozenArtifactError,
        match="NEW_MCQ_RAG_RUNTIME_SNAPSHOT_UNVERIFIED",
    ):
        phase13_new_mcq_rag.load_new_mcq_clean_rag_state(
            PACKAGE_ROOT,
            EVALUATION_ROOT,
            "gpqa_diamond",
            _BgeIdentityProvider(),
            allow_test_embedder=True,
        )


def test_package_contains_no_candidate_selection_or_dependent_branches() -> None:
    manifest = json.loads((PACKAGE_ROOT / "package_manifest_v1.json").read_text(encoding="utf-8"))
    leakage = json.loads((PACKAGE_ROOT / "leakage_report_v1.json").read_text(encoding="utf-8"))
    runtime = json.loads((PACKAGE_ROOT / "embedding_runtime_v1.json").read_text(encoding="utf-8"))

    assert not (PACKAGE_ROOT / "intervention_triplets_v1.json").exists()
    assert not (PACKAGE_ROOT / "relevance_universe_v1.json").exists()
    assert "task_local_intervention_triplets" not in manifest["required_artifacts"]
    assert leakage["status"] == "NOT_READY_REQUIRED_LEAKAGE_GATE_UNFROZEN"
    assert leakage["missing_objects"] == [
        "task_specific_canonicalizers",
        "displayed_permutation_equivalence",
        "near_duplicate_threshold",
        "structural_similarity_threshold",
        "lexical_overlap_threshold",
        "source_span_registry",
        "exclusion_manifest",
    ]
    assert runtime["status"] == "NOT_READY_SNAPSHOT_TREE_UNVERIFIED"
    assert runtime["missing_objects"] == [
        "measured_model_snapshot_tree_sha256",
        "production_query_snapshot_verification",
    ]
    for task in manifest["tasks"].values():
        assert set(task["index_hashes"]) == {"clean"}


def test_runtime_materializer_preserves_snapshot_blocker(tmp_path: Path) -> None:
    snapshot = (
        tmp_path
        / "models--BAAI--bge-m3"
        / "snapshots"
        / "5617a9f61b028005a4858fdac845db406aefb181"
    )
    (snapshot / "1_Pooling").mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "1_Pooling" / "config.json").write_text("{}", encoding="utf-8")
    artifact = phase13_new_mcq_rag_artifacts.runtime(tmp_path, _RuntimeProvider())

    assert artifact["status"] == "NOT_READY_SNAPSHOT_TREE_UNVERIFIED"
    assert artifact["missing_objects"] == [
        "measured_model_snapshot_tree_sha256",
        "production_query_snapshot_verification",
    ]
