from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest

from memcontam.readiness.phase13_legacy_rag_audit import build_opaque_exclusion_registry
from memcontam.readiness.phase13_legacy_rag_bytes import JsonValue, canonical_json_bytes
from memcontam.readiness.phase13_legacy_rag_errors import LegacyRagValidationError
from memcontam.readiness.phase13_legacy_rag_materialize import (
    LegacyRagMaterializationRequest,
    materialize_legacy_rag_package,
)
from memcontam.readiness.phase13_legacy_rag_runtime import (
    LegacyRagRuntimeRequest,
    load_legacy_rag_state,
)
from memcontam.readiness.phase13_legacy_rag_validate import validate_legacy_rag_package


ROOT = Path(__file__).resolve().parents[1]


class ContractEmbedder:
    metadata = {
        "model_id": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "embedding_library_version": "contract-test",
        "vector_dimension": 1024,
        "normalize_embeddings": True,
    }

    def encode_document(self, text: str) -> list[float]:
        return self._encode(text)

    def encode_query(self, text: str) -> list[float]:
        return self._encode(text)

    @staticmethod
    def _encode(text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < 1024:
            digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
            values.extend((value - 127.5) / 127.5 for value in digest)
            counter += 1
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values]


class DifferentLibraryEmbedder(ContractEmbedder):
    metadata = {**ContractEmbedder.metadata, "embedding_library_version": "different-runtime"}


@pytest.fixture(scope="module")
def frozen_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("legacy-rag-integrity")
    opaque = root / "opaque.json"
    build_opaque_exclusion_registry(ROOT / "data/phase13/main", opaque)
    output = root / "legacy"
    materialize_legacy_rag_package(
        LegacyRagMaterializationRequest(
            output,
            ROOT,
            opaque,
            ContractEmbedder(),
            allow_test_embedder=True,
            allow_unfrozen_meb_threshold_for_tests=True,
        )
    )
    return output


def _copy_package(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def _manifest_sha256(root: Path) -> str:
    return hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()


def _resign_manifest(root: Path, *relative_paths: str) -> None:
    compared = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {"manifest.json", "repeatability_report.json"}
    }
    aggregate_payload: dict[str, JsonValue] = dict(compared)
    aggregate = hashlib.sha256(canonical_json_bytes(aggregate_payload)).hexdigest()
    repeatability_path = root / "repeatability_report.json"
    repeatability = _load(repeatability_path)
    repeatability["compared_artifact_hashes"] = compared
    repeatability["first_materialization_sha256"] = aggregate
    repeatability["repeat_materialization_sha256"] = aggregate
    _write(repeatability_path, repeatability)
    manifest_path = root / "manifest.json"
    manifest = _load(manifest_path)
    for relative in (*relative_paths, "repeatability_report.json"):
        manifest["artifact_hashes"][relative] = hashlib.sha256(
            (root / relative).read_bytes()
        ).hexdigest()
    _write(manifest_path, manifest)


def _index_hash(branch: dict) -> str:
    payload = {
        "documents": branch["documents"],
        "embedding_contract": branch["embedding_contract"],
        "vectors": branch["vectors"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_validator_rejects_resigned_intervention_text_tamper(
    tmp_path: Path, frozen_package: Path
) -> None:
    package = _copy_package(frozen_package, tmp_path / "legacy")
    corpus_path = package / "game24/corpus.json"
    index_path = package / "game24/indices.json"
    corpus = _load(corpus_path)
    indices = _load(index_path)
    corpus["branches"]["contam"]["documents"][-1]["text"] = "tampered"
    index = indices["branches"]["contam"]
    index["documents"][-1]["text"] = "tampered"
    index["corpus_content_hash"] = hashlib.sha256(
        canonical_json_bytes(index["documents"])
    ).hexdigest()
    index["index_artifact_hash"] = _index_hash(index)
    _write(corpus_path, corpus)
    index_path.write_text(
        json.dumps(indices, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    _resign_manifest(package, "game24/corpus.json", "game24/indices.json")

    with pytest.raises(LegacyRagValidationError, match="LEGACY_RAG_CORPUS_INVALID"):
        validate_legacy_rag_package(
            package, ROOT, _manifest_sha256(package), allow_test_package=True
        )


def test_validator_rejects_resigned_status_tamper(
    tmp_path: Path, frozen_package: Path
) -> None:
    package = _copy_package(frozen_package, tmp_path / "legacy")
    status_path = package / "package_status.json"
    status = _load(status_path)
    status["tasks"]["game24"] = {"status": "BLOCKED", "reason_code": "tampered"}
    _write(status_path, status)
    _resign_manifest(package, "package_status.json")

    with pytest.raises(LegacyRagValidationError, match="LEGACY_RAG_PACKAGE_STATUS_INVALID"):
        validate_legacy_rag_package(
            package, ROOT, _manifest_sha256(package), allow_test_package=True
        )


def test_validator_rejects_resigned_audit_contract_tamper(
    tmp_path: Path, frozen_package: Path
) -> None:
    package = _copy_package(frozen_package, tmp_path / "legacy")
    audit_path = package / "opaque_exclusion_registry.json"
    audit = _load(audit_path)
    audit["audit_contracts"]["word_sorting"]["thresholds"]["token_overlap"] = "1/3"
    _write(audit_path, audit)
    _resign_manifest(package, "opaque_exclusion_registry.json")

    with pytest.raises(
        LegacyRagValidationError, match="LEGACY_RAG_MAIN_SOURCE_IDENTITY_MISMATCH"
    ):
        validate_legacy_rag_package(
            package, ROOT, _manifest_sha256(package), allow_test_package=True
        )


def test_validator_and_runtime_reject_resigned_test_package_promotion(
    tmp_path: Path, frozen_package: Path
) -> None:
    package = _copy_package(frozen_package, tmp_path / "legacy")
    status_path = package / "package_status.json"
    status = _load(status_path)
    status["package_status"] = "TRACK2_LEGACY_RAG_MATERIALIZATION_COMPLETE"
    for task in status["tasks"].values():
        task["status"] = "TRACK2_LEGACY_RAG_MATERIALIZATION_COMPLETE"
    _write(status_path, status)
    manifest_path = package / "manifest.json"
    manifest = _load(manifest_path)
    manifest["package_status"] = "TRACK2_LEGACY_RAG_MATERIALIZATION_COMPLETE"
    _write(manifest_path, manifest)
    _resign_manifest(package, "package_status.json")

    with pytest.raises(
        LegacyRagValidationError,
        match="MEB_STRUCTURAL_SIMILARITY_THRESHOLD_UNFROZEN",
    ):
        validate_legacy_rag_package(package, ROOT, _manifest_sha256(package))
    with pytest.raises(
        LegacyRagValidationError,
        match="MEB_STRUCTURAL_SIMILARITY_THRESHOLD_UNFROZEN",
    ):
        load_legacy_rag_state(
            LegacyRagRuntimeRequest(
                package,
                ROOT,
                "game24",
                "clean",
                ContractEmbedder(),
                _manifest_sha256(package),
                allow_test_embedder=True,
            )
        )


def test_validator_rejects_nonfinite_vector_even_when_resigned(
    tmp_path: Path, frozen_package: Path
) -> None:
    package = _copy_package(frozen_package, tmp_path / "legacy")
    index_path = package / "game24/indices.json"
    indices = _load(index_path)
    branch = indices["branches"]["clean"]
    first_id = next(iter(branch["vectors"]))
    branch["vectors"][first_id][0] = float("nan")
    branch["index_artifact_hash"] = _index_hash(branch)
    index_path.write_text(
        json.dumps(indices, allow_nan=True, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    _resign_manifest(package, "game24/indices.json")

    with pytest.raises(LegacyRagValidationError, match="LEGACY_RAG_INDEX_INVALID"):
        validate_legacy_rag_package(
            package, ROOT, _manifest_sha256(package), allow_test_package=True
        )


def test_validator_rejects_resigned_branch_specific_clean_vector(
    tmp_path: Path, frozen_package: Path
) -> None:
    package = _copy_package(frozen_package, tmp_path / "legacy")
    index_path = package / "game24/indices.json"
    indices = _load(index_path)
    branch = indices["branches"]["contam"]
    first_id = next(iter(indices["branches"]["clean"]["vectors"]))
    vector = branch["vectors"][first_id]
    position = next(index for index, value in enumerate(vector) if value != 0.0)
    vector[position] = -vector[position]
    branch["index_artifact_hash"] = _index_hash(branch)
    index_path.write_text(
        json.dumps(indices, allow_nan=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    _resign_manifest(package, "game24/indices.json")

    with pytest.raises(LegacyRagValidationError, match="LEGACY_RAG_INDEX_INVALID"):
        validate_legacy_rag_package(
            package, ROOT, _manifest_sha256(package), allow_test_package=True
        )


def test_validator_rejects_resigned_opaque_signature_removal(
    tmp_path: Path, frozen_package: Path
) -> None:
    package = _copy_package(frozen_package, tmp_path / "legacy")
    opaque_path = package / "opaque_exclusion_registry.json"
    opaque = _load(opaque_path)
    opaque["signature_hashes"]["game24"].pop()
    _write(opaque_path, opaque)
    _resign_manifest(package, "opaque_exclusion_registry.json")

    with pytest.raises(
        LegacyRagValidationError,
        match="LEGACY_RAG_MAIN_SOURCE_IDENTITY_MISMATCH",
    ):
        validate_legacy_rag_package(
            package, ROOT, _manifest_sha256(package), allow_test_package=True
        )


def test_validator_accepts_cross_runtime_score_roundoff(
    tmp_path: Path, frozen_package: Path
) -> None:
    package = _copy_package(frozen_package, tmp_path / "legacy")
    index_path = package / "game24/indices.json"
    indices = _load(index_path)
    scores = indices["retrieval_diagnostic"]["scores"]
    indices["retrieval_diagnostic"]["scores"] = [score + 1e-15 for score in scores]
    index_path.write_text(
        json.dumps(indices, allow_nan=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    _resign_manifest(package, "game24/indices.json")

    validate_legacy_rag_package(
        package, ROOT, _manifest_sha256(package), allow_test_package=True
    )


def test_runtime_validates_package_before_reconstruction(
    tmp_path: Path, frozen_package: Path
) -> None:
    package = _copy_package(frozen_package, tmp_path / "legacy")
    status_path = package / "package_status.json"
    status = _load(status_path)
    status["tasks"]["game24"] = {"status": "BLOCKED", "reason_code": "tampered"}
    _write(status_path, status)
    _resign_manifest(package, "package_status.json")

    with pytest.raises(LegacyRagValidationError, match="LEGACY_RAG_PACKAGE_STATUS_INVALID"):
        load_legacy_rag_state(
            LegacyRagRuntimeRequest(
                package,
                ROOT,
                "word_sorting",
                "clean",
                ContractEmbedder(),
                _manifest_sha256(package),
                allow_test_embedder=True,
                allow_test_package=True,
            )
        )


def test_runtime_loads_each_completed_task(frozen_package: Path) -> None:
    for task in ("game24", "math_equation_balancer", "word_sorting"):
        loaded = load_legacy_rag_state(
            LegacyRagRuntimeRequest(
                frozen_package,
                ROOT,
                task,
                "clean",
                ContractEmbedder(),
                _manifest_sha256(frozen_package),
                allow_test_embedder=True,
                allow_test_package=True,
            )
        )

        assert loaded.state.corpus is not None
        assert len(loaded.state.corpus.active_documents) == 24


def test_runtime_rejects_embedding_library_version_mismatch(frozen_package: Path) -> None:
    with pytest.raises(LegacyRagValidationError, match="LEGACY_RAG_RUNTIME_IDENTITY_INVALID"):
        load_legacy_rag_state(
            LegacyRagRuntimeRequest(
                frozen_package,
                ROOT,
                "game24",
                "clean",
                DifferentLibraryEmbedder(),
                _manifest_sha256(frozen_package),
                allow_test_embedder=True,
                allow_test_package=True,
            )
        )
