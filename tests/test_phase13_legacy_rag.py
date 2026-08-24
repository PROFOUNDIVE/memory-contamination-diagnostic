from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memcontam.readiness.phase13_legacy_rag_audit import build_opaque_exclusion_registry
from memcontam.readiness.phase13_legacy_rag_bytes import canonical_json_bytes
from memcontam.readiness.phase13_legacy_rag_generators import (
    game24_candidates,
    meb_candidates,
    word_sorting_candidates,
)
from memcontam.readiness.phase13_legacy_rag_materialize import (
    LegacyRagMaterializationError,
    LegacyRagMaterializationRequest,
    materialize_legacy_rag_package,
)
from memcontam.readiness.phase13_legacy_rag_errors import LegacyRagValidationError
from memcontam.readiness.phase13_legacy_rag_runtime import (
    LegacyRagRuntimeRequest,
    load_legacy_rag_state,
)
from memcontam.readiness.phase13_legacy_rag_semantics import render_semantic_record, semantic_records
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

    def __init__(self) -> None:
        self.document_calls = 0

    def encode_document(self, text: str) -> list[float]:
        self.document_calls += 1
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
        norm = sum(value * value for value in values) ** 0.5
        return [value / norm for value in values]


class WrongDimensionEmbedder(ContractEmbedder):
    metadata = {**ContractEmbedder.metadata, "vector_dimension": 8}

    @staticmethod
    def _encode(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()[:8]
        values = [(value - 127.5) / 127.5 for value in digest]
        norm = sum(value * value for value in values) ** 0.5
        return [value / norm for value in values]


def test_canonical_json_bytes_follow_registered_unicode_and_escape_law() -> None:
    payload = {"é": "e\u0301", "/": "\n", "array": [True, None, -2]}

    encoded = canonical_json_bytes(payload)

    assert encoded == b'{"/":"\\n","array":[true,null,-2],"\xc3\xa9":"\xc3\xa9"}'


def test_unaffected_candidate_streams_are_repeatable_and_digest_ordered() -> None:
    game24_first = game24_candidates(frozenset(), limit=64)
    game24_second = game24_candidates(frozenset(), limit=64)
    words_first = word_sorting_candidates(frozenset(), limit=64)
    words_second = word_sorting_candidates(frozenset(), limit=64)

    assert game24_first == game24_second
    assert words_first == words_second
    assert len(game24_first) == len(words_first) == 64
    assert tuple(row.digest for row in game24_first) == tuple(
        sorted(row.digest for row in game24_first)
    )
    assert tuple(row.digest for row in words_first) == tuple(
        sorted(row.digest for row in words_first)
    )
    assert all(row.response for row in game24_first)
    assert all(row.response == " ".join(sorted(row.input_words)) for row in words_first)
    projection = json.dumps(
        [(row.digest, row.response, row.canonical_signature) for row in game24_first],
        separators=(",", ":"),
    )
    assert hashlib.sha256(projection.encode()).hexdigest() == (
        "01fb803851d72e1c4eec4deea1fead6ad0aae57e5f721c3ce0f50dd1f33e3d50"
    )


def test_meb_stream_uses_repaired_order_and_current_domain() -> None:
    candidates = meb_candidates(frozenset(), limit=80)

    assert len(candidates) == 80
    assert tuple(row.digest for row in candidates) == tuple(sorted(row.digest for row in candidates))
    assert all(len(row.ordered_operands) in {3, 4} for row in candidates)
    assert all(set(row.canonical_operator_tuple) <= {"+", "-", "*", "/"} for row in candidates)
    assert all(row.response.endswith(f" = {row.target_value}") for row in candidates)
    assert len({row.canonical_signature for row in candidates}) == 80


def test_semantic_registry_has_exact_balanced_records() -> None:
    for task in ("game24", "math_equation_balancer", "word_sorting"):
        records = semantic_records(task)
        assert [sum(row.stratum == stratum for row in records) for stratum in "ABC"] == [6, 6, 6]
        assert render_semantic_record(records[0]).startswith(b"Title: ")
        assert b"\nRule: " in render_semantic_record(records[0])
        assert not render_semantic_record(records[0]).endswith(b"\n")


def test_auditor_emits_only_opaque_signatures(tmp_path: Path) -> None:
    output = tmp_path / "opaque.json"

    artifact = build_opaque_exclusion_registry(ROOT / "data/phase13/main", output)

    raw = output.read_text(encoding="utf-8")
    assert artifact.status == "NOT_READY"
    assert artifact.task_statuses == {
        "game24": "PASS",
        "math_equation_balancer": "NOT_READY",
        "word_sorting": "PASS",
    }
    assert set(artifact.signature_hashes) == {
        "game24",
        "math_equation_balancer",
        "word_sorting",
    }
    assert '"numbers"' not in raw
    assert '"words"' not in raw
    assert '"target"' not in raw
    assert all(len(value) == 64 for values in artifact.signature_hashes.values() for value in values)


def test_auditor_records_governed_word_sorting_threshold_application(tmp_path: Path) -> None:
    artifact = build_opaque_exclusion_registry(
        ROOT / "data/phase13/main", tmp_path / "opaque.json"
    )

    assert artifact.audit_contracts["word_sorting"].thresholds == {
        "lexical_signature": "1/6",
        "token_overlap": "1/4",
    }
    assert artifact.audit_contracts["word_sorting"].boundary_rule == (
        "similarity_greater_than_or_equal_to_threshold_rejects"
    )


def test_materialized_feasible_packages_validate_and_load_runtime(tmp_path: Path) -> None:
    output = tmp_path / "legacy"
    opaque = tmp_path / "opaque.json"
    build_opaque_exclusion_registry(ROOT / "data/phase13/main", opaque)

    embedder = ContractEmbedder()
    report = materialize_legacy_rag_package(
        LegacyRagMaterializationRequest(
            output,
            ROOT,
            opaque,
            embedder,
            allow_test_embedder=True,
            allow_unfrozen_meb_threshold_for_tests=True,
        )
    )
    manifest_sha256 = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    validated = validate_legacy_rag_package(output, ROOT, manifest_sha256)
    states = {
        task: load_legacy_rag_state(
            LegacyRagRuntimeRequest(
                output,
                ROOT,
                task,
                "clean",
                ContractEmbedder(),
                manifest_sha256,
                allow_test_embedder=True,
            )
        )
        for task in ("game24", "math_equation_balancer", "word_sorting")
    }

    assert report.package_status == "TRACK2_LEGACY_RAG_MATERIALIZATION_COMPLETE"
    assert all(
        task.status == "TRACK2_LEGACY_RAG_MATERIALIZATION_COMPLETE"
        for task in report.tasks.values()
    )
    assert all((output / task).is_dir() for task in report.tasks)
    assert (output / "math_equation_balancer/calibration_registry.json").is_file()
    leakage = json.loads((output / "word_sorting/leakage_calibration.json").read_text())
    assert leakage["thresholds"] == {"lexical_signature": "1/6", "token_overlap": "1/4"}
    assert leakage["separability_result"] == "PASS"
    assert embedder.document_calls == 162
    assert validated == report
    for state in states.values():
        assert state.state.corpus is not None
        assert state.state.index is not None
        assert len(state.state.corpus.active_documents) == 24
    game24_index = states["game24"].state.index
    meb_index = states["math_equation_balancer"].state.index
    word_sorting_index = states["word_sorting"].state.index
    assert game24_index is not None
    assert meb_index is not None
    assert word_sorting_index is not None
    assert len(game24_index.retrieve('{"numbers":[1,2,3,4],"target":24}', 3)) == 3
    assert len(meb_index.retrieve("1 ? 2 ? 3 = 0", 3)) == 3
    assert len(word_sorting_index.retrieve("List: pear apple banana", 3)) == 3


def test_materializer_rejects_non_bge_dimension(tmp_path: Path) -> None:
    output = tmp_path / "legacy"
    opaque = tmp_path / "opaque.json"
    build_opaque_exclusion_registry(ROOT / "data/phase13/main", opaque)

    with pytest.raises(LegacyRagValidationError, match="LEGACY_RAG_INDEX_INVALID"):
        materialize_legacy_rag_package(
            LegacyRagMaterializationRequest(
                output,
                ROOT,
                opaque,
                WrongDimensionEmbedder(),
                allow_test_embedder=True,
                allow_unfrozen_meb_threshold_for_tests=True,
            )
        )


def test_materializer_blocks_without_frozen_meb_structural_threshold(tmp_path: Path) -> None:
    opaque = tmp_path / "opaque.json"
    build_opaque_exclusion_registry(ROOT / "data/phase13/main", opaque)

    with pytest.raises(
        LegacyRagMaterializationError,
        match="MEB_STRUCTURAL_SIMILARITY_THRESHOLD_UNFROZEN",
    ):
        materialize_legacy_rag_package(
            LegacyRagMaterializationRequest(
                tmp_path / "legacy",
                ROOT,
                opaque,
                ContractEmbedder(),
                allow_test_embedder=True,
            )
        )


def test_validator_rejects_tampered_index(tmp_path: Path) -> None:
    output = tmp_path / "legacy"
    opaque = tmp_path / "opaque.json"
    build_opaque_exclusion_registry(ROOT / "data/phase13/main", opaque)
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
    manifest_sha256 = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    index = output / "game24" / "indices.json"
    payload = json.loads(index.read_text(encoding="utf-8"))
    payload["branches"]["clean"]["vectors"][next(iter(payload["branches"]["clean"]["vectors"]))][0] = 0
    index.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")

    with pytest.raises(LegacyRagValidationError, match="LEGACY_RAG_ARTIFACT_HASH_MISMATCH"):
        validate_legacy_rag_package(output, ROOT, manifest_sha256)
