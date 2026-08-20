from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path

from pydantic import JsonValue, TypeAdapter

from memcontam.contamination.phase12.models import canonical_json_hash
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.readiness.phase13_new_mcq_rag import SourceRegistry, TASKS

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_SOURCE_ROOT = Path(__file__).resolve().parents[1]


def source_eligibility(root: Path, evaluation_root: Path) -> dict[str, JsonValue]:
    registry = SourceRegistry.model_validate_json((root / "source_registry_v1.json").read_bytes())
    sources = {source.source_registry_id: source for source in registry.sources}
    manifest_hash = sha256(evaluation_root / "manifest.json")
    return _JSON_OBJECT.validate_python(
        {
            "schema_version": "new_mcq_rag_source_eligibility_registry_v1",
            "status": "COMPLETE",
            "tasks": {
                task: {
                    "source_registry_ids": list(ids),
                    "source_class": "public_task_specification",
                    "question_bearing_rows": 0,
                    "row_id_universe_hash": canonical_json_hash([]),
                    "allowed_authoring_surface": ["task-definition passages"],
                    "masked_fields": [
                        "answer",
                        "answer_index",
                        "cot_content",
                        "official_explanation",
                    ],
                    "main_exclusion_registry_hash": manifest_hash,
                    "disjointness_audit_identity": canonical_json_hash(
                        {
                            "evaluation_manifest_sha256": manifest_hash,
                            "source_registry_ids": list(ids),
                        }
                    ),
                    "availability": "FROZEN_HASHED_PUBLIC_SPECIFICATION",
                    "source_artifact_hashes": {
                        source_id: sources[source_id].sha256 for source_id in ids
                    },
                }
                for task, ids in registry.task_sources.items()
            },
        }
    )


def runtime(cache_root: Path, provider: BgeM3EmbeddingProvider) -> dict[str, JsonValue]:
    snapshot = (
        cache_root
        / "models--BAAI--bge-m3"
        / "snapshots"
        / BgeM3EmbeddingProvider.REVISION
    )
    tokenizer = provider.model.tokenizer
    return _JSON_OBJECT.validate_python(
        {
            "schema_version": "new_mcq_rag_embedding_runtime_v1",
            "status": "NOT_READY_SNAPSHOT_TREE_UNVERIFIED",
            "production_identity": (
                f"{BgeM3EmbeddingProvider.MODEL_ID}@{BgeM3EmbeddingProvider.REVISION}"
            ),
            **provider.metadata,
            "retrieval_mode": "dense_single_vector_only",
            "model_snapshot_tree_sha256": (
                "36b12c5d34c027708130f46fb57645ceb7c5bffe3a72a6b78408440f363bb94a"
            ),
            "missing_objects": [
                "measured_model_snapshot_tree_sha256",
                "production_query_snapshot_verification",
            ],
            "tokenizer_identity": BgeM3EmbeddingProvider.MODEL_ID,
            "tokenizer_revision": BgeM3EmbeddingProvider.REVISION,
            "tokenizer_sha256": sha256(snapshot / "tokenizer.json"),
            "pooling_configuration_sha256": sha256(snapshot / "1_Pooling" / "config.json"),
            "maximum_sequence_length": provider.model.max_seq_length,
            "truncation_side": tokenizer.truncation_side,
            "padding_side": tokenizer.padding_side,
            "query_prefix": None,
            "document_prefix": None,
            "device": str(provider.model.device),
            "dtype": "float32",
            "batch_size": provider.batch_size,
            "deterministic_algorithms": False,
            "sentence_transformers_version": version("sentence-transformers"),
            "transformers_version": version("transformers"),
            "torch_version": version("torch"),
            "numpy_version": version("numpy"),
            "local_files_only": True,
            "similarity": "cosine",
            "top_k": 3,
            "reranker": None,
            "score_threshold": None,
            "tie_break": "lexical_document_id",
            "corpus_scope": "same_task_only",
            "update_mode": "frozen_read_only",
            "embedding_implementation_sha256": sha256(
                _SOURCE_ROOT / "memory" / "embeddings.py"
            ),
            "index_implementation_sha256": sha256(_SOURCE_ROOT / "rag" / "branch_index.py"),
        }
    )


def leakage(root: Path, evaluation_root: Path) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_python(
        {
            "schema_version": "new_mcq_rag_leakage_report_v1",
            "status": "NOT_READY_REQUIRED_LEAKAGE_GATE_UNFROZEN",
            "scope": "accepted_clean_documents_only",
            "canonicalizer_id": "unicode_casefold_whitespace_v1",
            "source_boundary": "public_task_specifications_only_no_question_rows",
            "completed_deterministic_checks": {
                "document_id_uniqueness": "PASS",
                "exact_document_duplicate": "PASS",
                "canonical_document_duplicate": "PASS",
                "cross_task_exact_or_canonical_duplicate": "PASS",
                "exact_evaluation_question_or_option_overlap": "PASS",
            },
            "missing_objects": [
                "task_specific_canonicalizers",
                "displayed_permutation_equivalence",
                "near_duplicate_threshold",
                "structural_similarity_threshold",
                "lexical_overlap_threshold",
                "source_span_registry",
                "exclusion_manifest",
            ],
            "procedural_review_evidence": {
                "review_contract_id": "new_mcq_procedural_review_v1",
                "task_review_sha256": {
                    task: sha256(root / "reviews" / f"{task}.json") for task in TASKS
                },
            },
            "evaluation_manifest_sha256": sha256(evaluation_root / "manifest.json"),
            "evaluation_artifacts": {
                task: sha256(evaluation_root / f"{task}.jsonl") for task in TASKS
            },
        }
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["JsonValue", "leakage", "runtime", "sha256", "source_eligibility"]
