from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from memcontam.readiness import phase13_new_mcq_rag_materialize
from memcontam.readiness import phase13_new_mcq_rag_materialize_output


PACKAGE_ROOT = Path("data/phase13/rag/new_mcq")
STATUS_PATH = PACKAGE_ROOT.parent / "new_mcq_rag_status_v1.json"


def test_status_materialization_does_not_require_prior_status(tmp_path: Path) -> None:
    package = tmp_path / "new_mcq"
    shutil.copytree(PACKAGE_ROOT, package)

    phase13_new_mcq_rag_materialize_output.write_status(package)

    status = (tmp_path / STATUS_PATH.name).read_text(encoding="utf-8")
    assert "NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN" in status
    assert "clean_document_applicability_predicates_and_relevance_universe" in status
    assert "2026-08-22T18:00:00+09:00" in status


def test_materializer_preserves_live_package_when_staged_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_root = tmp_path / "rag" / "new_mcq"
    live_root.parent.mkdir()
    shutil.copytree(PACKAGE_ROOT, live_root)
    shutil.copy2(STATUS_PATH, live_root.parent / STATUS_PATH.name)
    manifest_before = (live_root / "package_manifest_v1.json").read_bytes()
    status_before = (live_root.parent / STATUS_PATH.name).read_bytes()

    def write_invalid_stage(root: Path, evaluation_root: Path, cache_root: Path) -> None:
        del evaluation_root, cache_root
        (root / "package_manifest_v1.json").write_text("staged-invalid\n", encoding="utf-8")

    def reject_stage(root: Path, evaluation_root: Path) -> None:
        del root, evaluation_root
        raise RuntimeError("late validation failure")

    monkeypatch.setattr(
        phase13_new_mcq_rag_materialize,
        "_materialize_staged_package",
        write_invalid_stage,
    )
    monkeypatch.setattr(
        phase13_new_mcq_rag_materialize,
        "validate_new_mcq_rag_package",
        reject_stage,
    )

    with pytest.raises(RuntimeError, match="late validation failure"):
        phase13_new_mcq_rag_materialize.materialize_new_mcq_rag_package(
            live_root,
            Path("data/phase13/core/materialized"),
            tmp_path / "cache",
        )

    assert (live_root / "package_manifest_v1.json").read_bytes() == manifest_before
    assert (live_root.parent / STATUS_PATH.name).read_bytes() == status_before
