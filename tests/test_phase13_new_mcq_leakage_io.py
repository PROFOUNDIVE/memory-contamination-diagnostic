from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from memcontam.readiness.phase13_new_mcq_leakage import LeakageArtifactError
from memcontam.readiness.phase13_new_mcq_leakage_io import load_leakage_inputs

PACKAGE_ROOT = Path("data/phase13/rag/new_mcq")
EVALUATION_ROOT = Path("data/phase13/core/materialized")


def test_current_inputs_bind_all_clean_documents_and_evaluation_exclusions() -> None:
    inputs = load_leakage_inputs(PACKAGE_ROOT, EVALUATION_ROOT)

    assert len(inputs.documents) == 72
    assert len(inputs.evaluation_items) == 698
    assert all(document.source_span_ids for document in inputs.documents)
    assert all(item.source_span_ids and item.identity_keys for item in inputs.evaluation_items)
    assert {key for key, _value in inputs.input_hashes} == {
        "accepted:gpqa_diamond",
        "accepted:mmlu_pro_engineering",
        "accepted:mmlu_pro_physics",
        "evaluation:gpqa_diamond",
        "evaluation:mmlu_pro_engineering",
        "evaluation:mmlu_pro_physics",
        "evaluation_manifest",
        "source_eligibility_registry",
    }


def test_evaluation_artifact_tamper_fails_closed(tmp_path: Path) -> None:
    package = tmp_path / "package"
    evaluation = tmp_path / "evaluation"
    shutil.copytree(PACKAGE_ROOT, package)
    shutil.copytree(EVALUATION_ROOT, evaluation)
    artifact = evaluation / "mmlu_pro_engineering.jsonl"
    artifact.write_text(
        artifact.read_text(encoding="utf-8").replace("\n", " \n", 1),
        encoding="utf-8",
    )

    with pytest.raises(LeakageArtifactError, match="NEW_MCQ_LEAKAGE_EVALUATION_INVALID"):
        load_leakage_inputs(package, evaluation)
