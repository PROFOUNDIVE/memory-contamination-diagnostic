import json
import hashlib
from pathlib import Path

import pytest

from memcontam.readiness import phase13_new_mcq_p0_4_evidence as candidate_evidence


ROOT = Path("data/phase13/rag/new_mcq")
SOURCE = ROOT / "sources" / "mmlu_pro_validation_475d58ba.parquet"
GPQA_TREE = ROOT / "sources" / "gpqa_tree_633f5ee8.json"
EVALUATION_ROOT = Path("data/phase13/core/materialized")


def test_mmlu_evidence_records_unpartitioned_observations_without_certification() -> None:
    evidence = candidate_evidence.build_candidate_evidence(ROOT, EVALUATION_ROOT)

    engineering = evidence.tasks["mmlu_pro_engineering"]
    physics = evidence.tasks["mmlu_pro_physics"]
    engineering_h1 = engineering.mechanical_observations["MCQ-H1-LEXICAL-OVERLAP-v1"]
    physics_h1 = physics.mechanical_observations["MCQ-H1-LEXICAL-OVERLAP-v1"]
    assert evidence.authority_stack == (
        "phase13_theory_revised_v1",
        "phase13_baseline_revised_v5",
        "phase13_protocol_revised_v7",
        "phase13_experiment_revised_v7",
    )
    assert evidence.split_registry_status == "NOT_ESTABLISHED"
    assert engineering.status == "NOT_READY_SPLIT_REGISTRY_UNFROZEN"
    assert engineering.mechanical_candidate_id is None
    assert engineering_h1.query_ids == ("40", "41", "42", "43", "44")
    assert engineering_h1.applicable_query_ids == ("41", "43")
    assert engineering_h1.counterexample_query_ids == ("41", "43")
    assert physics.mechanical_candidate_id is None
    assert physics_h1.query_ids == ("10", "11", "12", "13", "14")
    assert physics_h1.counterexample_query_ids == ("10", "12", "13", "14")


def test_gpqa_evidence_accounts_for_every_pinned_tree_entry() -> None:
    evidence = candidate_evidence.build_candidate_evidence(ROOT, EVALUATION_ROOT)

    gpqa = evidence.tasks["gpqa_diamond"]
    inventory = {entry.path: entry for entry in gpqa.source_inventory}
    assert gpqa.status == "NOT_READY_NO_ELIGIBLE_SOURCE"
    assert gpqa.source_sha256 == "3a722b406849c230a76cf797f0e5481a2dd17fe403be650b5798703ecfa54526"
    assert set(inventory) == {
        ".gitattributes",
        "README.md",
        "eval.yaml",
        "gpqa_diamond.csv",
        "gpqa_experts.csv",
        "gpqa_extended.csv",
        "gpqa_main.csv",
        "license.txt",
    }
    assert inventory["gpqa_experts.csv"].question_bearing is False
    assert inventory["gpqa_experts.csv"].eligibility == "INELIGIBLE_METADATA_ONLY"
    for path in ("gpqa_diamond.csv", "gpqa_main.csv", "gpqa_extended.csv"):
        assert inventory[path].question_bearing is True
        assert inventory[path].eligibility == "INELIGIBLE_EVALUATION_SET"


def test_terminal_blocker_ledger_covers_unfrozen_protocol_objects() -> None:
    evidence = candidate_evidence.build_candidate_evidence(ROOT, EVALUATION_ROOT)

    common = {
        "prospectively_frozen_build_calibration_split",
        "candidate_coverage_contract",
        "baseline_native_render_packets",
        "three_evaluator_blinded_plausibility_panel",
        "leakage_metric_threshold_registry",
        "full_leakage_conformance",
        "correct_i1_constructibility_and_validity",
        "unicode_source_test_vector_provenance",
        "candidate_freeze_identity",
        "deterministic_relevance_affinity_constructibility",
    }
    for task in ("mmlu_pro_engineering", "mmlu_pro_physics"):
        assert set(evidence.tasks[task].remaining_objects) == common
    assert set(evidence.tasks["gpqa_diamond"].remaining_objects) == common | {
        "eligible_question_bearing_construction_source",
        "prospective_displayed_permutation_identity",
    }


def test_evaluation_hashes_are_recomputed_from_exact_manifest_paths(tmp_path: Path) -> None:
    evaluation = tmp_path / "materialized"
    evaluation.mkdir()
    for path in EVALUATION_ROOT.iterdir():
        if path.is_file():
            (evaluation / path.name).write_bytes(path.read_bytes())
    manifest_path = evaluation / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["mmlu_pro_engineering"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        candidate_evidence.EvidenceError,
        match="NEW_MCQ_EVALUATION_ARTIFACT_MISMATCH",
    ):
        candidate_evidence.build_candidate_evidence(ROOT, evaluation)


def test_evaluation_hashes_reject_cotampered_artifact_and_manifest(tmp_path: Path) -> None:
    evaluation = tmp_path / "materialized"
    evaluation.mkdir()
    for path in EVALUATION_ROOT.iterdir():
        if path.is_file():
            (evaluation / path.name).write_bytes(path.read_bytes())
    artifact_path = evaluation / "mmlu_pro_engineering.jsonl"
    artifact_path.write_bytes(artifact_path.read_bytes() + b"\n")
    manifest_path = evaluation / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["mmlu_pro_engineering"]["sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        candidate_evidence.EvidenceError,
        match="NEW_MCQ_EVALUATION_ARTIFACT_MISMATCH",
    ):
        candidate_evidence.build_candidate_evidence(ROOT, evaluation)


def test_materialized_evidence_is_rebuilt_from_all_pinned_inputs(tmp_path: Path) -> None:
    root = tmp_path / "new_mcq"
    (root / "sources").mkdir(parents=True)
    for source in (SOURCE, GPQA_TREE):
        (root / "sources" / source.name).write_bytes(source.read_bytes())
    output = candidate_evidence.materialize_candidate_evidence(root, EVALUATION_ROOT)

    assert candidate_evidence.validate_candidate_evidence(
        root, EVALUATION_ROOT
    ) == candidate_evidence.build_candidate_evidence(ROOT, EVALUATION_ROOT)
    output.write_text(
        output.read_text(encoding="utf-8").replace(
            "NOT_READY_SPLIT_REGISTRY_UNFROZEN",
            "NOT_READY_NO_ELIGIBLE_SOURCE",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        candidate_evidence.EvidenceError,
        match="NEW_MCQ_CANDIDATE_EVIDENCE_INVALID",
    ):
        candidate_evidence.validate_candidate_evidence(root, EVALUATION_ROOT)


def test_missing_candidate_source_raises_domain_error(tmp_path: Path) -> None:
    root = tmp_path / "new_mcq"
    (root / "sources").mkdir(parents=True)
    (root / "sources" / GPQA_TREE.name).write_bytes(GPQA_TREE.read_bytes())

    with pytest.raises(
        candidate_evidence.EvidenceError,
        match="NEW_MCQ_CANDIDATE_SOURCE_INVALID",
    ):
        candidate_evidence.validate_candidate_evidence(root, EVALUATION_ROOT)


def test_mmlu_source_hash_mismatch_precedes_parquet_parse(tmp_path: Path) -> None:
    root = tmp_path / "new_mcq"
    (root / "sources").mkdir(parents=True)
    (root / "sources" / SOURCE.name).write_bytes(b"not parquet")
    (root / "sources" / GPQA_TREE.name).write_bytes(GPQA_TREE.read_bytes())

    with pytest.raises(
        candidate_evidence.EvidenceError,
        match="NEW_MCQ_BUILD_SOURCE_MISMATCH",
    ):
        candidate_evidence.build_candidate_evidence(root, EVALUATION_ROOT)
