from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from memcontam.readiness.phase13_new_mcq_phase1_freeze import (
    Phase1FreezeError,
    Phase1SourcePaths,
    build_phase1_freeze,
)
from memcontam.readiness.phase13_core_bundle import CoreTask


MMLU_SOURCE = Path("data/phase13/rag/new_mcq/sources/mmlu_pro_validation_475d58ba.parquet")
GPQA_MAIN = Path("data/phase13/core/materialized/gpqa_main_certification_633f5ee8.csv")
GPQA_DIAMOND = Path("data/phase13/core/materialized/gpqa_diamond.jsonl")
FROZEN_AT = "2026-08-20T00:00:00Z"


def _paths() -> Phase1SourcePaths:
    return Phase1SourcePaths(
        mmlu_source=MMLU_SOURCE,
        gpqa_main_source=GPQA_MAIN,
        gpqa_diamond_evaluation=GPQA_DIAMOND,
        gpqa_extended_source=None,
        frozen_at=FROZEN_AT,
    )


def test_real_sources_select_h1_in_both_outcome_blind_halves() -> None:
    freeze = build_phase1_freeze(_paths())

    assert set(freeze.tasks) == {
        "mmlu_pro_engineering",
        "mmlu_pro_physics",
        "gpqa_diamond",
    }
    expected_halves: dict[CoreTask, tuple[int, int]] = {
        "mmlu_pro_engineering": (2, 3),
        "mmlu_pro_physics": (2, 3),
        "gpqa_diamond": (125, 125),
    }
    for task_id, (build_rows, calibration_rows) in expected_halves.items():
        task = freeze.tasks[task_id]
        assert task.selected_candidate_id == "MCQ-H1-LEXICAL-OVERLAP-v1"
        assert task.status == "MECHANICALLY_CERTIFIED_PENDING_BLINDED_PANEL"
        assert (task.build.rows, task.calibration.rows) == (build_rows, calibration_rows)
        assert task.build.applicable_rows > 0
        assert task.build.counterexample_rows > 0
        assert task.calibration.applicable_rows > 0
        assert task.calibration.counterexample_rows > 0


def test_sources_bind_exact_hashes_and_exclude_every_diamond_row() -> None:
    freeze = build_phase1_freeze(_paths())

    engineering = freeze.tasks["mmlu_pro_engineering"].source
    gpqa = freeze.tasks["gpqa_diamond"].source
    assert engineering.source_sha256 == (
        "a6db33e44c7a8d6a0a9665aabe6596a5e7436bebb62412d1219821283835e457"
    )
    assert gpqa.source_sha256 == (
        "acdeeac8f622267f2cd727d7d474202ea08dec80f7d3c3593b3ef8644f19b8e3"
    )
    assert gpqa.exclusion_source_sha256 == (
        "d6413fa81bdbc1bf08a83cc81c1a369bcbaf9a51d27c027e0b3f219e584be372"
    )
    assert gpqa.selected_source_config == "gpqa_main"
    assert gpqa.excluded_canonical_rows == 198
    assert gpqa.eligible_rows == 250


def test_gpqa_public_freeze_never_contains_gated_source_text() -> None:
    freeze_json = build_phase1_freeze(_paths()).model_dump_json()
    with GPQA_MAIN.open(encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))

    gated_values = (
        row[field]
        for row in rows
        for field in ("Question", "Correct Answer", "Explanation")
    )
    assert all(json.dumps(value) not in freeze_json for value in gated_values if len(value) >= 16)
    for forbidden_key in ('"question"', '"options"', '"answer"', '"explanation"', '"stem"'):
        assert forbidden_key not in freeze_json
    assert '"official_gold' not in freeze_json
    assert '"heuristic_selected_option' not in freeze_json


def test_triplet_relevance_and_native_render_identities_are_complete() -> None:
    freeze = build_phase1_freeze(_paths())

    for task in freeze.tasks.values():
        assert task.normalizer_id == "MCQ-NORM-NFKC-CASEFOLD-WS-v1"
        assert task.tokenizer_id == "MCQ-TOK-UNICODE-LNM-RUN-v1"
        assert task.overlap_metric_id == "MCQ-H1-JACCARD-DISTINCT-TOKENS-v1"
        assert task.counterexample_certification_id == "MCQ-H1-GOLD-DISAGREEMENT-CERT-v1"
        assert task.triplet.false_candidate_id == task.selected_candidate_id
        assert task.triplet.correct_twin_id == "MCQ-H1-CORRECT-SUBSTANTIVE-CONTENT-v1"
        assert task.triplet.irrelevant_control_id == "MCQ-I1-SINGLETON-OPTION-v1"
        assert task.triplet.correct_shares_false_applicability is True
        assert task.triplet.irrelevant_is_valid is True
        assert task.triplet.irrelevant_is_inapplicable_to_target is True
        assert task.relevance.false_relation == "MCQ-H1-UNIQUE-MAX-APP-v1"
        assert task.relevance.correct_relation == "MCQ-H1-UNIQUE-MAX-APP-v1"
        assert task.relevance.irrelevant_relation == "MCQ-I1-SINGLETON-APP-v1"
        packets = {packet.baseline_id: packet for packet in task.render_packets}
        assert {key: value.native_kind for key, value in packets.items()} == {
            "FH-bounded": "raw_interaction",
            "RAG-Frozen": "retrieved_document",
            "BoT-style": "thought_template",
            "Reflexion-style": "reflection",
            "DC-RS adapted": "raw_interaction",
        }
        assert packets["RAG-Frozen"].cell_status == (
            "NOT_READY_NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN"
        )
        assert packets["DC-RS adapted"].role_invariant_query_identity is not None
        assert len({packet.packet_identity for packet in packets.values()}) == 5


def test_freeze_is_reproducible_and_pool_precedes_certification() -> None:
    first = build_phase1_freeze(_paths())
    second = build_phase1_freeze(_paths())

    assert first == second
    assert first.candidate_pool_frozen_before_certification is True
    assert len(first.candidate_pool_identity) == 64
    assert len(first.freeze_identity) == 64
    assert all(
        task.pool_identity == first.candidate_pool_identity for task in first.tasks.values()
    )
    assert all(len(task.candidate_freeze_identity) == 64 for task in first.tasks.values())


def test_gpqa_main_is_used_without_touching_fallback_when_eligible() -> None:
    paths = _paths().model_copy(
        update={"gpqa_extended_source": Path("does-not-exist-and-must-not-be-read.csv")}
    )

    freeze = build_phase1_freeze(paths)

    assert freeze.tasks["gpqa_diamond"].source.selected_source_config == "gpqa_main"


def test_source_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    corrupt = tmp_path / "mmlu.parquet"
    corrupt.write_bytes(MMLU_SOURCE.read_bytes() + b"corrupt")
    paths = _paths().model_copy(update={"mmlu_source": corrupt})

    with pytest.raises(Phase1FreezeError, match="PHASE1_MMLU_SOURCE_HASH_MISMATCH"):
        build_phase1_freeze(paths)
