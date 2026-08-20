from __future__ import annotations

from memcontam.readiness.phase13_new_mcq_candidate import (
    DisplayedMcq,
    InterventionRelevance,
    build_intervention_relevance,
    certify_task_candidate,
    h1_selection,
    h2_selection,
    mcq_tokens,
)


def test_h1_uses_exact_jaccard_and_requires_unique_maximum() -> None:
    item = DisplayedMcq(
        query_id="q1",
        stem="stable voltage source",
        options=("stable source", "voltage", "unrelated detail"),
        gold_index=1,
        display_identity="display-1",
    )

    selection = h1_selection(item)

    assert selection == 0


def test_h1_returns_none_for_exact_tie() -> None:
    item = DisplayedMcq("q1", "alpha", ("alpha", "alpha"), 0, "display-1")

    assert h1_selection(item) is None


def test_h2_uses_token_count_then_nonspace_codepoints() -> None:
    item = DisplayedMcq("q1", "stem", ("one two", "onetwo!"), 0, "display-1")

    assert h2_selection(item) == 0


def test_tokenizer_keeps_unicode_marks_only_after_letter_or_number() -> None:
    assert mcq_tokens(" ◌́ Á １２ ") == ("á", "12")


def test_candidate_certification_uses_h1_then_h2_without_gold_in_applicability() -> None:
    rows = (
        DisplayedMcq("q1", "alpha beta", ("alpha beta", "gamma"), 1, "display-1"),
        DisplayedMcq("q2", "delta", ("delta", "epsilon"), 0, "display-2"),
    )

    certification = certify_task_candidate("mmlu_pro_engineering", "build-v1", rows)

    assert certification.candidate_id == "MCQ-H1-LEXICAL-OVERLAP-v1"
    assert certification.applicable_query_ids == ("q1", "q2")
    assert certification.counterexample_query_ids == ("q1",)
    assert certification.irrelevant_control_id == "MCQ-I1-SINGLETON-OPTION-v1"


def test_candidate_certification_falls_back_to_h2() -> None:
    rows = (
        DisplayedMcq("q1", "same shared", ("same", "shared extra"), 0, "display-1"),
    )

    certification = certify_task_candidate("gpqa_diamond", "build-v1", rows)

    assert certification.candidate_id == "MCQ-H2-DETAIL-LENGTH-v1"
    assert certification.counterexample_query_ids == ("q1",)


def test_intervention_relevance_is_applicability_not_retrieval() -> None:
    rows = (
        DisplayedMcq("q1", "alpha", ("alpha", "beta"), 1, "display-1"),
        DisplayedMcq("q2", "same", ("same", "same"), 0, "display-2"),
    )
    certification = certify_task_candidate("mmlu_pro_physics", "build-v1", rows)

    relevance = build_intervention_relevance(certification, rows)

    assert relevance["q1"] == InterventionRelevance(True, True, False)
    assert relevance["q2"] == InterventionRelevance(False, False, False)
