from __future__ import annotations

from fractions import Fraction

from memcontam.experiment.phase12.filter_challenge.freeze_a import (
    game24_certificate,
    meb_certificate,
    word_sorting_certificate,
)


def test_game24_certificate_uses_minimum_canonical_expression() -> None:
    certificate = game24_certificate((3, 3, 8, 8))

    assert certificate is not None
    assert certificate["schema_version"] == "phase12_fv5_game24_certificate_v1"
    assert certificate["expression"] == "(8/(3-(8/3)))"
    assert certificate["postorder_trace"][-1]["result"] == [24, 1]


def test_meb_enumerates_all_16_pairs_and_uses_first_certificate() -> None:
    certificate = meb_certificate(1, 2, 3, 7)

    assert certificate is not None
    assert certificate["operator_pair"] == ["+", "*"]
    assert len(certificate["left_to_right_results"]) == 16
    assert certificate["standard_result"] == [7, 1]


def test_word_sorting_certificate_uses_first_qualifying_pair() -> None:
    certificate = word_sorting_certificate(("ayz", "aza", "bbb"))

    assert certificate is not None
    assert certificate["witness"]["left"] == "ayz"
    assert certificate["witness"]["right"] == "aza"
    assert certificate["witness"]["first_difference_index"] == 1


def test_word_sorting_rejects_first_character_only_difference() -> None:
    assert word_sorting_certificate(("ayz", "byz", "ccc")) is None


def test_word_sorting_rejects_final_position_first_difference() -> None:
    assert word_sorting_certificate(("aya", "ayz", "bbb")) is None


def test_fraction_pairs_are_reduced() -> None:
    certificate = game24_certificate((3, 3, 8, 8))

    assert certificate is not None
    for node in certificate["postorder_trace"]:
        for value in (node["left"], node["right"], node["result"]):
            assert Fraction(*value).denominator == value[1]
