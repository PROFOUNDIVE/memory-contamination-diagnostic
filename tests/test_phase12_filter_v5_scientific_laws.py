from __future__ import annotations

from fractions import Fraction

from memcontam.experiment.phase12.filter_challenge.calibration_laws import (
    game24_certificate,
    meb_certificate,
    word_sorting_certificate,
)


def test_game24_certificate_uses_minimum_canonical_expression() -> None:
    certificate = game24_certificate((3, 3, 8, 8))

    assert certificate is not None
    assert certificate["schema_version"] == "phase12_fv5_game24_certificate_v1"
    assert certificate["expression"] == "(8/(3-(8/3)))"
    trace = certificate["postorder_trace"]
    assert isinstance(trace, list)
    assert isinstance(trace[-1], dict)
    assert trace[-1]["result"] == [24, 1]


def test_meb_enumerates_all_16_pairs_and_uses_first_certificate() -> None:
    certificate = meb_certificate(1, 2, 3, 7)

    assert certificate is not None
    assert certificate["operator_pair"] == ["+", "*"]
    results = certificate["left_to_right_results"]
    assert isinstance(results, list)
    assert len(results) == 16
    assert certificate["standard_result"] == [7, 1]


def test_word_sorting_certificate_uses_first_qualifying_pair() -> None:
    certificate = word_sorting_certificate(("ayz", "aza", "bbb"))

    assert certificate is not None
    witness = certificate["witness"]
    assert isinstance(witness, dict)
    assert witness["left"] == "ayz"
    assert witness["right"] == "aza"
    assert witness["first_difference_index"] == 1


def test_word_sorting_rejects_first_character_only_difference() -> None:
    assert word_sorting_certificate(("ayz", "byz", "ccc")) is None


def test_word_sorting_rejects_final_position_first_difference() -> None:
    assert word_sorting_certificate(("aya", "ayz", "bbb")) is None


def test_fraction_pairs_are_reduced() -> None:
    certificate = game24_certificate((3, 3, 8, 8))

    assert certificate is not None
    trace = certificate["postorder_trace"]
    assert isinstance(trace, list)
    for node in trace:
        assert isinstance(node, dict)
        for value in (node["left"], node["right"], node["result"]):
            assert Fraction(*value).denominator == value[1]
