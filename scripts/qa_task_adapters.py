from __future__ import annotations

from memcontam.tasks.game24 import build_instance as build_game24_instance
from memcontam.tasks.math_equation_balancer import build_instance as build_meb_instance
from memcontam.tasks.word_sorting import build_instance as build_word_sorting_instance


def main() -> None:
    game24 = build_game24_instance({"sample_id": "game24_pilot_001", "numbers": [1, 3, 4, 6], "target": 24})
    assert game24.sample_id == "game24_pilot_001"
    assert game24.task_name == "game24"
    assert game24.input == {"numbers": [1, 3, 4, 6]}
    assert game24.verifier_spec == {"target": 24}

    word_sorting = build_word_sorting_instance(
        {
            "sample_id": "word_sorting_pilot_001",
            "words": ["pear", "apple", "banana"],
            "sorted_words": ["apple", "banana", "pear"],
        }
    )
    assert word_sorting.sample_id == "word_sorting_pilot_001"
    assert word_sorting.task_name == "word_sorting"
    assert word_sorting.input == {"words": ["pear", "apple", "banana"]}
    assert word_sorting.verifier_spec == {"sorted_words": ["apple", "banana", "pear"]}

    meb = build_meb_instance(
        {
            "sample_id": "meb_pilot_001",
            "input": "2 + 5 = ?",
            "verifier_spec": {"target": "2 + 5 = 7", "target_value": 7},
        }
    )
    assert meb.sample_id == "meb_pilot_001"
    assert meb.task_name == "math_equation_balancer"
    assert meb.input == {"input": "2 + 5 = ?"}
    assert meb.verifier_spec == {"target": "2 + 5 = 7", "target_value": 7}


if __name__ == "__main__":
    main()
