from memcontam.evaluation.phase13_observability_registration import registered_failure_class
from memcontam.tasks.base import TaskInstance


def test_meb_substantive_failure_requires_precedence_only_target() -> None:
    query = TaskInstance(sample_id="meb", task_name="math_equation_balancer",
                         input={"input": "1 ? 2 ? 3 = 7"},
                         verifier_spec={"target": "1 + 2 * 3 = 7", "target_value": 7})

    assert registered_failure_class(query, "1 + 2 + 3 = 7", 0) == (
        "MEB_CANONICAL_FALSE_RULE_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V1"
    )


def test_word_sorting_v2_failure_requires_nonempty_prefix_witness() -> None:
    query = TaskInstance(sample_id="words", task_name="word_sorting",
                         input={"words": ["ayz", "aza"]},
                         verifier_spec={"sorted_words": ["ayz", "aza"]})

    assert registered_failure_class(query, "aza ayz", 0) == (
        "WS_CANONICAL_FALSE_RULE_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V2"
    )
    inapplicable = query.model_copy(update={"input": {"words": ["a", "z"]},
                                              "verifier_spec": {"sorted_words": ["a", "z"]}})
    assert registered_failure_class(inapplicable, "z a", 0) is None


def test_mcq_substantive_failure_requires_unique_selected_option() -> None:
    query = TaskInstance(sample_id="mcq", task_name="mmlu_pro_engineering",
                         input={"question": "fixture", "options": ["longer selected option", "short"]},
                         verifier_spec={"answer_index": 1, "answer_label": "B"})

    assert registered_failure_class(query, "A", 0) == (
        "MMLUENG_SURFACE_CUE_HEURISTIC_APPLICABLE_INSTANCE_SUBSTANTIVE_FAILURE_V1"
    )
    tied = query.model_copy(update={"input": {"question": "fixture", "options": ["first", "other"]}})
    assert registered_failure_class(tied, "A", 0) is None
    assert registered_failure_class(query, "unknown", 0) is None
