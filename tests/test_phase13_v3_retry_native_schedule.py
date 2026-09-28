from memcontam.readiness.phase13_main_production import ProductionObject, _stages
from memcontam.readiness.phase13_v3_cost_models import CostUnit, StageOccurrences
from memcontam.readiness.phase13_v3_retry import dependency_fanout, schedule_unit


def test_reflexion_scheduled_occurrences_preserve_sparse_trial_slots() -> None:
    unit = ProductionObject(sequence=0, unit_id="a" * 64, kind="MEMORY_BEARING", seed=0,
                            task="game24", memory_baseline="reflexion_style", arm="contam",
                            prefix_unit_id="b" * 64, projected_cost_krw=0)
    cost = CostUnit(unit_id=unit.unit_id, stages=(
        StageOccurrences(stage_id="Reflexion_actor_generation", calls=100),
        StageOccurrences(stage_id="Reflexion_reflection", calls=100),
    ))

    actual = tuple((key.stage, key.ordinal) for key, _ in schedule_unit(unit, cost)[:8])

    assert actual == (
        ("reflexion_generate", 0), ("reflexion_reflect", 0),
        ("reflexion_generate", 1), ("reflexion_reflect", 1),
        ("reflexion_generate", 2), ("reflexion_reflect", 2),
        ("reflexion_generate", 3), ("reflexion_reflect", 3),
    )


def test_bot_dependency_order_is_native_not_cost_group_order() -> None:
    unit = ProductionObject(sequence=0, unit_id="a" * 64, kind="MEMORY_BEARING", seed=0,
                            task="game24", memory_baseline="bot_style", arm="contam",
                            prefix_unit_id="b" * 64, projected_cost_krw=0)
    cost = CostUnit(unit_id=unit.unit_id, stages=(
        StageOccurrences(stage_id="BoT_thought_distillation", calls=50),
        StageOccurrences(stage_id="BoT_solve", calls=50),
        StageOccurrences(stage_id="BoT_problem_distillation", calls=50),
    ))

    actual = tuple((key.stage, key.ordinal) for key, _ in schedule_unit(unit, cost)[:3])

    assert actual == (("bot_problem_distill", 0), ("bot_instantiate_solve", 0),
                      ("bot_thought_distill", 0))


def test_prefix_terminal_fanout_counts_dependent_branch_units_not_all_child_calls() -> None:
    prefix = ProductionObject(sequence=0, unit_id="a" * 64, kind="CLEAN_PREFIX", seed=0,
                              task="game24", memory_baseline="reflexion_style", arm="NOT_APPLICABLE",
                              prefix_unit_id=None, projected_cost_krw=0)
    children = tuple(ProductionObject(sequence=index + 1, unit_id=str(index) * 64,
                         kind="MEMORY_BEARING", seed=0, task="game24", memory_baseline="reflexion_style",
                         arm=arm, prefix_unit_id=prefix.unit_id, projected_cost_krw=0)
                     for index, arm in enumerate(("clean", "correct", "irrelevant", "contam"), start=1))

    assert dependency_fanout((prefix, *children), prefix.unit_id, remaining_occurrences=1) == 5


def test_reflexion_prefix_reserves_only_reachable_actor_and_reflection() -> None:
    unit = ProductionObject(sequence=0, unit_id="a" * 64, kind="CLEAN_PREFIX", seed=0,
                            task="game24", memory_baseline="reflexion_style", arm="NOT_APPLICABLE",
                            prefix_unit_id=None, projected_cost_krw=0)

    assert _stages(unit) == (("reflexion_generate", 1), ("reflexion_reflect", 1))
