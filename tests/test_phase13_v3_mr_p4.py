from __future__ import annotations

import pytest
from .test_phase13_v3_artifact_builder import builder_source as builder_source, staged as staged

from memcontam.readiness.phase13_main_live_runtime_support import verifier
from memcontam.tasks.base import TaskInstance
from memcontam.verifiers.math_equation_balancer import verify_answer


@pytest.mark.parametrize("answer", (
    "2 + 3 * 4 = 14", "2+3*4=14", "2 + 3*4 = 14", " 2\t+3 *4=14\n",
))
def test_meb_equivalent_whitespace_preserves_semantics(answer: str) -> None:
    task = TaskInstance(sample_id="synthetic-meb", task_name="math_equation_balancer",
                        input={"input": "2 ? 3 ? 4 = 14"},
                        verifier_spec={"target": "2 + 3 * 4 = 14", "target_value": 14})
    assert verifier("math_equation_balancer")(answer, task) is True


@pytest.mark.parametrize("answer", (
    "-2+-3*-4=10", "-2 + -3 * -4 = +10", "-2+ -3* -4=10",
))
def test_meb_signed_integers_remain_operands(answer: str) -> None:
    task = TaskInstance(sample_id="synthetic-signed-meb", task_name="math_equation_balancer",
                        input={"input": "-2 ? -3 ? -4 = 10"},
                        verifier_spec={"target": "-2 + -3 * -4 = 10", "target_value": 10})
    assert verify_answer(answer, task).is_correct


@pytest.mark.parametrize("answer", (
    "14", "(2+3)*4=14", "2+3**4=14", "2+3//4=14", "2+3%4=14",
    "2+3*4==14", "2+3*4=14=14", "2+3*4=14junk", "2+3*4=1 4",
    "2+3*4.0=14", "2+3*4e0=14", "2+3*4=14;print(1)", "2+3*4=15",
    "3+2*4=14", "2+3*4+0=14", "2+3/0=14", "2+3*4=", "2+3*=14",
))
def test_meb_rejects_extra_grammar_or_changed_task(answer: str) -> None:
    task = TaskInstance(sample_id="synthetic-meb-negative", task_name="math_equation_balancer",
                        input={"input": "2 ? 3 ? 4 = 14"},
                        verifier_spec={"target": "2 + 3 * 4 = 14", "target_value": 14})
    assert not verify_answer(answer, task).is_correct


def test_meb_accepts_alternative_operator_assignment_exactly() -> None:
    task = TaskInstance(sample_id="synthetic-alternative-meb", task_name="math_equation_balancer",
                        input={"input": "2 ? 2 ? 2 = 2"},
                        verifier_spec={"target": "2 + 2 - 2 = 2", "target_value": 2})
    assert verify_answer("2*2/2=2", task).is_correct


def test_mr_p4_only_publishes_its_six_files(staged):
    from memcontam.readiness.phase13_v3_publication import P4_PATHS
    module, root, output, authority = staged
    assert {path.relative_to(output).as_posix() for path in output.rglob("*.json")} == set(P4_PATHS)
    assert module.validate_stage(root, authority, output, stage="mr-p4").status == "CLOSED"


@pytest.mark.parametrize("field", ("concrete_seed_ids", "orders", "registry", "capacity"))
def test_each_first_freeze_mutation_fails_after_self_rehash(staged, field):
    import json
    from memcontam.readiness.phase13_v3_cost_models import digest, canonical_bytes
    from memcontam.readiness.phase13_v3_builder_models import MRP4Manifest
    module, root, output, authority = staged
    path = output / "mr_p4/corrected_v3/manifest_v3.json"
    payload = json.loads(path.read_bytes())
    first = payload["first_freeze"]
    if field == "concrete_seed_ids":
        first[field] = list(range(9))
    elif field == "capacity":
        first[field] = 8191
    elif field == "orders":
        first[field]["tasks"]["game24"]["seeds"][0]["ordered_sample_ids"].reverse()
    else:
        first[field]["tasks"]["game24"]["seeds"][0]["tau_star"] += 1
    model = MRP4Manifest.model_validate_json(json.dumps(payload))
    model = model.model_copy(update={"closure_hash": digest(model, "closure_hash")})
    path.write_bytes(canonical_bytes(model))
    with pytest.raises(ValueError, match="MAIN_MR_P4_FIRST_FREEZE_MISMATCH"):
        module.validate_stage(root, authority, output, stage="mr-p4")
