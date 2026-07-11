from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from memcontam.logging.schema import ContaminationExposure, TrialLog, VerifierResult


NOT_COMPUTED = "not_computed"


def _method_call(stage: str, **overrides):
    base = {
        "stage": stage,
        "messages": [{"role": "user", "content": f"{stage} prompt"}],
        "raw_response": f"{stage} response",
        "model": "replay",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 100,
        "latency_ms": 10,
        "token_usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        "retry_count": 0,
        "error_type": None,
    }
    base.update(overrides)
    return base


def _trial_row(**overrides):
    base = {
        "trial_id": "run1:game24:s1:no_memory:clean:replay",
        "run_id": "run1",
        "task_name": "game24",
        "sample_id": "s1",
        "baseline": "no_memory",
        "arm": "clean",
        "backbone": "replay",
        "input": {"numbers": [1, 3, 4, 6], "target": 24},
        "gold_or_verifier_spec": {"target": 24},
        "prompt_messages": [{"role": "user", "content": "solve"}],
        "raw_response": "final: 24",
        "verifier_result": VerifierResult(is_correct=True),
        "contamination_exposure": ContaminationExposure(),
        "bad_memory_uptake_label": "not_applicable",
        "repeated_failure_label": "not_applicable",
        "recovery_after_filter_label": "not_applicable",
        "latency_ms": 12,
        "token_usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }
    base.update(overrides)
    return TrialLog(**base).model_dump(mode="json")


def _bot_write_event(**overrides):
    base = {
        "event_type": "bot_write",
        "baseline": "bot_style",
        "parent_trial_id": "run1:game24:s1:bot_style:contaminated:replay",
        "source_entry_ids": ["bot_style_memory_1"],
        "new_entry_id": "bot_template:1",
        "update_reason": "distilled_thought_template_from_problem_solution_pair",
    }
    base.update(overrides)
    return base


def _write_trials_jsonl(run_dir: Path, rows: list[dict]) -> Path:
    trials_path = run_dir / "trials.jsonl"
    trials_path.write_text("".join(json.dumps(row) + chr(10) for row in rows), encoding="utf-8")
    return trials_path


def _cli_aggregate(run_dir: Path) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    return subprocess.run(
        [sys.executable, "-m", "memcontam.cli", "aggregate", str(run_dir)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_aggregate_run_computes_shallow_metrics_and_cli_prints_json(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "aggregate_run"
    run_dir.mkdir(parents=True)

    clean_row = _trial_row(
        trial_id="run1:game24:s1:no_memory:clean:replay",
        arm="clean",
        verifier_result=VerifierResult(is_correct=True),
        raw_response="final: 24",
        parsed_answer="24",
        latency_ms=12,
        token_usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    )
    contaminated_row = _trial_row(
        trial_id="run1:game24:s1:no_memory:contaminated:replay",
        arm="contaminated",
        verifier_result=VerifierResult(is_correct=False, reason="incorrect"),
        raw_response="final: wrong",
        parsed_answer="wrong",
        contamination_exposure=ContaminationExposure(
            condition="contaminated",
            is_exposed=True,
            source_entry_ids=["m1"],
            contamination_types=["wrong_solution"],
            memory_before_entry_ids=["m1"],
            retrieved_entry_ids=["m1"],
            exposure_mode="retrieved_memory",
            reason="retrieved contaminated memory source",
        ),
        bad_memory_uptake_label="uptake_detected",
        repeated_failure_label="repeated_failure",
        memory_write_event=_bot_write_event(
            parent_trial_id="run1:game24:s1:no_memory:clean:replay",
            source_entry_ids=["m1"],
            new_entry_id="bot_template:trial1",
        ),
        latency_ms=18,
        token_usage={"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
    )
    _write_trials_jsonl(run_dir, [clean_row, contaminated_row])

    from memcontam.evaluation.aggregate import aggregate_run

    result = aggregate_run(run_dir)
    assert result["run_dir"] == str(run_dir)
    assert result["n_trials"] == 2
    assert result["groups"] == [
        {
            "task_name": "game24",
            "baseline": "no_memory",
            "arm": "clean",
            "backbone": "replay",
            "n_trials": 1,
            "verified_success_count": 1,
            "verified_success_rate": 1.0,
            "contaminated_condition_count": 0,
            "contaminated_condition_rate": 0.0,
            "controlled_exposure_count": 0,
            "controlled_exposure_rate": 0.0,
            "contamination_exposure_rate": 0.0,
            "trial_level_uptake_count": "not_computed",
            "trial_level_uptake_rate": "not_computed",
            "contamination_uptake_rate": "not_computed",
            "contaminated_descendant_count": "not_computed",
            "contaminated_descendant_rate": "not_computed",
            "filter_drop_count": 0,
            "token_usage_total": 5,
            "latency_ms_min": 12,
            "latency_ms_mean": 12.0,
            "latency_ms_max": 12,
            "repeated_failure_count": "not_computed",
            "repeated_failure_rate": "not_computed",
            "method_call_count": "not_computed",
            "method_call_error_count": "not_computed",
            "prompt_token_total": "not_computed",
            "completion_token_total": "not_computed",
            "total_token_total": "not_computed",
            "latency_ms_total": "not_computed",
            "stage_histogram": "not_computed",
            "bot_update_accepted_count": "not_computed",
            "bot_update_rejected_count": "not_computed",
            "bot_update_incomplete_count": "not_computed",
            "bot_update_reused_count": "not_computed",
            "vanilla_to_contamination_degradation_rate": 1.0,
        },
        {
            "task_name": "game24",
            "baseline": "no_memory",
            "arm": "contaminated",
            "backbone": "replay",
            "n_trials": 1,
            "verified_success_count": 0,
            "verified_success_rate": 0.0,
            "contaminated_condition_count": 1,
            "contaminated_condition_rate": 1.0,
            "controlled_exposure_count": 1,
            "controlled_exposure_rate": 1.0,
            "contamination_exposure_rate": 1.0,
            "trial_level_uptake_count": 1,
            "trial_level_uptake_rate": 1.0,
            "contamination_uptake_rate": 1.0,
            "contaminated_descendant_count": 1,
            "contaminated_descendant_rate": 1.0,
            "filter_drop_count": 0,
            "token_usage_total": 7,
            "latency_ms_min": 18,
            "latency_ms_mean": 18.0,
            "latency_ms_max": 18,
            "repeated_failure_count": 1,
            "repeated_failure_rate": 1.0,
            "method_call_count": "not_computed",
            "method_call_error_count": "not_computed",
            "prompt_token_total": "not_computed",
            "completion_token_total": "not_computed",
            "total_token_total": "not_computed",
            "latency_ms_total": "not_computed",
            "stage_histogram": "not_computed",
            "bot_update_accepted_count": "not_computed",
            "bot_update_rejected_count": "not_computed",
            "bot_update_incomplete_count": "not_computed",
            "bot_update_reused_count": "not_computed",
            "vanilla_to_contamination_degradation_rate": 1.0,
        },
    ]

    cli_result = _cli_aggregate(run_dir)
    assert cli_result.returncode == 0
    assert json.loads(cli_result.stdout) == result


def test_aggregate_run_handles_empty_trials_jsonl(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "empty"
    run_dir.mkdir(parents=True)
    (run_dir / "trials.jsonl").write_text("", encoding="utf-8")

    from memcontam.evaluation.aggregate import aggregate_run

    result = aggregate_run(run_dir)
    assert result == {"run_dir": str(run_dir), "n_trials": 0, "groups": []}

    cli_result = _cli_aggregate(run_dir)
    assert cli_result.returncode == 0
    assert json.loads(cli_result.stdout) == result


def test_aggregate_run_sums_numeric_filter_drops(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "filter_drop"
    run_dir.mkdir(parents=True)
    _write_trials_jsonl(
        run_dir,
        [
            _trial_row(
                trial_id="run1:game24:s1:no_memory:contaminated_filter:replay",
                arm="contaminated_filter",
                filter_decision={"filter": "drop_known_contaminated", "dropped": 3},
            )
        ],
    )

    from memcontam.evaluation.aggregate import aggregate_run

    result = aggregate_run(run_dir)
    assert result["groups"][0]["filter_drop_count"] == 3


def test_aggregate_run_keeps_legacy_null_memory_write_event_rows(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "legacy_null_write_event"
    run_dir.mkdir(parents=True)
    _write_trials_jsonl(
        run_dir,
        [
            _trial_row(
                trial_id="run1:game24:s1:no_memory:clean:replay",
                arm="clean",
                memory_write_event=None,
            )
        ],
    )

    from memcontam.evaluation.aggregate import aggregate_run

    result = aggregate_run(run_dir)
    assert result["groups"][0]["contaminated_descendant_count"] == "not_computed"
    assert result["groups"][0]["contaminated_descendant_rate"] == "not_computed"


@pytest.mark.parametrize(
    "memory_write_event",
    [
        _bot_write_event(source_entry_ids=[]),
        _bot_write_event(parent_trial_id=None),
    ],
)
def test_aggregate_run_requires_both_parent_and_source_for_descendants(
    tmp_path, memory_write_event: dict
) -> None:
    run_dir = tmp_path / "runs" / "descendant"
    run_dir.mkdir(parents=True)
    _write_trials_jsonl(
        run_dir,
        [
            _trial_row(
                trial_id="run1:game24:s1:no_memory:contaminated:replay",
                arm="contaminated",
                memory_write_event=memory_write_event,
            )
        ],
    )

    from memcontam.evaluation.aggregate import aggregate_run

    result = aggregate_run(run_dir)
    assert result["groups"][0]["contaminated_descendant_count"] == "not_computed"
    assert result["groups"][0]["contaminated_descendant_rate"] == "not_computed"


def test_aggregate_run_counts_complete_bot_write_lineage(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "bot_lineage_complete"
    run_dir.mkdir(parents=True)
    _write_trials_jsonl(
        run_dir,
        [
            _trial_row(
                trial_id="run1:game24:s1:bot_style:contaminated:replay",
                baseline="bot_style",
                arm="contaminated",
                verifier_result=VerifierResult(is_correct=False, reason="incorrect"),
                memory_write_event=_bot_write_event(),
            )
        ],
    )

    from memcontam.evaluation.aggregate import aggregate_run

    result = aggregate_run(run_dir)
    assert result["groups"][0]["contaminated_descendant_count"] == 1
    assert result["groups"][0]["contaminated_descendant_rate"] == 1.0


@pytest.mark.parametrize(
    "memory_write_event",
    [
        _bot_write_event(parent_trial_id=None),
        _bot_write_event(source_entry_ids=[]),
    ],
)
def test_aggregate_run_ignores_incomplete_bot_write_lineage(
    tmp_path, memory_write_event: dict
) -> None:
    run_dir = tmp_path / "runs" / "bot_lineage_incomplete"
    run_dir.mkdir(parents=True)
    _write_trials_jsonl(
        run_dir,
        [
            _trial_row(
                trial_id="run1:game24:s1:bot_style:contaminated:replay",
                baseline="bot_style",
                arm="contaminated",
                verifier_result=VerifierResult(is_correct=False, reason="incorrect"),
                memory_write_event=memory_write_event,
            )
        ],
    )

    from memcontam.evaluation.aggregate import aggregate_run

    result = aggregate_run(run_dir)
    assert result["groups"][0]["contaminated_descendant_count"] == "not_computed"
    assert result["groups"][0]["contaminated_descendant_rate"] == "not_computed"


@pytest.mark.parametrize(
    ("fixture_name", "write_fixture"),
    [
        ("missing", False),
        ("malformed", True),
        ("invalid", True),
    ],
)
def test_aggregate_run_rejects_bad_trials_jsonl(tmp_path, fixture_name: str, write_fixture: bool) -> None:
    run_dir = tmp_path / "runs" / fixture_name
    run_dir.mkdir(parents=True)
    trials_path = run_dir / "trials.jsonl"
    if write_fixture and fixture_name == "malformed":
        trials_path.write_text("{" + chr(10), encoding="utf-8")
    if write_fixture and fixture_name == "invalid":
        trials_path.write_text(
            json.dumps(
                {
                    "trial_id": "bad",
                    "run_id": "run1",
                    "task_name": "game24",
                    "sample_id": "s1",
                    "baseline": "no_memory",
                    "arm": "clean",
                    "backbone": "replay",
                }
            )
            + chr(10),
            encoding="utf-8",
        )

    from memcontam.evaluation.aggregate import aggregate_run

    with pytest.raises(SystemExit):
        aggregate_run(run_dir)

    cli_result = _cli_aggregate(run_dir)
    assert cli_result.returncode != 0


def test_aggregate_reports_method_call_overhead(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "method_calls"
    run_dir.mkdir(parents=True)

    rag_row = _trial_row(
        trial_id="run1:game24:s1:retrieval_rag:clean:replay",
        baseline="retrieval_rag",
        arm="clean",
        method_calls=[_method_call("rag_generate")],
    )
    bot_row = _trial_row(
        trial_id="run1:game24:s1:bot_style:clean:replay",
        baseline="bot_style",
        arm="clean",
        method_calls=[
            _method_call("bot_problem_distill"),
            _method_call("bot_instantiate_solve"),
            _method_call("bot_thought_distill"),
            _method_call("bot_novelty_decide", token_usage={"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}),
        ],
        memory_write_event=_bot_write_event(status="accepted"),
    )
    _write_trials_jsonl(run_dir, [rag_row, bot_row])

    from memcontam.evaluation.aggregate import aggregate_run

    result = aggregate_run(run_dir)
    groups = {(g["task_name"], g["baseline"], g["arm"], g["backbone"]): g for g in result["groups"]}

    rag_group = groups[("game24", "retrieval_rag", "clean", "replay")]
    assert rag_group["method_call_count"] == 1
    assert rag_group["method_call_error_count"] == 0
    assert rag_group["prompt_token_total"] == 5
    assert rag_group["completion_token_total"] == 5
    assert rag_group["total_token_total"] == 10
    assert rag_group["latency_ms_total"] == 10
    assert rag_group["stage_histogram"] == {"rag_generate": 1}
    assert rag_group["bot_update_accepted_count"] == NOT_COMPUTED

    bot_group = groups[("game24", "bot_style", "clean", "replay")]
    assert bot_group["method_call_count"] == 4
    assert bot_group["method_call_error_count"] == 0
    assert bot_group["prompt_token_total"] == 23
    assert bot_group["completion_token_total"] == 17
    assert bot_group["total_token_total"] == 40
    assert bot_group["latency_ms_total"] == 40
    assert bot_group["stage_histogram"] == {
        "bot_problem_distill": 1,
        "bot_instantiate_solve": 1,
        "bot_thought_distill": 1,
        "bot_novelty_decide": 1,
    }
    assert bot_group["bot_update_accepted_count"] == 1
    assert bot_group["bot_update_rejected_count"] == 0
    assert bot_group["bot_update_incomplete_count"] == 0
    assert bot_group["bot_update_reused_count"] == 0


def test_aggregate_legacy_calls_not_computed(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "legacy_calls"
    run_dir.mkdir(parents=True)
    row = _trial_row(
        trial_id="run1:game24:s1:no_memory:clean:replay",
        baseline="no_memory",
        arm="clean",
    )
    row.pop("method_calls", None)
    _write_trials_jsonl(run_dir, [row])

    from memcontam.evaluation.aggregate import aggregate_run

    result = aggregate_run(run_dir)
    group = result["groups"][0]
    assert group["method_call_count"] == NOT_COMPUTED
    assert group["method_call_error_count"] == NOT_COMPUTED
    assert group["prompt_token_total"] == NOT_COMPUTED
    assert group["completion_token_total"] == NOT_COMPUTED
    assert group["total_token_total"] == NOT_COMPUTED
    assert group["latency_ms_total"] == NOT_COMPUTED
    assert group["stage_histogram"] == NOT_COMPUTED


def test_aggregate_marks_incomplete_bot_lineage(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "incomplete_lineage"
    run_dir.mkdir(parents=True)
    event = _bot_write_event()
    event.pop("status", None)
    _write_trials_jsonl(
        run_dir,
        [
            _trial_row(
                trial_id="run1:game24:s1:bot_style:contaminated:replay",
                baseline="bot_style",
                arm="contaminated",
                memory_write_event=event,
            )
        ],
    )

    from memcontam.evaluation.aggregate import aggregate_run

    result = aggregate_run(run_dir)
    group = result["groups"][0]
    assert group["bot_update_accepted_count"] == NOT_COMPUTED
    assert group["bot_update_rejected_count"] == NOT_COMPUTED
    assert group["bot_update_incomplete_count"] == NOT_COMPUTED
    assert group["bot_update_reused_count"] == NOT_COMPUTED


def test_aggregate_counts_bot_lineage_status_variants(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "lineage_variants"
    run_dir.mkdir(parents=True)
    _write_trials_jsonl(
        run_dir,
        [
            _trial_row(
                trial_id=f"run1:game24:s{i}:bot_style:contaminated:replay",
                baseline="bot_style",
                arm="contaminated",
                memory_write_event=_bot_write_event(status=status),
            )
            for i, status in enumerate(["accepted", "rejected", "reused", "incomplete"])
        ],
    )

    from memcontam.evaluation.aggregate import aggregate_run

    result = aggregate_run(run_dir)
    group = result["groups"][0]
    assert group["bot_update_accepted_count"] == 1
    assert group["bot_update_rejected_count"] == 1
    assert group["bot_update_reused_count"] == 1
    assert group["bot_update_incomplete_count"] == 1
