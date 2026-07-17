from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from memcontam.logging.schema import (
    ContaminationExposure,
    LOGGING_V1,
    TrialLog,
    VerifierResult,
)


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


def _trial_with_method_calls(**overrides):
    base = _trial_row(**overrides)
    return TrialLog.model_validate(base)


def _write_trials_jsonl(run_dir: Path, rows: list[dict]) -> Path:
    trials_path = run_dir / "trials.jsonl"
    trials_path.write_text("".join(json.dumps(row) + chr(10) for row in rows), encoding="utf-8")
    return trials_path


def _cli_aggregate(run_dir: Path, stage: str | None = None, allow_legacy: bool = False) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    cmd = [sys.executable, "-m", "memcontam.cli", "aggregate", str(run_dir)]
    if stage is not None:
        cmd.extend(["--stage", stage])
    if allow_legacy:
        cmd.append("--allow-legacy")
    return subprocess.run(
        cmd,
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
        contamination_exposure=ContaminationExposure.model_validate(
            {
                "condition": "contaminated",
                "is_exposed": True,
                "source_entry_ids": ["m1"],
                "contamination_types": ["wrong_solution"],
                "memory_before_entry_ids": ["m1"],
                "retrieved_entry_ids": ["m1"],
                "exposure_mode": "retrieved_memory",
                "reason": "retrieved contaminated memory source",
            }
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

    result = aggregate_run(run_dir, allow_legacy=True)
    assert result["run_dir"] == str(run_dir)
    assert result["n_trials"] == 2
    assert result["groups"] == [
        {
            "task_name": "game24",
            "baseline": "no_memory",
            "arm": "clean",
            "backbone": "replay",
            "n_trials": 1,
            "n_failed": 0,
            "n_evaluable": 1,
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
            "failure_origin_histogram": {},
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
            "n_failed": 0,
            "n_evaluable": 1,
            "verified_success_count": 0,
            "verified_success_rate": 0.0,
            "contaminated_condition_count": 1,
            "contaminated_condition_rate": 1.0,
            "controlled_exposure_count": 0,
            "controlled_exposure_rate": 0.0,
            "contamination_exposure_rate": 0.0,
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
            "failure_origin_histogram": {},
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

    cli_result = _cli_aggregate(run_dir, allow_legacy=True)
    assert cli_result.returncode == 0
    assert json.loads(cli_result.stdout) == result


def test_method_call_metrics_cover_native_full_history_reflexion_and_dc() -> None:
    from memcontam.evaluation.aggregate import _method_call_metrics

    trials = [
        _trial_with_method_calls(
            trial_id="run1:game24:s1:reflexion_style:clean:replay",
            baseline="reflexion_style",
            arm="clean",
            method_calls=[
                _method_call(
                    "reflexion_generate",
                    latency_ms=11,
                    token_usage={"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
                )
            ],
        ),
        _trial_with_method_calls(
            trial_id="run1:game24:s2:reflexion_style:contaminated:replay",
            sample_id="s2",
            baseline="reflexion_style",
            arm="contaminated",
            verifier_result=VerifierResult(is_correct=False, reason="incorrect"),
            raw_response="final: wrong",
            parsed_answer="wrong",
            repeated_failure_label="repeated_failure",
            method_calls=[
                _method_call(
                    "reflexion_generate",
                    latency_ms=12,
                    token_usage={"prompt_tokens": 7, "completion_tokens": 8, "total_tokens": 15},
                ),
                _method_call(
                    "reflexion_reflect",
                    latency_ms=13,
                    token_usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
                    error_type="RuntimeError",
                ),
            ],
        ),
        _trial_with_method_calls(
            trial_id="run1:game24:s3:full_history:clean:replay",
            sample_id="s3",
            baseline="full_history",
            method_calls=[
                _method_call(
                    "full_history_generate",
                    latency_ms=14,
                    token_usage={"prompt_tokens": 2, "completion_tokens": 9, "total_tokens": 11},
                )
            ],
        ),
        _trial_with_method_calls(
            trial_id="run1:game24:s4:dynamic_cheatsheet_optional:clean:replay",
            sample_id="s4",
            baseline="dynamic_cheatsheet_optional",
            method_calls=[
                _method_call(
                    "dynamic_cheatsheet_generate",
                    latency_ms=15,
                    token_usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                ),
                _method_call(
                    "dynamic_cheatsheet_curate",
                    latency_ms=16,
                    token_usage={"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
                ),
            ],
        ),
    ]

    metrics = _method_call_metrics(trials)

    assert metrics["method_call_count"] == 6
    assert metrics["method_call_error_count"] == 1
    assert metrics["prompt_token_total"] == 22
    assert metrics["completion_token_total"] == 30
    assert metrics["total_token_total"] == 52
    assert metrics["latency_ms_total"] == 81
    assert metrics["stage_histogram"] == {
        "full_history_generate": 1,
        "reflexion_generate": 2,
        "reflexion_reflect": 1,
        "dynamic_cheatsheet_generate": 1,
        "dynamic_cheatsheet_curate": 1,
    }


def test_aggregate_run_handles_empty_trials_jsonl(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "empty"
    run_dir.mkdir(parents=True)
    (run_dir / "trials.jsonl").write_text("", encoding="utf-8")

    from memcontam.evaluation.aggregate import aggregate_run

    result = aggregate_run(run_dir, allow_legacy=True)
    assert result == {"run_dir": str(run_dir), "status": "legacy", "n_trials": 0, "groups": []}

    cli_result = _cli_aggregate(run_dir, allow_legacy=True)
    assert cli_result.returncode == 0
    assert json.loads(cli_result.stdout) == result


def test_aggregate_run_excludes_warmup_rows(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "warmup_excluded"
    run_dir.mkdir(parents=True)
    _write_trials_jsonl(
        run_dir,
        [
            _trial_row(
                trial_id="run1:game24:warmup-1:no_memory:clean:replay",
                sample_id="warmup-1",
                metadata={"phase": "warmup", "exclude_from_aggregate": True},
            ),
            _trial_row(trial_id="run1:game24:s1:no_memory:clean:replay", sample_id="s1"),
        ],
    )

    from memcontam.evaluation.aggregate import aggregate_run

    result = aggregate_run(run_dir, allow_legacy=True)

    assert result["n_trials"] == 2
    assert result["groups"][0]["n_trials"] == 1


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

    result = aggregate_run(run_dir, allow_legacy=True)
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

    result = aggregate_run(run_dir, allow_legacy=True)
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

    result = aggregate_run(run_dir, allow_legacy=True)
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

    result = aggregate_run(run_dir, allow_legacy=True)
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

    result = aggregate_run(run_dir, allow_legacy=True)
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
        aggregate_run(run_dir, allow_legacy=True)

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

    result = aggregate_run(run_dir, allow_legacy=True)
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

    result = aggregate_run(run_dir, allow_legacy=True)
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

    result = aggregate_run(run_dir, allow_legacy=True)
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

    result = aggregate_run(run_dir, allow_legacy=True)
    group = result["groups"][0]
    assert group["bot_update_accepted_count"] == 1
    assert group["bot_update_rejected_count"] == 1
    assert group["bot_update_reused_count"] == 1
    assert group["bot_update_incomplete_count"] == 1



def _strict_run_metadata(run_id: str = "run1", stage: str = "replay") -> dict[str, Any]:
    return {
        "run_metadata_id": f"{run_id}:metadata",
        "run_id": run_id,
        "git_commit": "abc123",
        "config_hash": "deadbeef",
        "provider": "replay",
        "model_snapshots": {"replay": "replay"},
        "query_date": "2026-07-16",
        "start_date": "2026-07-16",
        "seed": 0,
        "order": "task-sample-baseline-arm-model",
        "decoding_defaults": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 100},
        "sample_set_hash": "set",
        "sample_order_hash": "order",
        "stage": stage,
        "schema_version": LOGGING_V1,
        "prompt_version": "v1",
        "memory_policy_version": "v1",
        "contamination_catalog_version": "v1",
        "retry_policy_version": "v1",
    }


def _strict_manifest(run_id: str = "run1", stage: str = "replay", status: str = "completed") -> dict[str, Any]:
    return {
        "run_metadata": _strict_run_metadata(run_id, stage),
        "status": status,
        "started_at": "2026-07-16T00:00:00Z",
        "ended_at": "2026-07-16T00:01:00Z",
        "counts": {"trials": 0, "calls": 0, "failures": 0, "filter_events": 0, "memory_events": 0},
    }


def _write_run_json(run_dir: Path, manifest: dict[str, Any]) -> None:
    (run_dir / "run.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_jsonl(run_dir: Path, filename: str, rows: list[dict]) -> None:
    (run_dir / filename).write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _strict_call_event(
    trial_id: str,
    call_id: str,
    trial_seq: int,
    event_seq: int,
    stage: str = "replay",
    method_stage: str = "no_memory_generate",
    token_usage: dict[str, int] | None = None,
    latency_ms: int = 10,
    error_type: str | None = None,
    source_spans: list[dict] | None = None,
    run_id: str = "run1",
) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "run_metadata_id": f"{run_id}:metadata",
        "run_id": run_id,
        "trial_id": trial_id,
        "trial_seq": trial_seq,
        "event_seq": event_seq,
        "stage": stage,
        "method_stage": method_stage,
        "messages": [{"role": "user", "content": f"{method_stage} prompt"}],
        "model": "replay",
        "decoding_params": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 100},
        "response_text": f"{method_stage} response",
        "token_usage": token_usage or {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        "latency_ms": latency_ms,
        "retry_count": 0,
        "source_spans": source_spans or [],
        "created_at": "2026-07-16T00:00:00Z",
        "error_type": error_type,
    }


def _strict_failure_event(
    trial_id: str,
    failure_id: str,
    trial_seq: int,
    event_seq: int,
    stage: str = "replay",
    origin: str = "runner",
    error_type: str = "RuntimeError",
    run_id: str = "run1",
) -> dict[str, Any]:
    return {
        "failure_id": failure_id,
        "run_metadata_id": f"{run_id}:metadata",
        "run_id": run_id,
        "trial_id": trial_id,
        "trial_seq": trial_seq,
        "event_seq": event_seq,
        "stage": stage,
        "origin": origin,
        "error_type": error_type,
        "failure_function": None,
        "failure_module": None,
        "failure_line": None,
        "retry_count": 0,
        "disposition": "continued",
        "created_at": "2026-07-16T00:00:00Z",
    }


def _strict_filter_event(
    trial_id: str,
    filter_id: str,
    trial_seq: int,
    event_seq: int,
    stage: str = "replay",
    baseline: str = "no_memory",
    arm: str = "clean",
    action: str = "apply",
    run_id: str = "run1",
) -> dict[str, Any]:
    return {
        "filter_id": filter_id,
        "run_metadata_id": f"{run_id}:metadata",
        "run_id": run_id,
        "trial_id": trial_id,
        "trial_seq": trial_seq,
        "event_seq": event_seq,
        "stage": stage,
        "arm": arm,
        "baseline": baseline,
        "decisions": [],
        "kept_source_ids": [],
        "removed_source_ids": [],
        "pre_source_ids": [],
        "post_source_ids": [],
        "ground_truth_contaminated_ids": [],
        "action": action,
        "final_answer_source_ids": [],
        "verdict": None,
        "created_at": "2026-07-16T00:00:00Z",
    }


def _strict_memory_event(
    trial_id: str,
    memory_id: str,
    trial_seq: int,
    event_seq: int,
    stage: str = "replay",
    baseline: str = "no_memory",
    run_id: str = "run1",
) -> dict[str, Any]:
    return {
        "memory_id": memory_id,
        "run_metadata_id": f"{run_id}:metadata",
        "run_id": run_id,
        "trial_id": trial_id,
        "trial_seq": trial_seq,
        "event_seq": event_seq,
        "stage": stage,
        "event_type": "memory_snapshot",
        "operation": "snapshot",
        "baseline": baseline,
        "source_trial_id": None,
        "parent_entry_ids": [],
        "source_entry_ids": [],
        "contaminated_source_ids": [],
        "before_entry_ids": [],
        "after_entry_ids": [],
        "before_snapshot_hash": None,
        "after_snapshot_hash": None,
        "new_entry_ids": [],
        "updated_entry_ids": [],
        "removed_entry_ids": [],
        "creation_origin": None,
        "memory_version": None,
        "status": "ok",
        "created_at": "2026-07-16T00:00:00Z",
    }


def _strict_trial_row(
    run_id: str = "run1",
    trial_seq: int = 0,
    event_seq: int = 1,
    stage: str = "replay",
    status: str = "succeeded",
    arm: str = "clean",
    baseline: str = "no_memory",
    backbone: str = "replay",
    trial_id: str | None = None,
    sample_id: str = "s1",
    answer_call_id: str | None = None,
    method_calls: list[dict] | None = None,
    contamination_exposure: dict[str, Any] | None = None,
    failure_id: str | None = None,
    **overrides,
) -> dict[str, Any]:
    trial_id = trial_id or f"{run_id}:game24:{sample_id}:{baseline}:{arm}:{backbone}"
    answer_call_id = answer_call_id or f"{trial_id}:call:1"
    method_calls = method_calls or [
        {
            "call_id": answer_call_id,
            "stage": "no_memory_generate",
            "messages": [{"role": "user", "content": "solve"}],
            "raw_response": "final: 24",
            "model": "replay",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 100,
            "latency_ms": 10,
            "token_usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            "retry_count": 0,
            "error_type": None,
            "source_spans": [],
        }
    ]
    if arm == "clean":
        exposure = contamination_exposure or {
            "condition": "clean",
            "status": "not_applicable",
            "is_exposed": None,
            "answer_call_id": None,
            "target_entry_ids": [],
            "source_entry_ids": [],
            "exposed_source_ids": [],
            "exposure_mode": "clean",
            "reason": "clean arm",
        }
    else:
        exposure = contamination_exposure or {
            "condition": arm,
            "status": "supported",
            "is_exposed": True,
            "answer_call_id": answer_call_id,
            "target_entry_ids": [],
            "source_entry_ids": ["m1"],
            "exposed_source_ids": ["m1"],
            "exposure_mode": "final_prompt",
            "reason": "supported exposure",
        }
    if status == "failed" and failure_id is None:
        failure_id = f"{trial_id}:failure:1"
    base = {
        "trial_id": trial_id,
        "run_id": run_id,
        "task_name": "game24",
        "sample_id": sample_id,
        "baseline": baseline,
        "arm": arm,
        "backbone": backbone,
        "input": {"numbers": [1, 3, 4, 6], "target": 24},
        "gold_or_verifier_spec": {"target": 24},
        "prompt_messages": [{"role": "user", "content": "solve"}],
        "raw_response": "final: 24",
        "parsed_answer": "24",
        "verifier_result": VerifierResult(is_correct=True),
        "metadata": {},
        "memory_before": [],
        "retrieved_memory": [],
        "retrieved_scores": [],
        "filter_decision": None,
        "memory_write_event": None,
        "memory_after": [],
        "method_calls": method_calls,
        "contamination_exposure": exposure,
        "bad_memory_uptake_label": "not_applicable",
        "repeated_failure_label": "not_applicable",
        "recovery_after_filter_label": "not_applicable",
        "latency_ms": 10,
        "token_usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        "retry_count": 0,
        "schema_version": LOGGING_V1,
        "stage": stage,
        "status": status,
        "run_metadata_id": f"{run_id}:metadata",
        "trial_seq": trial_seq,
        "event_seq": event_seq,
        "answer_call_id": answer_call_id,
        "failure_id": failure_id,
    }
    base.update(overrides)
    return TrialLog(**base).model_dump(mode="json")



def test_strict_aggregate_default_requires_stage_and_rejects_legacy(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "strict_no_stage"
    run_dir.mkdir(parents=True)
    _write_run_json(run_dir, _strict_manifest())
    _write_jsonl(run_dir, "trials.jsonl", [_strict_trial_row(trial_seq=0, event_seq=1)])
    _write_jsonl(run_dir, "calls.jsonl", [_strict_call_event(
        "run1:game24:s1:no_memory:clean:replay", "run1:game24:s1:no_memory:clean:replay:call:1", 0, 2
    )])
    _write_jsonl(run_dir, "failures.jsonl", [])
    _write_jsonl(run_dir, "filter_events.jsonl", [])
    _write_jsonl(run_dir, "memory_events.jsonl", [])

    from memcontam.evaluation.aggregate import aggregate_run

    with pytest.raises(SystemExit):
        aggregate_run(run_dir)

    with pytest.raises(SystemExit):
        aggregate_run(run_dir, allow_legacy=True)

    result = aggregate_run(run_dir, stage="replay")
    assert result["status"] == "completed"
    assert result["n_trials"] == 1


@pytest.mark.parametrize("stage", ["replay", "partial", "pilot", "main", "benchmark"])
def test_strict_aggregate_stage_mismatch_fails(stage: str, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / f"stage_mismatch_{stage}"
    run_dir.mkdir(parents=True)
    _write_run_json(run_dir, _strict_manifest(stage=stage))
    _write_jsonl(run_dir, "trials.jsonl", [_strict_trial_row(trial_seq=0, event_seq=1, stage=stage)])
    _write_jsonl(run_dir, "calls.jsonl", [_strict_call_event(
        "run1:game24:s1:no_memory:clean:replay", "run1:game24:s1:no_memory:clean:replay:call:1", 0, 2, stage=stage
    )])
    _write_jsonl(run_dir, "failures.jsonl", [])
    _write_jsonl(run_dir, "filter_events.jsonl", [])
    _write_jsonl(run_dir, "memory_events.jsonl", [])

    from memcontam.evaluation.aggregate import aggregate_run

    with pytest.raises(SystemExit):
        aggregate_run(run_dir, stage="debug")

    result = aggregate_run(run_dir, stage=stage)
    assert result["status"] == "completed"


def test_strict_aggregate_uses_calls_jsonl_for_telemetry(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "strict_calls_telemetry"
    run_dir.mkdir(parents=True)
    _write_run_json(run_dir, _strict_manifest())
    trial_id = "run1:game24:s1:no_memory:clean:replay"
    call_id = f"{trial_id}:call:1"
    _write_jsonl(run_dir, "trials.jsonl", [
        _strict_trial_row(trial_seq=0, event_seq=1, method_calls=[
            {
                "call_id": call_id,
                "stage": "no_memory_generate",
                "messages": [{"role": "user", "content": "solve"}],
                "raw_response": "final: 24",
                "model": "replay",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 100,
                "latency_ms": 10,
                "token_usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                "retry_count": 0,
                "error_type": None,
                "source_spans": [],
            }
        ])
    ])
    _write_jsonl(run_dir, "calls.jsonl", [
        _strict_call_event(trial_id, call_id, 0, 2, token_usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10})
    ])
    _write_jsonl(run_dir, "failures.jsonl", [])
    _write_jsonl(run_dir, "filter_events.jsonl", [])
    _write_jsonl(run_dir, "memory_events.jsonl", [])

    from memcontam.evaluation.aggregate import aggregate_run

    result = aggregate_run(run_dir, stage="replay")
    group = result["groups"][0]
    assert group["prompt_token_total"] == 7
    assert group["completion_token_total"] == 3
    assert group["method_call_count"] == 1


def test_strict_aggregate_fails_when_nested_calls_disagree_with_calls_jsonl(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "strict_calls_disagree"
    run_dir.mkdir(parents=True)
    _write_run_json(run_dir, _strict_manifest())
    trial_id = "run1:game24:s1:no_memory:clean:replay"
    call_id = f"{trial_id}:call:1"
    _write_jsonl(run_dir, "trials.jsonl", [
        _strict_trial_row(trial_seq=0, event_seq=1, method_calls=[
            {
                "call_id": call_id,
                "stage": "no_memory_generate",
                "messages": [{"role": "user", "content": "solve"}],
                "raw_response": "final: 24",
                "model": "replay",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 100,
                "latency_ms": 10,
                "token_usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                "retry_count": 0,
                "error_type": None,
                "source_spans": [],
            },
            {
                "call_id": f"{trial_id}:call:2",
                "stage": "no_memory_generate",
                "messages": [{"role": "user", "content": "solve again"}],
                "raw_response": "final: 24",
                "model": "replay",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 100,
                "latency_ms": 10,
                "token_usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                "retry_count": 0,
                "error_type": None,
                "source_spans": [],
            }
        ])
    ])
    _write_jsonl(run_dir, "calls.jsonl", [
        _strict_call_event(trial_id, call_id, 0, 2)
    ])
    _write_jsonl(run_dir, "failures.jsonl", [])
    _write_jsonl(run_dir, "filter_events.jsonl", [])
    _write_jsonl(run_dir, "memory_events.jsonl", [])

    from memcontam.evaluation.aggregate import aggregate_run

    with pytest.raises(SystemExit):
        aggregate_run(run_dir, stage="replay")


def test_strict_aggregate_rejects_missing_answer_call(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "strict_missing_answer_call"
    run_dir.mkdir(parents=True)
    _write_run_json(run_dir, _strict_manifest())
    trial_id = "run1:game24:s1:no_memory:clean:replay"
    _write_jsonl(run_dir, "trials.jsonl", [
        _strict_trial_row(trial_seq=0, event_seq=1, answer_call_id=f"{trial_id}:call:missing")
    ])
    _write_jsonl(run_dir, "calls.jsonl", [
        _strict_call_event(trial_id, f"{trial_id}:call:1", 0, 2)
    ])
    _write_jsonl(run_dir, "failures.jsonl", [])
    _write_jsonl(run_dir, "filter_events.jsonl", [])
    _write_jsonl(run_dir, "memory_events.jsonl", [])

    from memcontam.evaluation.aggregate import aggregate_run

    with pytest.raises(SystemExit):
        aggregate_run(run_dir, stage="replay")


def test_strict_aggregate_rejects_duplicate_event_seq(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "strict_duplicate_event_seq"
    run_dir.mkdir(parents=True)
    _write_run_json(run_dir, _strict_manifest())
    trial_id = "run1:game24:s1:no_memory:clean:replay"
    _write_jsonl(run_dir, "trials.jsonl", [
        _strict_trial_row(trial_seq=0, event_seq=1)
    ])
    _write_jsonl(run_dir, "calls.jsonl", [
        _strict_call_event(trial_id, f"{trial_id}:call:1", 0, 1)
    ])
    _write_jsonl(run_dir, "failures.jsonl", [])
    _write_jsonl(run_dir, "filter_events.jsonl", [])
    _write_jsonl(run_dir, "memory_events.jsonl", [])

    from memcontam.evaluation.aggregate import aggregate_run

    with pytest.raises(SystemExit):
        aggregate_run(run_dir, stage="replay")


def test_strict_aggregate_rejects_mixed_stage(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "strict_mixed_stage"
    run_dir.mkdir(parents=True)
    _write_run_json(run_dir, _strict_manifest(stage="replay"))
    trial_id = "run1:game24:s1:no_memory:clean:replay"
    _write_jsonl(run_dir, "trials.jsonl", [
        _strict_trial_row(trial_seq=0, event_seq=1, stage="pilot")
    ])
    _write_jsonl(run_dir, "calls.jsonl", [
        _strict_call_event(trial_id, f"{trial_id}:call:1", 0, 2, stage="pilot")
    ])
    _write_jsonl(run_dir, "failures.jsonl", [])
    _write_jsonl(run_dir, "filter_events.jsonl", [])
    _write_jsonl(run_dir, "memory_events.jsonl", [])

    from memcontam.evaluation.aggregate import aggregate_run

    with pytest.raises(SystemExit):
        aggregate_run(run_dir, stage="replay")


def test_strict_aggregate_rejects_unsupported_exposure_on_main(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "strict_main_unsupported_exposure"
    run_dir.mkdir(parents=True)
    _write_run_json(run_dir, _strict_manifest(stage="main"))
    trial_id = "run1:game24:s1:no_memory:contaminated:replay"
    call_id = f"{trial_id}:call:1"
    _write_jsonl(run_dir, "trials.jsonl", [
        _strict_trial_row(
            trial_seq=0,
            event_seq=1,
            stage="main",
            arm="contaminated",
            trial_id=trial_id,
            answer_call_id=call_id,
            contamination_exposure={
                "condition": "contaminated",
                "status": "not_evaluable",
                "is_exposed": None,
                "answer_call_id": None,
                "target_entry_ids": [],
                "source_entry_ids": ["m1"],
                "exposed_source_ids": [],
                "exposure_mode": "not_evaluable",
                "reason": "legacy proxy",
            },
        )
    ])
    _write_jsonl(run_dir, "calls.jsonl", [
        _strict_call_event(trial_id, call_id, 0, 2, stage="main")
    ])
    _write_jsonl(run_dir, "failures.jsonl", [])
    _write_jsonl(run_dir, "filter_events.jsonl", [])
    _write_jsonl(run_dir, "memory_events.jsonl", [])

    from memcontam.evaluation.aggregate import aggregate_run

    with pytest.raises(SystemExit):
        aggregate_run(run_dir, stage="main")


def test_strict_aggregate_excludes_failed_trials_from_accuracy(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "strict_failed_excluded"
    run_dir.mkdir(parents=True)
    _write_run_json(run_dir, _strict_manifest())
    success_trial_id = "run1:game24:s1:no_memory:clean:replay"
    failed_trial_id = "run1:game24:s2:no_memory:clean:replay"
    _write_jsonl(run_dir, "trials.jsonl", [
        _strict_trial_row(trial_seq=0, event_seq=1, trial_id=success_trial_id, sample_id="s1"),
        _strict_trial_row(
            trial_seq=1,
            event_seq=2,
            trial_id=failed_trial_id,
            sample_id="s2",
            status="failed",
            verifier_result=None,
            raw_response=None,
            parsed_answer=None,
        ),
    ])
    _write_jsonl(run_dir, "calls.jsonl", [
        _strict_call_event(success_trial_id, f"{success_trial_id}:call:1", 0, 3),
        _strict_call_event(failed_trial_id, f"{failed_trial_id}:call:1", 1, 4),
    ])
    _write_jsonl(run_dir, "failures.jsonl", [
        _strict_failure_event(failed_trial_id, f"{failed_trial_id}:failure:1", 1, 5, origin="verifier")
    ])
    _write_jsonl(run_dir, "filter_events.jsonl", [])
    _write_jsonl(run_dir, "memory_events.jsonl", [])

    from memcontam.evaluation.aggregate import aggregate_run

    result = aggregate_run(run_dir, stage="replay")
    group = result["groups"][0]
    assert group["n_trials"] == 2
    assert group["n_failed"] == 1
    assert group["n_evaluable"] == 1
    assert group["verified_success_count"] == 1
    assert group["verified_success_rate"] == 1.0
    assert group["failure_origin_histogram"] == {"verifier": 1}


def test_strict_aggregate_rejects_missing_filter_outcome(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "strict_missing_filter_outcome"
    run_dir.mkdir(parents=True)
    _write_run_json(run_dir, _strict_manifest())
    trial_id = "run1:game24:s1:no_memory:contaminated_filter:replay"
    call_id = f"{trial_id}:call:1"
    source_spans = [
        {
            "message_index": 0,
            "start": 0,
            "end": 5,
            "rendered_hash": "sha256:prompt",
            "entry_id": "m1",
            "source_ids": ["m1"],
            "parent_ids": [],
            "lineage_id": "lineage-1",
            "version": "v1",
            "origin": "memory_catalog",
            "clean_or_contaminated": "contaminated",
        }
    ]
    _write_jsonl(run_dir, "trials.jsonl", [
        _strict_trial_row(
            trial_seq=0,
            event_seq=3,
            trial_id=trial_id,
            arm="contaminated_filter",
            answer_call_id=call_id,
            filter_decision={"removed_count": 1, "dropped": 1},
            method_calls=[
                {
                    "call_id": call_id,
                    "stage": "no_memory_generate",
                    "messages": [{"role": "user", "content": "solve"}],
                    "raw_response": "final: 24",
                    "model": "replay",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 100,
                    "latency_ms": 10,
                    "token_usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                    "retry_count": 0,
                    "error_type": None,
                    "source_spans": source_spans,
                }
            ],
        )
    ])
    _write_jsonl(run_dir, "calls.jsonl", [
        _strict_call_event(trial_id, call_id, 0, 2, source_spans=source_spans)
    ])
    _write_jsonl(run_dir, "failures.jsonl", [])
    _write_jsonl(run_dir, "filter_events.jsonl", [
        _strict_filter_event(trial_id, f"{trial_id}:filter:1", 0, 1, arm="contaminated_filter")
    ])
    _write_jsonl(run_dir, "memory_events.jsonl", [])

    from memcontam.evaluation.aggregate import aggregate_run

    with pytest.raises(SystemExit, match=trial_id):
        aggregate_run(run_dir, stage="replay")


def test_strict_aggregate_rejects_missing_memory_event(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "strict_missing_memory_event"
    run_dir.mkdir(parents=True)
    _write_run_json(run_dir, _strict_manifest())
    trial_id = "run1:game24:s1:bot_style:clean:replay"
    call_id = f"{trial_id}:call:1"
    _write_jsonl(run_dir, "trials.jsonl", [
        _strict_trial_row(
            trial_seq=0,
            event_seq=2,
            trial_id=trial_id,
            baseline="bot_style",
            answer_call_id=call_id,
            memory_write_event=_bot_write_event(status="accepted", event_type="bot_write"),
        )
    ])
    _write_jsonl(run_dir, "calls.jsonl", [
        _strict_call_event(trial_id, call_id, 0, 1, method_stage="bot_instantiate_solve")
    ])
    _write_jsonl(run_dir, "failures.jsonl", [])
    _write_jsonl(run_dir, "filter_events.jsonl", [])
    _write_jsonl(run_dir, "memory_events.jsonl", [])

    from memcontam.evaluation.aggregate import aggregate_run

    with pytest.raises(SystemExit, match=trial_id):
        aggregate_run(run_dir, stage="replay")


def test_strict_aggregate_scopes_calls_jsonl_metrics_to_group(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "strict_group_call_scope"
    run_dir.mkdir(parents=True)
    _write_run_json(run_dir, _strict_manifest())
    first_trial_id = "run1:game24:s1:no_memory:clean:replay"
    second_trial_id = "run1:game24:s1:full_history:clean:replay"
    _write_jsonl(run_dir, "trials.jsonl", [
        _strict_trial_row(trial_seq=0, event_seq=1, trial_id=first_trial_id),
        _strict_trial_row(
            trial_seq=1,
            event_seq=2,
            trial_id=second_trial_id,
            baseline="full_history",
            answer_call_id=f"{second_trial_id}:call:1",
        ),
    ])
    _write_jsonl(run_dir, "calls.jsonl", [
        _strict_call_event(first_trial_id, f"{first_trial_id}:call:1", 0, 3),
        _strict_call_event(
            second_trial_id,
            f"{second_trial_id}:call:1",
            1,
            4,
            method_stage="full_history_generate",
        ),
    ])
    _write_jsonl(run_dir, "failures.jsonl", [])
    _write_jsonl(run_dir, "filter_events.jsonl", [])
    _write_jsonl(run_dir, "memory_events.jsonl", [])

    from memcontam.evaluation.aggregate import aggregate_run

    result = aggregate_run(run_dir, stage="replay")
    groups = {(g["baseline"], g["arm"]): g for g in result["groups"]}
    assert groups[("no_memory", "clean")]["method_call_count"] == 1
    assert groups[("full_history", "clean")]["method_call_count"] == 1


def test_legacy_aggregate_requires_allow_legacy_flag(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "legacy_flag"
    run_dir.mkdir(parents=True)
    _write_trials_jsonl(run_dir, [_trial_row()])

    from memcontam.evaluation.aggregate import aggregate_run

    with pytest.raises(SystemExit):
        aggregate_run(run_dir)

    with pytest.raises(SystemExit):
        aggregate_run(run_dir, stage="replay")

    result = aggregate_run(run_dir, allow_legacy=True)
    assert result["status"] == "legacy"
    assert result["n_trials"] == 1

    cli_result = _cli_aggregate(run_dir)
    assert cli_result.returncode != 0

    cli_result = _cli_aggregate(run_dir, allow_legacy=True)
    assert cli_result.returncode == 0
    assert json.loads(cli_result.stdout)["status"] == "legacy"
