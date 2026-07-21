from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, cast

from memcontam.evaluation.aggregate import aggregate_run
from memcontam.logging.schema import TrialLog, VerifierResult


REPO_ROOT = Path(__file__).resolve().parents[1]
INSPECTOR = REPO_ROOT / "scripts" / "inspect_g0_dc_rs_reflexion_fidelity.py"
CONFIG = REPO_ROOT / "configs" / "g0_dc_rs_reflexion_fidelity_followup_replay.yaml"
RUN_ID = "synthetic_g0_dc_rs_reflexion"
TASKS = {
    "game24": ["game24_pilot_001", "game24_pilot_002", "game24_pilot_003"],
    "math_equation_balancer": ["meb_pilot_001", "meb_pilot_002", "meb_pilot_003"],
    "word_sorting": ["word_sorting_pilot_001", "word_sorting_pilot_002", "word_sorting_pilot_003"],
}
ARMS = ["clean", "contaminated", "contaminated_filter"]
MODELS = ["gpt4o", "frontier_reasoning"]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return env


def _call(stage: str, model: str, content: str, *, records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "stage": stage,
        "messages": [{"role": "user", "content": content}],
        "raw_response": content,
        "model": model,
        "latency_ms": 2,
        "token_usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "retry_count": 1 if stage == "reflexion_generate" and "Retry" in content else 0,
        "error_type": None,
        "retrieved_records": records or [],
    }


def _exposure(arm: str, memory: list[dict[str, Any]]) -> dict[str, Any]:
    contaminated = [entry["entry_id"] for entry in memory if entry["clean_or_contaminated"] == "contaminated"]
    return {
        "condition": arm,
        "is_exposed": bool(contaminated),
        "source_entry_ids": contaminated,
        "contamination_types": ["dc_rs_io_pair"] if contaminated else [],
        "memory_before_entry_ids": [entry["entry_id"] for entry in memory],
        "retrieved_entry_ids": [],
        "exposure_mode": "memory_before" if contaminated else "none",
        "reason": "synthetic controlled exposure",
    }


def _record(entry_id: str, rank: int, text: str) -> dict[str, Any]:
    return {
        "document_id": entry_id,
        "rank": rank,
        "score": float(10 - rank),
        "text": text,
        "title_or_type": "dc_rs_io_pair",
        "clean_or_contaminated": "clean",
        "source": "pilot_warmup_dc_rs",
        "corpus_hash": "synthetic",
        "embedding_model_id": "fake",
        "embedding_revision": "synthetic",
        "embedding_library_version": "synthetic",
    }


def _trial_id(task: str, sample: str, baseline: str, arm: str, model: str) -> str:
    return ":".join([RUN_ID, task, sample, baseline, arm, model])


def _dc_rs_trial(
    task: str,
    sample: str,
    arm: str,
    model: str,
    memory: list[dict[str, Any]],
    trial_order: int,
) -> dict[str, Any]:
    baseline = "dynamic_cheatsheet_rs_optional"
    trial_id = _trial_id(task, sample, baseline, arm, model)
    records = [
        _record(entry["entry_id"], rank, entry["content"])
        for rank, entry in enumerate(memory[-3:], start=1)
    ]
    pair = {
        "entry_id": f"dc_rs_pair:{trial_id}",
        "content": f"input for {sample}",
        "memory_type": "dc_rs_io_pair",
        "clean_or_contaminated": "clean",
        "source_trial_id": trial_id,
        "metadata": {"output_text": "answer"},
    }
    calls = [
        _call("dc_rs_synthesize", model, "<cheatsheet>synthetic strategy</cheatsheet>", records=records),
        _call("dc_rs_generate", model, "final: answer"),
    ]
    return TrialLog(
        trial_id=trial_id,
        run_id=RUN_ID,
        task_name=task,
        sample_id=sample,
        baseline=baseline,
        arm=cast(Literal["clean", "contaminated", "contaminated_filter"], arm),
        backbone=model,
        input={"sample": sample},
        gold_or_verifier_spec={},
        prompt_messages=[message for call in calls for message in call["messages"]],
        memory_before=memory,
        retrieved_memory=[{**record, "entry_id": record["document_id"]} for record in records],
        retrieved_scores=[record["score"] for record in records],
        filter_decision={"dropped": 1} if arm == "contaminated_filter" else None,
        raw_response="final: answer",
        parsed_answer="answer",
        verifier_result=VerifierResult(is_correct=True, parsed_answer="answer", reason="ok"),
        metadata={"trial_order": trial_order},
        memory_write_event={
            "type": "dynamic_cheatsheet_rs_update",
            "status": "accepted",
            "source_trial_id": trial_id,
            "synthesis_update": {"status": "replaced", "parser_status": "accepted"},
            "pair_appended": {"entry_id": pair["entry_id"], "source_trial_id": trial_id},
        },
        memory_after=[*memory, pair],
        method_calls=cast(Any, calls),
        contamination_exposure=cast(Any, _exposure(arm, memory)),
    ).model_dump(mode="json")


def _reflexion_trial(
    task: str, sample: str, arm: str, model: str, trial_order: int
) -> dict[str, Any]:
    baseline = "reflexion_style"
    trial_id = _trial_id(task, sample, baseline, arm, model)
    is_retry = sample == "game24_pilot_001"
    reflection = "Reflection: try a useful denominator."
    calls = [_call("reflexion_generate", model, "Initial solve")]
    memory_after: list[dict[str, Any]] = []
    event: dict[str, Any] | None = None
    if is_retry:
        calls.extend(
            [
                _call("reflexion_reflect", model, reflection),
                _call("reflexion_generate", model, f"Retry with {reflection}"),
            ]
        )
        memory_after = [
            {
                "entry_id": f"reflexion:{trial_id}",
                "content": reflection,
                "memory_type": "verbal_reflection",
                "clean_or_contaminated": "clean",
                "source_trial_id": trial_id,
                "metadata": {},
            }
        ]
        event = {
            "type": "reflexion_append",
            "status": "accepted",
            "new_entry_id": memory_after[0]["entry_id"],
            "source_trial_id": trial_id,
        }
    final_response = calls[-1]["raw_response"]
    return TrialLog(
        trial_id=trial_id,
        run_id=RUN_ID,
        task_name=task,
        sample_id=sample,
        baseline=baseline,
        arm=cast(Literal["clean", "contaminated", "contaminated_filter"], arm),
        backbone=model,
        input={"sample": sample},
        gold_or_verifier_spec={},
        prompt_messages=[message for call in calls for message in call["messages"]],
        memory_before=[],
        retrieved_memory=[],
        retrieved_scores=[],
        filter_decision={"dropped": 1} if arm == "contaminated_filter" else None,
        raw_response=final_response,
        parsed_answer=final_response,
        verifier_result=VerifierResult(is_correct=True, parsed_answer=final_response, reason="ok"),
        metadata={"trial_order": trial_order},
        memory_write_event=event,
        memory_after=memory_after,
        method_calls=cast(Any, calls),
        contamination_exposure=cast(Any, _exposure(arm, [])),
    ).model_dump(mode="json")


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dc_states: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for task, samples in TASKS.items():
        for sample in samples:
            for arm in ARMS:
                for model in MODELS:
                    key = (task, arm, model)
                    memory = dc_states.setdefault(
                        key,
                        [
                            {
                                "entry_id": f"seed:{task}:{arm}:{model}",
                                "content": "seed input",
                                "memory_type": "dc_rs_io_pair",
                                "clean_or_contaminated": "clean",
                                "source_trial_id": None,
                                "metadata": {"output_text": "seed output"},
                            }
                        ],
                    )
                    dc_row = _dc_rs_trial(task, sample, arm, model, list(memory), len(rows))
                    rows.append(dc_row)
                    dc_states[key] = dc_row["memory_after"]
                    rows.append(_reflexion_trial(task, sample, arm, model, len(rows)))
    return rows


def _write_run(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trials.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (run_dir / "aggregate.json").write_text(
        json.dumps(aggregate_run(run_dir, allow_legacy=True)), encoding="utf-8"
    )


def _inspect(run_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(INSPECTOR), str(run_dir)],
        cwd=REPO_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_inspector_accepts_exact_synthetic_followup_gate(tmp_path: Path) -> None:
    run_dir = tmp_path / "synthetic"
    rows = _rows()
    assert len(rows) == 108
    _write_run(run_dir, rows)

    result = _inspect(run_dir)

    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout)
    assert report["overall"] == "pass"
    assert report["summary"] == {"trials": 108, "method_calls": 174, "dc_rs_calls": 108, "reflexion_calls": 66}


def test_inspector_rejects_swapped_dc_rs_stages(tmp_path: Path) -> None:
    rows = _rows()
    target = next(row for row in rows if row["baseline"] == "dynamic_cheatsheet_rs_optional")
    target["method_calls"] = list(reversed(target["method_calls"]))
    _write_run(tmp_path, rows)

    result = _inspect(tmp_path)

    assert result.returncode == 1
    assert "dc-rs stages" in result.stdout.lower()


def test_inspector_rejects_unconsumed_reflection(tmp_path: Path) -> None:
    rows = _rows()
    target = next(
        row
        for row in rows
        if row["baseline"] == "reflexion_style" and row["sample_id"] == "game24_pilot_001"
    )
    target["method_calls"][-1]["messages"] = [{"role": "user", "content": "Retry without memory"}]
    _write_run(tmp_path, rows)

    result = _inspect(tmp_path)

    assert result.returncode == 1
    assert "reflection" in result.stdout.lower()


def test_inspector_rejects_aggregate_call_mismatch(tmp_path: Path) -> None:
    rows = _rows()
    _write_run(tmp_path, rows)
    aggregate_path = tmp_path / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["groups"][0]["method_call_count"] = 0
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")

    result = _inspect(tmp_path)

    assert result.returncode == 1
    assert "aggregate method_call_count" in result.stdout


def test_inspector_real_followup_run_when_available() -> None:
    run_dir = REPO_ROOT / "runs" / "g0_dc_rs_reflexion_fidelity_followup_replay"
    if not (run_dir / "trials.jsonl").exists() or not (run_dir / "aggregate.json").exists():
        import pytest

        pytest.skip("canonical replay and aggregate artifacts are not present")
    result = _inspect(run_dir)
    assert result.returncode == 0, result.stderr + result.stdout
