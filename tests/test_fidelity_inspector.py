from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from memcontam.logging.schema import TrialLog, VerifierResult


REPO_ROOT = Path(__file__).resolve().parents[1]
INSPECTOR = REPO_ROOT / "scripts" / "inspect_g0_rag_bot_fidelity.py"


def _env():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return env


def _run_inspector(run_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(INSPECTOR), str(run_dir)],
        cwd=REPO_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def _rag_method_call(record_text: str = "Useful strategy for game24."):
    return {
        "stage": "rag_generate",
        "messages": [{"role": "user", "content": f"Retrieved: {record_text}"}],
        "raw_response": "final: 6 / (1 - 3/4)",
        "model": "replay",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
        "latency_ms": 10,
        "token_usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        "retry_count": 0,
        "error_type": None,
        "retrieved_records": [
            {
                "document_id": "rag-doc-1",
                "rank": 1,
                "score": 0.91,
                "text": record_text,
                "title_or_type": "game24_strategy",
                "clean_or_contaminated": "clean",
                "source": "memory_catalog_v1",
                "corpus_hash": "sha256:ragcorpus",
                "embedding_model_id": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                "embedding_library_version": "sentence-transformers-3.0.0",
            }
        ],
    }


def _bot_method_calls():
    return [
        {
            "stage": "bot_problem_distill",
            "messages": [{"role": "user", "content": "distill"}],
            "raw_response": "1. Key information: numbers = [1, 3, 4, 6]",
            "model": "replay",
        },
        {
            "stage": "bot_instantiate_solve",
            "messages": [{"role": "user", "content": "solve"}],
            "raw_response": "final: 6 / (1 - 3/4)",
            "model": "replay",
        },
        {
            "stage": "bot_thought_distill",
            "messages": [{"role": "user", "content": "distill thought"}],
            "raw_response": "High-level template for making 24.",
            "model": "replay",
        },
        {
            "stage": "bot_novelty_decide",
            "messages": [{"role": "user", "content": "decide"}],
            "raw_response": "True",
            "model": "replay",
        },
    ]


def _bot_template_entry(entry_id: str, source_trial_id: str):
    return {
        "entry_id": entry_id,
        "content": f"Template {entry_id}",
        "memory_type": "thought_template",
        "clean_or_contaminated": "clean",
        "source_trial_id": source_trial_id,
        "metadata": {},
    }


def _trial_row(**overrides) -> dict:
    base = {
        "trial_id": "run:game24:s1:no_memory:clean:replay",
        "run_id": "run",
        "task_name": "game24",
        "sample_id": "s1",
        "baseline": "no_memory",
        "arm": "clean",
        "backbone": "replay",
        "input": {"numbers": [1, 3, 4, 6]},
        "gold_or_verifier_spec": {"target": 24},
        "prompt_messages": [{"role": "user", "content": "solve"}],
        "raw_response": "final: 24",
        "verifier_result": VerifierResult(is_correct=True),
    }
    base.update(overrides)
    return TrialLog(**base).model_dump(mode="json")


def _write_trials_jsonl(run_dir: Path, rows: list[dict]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_path = run_dir / "trials.jsonl"
    trials_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return trials_path


def test_inspector_accepts_complete_synthetic_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "faithful_run"

    rag_row = _trial_row(
        trial_id="run:game24:s1:retrieval_rag:clean:replay",
        baseline="retrieval_rag",
        sample_id="s1",
        prompt_messages=[
            {"role": "user", "content": "Retrieved memory:\n#1 entry_id=rag-doc-1 Useful strategy for game24.\n\nSolve: ..."}
        ],
        retrieved_memory=[
            {
                "entry_id": "rag-doc-1",
                "content": "Useful strategy for game24.",
                "memory_type": "game24_strategy",
                "clean_or_contaminated": "clean",
                "source_trial_id": None,
                "metadata": {},
            }
        ],
        retrieved_scores=[0.91],
        memory_write_event=None,
        method_calls=[_rag_method_call()],
        metadata={"corpus_hash": "sha256:ragcorpus"},
    )

    bot_s1 = _trial_row(
        trial_id="run:game24:s1:bot_style:clean:replay",
        baseline="bot_style",
        sample_id="s1",
        prompt_messages=[{"role": "user", "content": "distill and solve"}],
        method_calls=_bot_method_calls(),
        memory_write_event={
            "event_type": "bot_write",
            "status": "accepted",
            "baseline": "bot_style",
            "parent_trial_id": "run:game24:s1:bot_style:clean:replay",
            "source_entry_ids": [],
            "new_entry_id": "bot-template:s1",
            "update_reason": "distilled_thought_template",
        },
        memory_after=[_bot_template_entry("bot-template:s1", "run:game24:s1:bot_style:clean:replay")],
    )

    bot_s2 = _trial_row(
        trial_id="run:game24:s2:bot_style:clean:replay",
        baseline="bot_style",
        sample_id="s2",
        prompt_messages=[{"role": "user", "content": "distill and solve"}],
        memory_before=[_bot_template_entry("bot-template:s1", "run:game24:s1:bot_style:clean:replay")],
        retrieved_memory=[_bot_template_entry("bot-template:s1", "run:game24:s1:bot_style:clean:replay")],
        retrieved_scores=[0.95],
        method_calls=_bot_method_calls(),
        memory_write_event={
            "event_type": "bot_write",
            "status": "accepted",
            "baseline": "bot_style",
            "parent_trial_id": "run:game24:s2:bot_style:clean:replay",
            "source_entry_ids": ["bot-template:s1"],
            "new_entry_id": "bot-template:s2",
            "update_reason": "distilled_thought_template",
        },
        memory_after=[
            _bot_template_entry("bot-template:s1", "run:game24:s1:bot_style:clean:replay"),
            _bot_template_entry("bot-template:s2", "run:game24:s2:bot_style:clean:replay"),
        ],
    )

    foreign_bot = _trial_row(
        trial_id="run:game24:s1:bot_style:contaminated:replay",
        baseline="bot_style",
        arm="contaminated",
        sample_id="s1",
        prompt_messages=[{"role": "user", "content": "distill and solve"}],
        method_calls=_bot_method_calls(),
        memory_write_event={
            "event_type": "bot_write",
            "status": "accepted",
            "baseline": "bot_style",
            "parent_trial_id": "run:game24:s1:bot_style:contaminated:replay",
            "source_entry_ids": [],
            "new_entry_id": "bot-template:s1-contaminated",
            "update_reason": "distilled_thought_template",
        },
        memory_after=[_bot_template_entry("bot-template:s1-contaminated", "run:game24:s1:bot_style:contaminated:replay")],
    )

    _write_trials_jsonl(run_dir, [rag_row, bot_s1, bot_s2, foreign_bot])

    result = _run_inspector(run_dir)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["rag"] == "pass"
    assert report["bot"] == "pass"
    assert report["persistence"] == "pass"
    assert report["isolation"] == "pass"
    assert report["logging"] == "pass"


def test_inspector_rejects_nonpersistent_and_misaligned_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad_run"

    misaligned_rag = _trial_row(
        trial_id="run:game24:s1:retrieval_rag:clean:replay",
        baseline="retrieval_rag",
        sample_id="s1",
        prompt_messages=[{"role": "user", "content": "Solve: ..."}],
        retrieved_memory=[
            {
                "entry_id": "rag-doc-1",
                "content": "Useful strategy for game24.",
                "memory_type": "game24_strategy",
                "clean_or_contaminated": "clean",
                "source_trial_id": None,
                "metadata": {},
            }
        ],
        retrieved_scores=[0.91],
        method_calls=[_rag_method_call()],
        memory_write_event=None,
    )

    nonpersistent_bot = _trial_row(
        trial_id="run:game24:s1:bot_style:clean:replay",
        baseline="bot_style",
        sample_id="s1",
        prompt_messages=[{"role": "user", "content": "distill and solve"}],
        method_calls=_bot_method_calls(),
        memory_write_event={
            "event_type": "bot_write",
            "status": "accepted",
            "baseline": "bot_style",
            "parent_trial_id": "run:game24:s1:bot_style:clean:replay",
            "source_entry_ids": [],
            "new_entry_id": "bot-template:s1",
            "update_reason": "distilled_thought_template",
        },
        memory_after=[_bot_template_entry("bot-template:s1", "run:game24:s1:bot_style:clean:replay")],
    )

    _write_trials_jsonl(run_dir, [misaligned_rag, nonpersistent_bot])

    result = _run_inspector(run_dir)
    assert result.returncode != 0
    report = json.loads(result.stdout)
    assert report["rag"] == "fail"
    assert report["bot"] == "pass"
    assert report["persistence"] == "fail"
    assert report["isolation"] == "pass"
    assert report["logging"] == "pass"
    assert "rag" in " ".join(report.get("reasons", [])).lower() or "prompt" in result.stdout.lower()
    assert "persist" in " ".join(report.get("reasons", [])).lower() or "reuse" in result.stdout.lower()
