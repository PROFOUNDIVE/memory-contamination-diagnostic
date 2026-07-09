from __future__ import annotations

import json
import tempfile
from pathlib import Path

from memcontam.cli import run_config
from memcontam.logging.schema import TrialLog


def _three_task_evidence(path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = {
            "run": {
                "name": "T4_three_task_dispatch",
                "task_order_seed": 1,
                "sample_order_seed": 1,
                "retry_policy_version": "retry_v0",
            },
            "models": ["replay"],
            "tasks": [
                {
                    "name": "game24",
                    "sample_path": "data/tasks/game24_pilot.jsonl",
                    "limit": 1,
                },
                {
                    "name": "word_sorting",
                    "sample_path": "data/tasks/word_sorting_pilot.jsonl",
                    "limit": 1,
                },
                {
                    "name": "math_equation_balancer",
                    "sample_path": "data/tasks/math_equation_balancer_pilot.jsonl",
                    "limit": 1,
                },
            ],
            "baselines": ["no_memory"],
            "arms": ["clean"],
            "logging": {
                "output_dir": tmp,
                "prompt_version": "prompt_v0",
                "memory_policy_version": "memory_policy_v0",
                "contamination_catalog_version": "contamination_v0",
            },
            "replay": {
                "responses_by_sample": {
                    "game24_pilot_001": "final: 6 / (1 - (3 / 4))",
                    "word_sorting_pilot_001": "final: apple banana pear",
                    "meb_pilot_001": "final: 2 + 5 = 7",
                }
            },
        }
        run_dir = run_config(config, "T4_dispatch_three_tasks")
        rows = []
        with (run_dir / "trials.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        assert len(rows) == 3, f"expected 3 rows, got {len(rows)}"
        task_names = {row["task_name"] for row in rows}
        assert task_names == {"game24", "word_sorting", "math_equation_balancer"}, task_names
        for row in rows:
            TrialLog.model_validate(row)

        path.write_text(
            f"T4 three-task dispatch QA passed\n"
            f"rows emitted: {len(rows)}\n"
            f"task names: {sorted(task_names)}\n"
            f"all rows validated as TrialLog\n"
        )


def _unsupported_task_evidence(path: Path) -> None:
    config = {
        "run": {
            "name": "T4_unsupported_task",
            "task_order_seed": 1,
            "sample_order_seed": 1,
            "retry_policy_version": "retry_v0",
        },
        "models": ["replay"],
        "tasks": [
            {
                "name": "unknown_task",
                "sample_path": "data/tasks/game24_pilot.jsonl",
                "limit": 1,
            },
        ],
        "baselines": ["no_memory"],
        "arms": ["clean"],
        "logging": {
            "output_dir": "/tmp",
            "prompt_version": "prompt_v0",
            "memory_policy_version": "memory_policy_v0",
            "contamination_catalog_version": "contamination_v0",
        },
        "replay": {
            "responses_by_sample": {"game24_pilot_001": "final: 6 / (1 - (3 / 4))"}
        },
    }
    try:
        run_config(config, "T4_unsupported_task")
    except SystemExit as exc:
        message = str(exc)
        assert "unsupported task" in message, message
        path.write_text(
            f"T4 unsupported task QA passed\n"
            f"SystemExit message: {message}\n"
        )
        return
    raise AssertionError("expected SystemExit for unsupported task")


def main() -> None:
    evidence_dir = Path(".sisyphus/evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    _three_task_evidence(evidence_dir / "task-T4-dispatch-three-tasks.txt")
    _unsupported_task_evidence(evidence_dir / "task-T4-unsupported-task.txt")
    print("T4 QA evidence saved")


if __name__ == "__main__":
    main()
