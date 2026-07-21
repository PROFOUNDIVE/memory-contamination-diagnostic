from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def report_resource_usage(run_dir: Path) -> dict[str, dict[str, int]]:
    trials = [
        json.loads(line)
        for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    calls = [
        json.loads(line)
        for line in (run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    baseline_by_trial = {trial["trial_id"]: trial["baseline"] for trial in trials}
    metrics: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "semantic_call_count": 0,
            "transport_retries": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0,
            "retrieval_count": 0,
            "memory_writes": 0,
        }
    )
    for call in calls:
        baseline = baseline_by_trial[call["trial_id"]]
        usage = call.get("token_usage", {})
        metrics[baseline]["semantic_call_count"] += 1
        metrics[baseline]["transport_retries"] += int(call.get("retry_count", 0))
        metrics[baseline]["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        metrics[baseline]["completion_tokens"] += int(usage.get("completion_tokens", 0))
        metrics[baseline]["latency_ms"] += int(call.get("latency_ms") or 0)
    for trial in trials:
        baseline = trial["baseline"]
        metrics[baseline]["retrieval_count"] += len(trial.get("retrieved_memory", []))
        if (trial.get("memory_write_event") or {}).get("status") == "accepted":
            metrics[baseline]["memory_writes"] += 1
    return dict(sorted(metrics.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report per-baseline method resource usage.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(report_resource_usage(args.run_dir), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
