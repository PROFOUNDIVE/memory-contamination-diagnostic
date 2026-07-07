from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from memcontam.baselines.bot_style import BotStylePolicy
from memcontam.baselines.full_history import FullHistoryPolicy
from memcontam.baselines.no_memory import NoMemoryPolicy
from memcontam.baselines.reflexion_style import ReflexionStylePolicy
from memcontam.baselines.retrieval_rag import RetrievalRagPolicy
from memcontam.clients.replay import ReplayClient
from memcontam.contamination.catalog import load_catalog
from memcontam.logging.schema import TrialLog
from memcontam.memory.filters import drop_known_contaminated
from memcontam.memory.retrieval import lexical_retrieve
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.game24 import build_instance as build_game24_instance
from memcontam.verifiers.game24 import verify_expression


BASELINE_POLICIES = {
    "no_memory": NoMemoryPolicy,
    "full_history": FullHistoryPolicy,
    "retrieval_rag": RetrievalRagPolicy,
    "reflexion_style": ReflexionStylePolicy,
    "bot_style": BotStylePolicy,
}


def validate_config(path: Path) -> None:
    load_config(path)
    print(f"valid config: {path}")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    required = ["run", "models", "tasks", "baselines", "arms"]
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"missing config keys: {', '.join(missing)}")
    return config


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _parse_answer(response: str) -> str:
    response = response.strip()
    if ":" in response:
        return response.split(":", 1)[1].strip()
    return response


def _validate_run_id(run_id: str) -> None:
    run_path = Path(run_id)
    if run_path.is_absolute() or ".." in run_path.parts or len(run_path.parts) != 1:
        raise SystemExit(f"invalid run id: {run_id}")


def _memory_entries_for_arm(arm: str, baseline: str) -> list[MemoryEntry]:
    if arm == "clean":
        return []
    catalog_path = Path("data/contamination/catalog_v0.jsonl")
    if not catalog_path.exists():
        raise SystemExit(f"contamination catalog not found: {catalog_path}")
    entries = []
    for item in load_catalog(catalog_path):
        if baseline in item.get("target_baselines", []):
            entries.append(
                MemoryEntry(
                    entry_id=item["entry_id"],
                    content=item["content"],
                    memory_type=item["type"],
                    clean_or_contaminated="contaminated",
                    metadata={"task": item.get("task"), "arm": arm},
                )
            )
    if arm == "contaminated_filter":
        return drop_known_contaminated(entries)[0]
    return entries


def _retrieved_memory(baseline: str, task_input: dict[str, Any], memory: MemoryState) -> tuple[list[dict[str, Any]], list[float]]:
    if baseline not in {"retrieval_rag", "bot_style"}:
        return [], []
    retrieved = lexical_retrieve(str(task_input), memory.entries, k=1 if baseline == "bot_style" else 3)
    return [entry.model_dump() for entry, _score in retrieved], [score for _entry, score in retrieved]


def run_config(config: dict[str, Any], run_id: str) -> Path:
    _validate_run_id(run_id)
    output_dir = Path(config.get("logging", {}).get("output_dir", "runs"))
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_path = run_dir / "trials.jsonl"

    replay_config = config.get("replay", {})
    replay_responses = replay_config.get("responses")
    client = ReplayClient(replay_responses)
    responses_by_sample = replay_config.get("responses_by_sample", {})
    with trials_path.open("w", encoding="utf-8") as f:
        for task_config in config["tasks"]:
            if task_config["name"] != "game24":
                raise SystemExit(f"unsupported task for replay spine: {task_config['name']}")
            rows = _load_jsonl(Path(task_config["sample_path"]), task_config.get("limit"))
            for row in rows:
                task = build_game24_instance(row)
                for baseline in config["baselines"]:
                    if baseline not in BASELINE_POLICIES:
                        raise SystemExit(f"unsupported baseline: {baseline}")
                    policy = BASELINE_POLICIES[baseline]()
                    for arm in config["arms"]:
                        memory = MemoryState(entries=_memory_entries_for_arm(arm, baseline))
                        memory_before = [entry.model_dump() for entry in memory.entries]
                        retrieved_memory, retrieved_scores = _retrieved_memory(baseline, task.input, memory)
                        prompt_messages = policy.build_prompt(task, memory)
                        for model in config["models"]:
                            if task.sample_id not in responses_by_sample and not replay_responses:
                                raise SystemExit(
                                    f"missing replay response for sample: {task.sample_id}"
                                )
                            trial_client = (
                                ReplayClient([responses_by_sample[task.sample_id]])
                                if task.sample_id in responses_by_sample
                                else client
                            )
                            response = trial_client.chat(prompt_messages, model=model, config={})
                            parsed_answer = _parse_answer(response.content)
                            verifier_result = verify_expression(
                                parsed_answer,
                                task.input["numbers"],
                                task.verifier_spec.get("target", 24),
                            )
                            trial = TrialLog(
                                trial_id=":".join(
                                    [run_id, task.task_name, task.sample_id, baseline, arm, model]
                                ),
                                run_id=run_id,
                                task_name=task.task_name,
                                sample_id=task.sample_id,
                                baseline=baseline,
                                arm=arm,
                                backbone=model,
                                input=task.input,
                                gold_or_verifier_spec=task.verifier_spec,
                                prompt_messages=prompt_messages,
                                memory_before=memory_before,
                                retrieved_memory=retrieved_memory,
                                retrieved_scores=retrieved_scores,
                                raw_response=response.content,
                                parsed_answer=verifier_result.parsed_answer,
                                verifier_result=verifier_result,
                                memory_after=[entry.model_dump() for entry in memory.entries],
                                latency_ms=response.latency_ms,
                                token_usage=response.token_usage,
                            )
                            f.write(trial.model_dump_json() + "\n")
    print(f"wrote replay trials: {trials_path}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(prog="memcontam")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config")
    validate.add_argument("config", type=Path)

    run = sub.add_parser("run")
    run.add_argument("config", type=Path)
    run.add_argument("--run-id", required=True)

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("run_dir", type=Path)

    args = parser.parse_args()

    if args.command == "validate-config":
        validate_config(args.config)
    elif args.command == "run":
        run_config(load_config(args.config), args.run_id)
    elif args.command == "aggregate":
        if not args.run_dir.exists():
            raise SystemExit(f"run dir not found: {args.run_dir}")
        print(f"aggregate skeleton: {args.run_dir}")


if __name__ == "__main__":
    main()
