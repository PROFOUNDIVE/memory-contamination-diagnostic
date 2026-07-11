from __future__ import annotations

import argparse
import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from memcontam.baselines.bot_style import BotStylePolicy, distill_thought_template
from memcontam.baselines.bot_runtime import BotRuntime
from memcontam.baselines.full_history import FullHistoryPolicy
from memcontam.baselines.no_memory import NoMemoryPolicy
from memcontam.baselines.reflexion_style import ReflexionStylePolicy
from memcontam.baselines.retrieval_rag import RetrievalRagPolicy
from memcontam.clients.base import LLMClient
from memcontam.clients.openai_compatible import OpenAICompatibleClient
from memcontam.clients.replay import ReplayClient
from memcontam.evaluation.aggregate import aggregate_run
from memcontam.contamination.catalog import load_catalog
from memcontam.logging.schema import (
    BadMemoryUptakeLabel,
    ContaminationExposure,
    RepeatedFailureLabel,
    TrialLog,
)
from memcontam.memory.bot_buffer import BotBufferIdentity, ThoughtTemplate
from memcontam.memory.corpus import CorpusRecord, build_arm_corpus, load_corpus
from memcontam.memory.embeddings import EmbeddingProvider, FakeEmbeddingProvider, SentenceTransformerProvider
from memcontam.memory.filters import drop_known_contaminated
from memcontam.memory.retrieval import retrieve_records
from memcontam.memory.run_state import RunState
from memcontam.memory.stores import MemoryEntry, MemoryState
from memcontam.tasks.game24 import build_instance as build_game24_instance
from memcontam.tasks.math_equation_balancer import build_instance as build_meb_instance
from memcontam.tasks.word_sorting import build_instance as build_word_sorting_instance
from memcontam.verifiers.game24 import verify_expression
from memcontam.verifiers.math_equation_balancer import verify_answer as verify_meb_answer
from memcontam.verifiers.word_sorting import verify_words


def _verify_game24(parsed_answer: str, task: Any) -> Any:
    return verify_expression(
        parsed_answer,
        task.input["numbers"],
        task.verifier_spec.get("target", 24),
    )


def _verify_meb(parsed_answer: str, task: Any) -> Any:
    return verify_meb_answer(parsed_answer, task.verifier_spec)


def _verify_word_sorting(parsed_answer: str, task: Any) -> Any:
    words = parsed_answer.split()
    return verify_words(words, task.verifier_spec["sorted_words"])


TASK_DISPATCH = {
    "game24": {
        "build": build_game24_instance,
        "verify": _verify_game24,
    },
    "math_equation_balancer": {
        "build": build_meb_instance,
        "verify": _verify_meb,
    },
    "word_sorting": {
        "build": build_word_sorting_instance,
        "verify": _verify_word_sorting,
    },
}


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
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"malformed replay input: {path}") from exc
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


def _memory_entries_for_arm(arm: str, baseline: str) -> tuple[list[MemoryEntry], dict[str, Any] | None]:
    if arm == "clean":
        return [], None
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
                    metadata={
                        "task": item.get("task"),
                        "arm": arm,
                        "contamination_type": item.get("contamination_type", item["type"]),
                    },
                )
            )
    if arm == "contaminated_filter":
        return drop_known_contaminated(entries)
    return entries, None


def _retrieved_memory(baseline: str, task_input: dict[str, Any], memory: MemoryState) -> tuple[list[dict[str, Any]], list[float]]:
    if baseline not in {"retrieval_rag", "bot_style"}:
        return [], []
    retrieved = retrieve_records(str(task_input), memory.entries, k=1 if baseline == "bot_style" else 3)
    return [record["memory_entry"].model_dump() for record in retrieved], [record["score"] for record in retrieved]


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _embedding_provider(config: dict[str, Any]) -> EmbeddingProvider:
    embedding_config = config.get("embedding", {})
    if embedding_config.get("offline_fallback", False):
        return FakeEmbeddingProvider()
    return SentenceTransformerProvider(
        cache_folder=embedding_config.get("cache_path"),
        local_files_only=True,
    )


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _trial_metadata(config: dict[str, Any], model: str, trial_order: int, run_started_at: str) -> dict[str, Any]:
    run_config = config.get("run", {})
    logging_config = config.get("logging", {})
    replay_config = config.get("replay", {})
    return {
        "git_commit": _git_commit(),
        "config_hash": _config_hash(config),
        "model_provider": "replay",
        "model_id": model,
        "model_snapshot_or_served_name": "replay",
        "query_date": run_started_at,
        "seed_or_order": run_config.get("sample_order_seed", run_config.get("task_order_seed", trial_order)),
        "temperature": replay_config.get("temperature"),
        "top_p": replay_config.get("top_p"),
        "max_tokens": replay_config.get("max_tokens"),
        "prompt_version": logging_config.get("prompt_version", "unknown"),
        "memory_policy_version": logging_config.get("memory_policy_version", "unknown"),
        "contamination_set_version": logging_config.get("contamination_catalog_version", "unknown"),
        "retry_policy_version": run_config.get("retry_policy_version", "unknown"),
    }


def _entry_id(entry: dict[str, Any]) -> str | None:
    entry_id = entry.get("entry_id") or entry.get("document_id")
    return entry_id if isinstance(entry_id, str) else None


def _contamination_type(entry: dict[str, Any]) -> str:
    metadata = entry.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("contamination_type"), str):
        return metadata["contamination_type"]
    return str(entry.get("memory_type", "unknown"))


def _contamination_exposure(
    arm: str, memory_before: list[dict[str, Any]], retrieved_memory: list[dict[str, Any]]
) -> ContaminationExposure:
    memory_before_ids = [entry_id for entry in memory_before if (entry_id := _entry_id(entry))]
    retrieved_ids = [entry_id for entry in retrieved_memory if (entry_id := _entry_id(entry))]
    source_entries = [entry for entry in memory_before if entry.get("clean_or_contaminated") == "contaminated"]
    source_ids = [entry_id for entry in source_entries if (entry_id := _entry_id(entry))]
    contamination_types = sorted({_contamination_type(entry) for entry in source_entries})
    if arm == "clean":
        is_exposed = False
        exposure_mode = "none"
        reason = "clean arm has no contaminated memory sources"
    elif retrieved_ids:
        is_exposed = True
        exposure_mode = "retrieved_memory"
        reason = "contaminated memory source was retrieved"
    elif source_ids:
        is_exposed = True
        exposure_mode = "memory_before"
        reason = "contaminated memory sources were available before prompting"
    else:
        is_exposed = False
        exposure_mode = "none"
        reason = "no contaminated memory sources remained after filtering"
    return ContaminationExposure(
        condition=cast(Literal["clean", "contaminated", "contaminated_filter"], arm),
        is_exposed=is_exposed,
        source_entry_ids=source_ids,
        contamination_types=contamination_types,
        memory_before_entry_ids=memory_before_ids,
        retrieved_entry_ids=retrieved_ids,
        exposure_mode=exposure_mode,
        reason=reason,
    )


def _bot_memory_writeback(
    trial_id: str,
    task: Any,
    raw_response: str,
    verifier_result: Any,
    retrieved_memory: list[dict[str, Any]],
    memory: MemoryState,
) -> dict[str, Any]:
    source_entry_ids = [entry_id for entry in retrieved_memory if (entry_id := _entry_id(entry))]
    new_entry_id = f"bot_template:{hashlib.sha256(trial_id.encode('utf-8')).hexdigest()[:12]}"
    memory.entries.append(
        MemoryEntry(
            entry_id=new_entry_id,
            content=distill_thought_template(
                task,
                raw_response,
                verifier_result,
                retrieved_memory[0] if retrieved_memory else None,
            ),
            memory_type="thought_template",
            clean_or_contaminated="clean",
            source_trial_id=trial_id,
            metadata={"distillation_source": "bot_writeback"},
        )
    )
    return {
        "event_type": "bot_write",
        "baseline": "bot_style",
        "parent_trial_id": trial_id,
        "source_entry_ids": source_entry_ids,
        "new_entry_id": new_entry_id,
        "update_reason": "distilled_thought_template_from_problem_solution_pair",
    }


def _bad_memory_uptake_label(arm: str, exposure: ContaminationExposure) -> BadMemoryUptakeLabel:
    if arm == "clean" or not exposure.source_entry_ids:
        return "not_applicable"
    return "not_evaluable"


def _is_faithful_config(config: dict[str, Any]) -> bool:
    return bool(config.get("embedding", {}).get("corpus_path") and config.get("bot_state"))


def _records_for_baseline(
    records: list[CorpusRecord], task_name: str, baseline: str
) -> list[CorpusRecord]:
    return [
        record
        for record in records
        if record.task == task_name
        and (not record.target_baselines or baseline in record.target_baselines)
    ]


def _bot_injection_entries(
    records: list[CorpusRecord], task_name: str, arm: str
) -> tuple[list[MemoryEntry], dict[str, Any] | None]:
    entries, filter_decision = build_arm_corpus(records, task_name, cast(Any, arm))
    return [entry for entry in entries if entry.clean_or_contaminated == "contaminated"], filter_decision


def _entry_to_template(entry: MemoryEntry) -> ThoughtTemplate:
    return ThoughtTemplate(
        entry_id=entry.entry_id,
        content=entry.content,
        source_trial_id=entry.source_trial_id or f"catalog:{entry.entry_id}",
        metadata=dict(entry.metadata),
    )


def _template_to_entry(template: ThoughtTemplate) -> MemoryEntry:
    return MemoryEntry(
        entry_id=template.entry_id,
        content=template.content,
        memory_type="thought_template",
        clean_or_contaminated="clean",
        source_trial_id=template.source_trial_id,
        metadata=dict(template.metadata),
    )


def _scoped_bot_entry_id(entry_id: str, identity: BotBufferIdentity) -> str:
    identity_key = ":".join(
        [identity.run_id, identity.task_name, identity.baseline, identity.arm, identity.backbone]
    )
    return f"{entry_id}:{hashlib.sha256(identity_key.encode('utf-8')).hexdigest()[:8]}"


def _retrieval_record_dict(record: Any) -> dict[str, Any]:
    data = record.model_dump()
    data["entry_id"] = data.get("document_id")
    return data


def _trial_client_for_sample(
    client: LLMClient, responses_by_sample: dict[str, Any], sample_id: str, replay_responses: Any
) -> LLMClient:
    if isinstance(client, ReplayClient) and sample_id not in responses_by_sample and not replay_responses:
        raise SystemExit(f"missing replay response for sample: {sample_id}")
    if sample_id in responses_by_sample and isinstance(responses_by_sample[sample_id], dict):
        return ReplayClient(responses_by_sample={sample_id: responses_by_sample[sample_id]})
    if sample_id in responses_by_sample:
        return ReplayClient([responses_by_sample[sample_id]])
    return client


def _method_call_messages(method_calls: list[Any]) -> list[dict[str, str]]:
    return [message for call in method_calls for message in call.messages]


def _faithful_result_trial(
    *,
    config: dict[str, Any],
    run_id: str,
    task: Any,
    baseline: str,
    arm: str,
    model: str,
    result: dict[str, Any],
    verifier_result: Any,
    filter_decision: dict[str, Any] | None,
    trial_order: int,
    run_started_at: str,
    repeated_failure_tracker: _RepeatedFailureTracker,
) -> TrialLog:
    trial_id = ":".join([run_id, task.task_name, task.sample_id, baseline, arm, model])
    method_calls = result.get("method_calls", [])
    retrieved_memory = result.get("retrieved_memory")
    if retrieved_memory is None:
        if "retrieved_records" in result:
            retrieved_memory = [_retrieval_record_dict(record) for record in result["retrieved_records"]]
        elif result.get("retrieved_template") is not None:
            retrieved_memory = [result["retrieved_template"]]
        else:
            retrieved_memory = []
    retrieved_scores = result.get("retrieved_scores") or [
        float(entry["score"]) for entry in retrieved_memory if "score" in entry
    ]
    metadata = {
        **_trial_metadata(config, model, trial_order, run_started_at),
        **result.get("metadata", {}),
    }
    contamination_exposure = _contamination_exposure(
        arm, result.get("memory_before", []), retrieved_memory
    )
    return TrialLog(
        trial_id=trial_id,
        run_id=run_id,
        task_name=task.task_name,
        sample_id=task.sample_id,
        baseline=baseline,
        arm=cast(Literal["clean", "contaminated", "contaminated_filter"], arm),
        backbone=model,
        input=task.input,
        gold_or_verifier_spec=task.verifier_spec,
        prompt_messages=_method_call_messages(method_calls),
        memory_before=result.get("memory_before", []),
        retrieved_memory=retrieved_memory,
        retrieved_scores=retrieved_scores,
        raw_response=result["final_response"],
        parsed_answer=result.get("parsed_answer"),
        verifier_result=verifier_result,
        metadata=metadata,
        filter_decision=filter_decision,
        contamination_exposure=contamination_exposure,
        bad_memory_uptake_label=_bad_memory_uptake_label(arm, contamination_exposure),
        repeated_failure_label=repeated_failure_tracker.label(
            verifier_result.is_correct,
            task.task_name,
            task.sample_id,
            baseline,
            arm,
            model,
        ),
        recovery_after_filter_label="not_applicable",
        memory_write_event=result.get("memory_write_event"),
        memory_after=result.get("memory_after", []),
        latency_ms=None,
        token_usage={},
        method_calls=method_calls,
    )


class _RepeatedFailureTracker:
    def __init__(self):
        self._seen_incorrect: set[tuple[str, str, str, str, str]] = set()

    def label(
        self,
        verifier_is_correct: bool,
        task_name: str,
        sample_id: str,
        baseline: str,
        arm: str,
        backbone: str,
    ) -> RepeatedFailureLabel:
        if verifier_is_correct:
            return "not_applicable"
        key = (task_name, sample_id, baseline, arm, backbone)
        if key in self._seen_incorrect:
            return "repeated_failure"
        self._seen_incorrect.add(key)
        return "first_failure"


def _run_faithful_config(
    config: dict[str, Any],
    run_id: str,
    client: LLMClient,
    replay_responses: Any,
    responses_by_sample: dict[str, Any],
    trials_path: Path,
    run_started_at: str,
    repeated_failure_tracker: _RepeatedFailureTracker,
) -> None:
    corpus_records = load_corpus(Path(config["embedding"]["corpus_path"]))
    embedding_provider = _embedding_provider(config)
    cache_dir = Path(config["embedding"].get("cache_path", "data/embedding_cache")) / run_id
    run_state = RunState(
        run_id,
        config_hash=_config_hash(config),
        evaluation_sample_ids=[
            row["sample_id"]
            for task_config in config["tasks"]
            for row in _load_jsonl(Path(task_config["sample_path"]), task_config.get("limit"))
        ],
    )
    bot_runtime = BotRuntime()
    bot_buffers: dict[BotBufferIdentity, list[MemoryEntry]] = {}

    with trials_path.open("w", encoding="utf-8") as f:
        trial_order = 0
        for task_config in config["tasks"]:
            task_name = task_config["name"]
            if task_name not in TASK_DISPATCH:
                raise SystemExit(f"unsupported task for replay spine: {task_name}")
            task_handler = TASK_DISPATCH[task_name]
            rows = _load_jsonl(Path(task_config["sample_path"]), task_config.get("limit"))
            if not rows:
                raise SystemExit(f'empty replay input: {task_config["sample_path"]}')
            for row in rows:
                task = task_handler["build"](row)
                for baseline in config["baselines"]:
                    if baseline not in {"retrieval_rag", "bot_style"}:
                        policy = BASELINE_POLICIES.get(baseline)
                        if policy is None:
                            raise SystemExit(f"unsupported baseline: {baseline}")
                        raise SystemExit(f"unsupported faithful baseline: {baseline}")
                    baseline_records = _records_for_baseline(corpus_records, task_name, baseline)
                    for arm in config["arms"]:
                        injection_entries: list[MemoryEntry] = []
                        if baseline == "retrieval_rag":
                            memory_entries, filter_decision = build_arm_corpus(
                                baseline_records, task_name, cast(Any, arm)
                            )
                            memory = MemoryState(entries=memory_entries)
                        else:
                            injection_entries, filter_decision = _bot_injection_entries(
                                baseline_records, task_name, arm
                            )
                            memory = MemoryState(entries=[])
                        for model in config["models"]:
                            trial_client = _trial_client_for_sample(
                                client, responses_by_sample, task.sample_id, replay_responses
                            )
                            if baseline == "retrieval_rag":
                                rag_cache_dir = cache_dir / task_name / arm
                                result = RetrievalRagPolicy().run(
                                    task,
                                    memory,
                                    client=trial_client,
                                    model=model,
                                    config={**config.get("replay", {}), "sample_id": task.sample_id},
                                    top_k=config["embedding"].get("top_k"),
                                    embedding_provider=embedding_provider,
                                    cache_dir=rag_cache_dir,
                                )
                                verifier_result = task_handler["verify"](result["parsed_answer"], task)
                            else:
                                identity = BotBufferIdentity(
                                    run_id, task.task_name, baseline, arm, model
                                )
                                if identity not in bot_buffers:
                                    if arm == "clean":
                                        snapshot = run_state.snapshot_clean_warmup(identity)
                                        bot_buffers[identity] = [
                                            _template_to_entry(entry) for entry in snapshot.entries
                                        ]
                                    else:
                                        injected_templates = [
                                            _entry_to_template(entry) for entry in injection_entries
                                        ]
                                        bot_buffers[identity] = [
                                            _template_to_entry(entry)
                                            for entry in run_state.clone_for_arm(
                                                identity, injected_templates
                                            )
                                        ]
                                result = bot_runtime.run(
                                    identity=identity,
                                    task=task,
                                    buffer_snapshot=bot_buffers[identity],
                                    client=trial_client,
                                    model=model,
                                    config={
                                        **config.get("replay", {}),
                                        "sample_id": task.sample_id,
                                        "embedding_provider": embedding_provider,
                                    },
                                    verifier=lambda response, task=task: task_handler["verify"](
                                        _parse_answer(response), task
                                    ),
                                )
                                verifier_result = result["verifier_result"]
                                event = result.get("memory_write_event")
                                if event and event.get("status") == "accepted":
                                    original_entry_id = str(event["new_entry_id"])
                                    scoped_entry_id = _scoped_bot_entry_id(original_entry_id, identity)
                                    event["new_entry_id"] = scoped_entry_id
                                    for entry in result["memory_after"]:
                                        if entry.get("entry_id") == original_entry_id:
                                            entry["entry_id"] = scoped_entry_id
                                    event["sample_id"] = ":".join(
                                        [
                                            run_id,
                                            task.task_name,
                                            task.sample_id,
                                            baseline,
                                            arm,
                                            model,
                                        ]
                                    )
                                    run_state.register_warmup_result(identity, event)
                                    bot_buffers[identity] = [
                                        MemoryEntry(
                                            entry_id=entry["entry_id"],
                                            content=entry["content"],
                                            memory_type="thought_template",
                                            clean_or_contaminated="clean",
                                            source_trial_id=entry.get("source_trial_id"),
                                            metadata=entry.get("metadata", {}),
                                        )
                                        for entry in result["memory_after"]
                                    ]
                            trial = _faithful_result_trial(
                                config=config,
                                run_id=run_id,
                                task=task,
                                baseline=baseline,
                                arm=arm,
                                model=model,
                                result=result,
                                verifier_result=verifier_result,
                                filter_decision=filter_decision,
                                trial_order=trial_order,
                                run_started_at=run_started_at,
                                repeated_failure_tracker=repeated_failure_tracker,
                            )
                            f.write(trial.model_dump_json() + "\n")
                            trial_order += 1


def run_config(
    config: dict[str, Any], run_id: str, _client_override: LLMClient | None = None
) -> Path:
    _validate_run_id(run_id)
    output_dir = Path(config.get("logging", {}).get("output_dir", "runs"))
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_path = run_dir / "trials.jsonl"

    replay_config = config.get("replay", {})
    replay_responses = replay_config.get("responses")
    live_smoke_enabled = config.get("live_smoke", {}).get("enabled", False)
    if live_smoke_enabled and _client_override is None:
        live_smoke = config.get("live_smoke", {})
        api_key_env = live_smoke.get("api_key_env", "OPENAI_API_KEY")
        try:
            client: LLMClient = OpenAICompatibleClient(
                base_url=live_smoke.get("base_url"),
                api_key_env=api_key_env,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    elif _client_override is not None:
        client = _client_override
    else:
        client = ReplayClient(replay_responses)
    responses_by_sample = replay_config.get("responses_by_sample", {})
    run_started_at = datetime.now(timezone.utc).isoformat()
    repeated_failure_tracker = _RepeatedFailureTracker()
    if _is_faithful_config(config):
        _run_faithful_config(
            config,
            run_id,
            client,
            replay_responses,
            responses_by_sample,
            trials_path,
            run_started_at,
            repeated_failure_tracker,
        )
        print(f"wrote replay trials: {trials_path}")
        return run_dir
    with trials_path.open("w", encoding="utf-8") as f:
        trial_order = 0
        for task_config in config["tasks"]:
            task_name = task_config["name"]
            if task_name not in TASK_DISPATCH:
                raise SystemExit(f"unsupported task for replay spine: {task_name}")
            task_handler = TASK_DISPATCH[task_name]
            rows = _load_jsonl(Path(task_config["sample_path"]), task_config.get("limit"))
            if not rows:
                raise SystemExit(f'empty replay input: {task_config["sample_path"]}')
            for row in rows:
                task = task_handler["build"](row)
                for baseline in config["baselines"]:
                    if baseline not in BASELINE_POLICIES:
                        raise SystemExit(f"unsupported baseline: {baseline}")
                    policy = BASELINE_POLICIES[baseline]()
                    for arm in config["arms"]:
                        memory_entries, filter_decision = _memory_entries_for_arm(arm, baseline)
                        memory = MemoryState(entries=memory_entries)
                        memory_before = [entry.model_dump() for entry in memory.entries]
                        retrieved_memory, retrieved_scores = _retrieved_memory(baseline, task.input, memory)
                        contamination_exposure = _contamination_exposure(
                            arm, memory_before, retrieved_memory
                        )
                        prompt_messages = policy.build_prompt(task, memory)
                        for model in config["models"]:
                            if isinstance(client, ReplayClient) and task.sample_id not in responses_by_sample and not replay_responses:
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
                            verifier_result = task_handler["verify"](parsed_answer, task)
                            trial_id = ":".join(
                                [run_id, task.task_name, task.sample_id, baseline, arm, model]
                            )
                            memory_write_event = (
                                _bot_memory_writeback(
                                    trial_id,
                                    task,
                                    response.content,
                                    verifier_result,
                                    retrieved_memory,
                                    memory,
                                )
                                if baseline == "bot_style"
                                else None
                            )
                            trial = TrialLog(
                                trial_id=trial_id,
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
                                metadata=_trial_metadata(config, model, trial_order, run_started_at),
                                filter_decision=filter_decision,
                                contamination_exposure=contamination_exposure,
                                bad_memory_uptake_label=_bad_memory_uptake_label(
                                    arm, contamination_exposure
                                ),
                                repeated_failure_label=repeated_failure_tracker.label(
                                    verifier_result.is_correct,
                                    task.task_name,
                                    task.sample_id,
                                    baseline,
                                    arm,
                                    model,
                                ),
                                recovery_after_filter_label="not_applicable",
                                memory_write_event=memory_write_event,
                                memory_after=[entry.model_dump() for entry in memory.entries],
                                latency_ms=response.latency_ms,
                                token_usage=response.token_usage,
                            )
                            f.write(trial.model_dump_json() + "\n")
                            trial_order += 1
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
        print(json.dumps(aggregate_run(args.run_dir)))


if __name__ == "__main__":
    main()
