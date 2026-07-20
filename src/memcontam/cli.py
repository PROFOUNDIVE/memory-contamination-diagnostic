from __future__ import annotations

import argparse
import copy
import json
import hashlib
import subprocess
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from memcontam.baselines.bot_style import BotStylePolicy, distill_thought_template
from memcontam.baselines.bot_runtime import BotRuntime
from memcontam.baselines.contracts import (
    BaselineExecutionOutcome,
    ErrorType,
    FailureDisposition,
    ScientificIneligibilityReason,
    validate_failure_triple,
)
from memcontam.baselines.dynamic_cheatsheet_optional import (
    DynamicCheatsheetOptionalPolicy,
    DynamicCheatsheetRetrievalSynthesisPolicy,
)
from memcontam.baselines.full_history import FullHistoryPolicy
from memcontam.baselines.no_memory import NoMemoryPolicy
from memcontam.baselines.reflexion_style import ReflexionStylePolicy
from memcontam.baselines.retrieval_rag import RetrievalRagPolicy
from memcontam.clients.base import LLMClient, LLMResponse
from memcontam.clients.config import ProviderConfig
from memcontam.clients.factory import build_llm_client, validate_provider_selection
from memcontam.clients.provider_profile import normalize_provider_profile
from memcontam.clients.recording import MethodCallRecorder, summarize_calls
from memcontam.clients.replay import ReplayClient
from memcontam.evaluation.aggregate import aggregate_run
from memcontam.contamination.catalog import load_catalog
from memcontam.logging.schema import (
    BadMemoryUptakeLabel,
    CallEvent,
    CheckpointRef,
    ContaminationExposure,
    EvaluationLawSpec,
    FailureEvent,
    FilterEvent,
    LOGGING_V1,
    LOGGING_V2,
    MethodCall,
    RepeatedFailureLabel,
    RunMetadata,
    TargetContaminationSetSpec,
    TrialLog,
    VerifierResult,
)
from memcontam.logging.provenance import (
    compute_exposure_from_spans,
    compute_exposure_from_spans_v2,
    normalize_memory_event,
)
from memcontam.logging.writer import RunLogWriter
from memcontam.logging.audit_artifacts import write_provider_profile_atomic, write_resolved_config_atomic
from memcontam.config.resolution import resolve_run_config
from memcontam.memory.bot_buffer import BotBufferIdentity, ThoughtTemplate
from memcontam.memory.corpus import CorpusRecord, build_arm_corpus, load_corpus
from memcontam.memory.embeddings import EmbeddingProvider, FakeEmbeddingProvider, SentenceTransformerProvider
from memcontam.memory.filters import FilterTelemetry, filter_legacy_replay_entries
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

BASELINE_ADAPTERS = {
    "no_memory": NoMemoryPolicy,
    "full_history": FullHistoryPolicy,
    "retrieval_rag": RetrievalRagPolicy,
    "reflexion_style": ReflexionStylePolicy,
    "bot_style": BotStylePolicy,
}

FAITHFUL_BASELINES = {
    "no_memory",
    "retrieval_rag",
    "bot_style",
    "full_history",
    "reflexion_style",
    "dynamic_cheatsheet_optional",
    "dynamic_cheatsheet_rs_optional",
}

WRITING_BASELINES = {
    "full_history",
    "reflexion_style",
    "bot_style",
    "dynamic_cheatsheet_optional",
    "dynamic_cheatsheet_rs_optional",
}

MEMORY_BASELINES = WRITING_BASELINES | {"retrieval_rag"}

def validate_config(path: Path) -> None:
    config = load_config(path)
    _validate_run_config(config)
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


def _memory_entries_for_arm(arm: str, baseline: str) -> tuple[list[MemoryEntry], FilterTelemetry | None]:
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
        return filter_legacy_replay_entries(entries)
    return entries, None


def _retrieved_memory(baseline: str, task_input: dict[str, Any], memory: MemoryState) -> tuple[list[dict[str, Any]], list[float]]:
    if baseline not in {"retrieval_rag", "bot_style"}:
        return [], []
    retrieved = retrieve_records(str(task_input), memory.entries, k=1 if baseline == "bot_style" else 3)
    return [record["memory_entry"].model_dump() for record in retrieved], [record["score"] for record in retrieved]


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_STAGES = {"debug", "replay", "partial", "pilot", "main", "benchmark"}


def _validate_protocol_gates(config: dict[str, Any]) -> None:
    run_config = config.get("run", {})
    live_smoke_enabled = config.get("live_smoke", {}).get("enabled", False)
    stage = run_config.get("stage", "pilot" if live_smoke_enabled else "replay")
    execution_class = run_config.get(
        "execution_class",
        "live" if live_smoke_enabled else "offline_contract_replay",
    )
    if _is_strict_config(config) and live_smoke_enabled:
        return
    try:
        provider_config = ProviderConfig.from_run_config(config)
        if stage in {"replay", "pilot", "main"}:
            validate_provider_selection(
                provider_config,
                stage=stage,
                execution_class=execution_class,
            )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    scientific_result = run_config.get("scientific_result", False)
    scientific_gate_id = run_config.get("scientific_gate_id")
    if not isinstance(scientific_result, bool):
        raise SystemExit("run.scientific_result must be a boolean")
    if scientific_gate_id is not None and (
        not isinstance(scientific_gate_id, str) or not scientific_gate_id.strip()
    ):
        raise SystemExit("run.scientific_gate_id must be a non-empty string or null")
    if scientific_result:
        if stage == "replay":
            raise SystemExit("replay runs cannot be scientific results")
        if stage not in {"pilot", "main"} or scientific_gate_id is None:
            raise SystemExit("scientific live runs require run.scientific_gate_id")
        raise SystemExit("scientific runs require an accepted scientific gate")
    elif scientific_gate_id is not None:
        raise SystemExit("run.scientific_gate_id requires run.scientific_result=true")

    if stage in {"pilot", "main"} and any(arm != "clean" for arm in config.get("arms", [])):
        raise SystemExit("pilot/main runs accept only the clean arm")


def _is_strict_config(config: dict[str, Any]) -> bool:
    return (
        config.get("logging", {}).get("schema_version") in {LOGGING_V1, LOGGING_V2}
        and _is_faithful_config(config)
    )


def _hash_values(values: list[str]) -> str:
    payload = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sample_hashes(config: dict[str, Any]) -> tuple[str, str]:
    sample_order = [
        str(row["sample_id"])
        for task_config in config["tasks"]
        for row in _load_jsonl(Path(task_config["sample_path"]), task_config.get("limit"))
    ]
    return _hash_values(sorted(sample_order)), _hash_values(sample_order)


def _run_metadata(config: dict[str, Any], run_id: str, run_started_at: str) -> RunMetadata:
    run_config = config["run"]
    logging_config = config["logging"]
    replay_config = config.get("replay", {})
    sample_set_hash, sample_order_hash = _sample_hashes(config)
    return RunMetadata(
        run_metadata_id=f"{run_id}:metadata",
        run_id=run_id,
        git_commit=_git_commit(),
        config_hash=_config_hash(config),
        provider=f"{run_config['provider']}:{run_config['provider_profile_id']}",
        model_snapshots=dict(run_config["model_snapshots"]),
        query_date=run_started_at[:10],
        start_date=run_started_at[:10],
        seed=run_config.get("sample_order_seed"),
        order=run_config.get("task_order_seed", "task-sample-baseline-arm-model"),
        decoding_defaults={
            key: replay_config[key]
            for key in ("temperature", "top_p", "max_tokens")
            if key in replay_config
        },
        sample_set_hash=sample_set_hash,
        sample_order_hash=sample_order_hash,
        stage=run_config["stage"],
        schema_version=logging_config["schema_version"],
        contract_level=run_config.get("contract_level", "phase10"),
        evaluation_law=config.get("evaluation"),
        target_contamination_set=config.get("target_contamination_set"),
        prompt_version=logging_config["prompt_version"],
        memory_policy_version=logging_config["memory_policy_version"],
        contamination_catalog_version=logging_config["contamination_catalog_version"],
        retry_policy_version=run_config["retry_policy_version"],
    )


def _validate_run_config(config: dict[str, Any]) -> None:
    _validate_placeholder_main_config(config)
    run_config = config.get("run", {})
    stage = run_config.get("stage", "replay")
    if stage not in _STAGES:
        raise SystemExit(f"unsupported run.stage: {stage}")
    _validate_protocol_gates(config)

    faithful = _is_faithful_config(config)
    if not _is_strict_config(config):
        if not faithful and stage not in {"debug", "replay"}:
            raise SystemExit("legacy run.mode is limited to debug or replay")
        return

    schema_version = config.get("logging", {}).get("schema_version")
    if "stage" not in run_config:
        raise SystemExit(f"{schema_version} requires run.stage")
    if not faithful:
        raise SystemExit(f"{schema_version} requires run.mode=faithful")
    if config.get("live_smoke", {}).get("enabled", False):
        raise SystemExit("strict offline configs require live_smoke.enabled=false")
    if schema_version == LOGGING_V2:
        _validate_phase11_config_sections(config)

    required_versions = {
        "logging.prompt_version": config.get("logging", {}).get("prompt_version"),
        "logging.memory_policy_version": config.get("logging", {}).get("memory_policy_version"),
        "logging.contamination_catalog_version": config.get("logging", {}).get(
            "contamination_catalog_version"
        ),
        "run.retry_policy_version": run_config.get("retry_policy_version"),
    }
    unresolved = [
        key
        for key, value in required_versions.items()
        if not isinstance(value, str) or not value or value.lower() in {"unknown", "todo"}
    ]
    if unresolved:
        raise SystemExit(f"{schema_version} requires resolved versions: {', '.join(unresolved)}")

    provider = run_config.get("provider")
    snapshots = run_config.get("model_snapshots")
    if not isinstance(provider, str) or not provider:
        raise SystemExit(f"{schema_version} requires run.provider")
    if not isinstance(snapshots, dict):
        raise SystemExit(f"{schema_version} requires run.model_snapshots")
    for model in config["models"]:
        snapshot = snapshots.get(model)
        if not isinstance(snapshot, str) or not snapshot or snapshot.lower() in {"unknown", "todo"}:
            raise SystemExit(f"{schema_version} requires resolved snapshot for model: {model}")
    for task in config["tasks"]:
        limit = task.get("limit")
        if not isinstance(limit, int) or limit <= 0:
            raise SystemExit(f"{schema_version} requires positive task limit: {task.get('name', 'unknown')}")

    if schema_version == LOGGING_V2:
        load_corpus(Path(_corpus_path(config)))


def _validate_phase11_config_sections(config: dict[str, Any]) -> None:
    run_config = config.get("run", {})
    if run_config.get("contract_level") != "phase11":
        raise SystemExit("logging_v2 requires run.contract_level=phase11")
    _validate_typed_config_section(
        "evaluation", config.get("evaluation"), EvaluationLawSpec
    )
    _validate_typed_config_section(
        "target_contamination_set",
        config.get("target_contamination_set"),
        TargetContaminationSetSpec,
    )
    _validate_phase11_runtime_config(config)


def _validate_phase11_runtime_config(config: dict[str, Any]) -> None:
    if "update_mode" in config.get("memory", {}):
        raise SystemExit("logging_v2 does not accept memory.update_mode; use evaluation.regime")
    regime = config["evaluation"]["regime"]
    checkpoint_ref = config.get("checkpoint_ref")
    if regime == "online":
        if checkpoint_ref is not None:
            raise SystemExit("online logging_v2 configs must not set checkpoint_ref")
        return
    if regime != "frozen":
        return
    invalid = sorted(set(config.get("baselines", [])) & WRITING_BASELINES)
    if invalid:
        raise SystemExit(f"frozen logging_v2 rejects memory-writing baselines: {', '.join(invalid)}")
    unsupported = sorted(set(config.get("baselines", [])) - {"no_memory", "retrieval_rag"})
    if unsupported:
        raise SystemExit(f"frozen logging_v2 supports only no_memory and retrieval_rag: {', '.join(unsupported)}")
    if checkpoint_ref is None:
        raise SystemExit("frozen logging_v2 requires checkpoint_ref")
    try:
        CheckpointRef.model_validate(checkpoint_ref)
    except ValueError as exc:
        errors = getattr(exc, "errors", lambda: [])()
        if errors:
            loc = ".".join(str(part) for part in errors[0].get("loc", ()))
            raise SystemExit(f"invalid checkpoint_ref.{loc}: {errors[0].get('msg')}") from exc
        raise SystemExit(f"invalid checkpoint_ref: {exc}") from exc


def _validate_typed_config_section(section: str, value: Any, model: Any) -> None:
    if value is None:
        raise SystemExit(f"logging_v2 requires {section}")
    try:
        model.model_validate(value)
    except ValueError as exc:
        errors = getattr(exc, "errors", lambda: [])()
        if errors:
            loc = ".".join(str(part) for part in errors[0].get("loc", ()))
            raise SystemExit(f"invalid {section}.{loc}: {errors[0].get('msg')}") from exc
        raise SystemExit(f"invalid {section}: {exc}") from exc


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
    return ContaminationExposure.model_validate(
        {
            "condition": cast(Literal["clean", "contaminated", "contaminated_filter"], arm),
            "is_exposed": is_exposed,
            "source_entry_ids": source_ids,
            "contamination_types": contamination_types,
            "memory_before_entry_ids": memory_before_ids,
            "retrieved_entry_ids": retrieved_ids,
            "exposure_mode": exposure_mode,
            "reason": reason,
        }
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


def _validate_reflexion_config(config: dict[str, Any]) -> None:
    reflexion = config.get("reflexion")
    if reflexion is not None:
        max_attempts = reflexion.get("max_attempts")
        if max_attempts not in {1, 2}:
            raise SystemExit("reflexion.max_attempts must be 1 or 2")


def _validate_placeholder_main_config(config: dict[str, Any]) -> None:
    if config.get("run", {}).get("stage") != "main":
        return
    unresolved_limits = [
        task.get("name", "unknown")
        for task in config.get("tasks", [])
        if task.get("limit") == "TODO"
    ]
    unresolved_snapshots = [
        model
        for model, snapshot in config.get("run", {}).get("model_snapshots", {}).items()
        if snapshot == "TODO"
    ]
    if unresolved_limits or unresolved_snapshots:
        parts = []
        if unresolved_limits:
            parts.append(f"unresolved task limits: {', '.join(unresolved_limits)}")
        if unresolved_snapshots:
            parts.append(f"unresolved model snapshots: {', '.join(unresolved_snapshots)}")
        raise SystemExit("; ".join(parts))


def _valid_arms_for_baseline(baseline: str, requested_arms: list[str]) -> list[str]:
    if baseline == "no_memory":
        return [arm for arm in requested_arms if arm == "clean"]
    return list(requested_arms)


def _is_faithful_config(config: dict[str, Any]) -> bool:
    valid_pairs = [
        (baseline, arm)
        for baseline in config.get("baselines", [])
        for arm in _valid_arms_for_baseline(baseline, config.get("arms", []))
    ]
    if not valid_pairs:
        raise SystemExit("config has zero valid baseline×arm combinations")
    run_mode = config.get("run", {}).get("mode")
    if run_mode == "faithful":
        _validate_reflexion_config(config)
        return True
    if run_mode == "legacy":
        if {"retrieval_rag", "bot_style"}.intersection(config.get("baselines", [])):
            raise SystemExit("legacy run.mode does not support retrieval_rag or bot_style")
        return False
    if run_mode is not None:
        raise SystemExit(f"unsupported run.mode: {run_mode}")
    if (
        {"retrieval_rag", "bot_style"}.intersection(config.get("baselines", []))
        and config.get("arms", []) == ["clean"]
    ):
        return True
    return bool(config.get("embedding", {}).get("corpus_path") and config.get("bot_state"))


def _corpus_path(config: dict[str, Any]) -> str:
    if "corpus_path" in config.get("memory", {}):
        return config["memory"]["corpus_path"]
    if "corpus_path" in config.get("embedding", {}):
        return config["embedding"]["corpus_path"]
    raise SystemExit("faithful config requires memory.corpus_path or embedding.corpus_path")


def _records_for_baseline(
    records: list[CorpusRecord], task_name: str, baseline: str
) -> list[CorpusRecord]:
    return [
        record
        for record in records
        if record.task == task_name
        and (not record.target_baselines or baseline in record.target_baselines)
    ]


def _apply_phase11_target_metadata(config: dict[str, Any], entries: list[MemoryEntry]) -> None:
    if config.get("logging", {}).get("schema_version") != LOGGING_V2:
        return
    target_set = config["target_contamination_set"]
    included = set(target_set["included_classes"])
    require_exact = target_set["require_exact_lineage"]
    for entry in entries:
        contamination_class = entry.metadata.get("contamination_class")
        entry.metadata["target_set_id"] = target_set["target_set_id"]
        entry.metadata["is_target_contamination"] = (
            contamination_class in included
            and (not require_exact or entry.metadata.get("lineage_status") == "exact")
        )


def _bot_injection_entries(
    records: list[CorpusRecord], task_name: str, arm: str
) -> tuple[list[MemoryEntry], FilterTelemetry | None]:
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
    contamination_class = template.metadata.get("contamination_class")
    return MemoryEntry(
        entry_id=template.entry_id,
        content=template.content,
        memory_type="thought_template",
        clean_or_contaminated=(
            "clean" if contamination_class in {None, "clean"} else "contaminated"
        ),
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


class _ReplayBotSolveCompatibilityClient:
    """Adapt the locked legacy replay solve fixture, never a live response."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def chat(self, messages: list[dict[str, str]], model: str, config: dict[str, Any]) -> LLMResponse:
        response = self._client.chat(messages, model, config)
        stage = config.get("method_stage")
        if stage == "bot_problem_distill" and not response.content.lstrip().startswith("{"):
            content = json.dumps(
                {
                    "key_information": "legacy replay fixture",
                    "restrictions": "Follow the task constraints.",
                    "distilled_task": "Solve the current task.",
                }
            )
        elif stage == "bot_thought_distill" and not response.content.lstrip().startswith("{"):
            content = json.dumps(
                {
                    "description": "legacy replay template",
                    "template": response.content.strip(),
                    "category": "procedure-based",
                    "explicitly_used_memory_ids": config.get("visible_memory_ids", [])[:1],
                }
            )
        elif stage == "bot_instantiate_solve" and response.content.lower().startswith("final:"):
            content = json.dumps({"solution_trace": "legacy replay", "final_answer": response.content})
        elif stage == "reflexion_reflect" and not response.content.lstrip().startswith("{"):
            content = json.dumps(
                {
                    "mode": "corrective",
                    "failure_class": "incorrect_answer",
                    "reflection_text": response.content.strip(),
                    "explicitly_used_memory_ids": config.get("visible_memory_ids", [])[:1],
                }
            )
        else:
            return response
        return LLMResponse(
            content=content,
            raw=response.raw,
            token_usage=response.token_usage,
            latency_ms=response.latency_ms,
        )


def _legacy_method_call_messages(method_calls: list[Any]) -> list[dict[str, str]]:
    return [message for call in method_calls for message in call.messages]


def _trial_id(run_id: str, task: Any, baseline: str, arm: str, model: str) -> str:
    return ":".join([run_id, task.task_name, task.sample_id, baseline, arm, model])


def _phase11_trial_context(
    config: dict[str, Any], task: Any, baseline: str, model: str, checkpoint_index: int
) -> dict[str, Any]:
    if config.get("logging", {}).get("schema_version") != LOGGING_V2:
        return {}
    trajectory_values = [
        str(config.get("run", {}).get("sample_order_seed")),
        str(config.get("run", {}).get("task_order_seed")),
        task.task_name,
        baseline,
        model,
    ]
    checkpoint_ref = config.get("checkpoint_ref")
    if checkpoint_ref is not None:
        trajectory_values.append(str(checkpoint_ref.get("checkpoint_id")))
    trajectory_pair_id = "traj:" + _hash_values(trajectory_values)[:16]
    memory_update_mode = _phase11_memory_update_mode(config, baseline)
    return {
        "evaluation_law_id": config["evaluation"]["evaluation_law_id"],
        "target_set_id": config["target_contamination_set"]["target_set_id"],
        "memory_update_mode": memory_update_mode,
        "trajectory_pair_id": trajectory_pair_id,
        "checkpoint_index": checkpoint_index,
        "pair_id": ":".join([trajectory_pair_id, str(checkpoint_index), task.sample_id]),
        "checkpoint_ref": (
            CheckpointRef.model_validate(checkpoint_ref)
            if memory_update_mode == "disabled" and checkpoint_ref is not None
            else None
        ),
    }


def _phase11_memory_update_mode(config: dict[str, Any], baseline: str) -> str:
    if baseline == "no_memory":
        return "not_applicable"
    if config["evaluation"]["regime"] == "frozen":
        return "disabled"
    if baseline in MEMORY_BASELINES:
        return "enabled"
    return "not_applicable"


def _outcome_result_dict(outcome: BaselineExecutionOutcome) -> dict[str, Any]:
    return {
        "status": outcome.status,
        "final_response": outcome.final_response,
        "parsed_answer": outcome.parsed_answer,
        "verifier_result": outcome.verifier_result,
        "answer_call_id": outcome.answer_call_id,
        "method_calls": list(outcome.method_calls),
        "memory_before": list(outcome.memory_before),
        "memory_after": list(outcome.memory_after),
        "retrieved_memory": list(outcome.retrieved_memory),
        "retrieved_scores": list(outcome.retrieved_scores),
        "memory_write_event": outcome.memory_write_event,
        "error_type": outcome.error_type,
        "failure_disposition": outcome.failure_disposition,
        "scientific_ineligibility_reason": outcome.scientific_ineligibility_reason,
        "metadata": dict(outcome.metadata),
    }


def _reject_frozen_memory_drift(result: dict[str, Any], phase11_context: dict[str, Any]) -> None:
    if phase11_context.get("memory_update_mode") != "disabled":
        return
    if result.get("memory_write_event") is not None or result.get("memory_before", []) != result.get(
        "memory_after", []
    ):
        raise SystemExit("frozen logging_v2 trial changed memory")


def _event_context(metadata: RunMetadata, trial_id: str, trial_seq: int) -> dict[str, Any]:
    return {
        "run_metadata_id": metadata.run_metadata_id,
        "run_id": metadata.run_id,
        "trial_id": trial_id,
        "trial_seq": trial_seq,
        "event_seq": 0,
        "stage": metadata.stage,
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _filter_event(
    metadata: RunMetadata,
    trial_id: str,
    trial_seq: int,
    baseline: str,
    arm: str,
    telemetry: FilterTelemetry,
    action: Literal["apply", "outcome"],
    *,
    final_answer_source_ids: list[str] | None = None,
    verdict: str | None = None,
) -> FilterEvent:
    decisions = telemetry["decisions"]
    return FilterEvent(
        filter_id="",
        **_event_context(metadata, trial_id, trial_seq),
        arm=cast(Literal["clean", "contaminated", "contaminated_filter"], arm),
        baseline=baseline,
        decisions=[dict(decision) for decision in decisions],
        kept_source_ids=telemetry["kept_source_ids"],
        removed_source_ids=telemetry["removed_source_ids"],
        pre_source_ids=telemetry["input_source_ids"],
        post_source_ids=telemetry["kept_source_ids"],
        ground_truth_contaminated_ids=[
            decision["entry_id"] for decision in decisions if decision["ground_truth"] == "contaminated"
        ],
        action=action,
        final_answer_source_ids=final_answer_source_ids or [],
        verdict=verdict,
        created_at=_timestamp(),
    )


def _answer_call(method_calls: list[Any], answer_call_id: str) -> Any:
    answer_call = next((call for call in method_calls if call.call_id == answer_call_id), None)
    if answer_call is None:
        raise RuntimeError("faithful result has no answer call")
    return answer_call


def _failure_location(exc: Exception) -> tuple[str | None, str | None, int | None]:
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return None, None, None
    frame = frames[-1]
    path = Path(frame.filename)
    try:
        module = str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        module = path.name
    return frame.name, module, frame.lineno


def _failure_origin(call_events: list[CallEvent], exc: Exception) -> Literal[
    "provider_call", "parser", "verifier", "runner"
]:
    if call_events and call_events[-1].origin == "provider_call":
        return "provider_call"
    names = " ".join(frame.name.lower() for frame in traceback.extract_tb(exc.__traceback__))
    if "verify" in names:
        return "verifier"
    if "parse" in names:
        return "parser"
    return "runner"


def _failure_event(
    metadata: RunMetadata,
    trial_id: str,
    trial_seq: int,
    call_events: list[CallEvent],
    exc: Exception,
) -> FailureEvent:
    origin = _failure_origin(call_events, exc)
    function, module, line = _failure_location(exc)
    if origin == "provider_call" and call_events:
        call = call_events[-1]
        function = call.failure_function or function
        module = call.failure_module or module
        line = call.failure_line if call.failure_line is not None else line
    return FailureEvent(
        failure_id="",
        **_event_context(metadata, trial_id, trial_seq),
        origin=origin,
        error_type=type(exc).__name__,
        failure_function=function,
        failure_module=module,
        failure_line=line,
        retry_count=summarize_calls(call_events)["retry_count"],
        disposition="continued",
        created_at=_timestamp(),
    )


def _outcome_failure_event(
    metadata: RunMetadata,
    trial_id: str,
    trial_seq: int,
    call_events: list[CallEvent],
    result: dict[str, Any],
) -> FailureEvent:
    error_type = result.get("error_type")
    disposition = result.get("failure_disposition")
    if not isinstance(error_type, str) or not isinstance(disposition, str):
        raise RuntimeError("failed baseline outcome is missing its closed failure triple")
    origin = cast(Literal["provider_call", "parser", "verifier", "runner"], {
        "ProviderCallFailure": "provider_call",
        "BaselineOutputError": "parser",
        "VerifierContractError": "verifier",
    }.get(error_type, "runner"))
    return FailureEvent(
        failure_id="",
        **_event_context(metadata, trial_id, trial_seq),
        origin=origin,
        error_type=error_type,
        failure_function=None,
        failure_module=None,
        failure_line=None,
        retry_count=summarize_calls(call_events)["retry_count"],
        disposition=disposition,
        created_at=_timestamp(),
    )


def _write_unrecorded_calls(
    writer: RunLogWriter,
    trial_id: str,
    trial_seq: int,
    result: dict[str, Any],
    call_events: list[CallEvent],
) -> None:
    if call_events:
        return
    old_answer_call_id = result.get("answer_call_id")
    call_id_map: dict[str | None, str] = {}
    method_calls = list(result.get("method_calls", []))
    normalized_calls = []
    for index, call in enumerate(method_calls, start=1):
        call_id = f"{trial_id}:call:{index}"
        call_id_map[call.call_id] = call_id
        normalized = call.model_copy(update={"call_id": call_id})
        normalized_calls.append(normalized)
        call_events.append(
            writer.write_call(
                CallEvent(
                    call_id=call_id,
                    **_event_context(writer.run_metadata, trial_id, trial_seq),
                    method_stage=normalized.stage,
                    messages=normalized.messages,
                    model=normalized.model,
                    decoding_params={
                        key: value
                        for key, value in {
                            "temperature": normalized.temperature,
                            "top_p": normalized.top_p,
                            "max_tokens": normalized.max_tokens,
                        }.items()
                        if value is not None
                    },
                    response_text=normalized.raw_response,
                    token_usage=normalized.token_usage,
                    latency_ms=normalized.latency_ms,
                    retry_count=normalized.retry_count,
                    source_spans=normalized.source_spans,
                    created_at=_timestamp(),
                    error_type=normalized.error_type,
                )
            )
        )
    result["method_calls"] = normalized_calls
    if old_answer_call_id in call_id_map:
        result["answer_call_id"] = call_id_map[old_answer_call_id]


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
    filter_decision: FilterTelemetry | None,
    trial_order: int,
    run_started_at: str,
    repeated_failure_tracker: _RepeatedFailureTracker,
    run_metadata: RunMetadata | None = None,
    call_events: list[CallEvent] | None = None,
    trial_id: str | None = None,
    phase11_context: dict[str, Any] | None = None,
) -> TrialLog:
    trial_id = trial_id or _trial_id(run_id, task, baseline, arm, model)
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
    raw_response = result["final_response"]
    if run_metadata is None:
        metadata = {
            **_trial_metadata(config, model, trial_order, run_started_at),
            **result.get("metadata", {}),
        }
        contamination_exposure = _contamination_exposure(
            arm, result.get("memory_before", []), retrieved_memory
        )
        prompt_messages = _legacy_method_call_messages(method_calls)
        telemetry = {"latency_ms": None, "token_usage": {}, "retry_count": 0}
        schema_version = "legacy"
        stage = "legacy"
        status = "legacy"
        run_metadata_id = None
        strict_trial_seq = None
        answer_call_id = result.get("answer_call_id")
    else:
        answer_call_id = result.get("answer_call_id")
        if not isinstance(answer_call_id, str):
            raise RuntimeError("faithful result has no answer_call_id")
        answer_call = _answer_call(method_calls, answer_call_id)
        raw_response = answer_call.raw_response
        prompt_messages = answer_call.messages
        if run_metadata.schema_version == LOGGING_V2:
            assert run_metadata.target_contamination_set is not None
            contamination_exposure = compute_exposure_from_spans_v2(
                answer_call_id,
                answer_call.source_spans,
                cast(Literal["clean", "contaminated", "contaminated_filter"], arm),
                result.get("memory_before", []),
                run_metadata.target_contamination_set,
            )
        else:
            contamination_exposure = compute_exposure_from_spans(
                answer_call_id,
                answer_call.source_spans,
                cast(Literal["clean", "contaminated", "contaminated_filter"], arm),
            )
        telemetry = summarize_calls(call_events or [])
        metadata = result.get("metadata", {})
        schema_version = LOGGING_V1
        stage = run_metadata.stage
        status = "succeeded"
        run_metadata_id = run_metadata.run_metadata_id
        strict_trial_seq = trial_order
    phase11_context = phase11_context or {}
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
        prompt_messages=prompt_messages,
        memory_before=result.get("memory_before", []),
        retrieved_memory=retrieved_memory,
        retrieved_scores=retrieved_scores,
        raw_response=raw_response,
        parsed_answer=result.get("parsed_answer"),
        verifier_result=verifier_result,
        metadata=metadata,
        filter_decision=cast(dict[str, Any] | None, filter_decision),
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
        latency_ms=telemetry["latency_ms"],
        token_usage=telemetry["token_usage"],
        retry_count=telemetry["retry_count"],
        method_calls=method_calls,
        answer_call_id=answer_call_id,
        schema_version=(
            cast(Literal["legacy", "logging_v1", "logging_v2"], run_metadata.schema_version)
            if run_metadata is not None
            else schema_version
        ),
        stage=stage,
        status=status,
        run_metadata_id=run_metadata_id,
        trial_seq=strict_trial_seq,
        event_seq=0 if run_metadata is not None else None,
        **phase11_context,
    )


def _failed_faithful_trial(
    *,
    metadata: RunMetadata,
    trial_id: str,
    trial_seq: int,
    task: Any,
    baseline: str,
    arm: str,
    model: str,
    method_calls: list[Any],
    call_events: list[CallEvent],
    memory_before: list[dict[str, Any]],
    memory_after: list[dict[str, Any]],
    filter_decision: FilterTelemetry | None,
    failure_id: str,
    error_type: str,
    failure_disposition: str | None = None,
    scientific_ineligibility_reason: str | None = None,
    phase11_context: dict[str, Any] | None = None,
    memory_write_event: dict[str, Any] | None = None,
) -> TrialLog:
    if not method_calls:
        raise RuntimeError("failed faithful trial has no provider call")
    answer_stages = {
        "no_memory": {"no_memory_generate"},
        "retrieval_rag": {"rag_generate"},
        "full_history": {"full_history_generate"},
        "reflexion_style": {"reflexion_generate"},
        "bot_style": {"bot_instantiate_solve"},
        "dynamic_cheatsheet_optional": {"dynamic_cheatsheet_generate"},
        "dynamic_cheatsheet_rs_optional": {"dc_rs_generate"},
    }
    answer_call = next(
        (call for call in reversed(method_calls) if call.stage in answer_stages[baseline]), None
    )
    if answer_call is None:
        raise RuntimeError("failed faithful trial has no answer-stage provider call")
    if not isinstance(answer_call.call_id, str):
        raise RuntimeError("failed faithful trial has no answer call id")
    answer_call_id = answer_call.call_id
    if metadata.schema_version == LOGGING_V2:
        assert metadata.target_contamination_set is not None
        exposure = compute_exposure_from_spans_v2(
            answer_call_id,
            answer_call.source_spans,
            cast(Literal["clean", "contaminated", "contaminated_filter"], arm),
            memory_before,
            metadata.target_contamination_set,
        )
    else:
        exposure = compute_exposure_from_spans(
            answer_call_id,
            answer_call.source_spans,
            cast(Literal["clean", "contaminated", "contaminated_filter"], arm),
        )
    phase11_context = phase11_context or {}
    telemetry = summarize_calls(call_events)
    failure_metadata = {}
    if failure_disposition is not None or scientific_ineligibility_reason is not None:
        if not isinstance(failure_disposition, str) or not isinstance(scientific_ineligibility_reason, str):
            raise RuntimeError("failed baseline outcome is missing its closed failure triple")
        validate_failure_triple(
            cast(ErrorType, error_type),
            cast(FailureDisposition, failure_disposition),
            cast(ScientificIneligibilityReason, scientific_ineligibility_reason),
        )
        failure_metadata = {
            "failure_disposition": failure_disposition,
            "scientific_ineligibility_reason": scientific_ineligibility_reason,
        }
    return TrialLog(
        trial_id=trial_id,
        run_id=metadata.run_id,
        task_name=task.task_name,
        sample_id=task.sample_id,
        baseline=baseline,
        arm=cast(Literal["clean", "contaminated", "contaminated_filter"], arm),
        backbone=model,
        input=task.input,
        gold_or_verifier_spec=task.verifier_spec,
        prompt_messages=answer_call.messages,
        memory_before=memory_before,
        retrieved_memory=[],
        retrieved_scores=[],
        filter_decision=cast(dict[str, Any] | None, filter_decision),
        raw_response=None,
        parsed_answer=None,
        verifier_result=None,
        metadata=failure_metadata,
        memory_write_event=memory_write_event,
        memory_after=memory_after,
        method_calls=method_calls,
        contamination_exposure=exposure,
        bad_memory_uptake_label="not_evaluable" if arm != "clean" else "not_applicable",
        repeated_failure_label="not_applicable",
        recovery_after_filter_label="not_applicable",
        latency_ms=telemetry["latency_ms"],
        token_usage=telemetry["token_usage"],
        retry_count=telemetry["retry_count"],
        error_type=error_type,
        schema_version=cast(Literal["legacy", "logging_v1", "logging_v2"], metadata.schema_version),
        stage=metadata.stage,
        status="failed",
        run_metadata_id=metadata.run_metadata_id,
        trial_seq=trial_seq,
        event_seq=0,
        answer_call_id=answer_call_id,
        failure_id=failure_id,
        **phase11_context,
    )


def _method_calls_from_events(call_events: list[CallEvent]) -> list[MethodCall]:
    return [
        MethodCall(
            call_id=event.call_id,
            stage=event.method_stage,
            messages=event.messages,
            raw_response=event.response_text,
            model=event.model,
            temperature=event.decoding_params.get("temperature"),
            top_p=event.decoding_params.get("top_p"),
            max_tokens=event.decoding_params.get("max_tokens"),
            latency_ms=event.latency_ms,
            token_usage=event.token_usage,
            retry_count=event.retry_count,
            error_type=event.error_type,
            source_spans=event.source_spans,
        )
        for event in call_events
    ]


def _snapshot_memory_entries(snapshot: list[dict[str, Any]]) -> list[MemoryEntry]:
    entries: list[MemoryEntry] = []
    for entry in snapshot:
        if "memory_type" in entry:
            entries.append(MemoryEntry.model_validate(entry))
            continue
        entries.append(
            MemoryEntry(
                entry_id=entry["entry_id"],
                content=entry["content"],
                memory_type="thought_template",
                clean_or_contaminated=(
                    "clean"
                    if entry.get("metadata", {}).get("contamination_class") in {None, "clean"}
                    else "contaminated"
                ),
                source_trial_id=entry.get("source_trial_id"),
                metadata=entry.get("metadata", {}),
            )
        )
    return entries


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
    run_metadata: RunMetadata | None = None,
    run_dir: Path | None = None,
    audit_dir: Path | None = None,
) -> None:
    writer: RunLogWriter | None = None
    trial_file = None
    if run_metadata is not None:
        if run_dir is None:
            raise ValueError("strict faithful run requires final run directory")
        writer = RunLogWriter(run_dir, run_metadata)
        if audit_dir is not None:
            for filename in ("provider_profile.json", "resolved_config.json"):
                (audit_dir / filename).replace(writer.temp_dir / filename)
            audit_dir.rmdir()
    else:
        trials_path.parent.mkdir(parents=True, exist_ok=True)
        trial_file = trials_path.open("w", encoding="utf-8")

    try:
        corpus_records = (
            load_corpus(Path(_corpus_path(config)))
            if "corpus_path" in config.get("memory", {})
            or "corpus_path" in config.get("embedding", {})
            else []
        )
        needs_embedding = any(
            baseline in {"retrieval_rag", "bot_style", "dynamic_cheatsheet_rs_optional"}
            for baseline in config["baselines"]
        )
        needs_bot = "bot_style" in config["baselines"]
        embedding_provider: EmbeddingProvider | None = None
        cache_dir: Path | None = None
        if needs_embedding:
            embedding_provider = (
                FakeEmbeddingProvider()
                if not config.get("embedding") and config.get("arms", []) == ["clean"]
                else _embedding_provider(config)
            )
            cache_dir = Path(
                config.get("embedding", {}).get("cache_path", "data/embedding_cache")
            ) / run_id

        run_state: RunState | None = None
        bot_runtime: BotRuntime | None = None
        bot_buffers: dict[BotBufferIdentity, list[MemoryEntry]] | None = None
        if needs_bot:
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
            bot_buffers = {}

        transcript_states: dict[tuple[str, ...], list[MemoryEntry]] = {}
        reflection_states: dict[tuple[str, ...], list[MemoryEntry]] = {}
        cheatsheet_states: dict[tuple[str, ...], list[MemoryEntry]] = {}
        dc_rs_states: dict[tuple[str, ...], list[MemoryEntry]] = {}
        filter_decisions: dict[tuple[str, ...], FilterTelemetry | None] = {}
        trial_order = 0

        for task_config in config["tasks"]:
            task_name = task_config["name"]
            if task_name not in TASK_DISPATCH:
                raise SystemExit(f"unsupported task for replay spine: {task_name}")
            task_handler = TASK_DISPATCH[task_name]
            rows = _load_jsonl(Path(task_config["sample_path"]), task_config.get("limit"))
            if not rows:
                raise SystemExit(f'empty replay input: {task_config["sample_path"]}')
            for checkpoint_index, row in enumerate(rows):
                task = task_handler["build"](row)
                for baseline in config["baselines"]:
                    if baseline not in FAITHFUL_BASELINES:
                        raise SystemExit(f"unsupported faithful baseline: {baseline}")
                    baseline_records = _records_for_baseline(corpus_records, task_name, baseline)
                    for arm in _valid_arms_for_baseline(baseline, config["arms"]):
                        injection_entries: list[MemoryEntry] = []
                        if baseline == "retrieval_rag":
                            memory_entries, filter_decision = build_arm_corpus(
                                baseline_records, task_name, cast(Any, arm)
                            )
                            _apply_phase11_target_metadata(config, memory_entries)
                            memory = MemoryState(entries=memory_entries)
                        else:
                            injection_entries, filter_decision = _bot_injection_entries(
                                baseline_records, task_name, arm
                            )
                            _apply_phase11_target_metadata(config, injection_entries)
                            memory = MemoryState(entries=[])

                        for model in config["models"]:
                            trial_id = _trial_id(run_id, task, baseline, arm, model)
                            phase11_context = _phase11_trial_context(
                                config, task, baseline, model, checkpoint_index
                            )
                            call_events: list[CallEvent] = []
                            strict_context = (
                                _event_context(run_metadata, trial_id, trial_order)
                                if run_metadata is not None
                                else {}
                            )

                            def record_call(event: CallEvent) -> None:
                                if writer is not None:
                                    call_events.append(writer.write_call(event))

                            policy_context = {
                                **config.get("replay", {}),
                                "sample_id": task.sample_id,
                                "run_id": run_id,
                                "baseline": baseline,
                                "arm": arm,
                                "model": model,
                                "_logging_target_set_id": phase11_context.get("target_set_id"),
                                "_logging_target_contamination_set": config.get(
                                    "target_contamination_set"
                                ),
                                "_logging_trial_context": strict_context,
                                "_logging_event_callback": record_call if writer is not None else None,
                            }
                            trial_memory_before: list[dict[str, Any]] | None = None
                            if writer is not None and filter_decision is not None:
                                writer.write_filter(
                                    _filter_event(
                                        writer.run_metadata,
                                        trial_id,
                                        trial_order,
                                        baseline,
                                        arm,
                                        filter_decision,
                                        "apply",
                                    )
                                )

                            try:
                                trial_client = _trial_client_for_sample(
                                    client, responses_by_sample, task.sample_id, replay_responses
                                )
                                if config["run"]["execution_class"] == "offline_contract_replay":
                                    trial_client = _ReplayBotSolveCompatibilityClient(trial_client)
                                if baseline == "retrieval_rag":
                                    assert embedding_provider is not None
                                    assert cache_dir is not None
                                    trial_memory_before = [entry.model_dump() for entry in memory.entries]
                                    result = RetrievalRagPolicy().run(
                                        task,
                                        memory,
                                        client=trial_client,
                                        model=model,
                                        config=policy_context,
                                        top_k=config.get("embedding", {}).get("top_k"),
                                        embedding_provider=embedding_provider,
                                        cache_dir=cache_dir / task_name / arm,
                                    )
                                    verifier_result = task_handler["verify"](result["parsed_answer"], task)
                                elif baseline == "no_memory":
                                    trial_memory_before = [entry.model_dump() for entry in memory.entries]
                                    captured_verifier_result: Any = None

                                    def verify_no_memory(answer: str, seen_task: Any) -> Any:
                                        nonlocal captured_verifier_result
                                        captured_verifier_result = task_handler["verify"](answer, seen_task)
                                        return captured_verifier_result

                                    result = NoMemoryPolicy().run(
                                        task,
                                        memory,
                                        client=trial_client,
                                        model=model,
                                        config=policy_context,
                                        verifier=verify_no_memory,
                                    )
                                    verifier_result = captured_verifier_result or result["verifier_result"]
                                elif baseline == "bot_style":
                                    assert bot_runtime is not None
                                    assert run_state is not None
                                    assert bot_buffers is not None
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
                                            clean_identity = BotBufferIdentity(
                                                run_id, task.task_name, baseline, "clean", model
                                            )
                                            bot_buffers[identity] = [
                                                *copy.deepcopy(bot_buffers[clean_identity]),
                                                *copy.deepcopy(injection_entries),
                                            ]
                                    trial_memory_before = [
                                        entry.model_dump() for entry in bot_buffers[identity]
                                    ]
                                    result = _outcome_result_dict(
                                        bot_runtime.run(
                                            identity=identity,
                                            task=task,
                                            buffer_snapshot=bot_buffers[identity],
                                            client=(
                                                _ReplayBotSolveCompatibilityClient(trial_client)
                                                if config["run"]["execution_class"]
                                                == "offline_contract_replay"
                                                else trial_client
                                            ),
                                            model=model,
                                            config={
                                                **policy_context,
                                                "embedding_provider": embedding_provider,
                                                "visible_memory_ids": [
                                                    entry.entry_id for entry in bot_buffers[identity]
                                                ],
                                            },
                                            verifier=lambda response, task=task: task_handler["verify"](
                                                _parse_answer(response), task
                                            ),
                                        )
                                    )
                                    verifier_result = result["verifier_result"]
                                    if isinstance(verifier_result, bool):
                                        verifier_result = VerifierResult(
                                            is_correct=verifier_result,
                                            parsed_answer=result.get("parsed_answer"),
                                        )
                                    event = result.get("memory_write_event")
                                    if event and event.get("status") == "accepted":
                                        original_entry_id = str(event["new_entry_id"])
                                        scoped_entry_id = _scoped_bot_entry_id(original_entry_id, identity)
                                        event["new_entry_id"] = scoped_entry_id
                                        for entry in result["memory_after"]:
                                            if entry.get("entry_id") == original_entry_id:
                                                entry["entry_id"] = scoped_entry_id
                                        event["sample_id"] = trial_id
                                        run_state.register_warmup_result(identity, event)
                                        bot_buffers[identity] = [
                                            MemoryEntry(
                                            entry_id=entry["entry_id"],
                                            content=entry["content"],
                                            memory_type="thought_template",
                                            clean_or_contaminated=(
                                                "clean"
                                                if entry.get("metadata", {}).get(
                                                    "contamination_class"
                                                )
                                                in {None, "clean"}
                                                else "contaminated"
                                            ),
                                                source_trial_id=entry.get("source_trial_id"),
                                                metadata=entry.get("metadata", {}),
                                            )
                                            for entry in result["memory_after"]
                                        ]
                                else:
                                    identity = (run_id, task_name, baseline, arm, model)
                                    if baseline == "full_history":
                                        state = transcript_states
                                        policy = FullHistoryPolicy()
                                    elif baseline == "reflexion_style":
                                        state = reflection_states
                                        policy = ReflexionStylePolicy()
                                    elif baseline == "dynamic_cheatsheet_rs_optional":
                                        state = dc_rs_states
                                        policy = DynamicCheatsheetRetrievalSynthesisPolicy(
                                            embedding_provider=(
                                                embedding_provider
                                                if embedding_provider is not None
                                                else FakeEmbeddingProvider()
                                            ),
                                            cache_dir=(
                                                cache_dir / "dc_rs" / task_name / arm / model
                                                if cache_dir is not None
                                                else None
                                            ),
                                        )
                                    else:
                                        state = cheatsheet_states
                                        policy = DynamicCheatsheetOptionalPolicy()
                                    if identity not in state:
                                        entries, filter_decision = build_arm_corpus(
                                            baseline_records, task_name, cast(Any, arm)
                                        )
                                        _apply_phase11_target_metadata(config, entries)
                                        state[identity] = entries
                                        filter_decisions[identity] = filter_decision
                                    else:
                                        filter_decision = filter_decisions[identity]
                                    memory = MemoryState(entries=state[identity])
                                    trial_memory_before = [entry.model_dump() for entry in memory.entries]
                                    if baseline == "reflexion_style":
                                        policy_context["max_attempts"] = config.get(
                                            "reflexion", {}
                                        ).get("max_attempts", 1)
                                        policy_context["visible_memory_ids"] = [
                                            entry.entry_id for entry in memory.entries
                                        ]
                                    result = policy.run(
                                        task,
                                        memory,
                                        client=trial_client,
                                        model=model,
                                        config=policy_context,
                                        verifier=lambda response, task: task_handler["verify"](
                                            _parse_answer(response), task
                                        ),
                                    )
                                    verifier_result = result["verifier_result"]
                                    state[identity] = [
                                        MemoryEntry(**entry) for entry in result["memory_after"]
                                    ]
                            except Exception as exc:
                                if writer is None:
                                    raise
                                if not call_events:
                                    raise
                                failure = writer.write_failure(
                                    _failure_event(
                                        writer.run_metadata,
                                        trial_id,
                                        trial_order,
                                        call_events,
                                        exc,
                                    )
                                )
                                writer.write_trial(
                                    _failed_faithful_trial(
                                        metadata=writer.run_metadata,
                                        trial_id=trial_id,
                                        trial_seq=trial_order,
                                        task=task,
                                        baseline=baseline,
                                        arm=arm,
                                        model=model,
                                        method_calls=_method_calls_from_events(call_events),
                                        call_events=call_events,
                                        memory_before=(
                                            trial_memory_before
                                            if trial_memory_before is not None
                                            else [entry.model_dump() for entry in memory.entries]
                                        ),
                                        memory_after=[entry.model_dump() for entry in memory.entries],
                                        filter_decision=filter_decision,
                                        failure_id=failure.failure_id,
                                        error_type=type(exc).__name__,
                                        phase11_context=phase11_context,
                                    )
                                )
                                trial_order += 1
                                continue

                            if writer is not None:
                                _write_unrecorded_calls(
                                    writer, trial_id, trial_order, result, call_events
                                )
                            if result.get("status") == "failed":
                                if writer is None:
                                    raise RuntimeError("legacy adapter outcome failed")
                                failure = writer.write_failure(
                                    _outcome_failure_event(
                                        writer.run_metadata,
                                        trial_id,
                                        trial_order,
                                        call_events,
                                        result,
                                    )
                                )
                                trial = _failed_faithful_trial(
                                    metadata=writer.run_metadata,
                                    trial_id=trial_id,
                                    trial_seq=trial_order,
                                    task=task,
                                    baseline=baseline,
                                    arm=arm,
                                    model=model,
                                    method_calls=result["method_calls"],
                                    call_events=call_events,
                                    memory_before=result.get("memory_before", []),
                                    memory_after=result.get("memory_after", []),
                                    memory_write_event=result.get("memory_write_event"),
                                    filter_decision=filter_decision,
                                    failure_id=failure.failure_id,
                                    error_type=result["error_type"],
                                    failure_disposition=result["failure_disposition"],
                                    scientific_ineligibility_reason=result[
                                        "scientific_ineligibility_reason"
                                    ],
                                    phase11_context=phase11_context,
                                )
                                memory_event = normalize_memory_event(
                                    baseline,
                                    trial_id,
                                    _snapshot_memory_entries(result.get("memory_before", [])),
                                    _snapshot_memory_entries(result.get("memory_after", [])),
                                    result.get("memory_write_event"),
                                )
                                if memory_event is not None:
                                    writer.write_memory(
                                        memory_event.model_copy(
                                            update=_event_context(
                                                writer.run_metadata, trial_id, trial_order
                                            )
                                        )
                                    )
                                writer.write_trial(trial)
                                trial_order += 1
                                continue
                            _reject_frozen_memory_drift(result, phase11_context)
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
                                run_metadata=writer.run_metadata if writer is not None else None,
                                call_events=call_events,
                                trial_id=trial_id,
                                phase11_context=phase11_context,
                            )
                            if writer is not None:
                                answer_call = _answer_call(result["method_calls"], trial.answer_call_id or "")
                                final_source_ids = [
                                    source_id
                                    for span in answer_call.source_spans
                                    for source_id in (span.source_ids or [span.entry_id])
                                ]
                                if filter_decision is not None:
                                    writer.write_filter(
                                        _filter_event(
                                            writer.run_metadata,
                                            trial_id,
                                            trial_order,
                                            baseline,
                                            arm,
                                            filter_decision,
                                            "outcome",
                                            final_answer_source_ids=final_source_ids,
                                            verdict=str(verifier_result.is_correct).lower(),
                                        )
                                    )
                                memory_event = normalize_memory_event(
                                    baseline,
                                    trial_id,
                                    _snapshot_memory_entries(result["memory_before"]),
                                    _snapshot_memory_entries(result["memory_after"]),
                                    result.get("memory_write_event"),
                                )
                                if memory_event is not None:
                                    writer.write_memory(
                                        memory_event.model_copy(
                                            update=_event_context(
                                                writer.run_metadata, trial_id, trial_order
                                            )
                                        )
                                    )
                                writer.write_trial(trial)
                            else:
                                assert trial_file is not None
                                trial_file.write(trial.model_dump_json() + "\n")
                            trial_order += 1
        if writer is not None:
            writer.finalize()
    except BaseException:
        if writer is not None:
            writer.finalize(status="failed")
        raise
    finally:
        if trial_file is not None:
            trial_file.close()


def run_config(
    config: dict[str, Any], run_id: str, _client_override: LLMClient | None = None
) -> Path:
    _validate_run_id(run_id)
    _validate_run_config(config)
    provider_config = ProviderConfig.from_run_config(config)
    profile = normalize_provider_profile(
        provider_config,
        served_models=config["models"],
        model_snapshots=config.get("run", {}).get("model_snapshots", {}),
    )
    config = resolve_run_config(config, provider_profile=profile)
    run_config = config["run"]
    replay_config = config.get("replay", {})
    replay_responses = replay_config.get("responses")
    try:
        validate_provider_selection(
            provider_config,
            stage=run_config["stage"],
            execution_class=run_config["execution_class"],
        )
        client: LLMClient = (
            _client_override
            if _client_override is not None
            else build_llm_client(
                provider_config,
                stage=run_config["stage"],
                execution_class=run_config["execution_class"],
                replay_responses=replay_responses,
            )
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    output_dir = Path(config.get("logging", {}).get("output_dir", "runs"))
    run_dir = output_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"final run path already exists: {run_dir}")
    trials_path = run_dir / "trials.jsonl"

    responses_by_sample = replay_config.get("responses_by_sample", {})
    run_started_at = datetime.now(timezone.utc).isoformat()
    repeated_failure_tracker = _RepeatedFailureTracker()
    audit_dir: Path | None = None
    if _is_strict_config(config):
        output_dir.mkdir(parents=True, exist_ok=True)
        audit_dir = Path(tempfile.mkdtemp(prefix=f"{run_id}.audit-", dir=output_dir))
        write_provider_profile_atomic(audit_dir, profile)
        write_resolved_config_atomic(audit_dir, config)
    else:
        run_dir.mkdir(parents=True)
        write_provider_profile_atomic(run_dir, profile)
        write_resolved_config_atomic(run_dir, config)
    if _is_faithful_config(config):
        run_metadata = _run_metadata(config, run_id, run_started_at) if _is_strict_config(config) else None
        _run_faithful_config(
            config,
            run_id,
            client,
            replay_responses,
            responses_by_sample,
            trials_path,
            run_started_at,
            repeated_failure_tracker,
            run_metadata=run_metadata,
            run_dir=run_dir if run_metadata is not None else None,
            audit_dir=audit_dir,
        )
        print(f"wrote replay trials: {trials_path}")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
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
                    for arm in _valid_arms_for_baseline(baseline, config["arms"]):
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
                            trial_id = _trial_id(run_id, task, baseline, arm, model)
                            recorder = MethodCallRecorder(
                                trial_client,
                                trial_context={"trial_id": trial_id},
                            )
                            response = recorder.chat(
                                prompt_messages,
                                model=model,
                                config={
                                    **config.get("replay", {}),
                                    "sample_id": task.sample_id,
                                    "method_stage": (
                                        "full_history_generate"
                                        if baseline == "full_history"
                                        else "legacy_generate"
                                    ),
                                },
                            )
                            parsed_answer = _parse_answer(response.content)
                            verifier_result = task_handler["verify"](parsed_answer, task)
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
                            arm_literal = cast(Literal["clean", "contaminated", "contaminated_filter"], arm)
                            trial = TrialLog(
                                trial_id=trial_id,
                                run_id=run_id,
                                task_name=task.task_name,
                                sample_id=task.sample_id,
                                baseline=baseline,
                                arm=arm_literal,
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
                                filter_decision=cast(dict[str, Any] | None, filter_decision),
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
                                retry_count=0,
                                method_calls=recorder.get_records(),
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
    aggregate.add_argument("--stage", type=str, default=None)
    aggregate.add_argument("--allow-legacy", action="store_true")
    aggregate.add_argument("--contract", choices=["phase10", "phase11"], default=None)

    args = parser.parse_args()

    if args.command == "validate-config":
        validate_config(args.config)
    elif args.command == "run":
        run_config(load_config(args.config), args.run_id)
    elif args.command == "aggregate":
        if not args.run_dir.exists():
            raise SystemExit(f"run dir not found: {args.run_dir}")
        stage = args.stage
        if stage is None and (args.run_dir / "run.json").exists():
            stage = RunLogWriter.read_manifest(args.run_dir)["run_metadata"]["stage"]
        print(
            json.dumps(
                aggregate_run(
                    args.run_dir,
                    stage=stage,
                    allow_legacy=args.allow_legacy,
                    contract=args.contract,
                )
            )
        )


if __name__ == "__main__":
    main()
