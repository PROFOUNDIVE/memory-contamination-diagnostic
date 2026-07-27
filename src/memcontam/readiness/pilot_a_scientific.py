from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, cast

import yaml  # type: ignore[import-untyped]

from memcontam.baselines.reflexion_phase12 import ReflexionStateV3
from memcontam.clients.base import LLMClient
from memcontam.clients.config import ProviderConfig
from memcontam.clients.cost_guard import CostLimitExceeded
from memcontam.clients.factory import build_llm_client
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.experiment.phase12.contracts import BaselineConditionSpec, MemoryArmExecutionKey
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.live_branch import build_live_three_arm_branches
from memcontam.experiment.phase12.live_prefix import run_live_clean_prefix
from memcontam.experiment.phase12.live_suffix import run_live_matched_suffix
from memcontam.experiment.phase12.runtime_registry import RuntimeTrialResult
from memcontam.memory.admission import AdmissionContext
from memcontam.memory.cards_v3 import MemoryCardEnvelopeV3, canonical_content_hash
from memcontam.memory.checkpoint_v3 import NATIVE_ENTRY_V1, NativeEntry
from memcontam.readiness.pilot_a_scientific_archive import (
    validate_scientific_archive,
    write_scientific_archive,
)
from memcontam.readiness.pilot_a_scientific_records import (
    ROW_NAMES,
    artifacts,
    record_prefix,
    record_suffix,
    record_terminal_failure,
)
from memcontam.tasks.game24 import build_instance


class ScientificPilotAError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def run_scientific_pilot_a(
    config_path: Path,
    *,
    allow_live_calls: bool,
    artifact_root: Path | None = None,
    client_factory: Callable[[], object] | None = None,
    context_factory: Callable[[LLMClient, str, str], Game24RuntimeContext] | None = None,
    run_id: str | None = None,
    parent_run_id: str | None = None,
) -> dict[str, Any]:
    if not allow_live_calls:
        raise ScientificPilotAError("LIVE_CALL_AUTHORIZATION_REQUIRED")
    config = _load_config(config_path)
    provider = ProviderConfig.from_run_config(config)
    if provider.provider != "openai_responses" or not provider.live_calls_enabled:
        raise ScientificPilotAError("LIVE_CALL_CONFIG_REQUIRED")
    identity = run_id or f"pilot-a-game24-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    _validate_run_id(identity)
    if parent_run_id is not None:
        _validate_run_id(parent_run_id)
    root = artifact_root or _artifact_root()
    run_dir = root / "runs" / identity
    if run_dir.exists():
        raise ScientificPilotAError("RUN_ID_ALREADY_EXISTS")
    run_dir.mkdir(parents=True)
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ROW_NAMES}

    def seal(status: str, reason: str | None) -> None:
        write_scientific_archive(
            run_dir,
            artifacts(
                config_path,
                config,
                identity,
                provider,
                rows,
                parent_run_id=parent_run_id,
                run_status=status,
                status_reason=reason,
            ),
        )

    seal("interrupted", "not_started")
    try:
        client = cast(
            LLMClient,
            client_factory()
            if client_factory is not None
            else build_llm_client(
                provider, stage="pilot", execution_class="live", allow_live_calls=True
            ),
        )
        rows = _run_seeds(config, identity, client, provider, context_factory, rows=rows)
        if rows["seed_status"] and not any(item.get("eligible") for item in rows["seed_status"]) and not any(
            "status" in item for item in rows["seed_status"]
        ):
            raise ScientificPilotAError("JOINT_CHECKPOINT_ELIGIBILITY_EMPTY")
    except BaseException as error:
        terminal_status, status_reason = _failure_terminal_status(error)
        record_terminal_failure(rows, error, terminal_status, status_reason)
        seal(terminal_status, status_reason)
        raise

    final_status, final_status_reason = _terminal_status(rows["seed_status"])
    seal(final_status, final_status_reason)
    report = validate_scientific_archive(run_dir)
    if report["overall"] != "pass":
        raise ScientificPilotAError(str(report["reason_code"]))
    return report


def validate_scientific_pilot_a_archive(run_dir: Path) -> dict[str, Any]:
    return validate_scientific_archive(run_dir)


def _run_seeds(
    config: dict[str, Any],
    run_id: str,
    client: LLMClient,
    provider: ProviderConfig,
    context_factory: Callable[[LLMClient, str, str], Game24RuntimeContext] | None,
    *,
    rows: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    instances = _instances(config)
    registry = load_candidate_registry(Path(config["candidate_registry"]["path"]))
    for seed_spec in config["trajectory_seeds"]:
        seed = int(seed_spec["seed"])
        factory = context_factory or _default_context
        base = _mature_context(factory(client, run_id, config["provider"]["model_id"]))
        contexts = tuple(
            replace(
                base,
                task=instances[task_id],
                identities=RuntimeIdentities(run_id, f"{run_id}:seed:{seed}:prefix:{index}", index),
                initial_states=copy.deepcopy(base.initial_states),
            )
            for index, task_id in enumerate(seed_spec["ordered_prefix_task_ids"], start=1)
        )
        prefix = run_live_clean_prefix(
            seed=seed,
            contexts=contexts,
            conditions=_conditions(),
            suffix_horizon=int(config["suffix_horizon"]),
        )
        record_prefix(rows, seed, contexts, prefix, provider)
        if prefix.selection.blocked:
            continue
        selected_index = cast(int, prefix.selection.selected_trial_index)
        frozen_suffix_order = tuple(seed_spec["ordered_suffix_task_ids"])
        expected_suffix = frozen_suffix_order[
            selected_index : selected_index + int(config["suffix_horizon"])
        ]
        if tuple(task.sample_id for task in prefix.suffix_tasks) != expected_suffix:
            raise ScientificPilotAError("FROZEN_SUFFIX_MISMATCH")
        selected_context = contexts[selected_index - 1]
        branches = {
            baseline: build_live_three_arm_branches(
                prefix=checkpoint,
                context=selected_context,
                candidate_registry=registry,
                filter_policy=_admission_context(
                    baseline, prefix.trial_results_by_baseline[baseline][:selected_index]
                ),
            )
            for baseline, checkpoint in prefix.selection.selected_checkpoints.items()
        }
        suffix_contexts = tuple(
            context
            for context in contexts
            if context.task.sample_id in expected_suffix
        )
        suffix = run_live_matched_suffix(branches_by_baseline=branches, contexts=suffix_contexts)
        record_suffix(rows, seed, suffix.memory_runs, suffix.nomem.trials, branches, provider)
    return rows


def _terminal_status(seed_status: list[dict[str, Any]]) -> tuple[str, str | None]:
    if not seed_status:
        return "interrupted", "no_seed_rows"

    if any(item.get("eligible") for item in seed_status):
        return "completed", None

    blocked = [item.get("status") for item in seed_status if item.get("status") == "blocked"]
    if blocked and len(blocked) == len(seed_status):
        return "blocked", "all_joint_checkpoint_blocked"

    return "invalidated", "joint_checkpoint_eligibility_empty"


def _failure_terminal_status(error: BaseException) -> tuple[str, str]:
    if isinstance(error, CostLimitExceeded):
        return "blocked", "cost_limit_exceeded"
    if isinstance(error, KeyboardInterrupt):
        return "interrupted", "interrupted"
    if isinstance(error, ScientificPilotAError):
        return "invalidated", error.code.lower()
    return "interrupted", "provider_failure"


def _conditions() -> dict[str, BaselineConditionSpec]:
    families: dict[str, Literal["full_history", "rag", "bot", "reflexion"]] = {
        "fh_bounded": "full_history", "rag_frozen": "rag", "bot_style": "bot", "reflexion_style": "reflexion",
    }
    return {
        baseline: BaselineConditionSpec(
            condition_id=f"{baseline}-clean", baseline_family=family, fidelity_label="bounded",
            rag_mode="frozen" if family == "rag" else "not_applicable", fh_mode="bounded",
            execution_key_example=MemoryArmExecutionKey(kind="memory_arm", arm="clean"),
        )
        for baseline, family in families.items()
    }


def _instances(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["instance_registry"]["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != config["instance_registry"]["sha256"]:
        raise ScientificPilotAError("INSTANCE_REGISTRY_HASH_MISMATCH")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    instances = {row["sample_id"]: build_instance(row) for row in rows}
    used = {task for seed in config["trajectory_seeds"] for task in seed["ordered_prefix_task_ids"]}
    if not used <= set(instances) or used & set(config["instance_registry"]["reserved_main_ids"]):
        raise ScientificPilotAError("CALIBRATION_SPLIT_INVALID")
    return instances


def _mature_context(context: Game24RuntimeContext) -> Game24RuntimeContext:
    states = dict(context.initial_states)
    reflexion = states.get("reflexion_style")
    if isinstance(reflexion, ReflexionStateV3) and not reflexion.reflections:
        states["reflexion_style"] = ReflexionStateV3(
            reflections=[
                NativeEntry(
                    entry_id="pilot-a-clean-reflection-v1",
                    semantic_kind="verbal_reflection",
                    schema_version=NATIVE_ENTRY_V1,
                    native_component="reflections",
                    content="Verify arithmetic before finalizing.",
                    content_hash=canonical_content_hash("Verify arithmetic before finalizing."),
                )
            ],
            active_capacity=reflexion.active_capacity,
        )
    return replace(context, initial_states=states)


def _default_context(client: LLMClient, run_id: str, model: str) -> Game24RuntimeContext:
    from memcontam.readiness.pilot_a_launch import _live_context

    return _live_context(client, run_id, model)


def _admission_context(
    baseline: str, prefix_results: tuple[RuntimeTrialResult, ...]
) -> AdmissionContext:
    envelopes = tuple(
        envelope for result in prefix_results for envelope in result.write_envelopes
    )
    if any(not isinstance(envelope, MemoryCardEnvelopeV3) for envelope in envelopes):
        raise ScientificPilotAError("INVALID_PREFIX_WRITE_EVIDENCE")
    evidence = tuple(envelope for envelope in envelopes if isinstance(envelope, MemoryCardEnvelopeV3))
    if any(envelope.baseline != baseline for envelope in evidence):
        raise ScientificPilotAError("PREFIX_WRITE_BASELINE_MISMATCH")
    return AdmissionContext(
        writer_event_ids=frozenset(envelope.writer_event_id for envelope in evidence),
        trial_record_ids=frozenset(
            trial_id for envelope in evidence for trial_id in envelope.trial_support_ids
        ),
        evidence_envelopes=evidence,
        active_envelopes=evidence,
    )


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("config_kind") != "phase12_pilot_a_scientific_v1":
        raise ScientificPilotAError("INVALID_SCIENTIFIC_PILOT_A_CONFIG")
    if payload.get("scientific_result") is not True or payload.get("arms") != ["Clean", "Contam", "Filter"]:
        raise ScientificPilotAError("INVALID_SCIENTIFIC_PILOT_A_CONFIG")
    if [item.get("seed") for item in payload.get("trajectory_seeds", [])] != [0, 1]:
        raise ScientificPilotAError("TWO_CALIBRATION_SEEDS_REQUIRED")
    return payload


def _artifact_root() -> Path:
    value = os.environ.get("MEMCONTAM_ARTIFACT_ROOT")
    if not value:
        raise ScientificPilotAError("MEMCONTAM_ARTIFACT_ROOT_REQUIRED")
    return Path(value)


def _validate_run_id(run_id: str) -> None:
    path = Path(run_id)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ScientificPilotAError("INVALID_RUN_ID")


__all__ = ["ScientificPilotAError", "run_scientific_pilot_a", "validate_scientific_archive"]
