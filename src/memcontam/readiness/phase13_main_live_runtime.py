from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Final

from memcontam.baselines.bot_phase12 import BoTStateV3
from memcontam.baselines.retrieval_rag_phase12 import RagFrozenStateV3
from memcontam.clients.base import LLMClient
from memcontam.clients.config import ProviderConfig
from memcontam.clients.openai_responses import OpenAIResponsesClient
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.experiment.phase12.game24_runner import Branch, Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.live_branch import build_live_reduced_main_branches
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.experiment.phase13_ordinary_runtime import (
    ProspectiveOrdinaryRun,
    execute_prospective_ordinary,
)
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.memory.embeddings import BgeM3EmbeddingProvider, EmbeddingProvider
from memcontam.readiness.phase13_core_datasets import load_core_task
from memcontam.readiness.phase13_cost_policy import bind_cost_policy_client
from memcontam.readiness.phase13_legacy_rag_runtime import (
    LegacyRagRuntimeRequest,
    load_legacy_rag_state,
)
from memcontam.readiness.phase13_main_checkpoint import CommonCheckpointRegistry
from memcontam.readiness.phase13_main_live_dispatch import MainUnitDispatchOutput
from memcontam.readiness.phase13_main_live_runtime_support import (
    MainLiveRuntimeError,
    core_task_name,
    dispatch_output,
    legacy_task_name,
    production_identity,
    task_name,
    verifier,
)
from .phase13_main_new_mcq_runtime import (
    build_new_mcq_live_branches,
    load_new_mcq_runtime_registry,
)
from memcontam.readiness.phase13_main_production import ProductionObject
from memcontam.readiness.phase13_main_production_backend import (
    OrdinaryRuntimeRequest,
    PrefixRuntimeOutput,
)
from memcontam.evaluation.phase13_observability_registration import ObservabilityRegistrationPacket
from memcontam.readiness.phase13_production_observability import validate_production_archive
from memcontam.readiness.phase13_production_runtime_join import production_archive_from_ordinary
from memcontam.readiness.phase13_route_capacity import bind_capacity_configs
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.game24 import build_instance as build_game24
from memcontam.tasks.math_equation_balancer import build_instance as build_equation
from memcontam.tasks.word_sorting import build_instance as build_word_sorting


_CORE_TASKS: Final = frozenset({"mmlu_pro_engineering", "mmlu_pro_physics"})


class ProductionMainRuntime:
    def __init__(
        self,
        repository_root: Path,
        cache_root: Path,
        *,
        client: LLMClient | None = None,
    ) -> None:
        self._root = repository_root
        self._core = repository_root / "data/phase13/core/materialized"
        self._cache = cache_root
        self._embedder_instance: BgeM3EmbeddingProvider | None = None
        self._client: LLMClient = client or OpenAIResponsesClient(
            ProviderConfig(
                provider="openai_responses",
                api_key_env="OPENAI_API_KEY",
                timeout_seconds=180,
                live_calls_enabled=True,
                service_tier="default",
                store=False,
                max_output_tokens=512,
                retries_after_initial_attempt=2,
                retry_delays_seconds=(1, 2),
                input_per_million_usd=0.20,
                cached_input_per_million_usd=0.02,
                output_per_million_usd=1.20,
            ),
            allow_live_calls=True,
        )
        self._checkpoint_registry = CommonCheckpointRegistry.model_validate_json(
            (repository_root / "data/phase13/main/mr_p4/main_a_common_checkpoint_registry_v1.json")
            .read_bytes()
        )
        packet_raw = (
            repository_root / "data/phase13/observability/registration_packet_v1.json"
        ).read_bytes()
        self._packet = ObservabilityRegistrationPacket.model_validate_json(packet_raw)
        self._candidate_registry = load_candidate_registry(
            repository_root / "data/phase12/registries/candidate_registry_v1.json"
        )
        self._new_mcq_registry = load_new_mcq_runtime_registry(repository_root)

    def preflight(self, units: tuple[ProductionObject, ...]) -> None:
        checked: set[tuple[str, str]] = set()
        for unit in units:
            if unit.kind != "CLEAN_PREFIX" or unit.memory_baseline is None:
                continue
            key = (unit.task, unit.memory_baseline)
            if key in checked:
                continue
            checked.add(key)
            try:
                task = self._tasks(unit.task, unit.seed, prefix=True)[0]
                context = self._prefix_context(unit, task)
                if not context.identities.condition_id:
                    raise MainLiveRuntimeError("MAIN_PREFIX_CONDITION_ID_REQUIRED")
                entry = PHASE13_CORE_BASELINE_REGISTRY[unit.memory_baseline]
                snapshot = entry.serialize_state(entry.initial_state(context))
            except (RuntimeError, ValueError) as error:
                raise MainLiveRuntimeError(
                    f"MAIN_PREFIX_PREFLIGHT_INVALID:{error}"
                ) from error
            if not isinstance(snapshot, NativeState):
                raise MainLiveRuntimeError("MAIN_PREFIX_PREFLIGHT_CHECKPOINT_INVALID")

    def execute_prefix(self, unit: ProductionObject) -> PrefixRuntimeOutput:
        if unit.memory_baseline is None:
            raise MainLiveRuntimeError("MAIN_PREFIX_BASELINE_REQUIRED")
        task = self._tasks(unit.task, unit.seed, prefix=True)[0]
        context = self._prefix_context(unit, task)
        entry = PHASE13_CORE_BASELINE_REGISTRY[unit.memory_baseline]
        result = entry.execute_trial(context, entry.initial_state(context))
        if unit.memory_baseline == "reflexion_style" and result.outcome.status != "succeeded":
            raise MainLiveRuntimeError("MAIN_PREFIX_EXECUTION_FAILED")
        snapshot = entry.serialize_state(result.state)
        if not isinstance(snapshot, NativeState):
            raise MainLiveRuntimeError("MAIN_PREFIX_CHECKPOINT_INVALID")
        checkpoint = serialize_checkpoint(
            NativeState(
                snapshot.baseline,
                snapshot.entries,
                {**snapshot.native_state, "checkpoint_index": 1},
                snapshot.schema_version,
            )
        )
        return PrefixRuntimeOutput(checkpoint, dispatch_output(unit, (result,), production_identity(unit)))

    def _prefix_context(
        self,
        unit: ProductionObject,
        task: TaskInstance,
    ) -> Game24RuntimeContext:
        context = self._context(unit, task, "clean").for_condition(
            f"{unit.memory_baseline}-clean"
        )
        return replace(
            context,
            baseline_configs={
                **context.baseline_configs,
                "reflexion_style": {
                    **context.baseline_configs.get("reflexion_style", {}),
                    "max_attempts": 1,
                },
            },
        )

    def execute_ordinary(self, request: OrdinaryRuntimeRequest) -> MainUnitDispatchOutput:
        unit = request.unit
        tasks = self._tasks(unit.task, unit.seed, prefix=False)
        identity = production_identity(unit)
        branch = None
        if request.checkpoint is not None:
            if unit.task in _CORE_TASKS:
                branch = build_new_mcq_live_branches(
                    prefix=request.checkpoint,
                    context=self._context(
                        unit,
                        tasks[0],
                        "clean",
                        run_id=f"main-a-{request.prefix_unit_id}",
                        history_index=2,
                    ),
                    task=core_task_name(unit.task),
                    registry=self._new_mcq_registry,
                    runtime_registry=PHASE13_CORE_BASELINE_REGISTRY,
                ).arms[request.arm]
            else:
                branch = build_live_reduced_main_branches(
                    prefix=request.checkpoint,
                    context=self._context(
                        unit,
                        tasks[0],
                        "clean",
                        run_id=f"main-a-{request.prefix_unit_id}",
                        history_index=2,
                    ),
                    candidate_registry=self._candidate_registry,
                    registry=PHASE13_CORE_BASELINE_REGISTRY,
                ).arms[request.arm]
        run = ProspectiveOrdinaryRun(
            task_name=task_name(unit.task),
            baseline=request.baseline,
            run_id=f"main-a-{request.prefix_unit_id or unit.unit_id}",
            model="gpt-5.6-luna",
            client=self._client,
            verifier=verifier(task_name(unit.task)),
            decoding={"temperature": 0.0, "top_p": 1.0},
            arm=request.arm,
            branch=branch,
            tasks=() if unit.task in _CORE_TASKS else tasks,
            core_bundle=self._core if unit.task in _CORE_TASKS else None,
            trajectory_seed=unit.seed,
            embedding_provider=self._embedder(),
            baseline_configs=self._configs(),
            initial_states=self._initial_states(unit.task),
            production_identity=identity,
        )
        result = execute_prospective_ordinary(run)
        archive = production_archive_from_ordinary(run, result, identity)
        validate_production_archive(archive, self._packet, identity.registration_packet_sha256)
        return dispatch_output(unit, result.trials, identity, archive)

    def _context(
        self,
        unit: ProductionObject,
        task: TaskInstance,
        arm: Branch,
        *,
        run_id: str | None = None,
        history_index: int = 1,
    ) -> Game24RuntimeContext:
        identity = run_id or f"main-a-{unit.unit_id}"
        return Game24RuntimeContext(
            task=task,
            client=bind_cost_policy_client(self._client, self._root),
            model="gpt-5.6-luna",
            verifier=verifier(task_name(unit.task)),
            decoding={
                "temperature": 0.0,
                "top_p": 1.0,
                "max_output_tokens": 512,
                "service_tier": "default",
            },
            branch=arm,
            identities=RuntimeIdentities(
                identity,
                f"{identity}:trial:{history_index}:{arm}:{task.sample_id}",
                history_index,
            ),
            embedding_provider=self._embedder(),
            baseline_configs=bind_capacity_configs(self._configs(), 8192),
            initial_states=self._initial_states(unit.task),
        )

    def _tasks(self, task: str, seed: int, *, prefix: bool) -> tuple[TaskInstance, ...]:
        seed_row = self._checkpoint_registry.tasks[task].seeds[seed]
        sample_ids = seed_row.clean_prefix_sample_ids if prefix else seed_row.suffix_sample_ids
        if task in _CORE_TASKS:
            rows = load_core_task(self._core, core_task_name(task))
        else:
            path = self._root / f"data/phase13/main/{task}_main_v1.jsonl"
            builder = {
                "game24": build_game24,
                "math_equation_balancer": build_equation,
                "word_sorting": build_word_sorting,
            }[task]
            rows = tuple(builder(json.loads(line)) for line in path.read_text().splitlines())
        by_id = {row.sample_id: row for row in rows}
        return tuple(by_id[sample_id] for sample_id in sample_ids)

    def _initial_states(self, task: str) -> dict[str, BoTStateV3 | RagFrozenStateV3]:
        states: dict[str, BoTStateV3 | RagFrozenStateV3] = {
            "bot_style": BoTStateV3(entries=[], clean_competitor_ids=())
        }
        if task not in _CORE_TASKS:
            seal = json.loads(
                (self._root / "data/phase13/rag/legacy_seal_v1.json").read_text()
            )
            states["rag_frozen"] = load_legacy_rag_state(
                LegacyRagRuntimeRequest(
                    self._root / "data/phase13/rag/legacy",
                    self._root,
                    legacy_task_name(task),
                    "clean",
                    self._embedder(),
                    seal["manifest_sha256"],
                )
            ).state
        return states

    def _configs(self) -> dict[str, dict[str, str | Path]]:
        return {"dc_rs": {"cache_dir": self._cache, "tool_mode": "text_only"}}

    def _embedder(self) -> EmbeddingProvider:
        if self._embedder_instance is None:
            self._embedder_instance = BgeM3EmbeddingProvider(
                cache_folder=self._cache,
                local_files_only=True,
            )
        return self._embedder_instance

__all__ = ["MainLiveRuntimeError", "ProductionMainRuntime"]
