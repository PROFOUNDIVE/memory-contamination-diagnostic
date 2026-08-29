from __future__ import annotations

import hashlib
import json
from pathlib import Path

from memcontam.baselines.bot_phase12 import BoTStateV3
from memcontam.clients.config import ProviderConfig
from memcontam.clients.openai_responses import OpenAIResponsesClient
from memcontam.experiment.phase13_ordinary_runtime import (
    ProspectiveOrdinaryRun,
    execute_readiness0_trial,
)
from memcontam.memory.embeddings import BgeM3EmbeddingProvider, EmbeddingProvider
from memcontam.memory.checkpoint_v3 import NativeEntry
from memcontam.memory.stores import MemoryEntry
from memcontam.readiness.phase13_legacy_rag_runtime import (
    LegacyRagRuntimeRequest,
    load_legacy_rag_state,
)
from memcontam.readiness.phase13_main_checkpoint import production_identity_from_checkpoint
from memcontam.readiness.phase13_readiness0_case_evidence import (
    CaseEvidenceInput,
    build_case_evidence,
)
from memcontam.readiness.phase13_readiness0_budget import (
    BudgetedResponses,
    ProviderCallBudgetError,
)
from memcontam.tasks.base import TaskInstance
from memcontam.verifiers.game24 import verify_expression

from memcontam.readiness.phase13_readiness0_live_models import CaseEvidence, Readiness0Case


class ProductionCaseExecutor:
    def __init__(
        self,
        repository_root: Path,
        core_root: Path,
        cache_root: Path,
        *,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._root = repository_root
        self._core_bundle = core_root / "materialized"
        self._cache_root = cache_root
        self._allow_test_embedder = embedding_provider is not None
        self._embedder = embedding_provider or BgeM3EmbeddingProvider(
            cache_folder=cache_root, local_files_only=True
        )
        self._client = OpenAIResponsesClient(
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
            maximum_provider_calls=12,
        )

    def __call__(self, case: Readiness0Case) -> CaseEvidence:
        routing_results: list[bool] = []
        actual_results: list[bool] = []

        def verifier(answer: str, task: TaskInstance) -> bool:
            actual = (
                verify_expression(
                    answer,
                    list(task.input["numbers"]),
                    int(task.verifier_spec.get("target", 24)),
                ).is_correct
                if task.task_name == "game24"
                else answer == task.verifier_spec["answer_label"]
            )
            actual_results.append(actual)
            if case.baseline == "reflexion_style":
                route = bool(routing_results)
                routing_results.append(route)
                return route
            return actual

        run = self._run(case, verifier)
        result = execute_readiness0_trial(run)
        identity = run.production_identity
        if identity is None:
            raise RuntimeError("READINESS0_PRODUCTION_IDENTITY_MISSING")
        return build_case_evidence(
            CaseEvidenceInput(
                case,
                result.trials[0],
                identity,
                result.sample_ids[0],
                tuple(routing_results),
                tuple(actual_results),
                self._root,
            )
        )

    def _run(self, case: Readiness0Case, verifier) -> ProspectiveOrdinaryRun:
        identity = production_identity_from_checkpoint(
            self._root / "data/phase13/main/mr_p4",
            self._root,
            task=case.task,
            trajectory_seed=0,
            execution_template_id=f"readiness0|{case.task}|{case.baseline}|clean",
            registration_packet_sha256=self._sha256(
                self._root / "data/phase13/observability/registration_packet_v1.json"
            ),
        )
        initial_states = {}
        configs = {}
        if case.baseline == "rag_frozen":
            seal = json.loads(
                (self._root / "data/phase13/rag/legacy_seal_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            loaded = load_legacy_rag_state(
                LegacyRagRuntimeRequest(
                    package_root=self._root / "data/phase13/rag/legacy",
                    repository_root=self._root,
                    task="game24",
                    branch="clean",
                    embedder=self._embedder,
                    expected_manifest_sha256=seal["manifest_sha256"],
                    allow_test_embedder=self._allow_test_embedder,
                )
            )
            initial_states["rag_frozen"] = loaded.state
        if case.baseline == "bot_style":
            entries: list[MemoryEntry | NativeEntry] = [
                MemoryEntry(
                    entry_id=f"readiness0-clean-competitor-{index}",
                    content=content,
                    memory_type="thought_template",
                    metadata={"description": content, "category": "procedure-based"},
                )
                for index, content in enumerate(
                    (
                        "Combine multiplication before addition and verify all four numbers.",
                        "Search reversible arithmetic combinations and verify the final expression.",
                    )
                )
            ]
            initial_states["bot_style"] = BoTStateV3(
                entries=entries,
                clean_competitor_ids=tuple(entry.entry_id for entry in entries),
            )
        if case.baseline == "dc_rs":
            configs["dc_rs"] = {
                "serialized_cheatsheet_budget_tokens": 8192,
                "tool_mode": "text_only",
                "cache_dir": self._cache_root,
            }
        return ProspectiveOrdinaryRun(
            task_name=case.task,
            baseline=case.baseline,
            run_id=f"readiness0-{case.case_id}",
            model="gpt-5.6-luna",
            client=self._client,
            verifier=verifier,
            decoding={"temperature": 0.0},
            tasks=() if case.task != "game24" else self._game24_suffix(),
            core_bundle=self._core_bundle if case.task != "game24" else None,
            trajectory_seed=0,
            embedding_provider=self._embedder,
            baseline_configs=configs,
            initial_states=initial_states,
            production_identity=identity,
        )

    def _game24_suffix(self) -> tuple[TaskInstance, ...]:
        registry = json.loads(
            (self._root / "data/phase13/main/mr_p4/main_a_common_checkpoint_registry_v1.json")
            .read_text(encoding="utf-8")
        )
        sample_ids = registry["tasks"]["game24"]["seeds"][0]["suffix_sample_ids"]
        rows = {
            row["sample_id"]: row
            for row in (
                json.loads(line)
                for line in (self._root / "data/phase13/main/game24_main_v1.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        }
        return tuple(
            TaskInstance(
                sample_id=sample_id,
                task_name="game24",
                input={"numbers": rows[sample_id]["numbers"]},
                verifier_spec={"target": rows[sample_id]["target"]},
            )
            for sample_id in sample_ids
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "BudgetedResponses",
    "CaseEvidenceInput",
    "ProductionCaseExecutor",
    "ProviderCallBudgetError",
    "build_case_evidence",
]
