from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import assert_never

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.memory.embeddings import BgeM3EmbeddingProvider, EmbeddingProvider
from memcontam.readiness.phase13_main_live_runtime import ProductionMainRuntime
from memcontam.readiness.phase13_v3_builder_inputs import first_freeze, production
from memcontam.readiness.phase13_v3_request import (
    CompiledProviderRequestV3,
    PackageBindingV3,
)
from memcontam.readiness.phase13_v3_resource_files import read_files

from .phase13_runner_safety_fixture import FakeProvider, open_run
from .test_phase13_v3_entrypoint_fixture import (
    REPAIR_ROOT,
    RESOURCE_ROOT,
    STATIC_PATHS,
    build_entrypoint_bytes,
    seal_fixture_closure,
)
from .test_phase13_v3_entrypoint_integration import deny_external as deny_external
from .test_phase13_runner_safety import local_authority as local_authority


@pytest.fixture(scope="session")
def local_embedder() -> EmbeddingProvider:
    return BgeM3EmbeddingProvider(
        cache_folder=Path.home() / ".cache/huggingface/hub",
        local_files_only=True,
    )


class SeedZeroShadowProvider(FakeProvider):
    def factory(self, binding: PackageBindingV3) -> SeedZeroShadowProvider:
        super().factory(binding)
        return self

    def send_compiled_v3(
        self,
        compiled: CompiledProviderRequestV3,
        before_request: Callable[[], None],
    ) -> LLMResponse:
        before_request()
        self.requests.append(compiled.key.dispatch_id)
        match compiled.key.stage:
            case "bot_problem_distill":
                content = json.dumps(
                    {
                        "key_information": "provider-free shadow",
                        "restrictions": "text only",
                        "distilled_task": "return a final answer",
                    }
                )
            case "bot_instantiate_solve":
                prompt = "\n".join(message.content for message in compiled.material.messages)
                content = json.dumps(
                    {
                        "selected_structure": (
                            "retrieved-template"
                            if "Set selected_structure to retrieved-template." in prompt
                            else "procedure-based"
                        ),
                        "solution_trace": "deterministic shadow",
                        "final_answer": "final: 0",
                    }
                )
            case "bot_thought_distill":
                content = json.dumps(
                    {
                        "description": "provider-free procedure",
                        "template": "return the deterministic shadow answer",
                        "category": "procedure-based",
                        "explicitly_used_memory_ids": [],
                    }
                )
            case "reflexion_reflect":
                content = json.dumps(
                    {
                        "mode": "corrective",
                        "failure_class": "incorrect_answer",
                        "reflection_text": "Use the deterministic shadow answer.",
                        "explicitly_used_memory_ids": [],
                    }
                )
            case "dc_rs_synthesize":
                content = "<cheatsheet>provider-free strategy</cheatsheet>"
            case (
                "full_history_generate"
                | "rag_generate"
                | "reflexion_generate"
                | "dc_rs_generate"
                | "no_memory_generate"
            ):
                content = "final: 0"
            case unreachable:
                assert_never(unreachable)
        return LLMResponse(
            content,
            {
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "attempts": 1,
                "authoritative_provider_cost_usd": "0",
                "currency": "USD",
                "status": "completed",
                "response_id": f"shadow-{compiled.key.dispatch_id}",
                "model": "gpt-5.6-luna",
                "service_tier": "default",
            },
            {"prompt_tokens": 0, "completion_tokens": 0},
            0,
        )


def test_seed_zero_shadow_commits_every_production_unit_without_external_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deny_external: dict[str, int],
    local_authority: Path,
    local_embedder: EmbeddingProvider,
) -> None:
    del deny_external
    source_root = tmp_path / "source"
    repair_path = "data/phase13/main/legacy_dc_rs_intervention_registry_v1.json"
    for path in STATIC_PATHS:
        source = (REPAIR_ROOT if path == repair_path else RESOURCE_ROOT) / path
        target = source_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    source_rows = read_files(source_root, STATIC_PATHS)
    units = tuple(
        unit
        for unit in production(
            first_freeze(source_root), tuple(row.binding for row in source_rows)
        )
        if unit.seed == 0
    )
    entrypoint_bytes = build_entrypoint_bytes((0,), production_units=units)
    for path, raw in entrypoint_bytes.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    shutil.copytree(local_authority, tmp_path / "authority")
    seal_fixture_closure(tmp_path)
    from memcontam.readiness.phase13_v3_entrypoint import SelectionRequest

    from .phase13_corrective_identity import corrective_identity

    request = SelectionRequest(
        tmp_path,
        tmp_path / "package.json",
        tmp_path / "authorization.json",
        tmp_path / "authority",
        tmp_path / "authorization.sha256",
        corrective_identity().run_id,
    )
    monkeypatch.setattr(ProductionMainRuntime, "_embedder", lambda _self: local_embedder)
    provider = SeedZeroShadowProvider()
    run = open_run(request, create=True)
    try:
        status = run.execute(
            tmp_path / "cache",
            max_units=None,
            tranche_ceiling_krw=450000,
            provider_factory=provider.factory,
        )
        assert status.completed_count == 120
        assert status.terminal_technical_missing_count == 0
        assert status.pending_count == 0
        assert len(provider.requests) == 10893
        assert len(set(provider.requests)) == 10893
    finally:
        run.close()
