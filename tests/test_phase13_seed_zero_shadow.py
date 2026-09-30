from __future__ import annotations

import hashlib
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
from memcontam.readiness.phase13_v3_terminal_ledger import TerminalLedgerV3

from .phase13_runner_safety_fixture import FakeProvider, open_run
from .test_phase13_v3_entrypoint_fixture import (
    REPAIR_ROOT,
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
    target_dispatch_id: str | None = None
    failed_once = False
    target_request_hash: str | None = None

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
        if compiled.key.dispatch_id == self.target_dispatch_id:
            request_hash = hashlib.sha256(compiled.request_bytes).hexdigest()
            if not self.failed_once:
                self.failed_once = True
                self.target_request_hash = request_hash
                raise ShadowPrePayloadTimeout()
            assert request_hash == self.target_request_hash
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


class ShadowPrePayloadTimeout(TimeoutError):
    phase13_retry_class = "TIMEOUT_BEFORE_SEMANTIC_PAYLOAD"


@pytest.mark.parametrize("bounded", [True, False], ids=["first-entitled-unit", "full-seed-zero"])
def test_seed_zero_shadow_commits_every_production_unit_without_external_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deny_external: dict[str, int],
    local_authority: Path,
    local_embedder: EmbeddingProvider,
    bounded: bool,
) -> None:
    del deny_external
    source_root = tmp_path / "source"
    for path in STATIC_PATHS:
        source = REPAIR_ROOT / path
        target = source_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    source_rows = read_files(source_root, STATIC_PATHS)
    all_units = production(first_freeze(source_root), tuple(row.binding for row in source_rows))
    seed_zero_units = tuple(unit for unit in all_units if unit.seed == 0)
    entrypoint_bytes = build_entrypoint_bytes(tuple(range(10)), production_units=all_units)
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
    entitlements = run.dispatcher().retry_entitlements
    target_unit_index, target_unit = next(
        (index, unit)
        for index, unit in enumerate(seed_zero_units)
        if any(
            row.unit_id == unit.unit_id and row.dispatch_id in entitlements
            for row in run.selected.costs.resources.phase4.base.retry_reservations
        )
    )
    limit = target_unit_index + 1 if bounded else None
    provider.target_dispatch_id = next(
        row.dispatch_id
        for row in run.selected.costs.resources.phase4.base.retry_reservations
        if row.unit_id == target_unit.unit_id and row.dispatch_id in entitlements
    )
    append = TerminalLedgerV3.append

    def reconcile_shadow_timeout(ledger: TerminalLedgerV3, event: dict) -> None:
        append(ledger, event)
        if (
            ledger is run.ledger
            and event["kind"] == "COMPLETED"
            and event["unit_id"] == provider.target_dispatch_id
        ):
            # The fake transport's first, pre-payload timeout consumed zero tokens.
            ledger.reconcile_cost(
                event["unit_id"],
                {"usage": {"input_tokens": 0, "output_tokens": 0}},
                "f" * 64,
                attempt_index=0,
            )

    monkeypatch.setattr(TerminalLedgerV3, "append", reconcile_shadow_timeout)
    expected_dispatches = sum(
        group.calls
        for unit in run.selected.costs.resources.phase4.base.units
        if unit.unit_id in {item.unit_id for item in seed_zero_units[:limit]}
        for group in unit.stages
    )
    try:
        status = run.execute(
            tmp_path / "cache",
            max_units=limit,
            tranche_ceiling_krw=450000,
            provider_factory=provider.factory,
        )
        assert status.completed_count == (limit or len(seed_zero_units))
        assert status.terminal_technical_missing_count == 0
        assert status.pending_count == len(all_units) - (limit or len(seed_zero_units))
        assert len(provider.requests) == expected_dispatches + 1
        assert len(set(provider.requests)) == expected_dispatches
        assert status.provider_calls_issued == len(provider.requests)
        events = tuple(json.loads(raw) for raw in run.ledger.rows())
        assert sum(row["kind"] == "RETRYABLE_ATTEMPT_FAILURE" for row in events) == 1
        target_events = [row for row in events if row["unit_id"] == provider.target_dispatch_id]
        assert [
            row["attempt_index"] for row in target_events if row["kind"] == "ATTEMPT_STARTED"
        ] == [0, 1]
        assert (
            next(row for row in target_events if row["kind"] == "COMPLETED")["transport_attempts"]
            == 2
        )
        assert sum(row["kind"] == "REQUEST_COMPILED" for row in target_events) == 1
        assert provider.requests.count(provider.target_dispatch_id) == 2
        with run.private.connect() as connection:
            parents = tuple(
                connection.execute("SELECT unit_id, raw, sha256 FROM parents WHERE raw IS NOT NULL")
            )
        assert len(parents) == status.completed_count
        assert all(hashlib.sha256(raw).hexdigest() == checksum for _, raw, checksum in parents)
        print(
            json.dumps(
                {
                    "package_sha256": run.selected.package_sha256,
                    "resources_sha256": [
                        (row.binding.path, row.binding.sha256) for row in source_rows
                    ],
                    "semantic_requests": len(set(provider.requests)),
                    "physical_attempts": len(provider.requests),
                    "completed_parents": len(parents),
                    "terminal_technical_missing": status.terminal_technical_missing_count,
                    "retry_dispatch_id": provider.target_dispatch_id,
                },
                sort_keys=True,
            )
        )
    finally:
        run.close()
    reopened = open_run(request, create=False)
    try:
        before_resume = tuple(provider.requests)
        assert reopened.status() == status
        if not bounded:
            resumed = reopened.execute(
                tmp_path / "cache",
                max_units=None,
                tranche_ceiling_krw=450000,
                provider_factory=provider.factory,
            )
            assert resumed == status
        assert tuple(provider.requests) == before_resume
    finally:
        reopened.close()
    later_seed = open_run(request, create=False, seed=1)
    try:
        assert later_seed.status().pending_count == len(all_units) - (limit or len(seed_zero_units))
        assert tuple(provider.requests) == before_resume
    finally:
        later_seed.close()
