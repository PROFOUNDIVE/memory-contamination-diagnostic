from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import assert_never

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.memory.embeddings import BgeM3EmbeddingProvider, EmbeddingProvider
from memcontam.readiness.phase13_main_live_runtime import ProductionMainRuntime
from memcontam.readiness.phase13_v3_builder import build_mr_p4, build_mr_p5, validate_mr_p5, _authorization
from memcontam.readiness.phase13_v3_builder_inputs import PREFIX, STATIC_PATHS, phase4_costs
from memcontam.readiness.phase13_v3_cost_models import CostError, canonical_bytes, digest
from memcontam.readiness.phase13_v3_entrypoint import SelectionRequest, SelectedExecutionV3, select_execution, EntrypointError
from memcontam.readiness.phase13_v3_entrypoint_models import MainAuthorizationV3, MainExecutionPackageV3
from memcontam.readiness.phase13_v3_publication import P4_PATHS, P5_PATHS
from memcontam.readiness.phase13_v3_request import (
    CompiledProviderRequestV3,
    PackageBindingV3,
)
from memcontam.readiness.phase13_v3_resource_files import read_files
from memcontam.readiness.phase13_v3_terminal_ledger import TerminalLedgerV3

from .phase13_runner_safety_fixture import FakeProvider, open_run
from .test_phase13_v3_entrypoint_fixture import REPAIR_ROOT
from .test_phase13_v3_artifact_builder import builder_source as builder_source, synthetic_pricing
from .phase13_corrective_identity import corrective_identity
from .test_phase13_v3_entrypoint_integration import deny_external as deny_external


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
    provider_failure_acknowledged = True


def staging_shadow_request(tmp_path: Path, builder_source) -> SelectionRequest:
    root, commit, authority = builder_source
    output = tmp_path / "staging-output"
    output.mkdir()
    manifest = build_mr_p4(root, authority, output, governed_source_commit=commit,
                           identity=corrective_identity(), count_pricing=synthetic_pricing(root, authority))
    package = build_mr_p5(root, authority, output)
    assert validate_mr_p5(root, authority, output) == package
    for path in (*P4_PATHS, *P5_PATHS):
        target = root / PREFIX / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((output / path).read_bytes())
    # Disposable selection fixture only; no MR-P6 publication or live authorization.
    authorization = _authorization(package)
    auth_path = root / "staging-shadow-authorization.json"
    auth_path.write_bytes(canonical_bytes(authorization))
    sidecar = root / "staging-shadow-authorization.sha256"
    sidecar.write_text(digest(authorization) + "\n")
    assert phase4_costs(manifest).base.bindings.request_compiler_hash == package.final_order.request_hash
    return SelectionRequest(root, root / PREFIX / P5_PATHS[-1], auth_path, authority,
                            sidecar, package.identity.run_id)


def test_shadow_package_uses_candidate_governed_prefreeze(tmp_path: Path, builder_source) -> None:
    request = staging_shadow_request(tmp_path, builder_source)
    root, commit, _authority = builder_source
    selected = select_execution(request, "run")
    assert isinstance(selected, SelectedExecutionV3)
    assert selected.package.governed_source is not None
    assert selected.package.governed_source.governed_source_commit == commit
    assert selected.package.final_order.request_hash != "c" * 64
    assert selected.package.final_order.tokenizer_hash != "d" * 64
    assert all(row.binding.sha256 == hashlib.sha256((REPAIR_ROOT / row.binding.path).read_bytes()).hexdigest()
               for row in read_files(root, STATIC_PATHS))
    selected.close()
    target = root / "data/phase13/main/legacy_dc_rs_intervention_registry_v2.json"
    original = target.read_bytes()
    target.write_bytes(original + b" ")
    with pytest.raises(EntrypointError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
        select_execution(request, "run")
    target.write_bytes(original)
    package = MainExecutionPackageV3.model_validate_json(request.package_path.read_bytes())
    package = package.model_copy(update={"final_order": package.final_order.model_copy(
        update={"request_hash": "c" * 64})})
    package = package.model_copy(update={"package_hash": digest(package, "package_hash")})
    request.package_path.write_bytes(canonical_bytes(package))
    auth = MainAuthorizationV3.model_validate_json(request.authorization_path.read_bytes())
    auth = auth.model_copy(update={"execution_package_sha256": digest(package),
                                   "execution_package_hash": package.package_hash})
    auth = auth.model_copy(update={"authorization_hash": digest(auth, "authorization_hash")})
    request.authorization_path.write_bytes(canonical_bytes(auth))
    assert request.expected_authorization_sha256_file is not None
    request.expected_authorization_sha256_file.write_text(digest(auth) + "\n")
    with pytest.raises(CostError, match="MAIN_COST_PROOF_MISMATCH"):
        select_execution(request, "run")


@pytest.mark.parametrize("bounded", [True, False], ids=["first-entitled-unit", "full-seed-zero"])
def test_seed_zero_shadow_commits_every_production_unit_without_external_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deny_external: dict[str, int],
    builder_source,
    local_embedder: EmbeddingProvider,
    bounded: bool,
) -> None:
    del deny_external
    request = staging_shadow_request(tmp_path, builder_source)
    source_rows = read_files(request.repository_root, STATIC_PATHS)
    selected = select_execution(request, "run")
    assert isinstance(selected, SelectedExecutionV3)
    try:
        production_units = selected.package.production
    finally:
        selected.close()
    seed_zero_units = tuple(unit for unit in production_units if unit.seed == 0)
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
        assert status.pending_count == len(production_units) - (limit or len(seed_zero_units))
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
        assert later_seed.status().pending_count == len(production_units) - (limit or len(seed_zero_units))
        assert tuple(provider.requests) == before_resume
    finally:
        later_seed.close()
