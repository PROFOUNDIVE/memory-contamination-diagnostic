from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import httpx
from openai import OpenAI, AsyncOpenAI

from memcontam.clients.base import LLMResponse
from memcontam.evaluation.phase13_observability_models import Phase13TrialEvidence
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.readiness.phase13_main_checkpoint import CommonCheckpointRegistry
from memcontam.readiness.phase13_main_live_evidence import MemoryUnitEvidence
from memcontam.readiness.phase13_main_live_runtime import ProductionMainRuntime
from memcontam.readiness.phase13_main_production import ProductionObject, UnitKind
from memcontam.readiness.phase13_main_resource_contract import RESOURCE_PATHS
from memcontam.readiness.phase13_main_terminal_partial import TerminalPartialParent, validate_terminal_partial
from memcontam.readiness.phase13_main_v3_runner import DurableParentRecordV3, V3MainRun
from memcontam.readiness.phase13_v3_entrypoint import SelectedExecutionV3, SelectionRequest, select_execution
from memcontam.readiness.phase13_v3_request import RequestKeyV3
from memcontam.readiness.phase13_v3_terminal_models import TerminalEvidenceError
from memcontam.readiness.phase13_v3_entrypoint import EntrypointError

from .phase13_corrective_identity import corrective_identity
from .phase13_count_fake import CountedProvider
from .test_phase13_readiness0_production_dry_run import _ContractFakeEmbeddingProvider
from .test_phase13_v3_entrypoint_fixture import (
    AUTHORITY,
    REPAIR_ROOT,
    RESOURCE_ROOT,
    build_entrypoint_bytes,
    seal_fixture_closure,
)


class NativeProvider(CountedProvider):
    def send_compiled_v3(self, compiled, before_request):
        before_request()
        content = {
            "bot_problem_distill": json.dumps({
                "key_information": "four numbers",
                "restrictions": "exact use",
                "distilled_task": "make 24",
            }),
            "bot_instantiate_solve": json.dumps({
                        "selected_structure": "procedure-based",
                "solution_trace": "divide",
                "final_answer": "final: 8*((7+8)-12)",
            }),
            "bot_thought_distill": json.dumps({
                "description": "arithmetic",
                "template": "check all numbers",
                "category": "procedure-based",
                "explicitly_used_memory_ids": [],
            }),
            "dc_rs_synthesize": "<cheatsheet>ordinary strategy</cheatsheet>",
        }.get(compiled.key.stage, "final: 8*((7+8)-12)")
        return LLMResponse(content, {
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "attempts": 1,
            "authoritative_provider_cost_usd": "0.0000014",
            "currency": "USD",
            "status": "completed",
            "response_id": f"fake-{compiled.key.dispatch_id}",
            "model": "gpt-5.6-luna",
            "service_tier": "default",
        }, {"prompt_tokens": 1, "completion_tokens": 1}, 0)


@pytest.fixture(autouse=True)
def deny_main_external(monkeypatch: pytest.MonkeyPatch) -> Callable[[], None]:
    def denied(*_args, **_kwargs):
        pytest.fail("Main runtime attempted external process or network access")

    monkeypatch.setattr(socket, "socket", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(httpx.Client, "send", denied)
    monkeypatch.setattr(httpx.AsyncClient, "send", denied)
    monkeypatch.setattr(OpenAI, "__init__", denied)
    monkeypatch.setattr(AsyncOpenAI, "__init__", denied)

    def arm() -> None:
        popen = subprocess.Popen
        def controlled_popen(args, *rest, **kwargs):
            if args[0] == "git" and args[1] == "--no-replace-objects":
                return popen(args, *rest, **kwargs)
            return denied(args, *rest, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", controlled_popen)
        monkeypatch.setattr(os, "system", denied)
        monkeypatch.setattr(os, "popen", denied)
    return arm


@pytest.mark.parametrize("failure", ("semantic", "provider", "pre_result", "zero_prefix", "late_provider"))
def test_terminal_ordinary_preserves_only_real_trials_and_reopens(
    tmp_path: Path, deny_main_external: Callable[[], None], failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit = replace(_units()[-1], sequence=0)
    for path, raw in build_entrypoint_bytes((0,), production_units=(unit,)).items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    seal_fixture_closure(tmp_path)
    request = SelectionRequest(tmp_path, tmp_path / "package.json", tmp_path / "authorization.json",
        AUTHORITY, tmp_path / "authorization.sha256", corrective_identity().run_id)
    calls: list[int] = []
    if failure == "pre_result":
        entry = PHASE13_CORE_BASELINE_REGISTRY["nomem"]

        def execute_trial(context, state):
            if context.identities.order_key == 2:
                context.client.chat([{"role": "user", "content": "fixture"}], context.model,
                                    {**context.decoding, "method_stage": "no_memory_generate"})
            return entry.execute_trial(context, state)

        monkeypatch.setitem(PHASE13_CORE_BASELINE_REGISTRY, "nomem", replace(entry, execute_trial=execute_trial))

    class TerminalProvider(NativeProvider):
        def send_compiled_v3(self, compiled, before_request):
            calls.append(compiled.key.ordinal)
            if ((compiled.key.ordinal == 1 and failure == "provider")
                or (compiled.key.ordinal == 2 and failure == "late_provider")
                or failure == "zero_prefix"):
                before_request()
                raise RuntimeError("provider unavailable")
            if compiled.key.ordinal == 1 and failure == "pre_result":
                before_request()
                raise RuntimeError("dispatch before baseline result")
            response = super().send_compiled_v3(compiled, before_request)
            if compiled.key.ordinal == 1 and failure != "late_provider":
                return LLMResponse("not a final answer", response.raw, response.token_usage, response.latency_ms)
            return response

    selected = select_execution(request, "run")
    assert isinstance(selected, SelectedExecutionV3)
    directory = tmp_path / "terminal-run"
    run = V3MainRun.open(selected, directory, create=True, seed=0)
    deny_main_external()
    try:
        status = run.execute(Path.home() / ".cache/huggingface/hub", max_units=1,
            tranche_ceiling_krw=450000, provider_factory=lambda _binding: TerminalProvider())
        assert calls == ([0] if failure == "zero_prefix" else [0, 1, 2] if failure == "late_provider" else [0, 1])
        assert status.provider_calls_issued == len(calls)
        assert run.ledger.state(RequestKeyV3(parent_id=unit.unit_id,
            stage="no_memory_generate", ordinal=0).dispatch_id).kind == (
                "ATTEMPTED_PROVIDER_FAILURE" if failure == "zero_prefix" else "COMPLETED")
        if failure != "zero_prefix":
            assert run.ledger.state(RequestKeyV3(parent_id=unit.unit_id,
                stage="no_memory_generate", ordinal=1).dispatch_id).kind == (
                    "COMPLETED" if failure == "late_provider" else "ATTEMPTED_PROVIDER_FAILURE")
        assert status.completed_count == 0
        assert status.terminal_technical_missing_count == 1
        assert status.pending_count == 0
        with run.private.connect() as connection:
            raw = connection.execute("SELECT raw FROM parents WHERE unit_id=?", (unit.unit_id,)).fetchone()[0]
        assert raw is not None
        if failure == "semantic":
            archive = json.loads(raw)["unit_evidence"]["evidence"]["runtime_evidence"]["production_observability_archive"]
            assert [row["evidence"]["trial"]["execution_status"] for row in archive["records"]] == ["completed", "failed"]
            assert archive["records"][-1]["terminal_failure_code"] == "no_memory_invalid_final_answer"
            assert archive["records"][-1]["terminal_provider_evidence"]["trigger_class"] == "post_response_semantic_failure"
        else:
            partial = json.loads(raw)
            assert partial["schema_version"] == "phase13_main_terminal_partial_parent_v1"
            assert len(partial["archive"]["records"]) == (0 if failure == "zero_prefix" else 2 if failure == "late_provider" else 1)
            assert partial["whole_unit_cost_krw"] is None
            assert partial["observation_cost_krw"] == (0 if failure == "zero_prefix" else 6 if failure == "late_provider" else 3)
            assert partial["terminal_key"]["ordinal"] == (0 if failure == "zero_prefix" else 2 if failure == "late_provider" else 1)
            if failure == "late_provider":
                parsed = TerminalPartialParent.model_validate_json(raw)
                assert parsed.archive.records[1].task_instance is not None
                suppressed = parsed.model_copy(update={
                    "archive": parsed.archive.model_copy(update={"records": parsed.archive.records[:1]}),
                    "observed_calls": parsed.observed_calls[:1],
                    "interrupted_keys": (RequestKeyV3(parent_id=unit.unit_id,
                        stage="no_memory_generate", ordinal=1), *parsed.interrupted_keys),
                    "terminal_sample_id": parsed.archive.records[1].task_instance.sample_id,
                    "observation_cost_krw": 1,
                })
                with pytest.raises(TerminalEvidenceError):
                    validate_terminal_partial(run, suppressed, unit)
            if failure == "provider":
                parsed = TerminalPartialParent.model_validate_json(raw)
                assert parsed.archive.records[0].task_instance is not None
                for field, value in (("trajectory_seed", 1), ("concrete_seed_id", "1"),
                                     ("order_key", 9), ("analysis_inclusion", "excluded")):
                    row = parsed.archive.records[0]
                    evidence = row.evidence
                    if field == "analysis_inclusion":
                        evidence = evidence.model_copy(update={"trial": evidence.trial.model_copy(
                            update={field: value})})
                    else:
                        evidence = evidence.model_copy(update={field: value})
                    altered_archive = parsed.archive.model_copy(update={"records": (
                        row.model_copy(update={"evidence": evidence}),)})
                    with pytest.raises((TerminalEvidenceError, ValueError)):
                        validate_terminal_partial(run, parsed.model_copy(update={"archive": altered_archive}), unit)
                altered_record = parsed.archive.records[0].model_copy(update={
                    "task_instance": parsed.archive.records[0].task_instance.model_copy(
                        update={"sample_id": "forged"}),
                })
                forged_archive = parsed.archive.model_copy(update={"records": (altered_record,)})
                altered = (
                    parsed.model_copy(update={"terminal_sample_id": "forged"}),
                    parsed.model_copy(update={"terminal_event_hash": "0" * 64}),
                    parsed.model_copy(update={"interrupted_keys": ()}),
                    parsed.model_copy(update={"observed_calls": ()}),
                    parsed.model_copy(update={"archive": forged_archive}),
                    parsed.model_copy(update={"observation_cost_krw": 0}),
                    parsed.model_copy(update={"whole_unit_cost_krw": 0}),
                    parsed.model_copy(update={"interrupted_keys": (parsed.terminal_key, parsed.terminal_key)}),
                )
                for forged in altered:
                    with pytest.raises(TerminalEvidenceError):
                        validate_terminal_partial(run, forged, unit)
                contractless_call = parsed.observed_calls[0].model_copy(update={
                    "provider_request_contract": {},
                    "provider_authority_contract": {},
                })
                with pytest.raises(TerminalEvidenceError):
                    validate_terminal_partial(run, parsed.model_copy(
                        update={"observed_calls": (contractless_call,)}), unit)
                original_read = run.ledger.read_record
                with monkeypatch.context() as tamper:
                    tamper.setattr(type(run.ledger), "read_record", lambda _ledger, name: (
                        b"tampered" if name == f"{parsed.terminal_key.dispatch_id}.observation.json"
                        else original_read(name)))
                    with pytest.raises(TerminalEvidenceError):
                        validate_terminal_partial(run, parsed, unit)
    finally:
        run.close()

    selected = select_execution(request, "resume")
    assert isinstance(selected, SelectedExecutionV3)
    reopened = V3MainRun.open(selected, directory, create=False, seed=0)
    try:
        assert reopened.status().completed_count == 0
        assert reopened.status().terminal_technical_missing_count == 1
        if failure == "provider":
            terminal_key = RequestKeyV3(parent_id=unit.unit_id, stage="no_memory_generate", ordinal=1)
            reopened.ledger.reconcile_cost(terminal_key.dispatch_id,
                {"usage": {"input_tokens": 1, "output_tokens": 1}}, "f" * 64)
            assert reopened.status().terminal_technical_missing_count == 1
        reopened.execute(Path.home() / ".cache/huggingface/hub", max_units=1,
            tranche_ceiling_krw=450000, provider_factory=lambda _binding: TerminalProvider())
        assert calls == ([0] if failure == "zero_prefix" else [0, 1, 2] if failure == "late_provider" else [0, 1])
    finally:
        reopened.close()


def _units() -> tuple[ProductionObject, ...]:
    checkpoint_raw = (
        RESOURCE_ROOT / RESOURCE_PATHS["common_checkpoint_registry"]
    ).read_bytes()
    checkpoint = CommonCheckpointRegistry.model_validate_json(checkpoint_raw)
    seed = checkpoint.tasks["game24"].seeds[0]
    packet_raw = (REPAIR_ROOT / RESOURCE_PATHS["observability_packet"]).read_bytes()
    packet_sha256 = hashlib.sha256(packet_raw).hexdigest()
    checkpoint_sha256 = hashlib.sha256(checkpoint_raw).hexdigest()
    rows: tuple[tuple[UnitKind, str | None, str], ...] = (
        ("CLEAN_PREFIX", "rag_frozen", "NOT_APPLICABLE"),
        ("CLEAN_PREFIX", "reflexion_style", "NOT_APPLICABLE"),
        ("CLEAN_PREFIX", "dc_rs", "NOT_APPLICABLE"),
        ("CLEAN_PREFIX", "bot_style", "NOT_APPLICABLE"),
        ("NO_MEMORY_SINGLETON", None, "NOT_APPLICABLE"),
    )
    return tuple(
        ProductionObject(
            sequence=sequence,
            unit_id=hashlib.sha256(json.dumps([
                "phase13-main-a-disjoint-unit-id-v1", kind, 0, "game24", baseline, arm,
            ], separators=(",", ":")).encode()).hexdigest(),
            kind=kind,
            seed=0,
            task="game24",
            memory_baseline=baseline,
            arm=arm,
            prefix_unit_id=None,
            projected_cost_krw=0,
            execution_template_id=(
                "game24|nomem" if baseline is None else f"game24|{baseline}|prefix"
            ),
            ordered_sample_ids_sha256=seed.suffix_sample_ids_sha256,
            registration_packet_sha256=packet_sha256,
            checkpoint_registry_sha256=checkpoint_sha256,
        )
        for sequence, (kind, baseline, arm) in enumerate(rows)
    )


def test_native_families_persist_sql_joined_parents_that_revalidate_after_reopen(
    tmp_path: Path,
) -> None:
    units = _units()
    for path, raw in build_entrypoint_bytes((0,), production_units=units).items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    seal_fixture_closure(tmp_path)
    request = SelectionRequest(
        tmp_path,
        tmp_path / "package.json",
        tmp_path / "authorization.json",
        AUTHORITY,
        tmp_path / "authorization.sha256",
        corrective_identity().run_id,
    )
    selected = select_execution(request, "run")
    assert isinstance(selected, SelectedExecutionV3)
    directory = tmp_path / "native-parent-run"
    run = V3MainRun.open(selected, directory, create=True, seed=0)
    try:
        report = run.execute(
            Path.home() / ".cache/huggingface/hub",
            max_units=len(units),
            tranche_ceiling_krw=450000,
            provider_factory=lambda _binding: NativeProvider(),
        )
        assert report.completed_count == len(units)
    finally:
        run.close()

    selected = select_execution(request, "resume")
    assert isinstance(selected, SelectedExecutionV3)
    reopened = V3MainRun.open(selected, directory, create=False, seed=0)
    try:
        with reopened.private.connect() as connection:
            parents = tuple(connection.execute(
                "SELECT unit_id, raw, sha256 FROM parents ORDER BY unit_id"
            ))
        assert tuple(row[0] for row in parents) == tuple(sorted(unit.unit_id for unit in units))
        for unit_id, raw, sha256 in parents:
            assert raw is not None
            assert hashlib.sha256(raw).hexdigest() == sha256
            assert reopened.ledger.read_record(f"{unit_id}.parent.json") == raw
            parent = reopened._load_parent(unit_id, raw, sha256)
            assert not isinstance(parent, TerminalPartialParent)
            assert parent.unit_evidence.unit_id == unit_id
    finally:
        reopened.close()


def test_dc_rs_memory_suffix_archive_and_prefix_ancestry_survive_recursive_reopen(
    tmp_path: Path, deny_main_external: Callable[[], None], monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = replace(_units()[2], sequence=0)
    suffix = replace(prefix, sequence=1, kind="MEMORY_BEARING", arm="contam",
        prefix_unit_id=prefix.unit_id, execution_template_id="game24|dc_rs|contam",
        unit_id=hashlib.sha256(json.dumps([
            "phase13-main-a-disjoint-unit-id-v1", "MEMORY_BEARING", 0,
            "game24", "dc_rs", "contam",
        ], separators=(",", ":")).encode()).hexdigest())
    for path, raw in build_entrypoint_bytes((0,), production_units=(prefix, suffix)).items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    seal_fixture_closure(tmp_path)
    request = SelectionRequest(tmp_path, tmp_path / "package.json", tmp_path / "authorization.json",
        AUTHORITY, tmp_path / "authorization.sha256", corrective_identity().run_id)
    selected = select_execution(request, "run")
    assert isinstance(selected, SelectedExecutionV3)
    directory = tmp_path / "dc-rs-suffix-run"
    calls: list[str] = []

    class Provider(NativeProvider):
        def send_compiled_v3(self, compiled, before_request):
            calls.append(compiled.key.dispatch_id)
            return super().send_compiled_v3(compiled, before_request)

    run = V3MainRun.open(selected, directory, create=True, seed=0)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    embedder = _ContractFakeEmbeddingProvider(vector_dimension=1024)
    monkeypatch.setattr(ProductionMainRuntime, "_embedder", lambda _self: embedder)
    deny_main_external()
    try:
        report = run.execute(tmp_path / "cache", max_units=2, tranche_ceiling_krw=450000,
            provider_factory=lambda _binding: Provider())
        assert report.completed_count == 2
        assert len(calls) == len(set(calls)) == 102
    finally:
        run.close()

    selected = select_execution(request, "resume")
    assert isinstance(selected, SelectedExecutionV3)
    reopened = V3MainRun.open(selected, directory, create=False, seed=0)
    try:
        with reopened.private.connect() as connection:
            raw, sha256 = connection.execute("SELECT raw, sha256 FROM parents WHERE unit_id=?",
                (suffix.unit_id,)).fetchone()
        assert raw == reopened.ledger.read_record(f"{suffix.unit_id}.parent.json")
        assert hashlib.sha256(raw).hexdigest() == sha256
        parent = reopened._load_parent(suffix.unit_id, raw, sha256)
        assert isinstance(parent, DurableParentRecordV3)
        assert parent.unit_evidence.kind == "MEMORY_BEARING"
        evidence = parent.unit_evidence.evidence
        assert isinstance(evidence, MemoryUnitEvidence)
        checkpoint = reopened.checkpoint(suffix)
        assert checkpoint is not None
        assert evidence.consumed_checkpoint_canonical_sha256 == checkpoint.canonical_sha256
        archive = evidence.runtime_evidence.production_observability_archive
        assert archive is not None and len(archive.records) == 50
        first = archive.records[0].evidence
        second = archive.records[1].evidence
        assert isinstance(first, Phase13TrialEvidence)
        assert isinstance(second, Phase13TrialEvidence)
        assert first.new_entry_ids
        assert set(first.new_entry_ids) <= set(second.memory_before_ids)
        assert reopened.execute(tmp_path / "cache", max_units=2, tranche_ceiling_krw=450000,
            provider_factory=lambda _binding: Provider()).provider_calls_issued == 102
        assert len(calls) == 102
    finally:
        reopened.close()


def test_rehashed_normal_parent_rejects_forged_trial_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    unit = replace(_units()[-1], sequence=0)
    for path, raw in build_entrypoint_bytes((0,), production_units=(unit,)).items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    seal_fixture_closure(tmp_path)
    request = SelectionRequest(tmp_path, tmp_path / "package.json", tmp_path / "authorization.json",
        AUTHORITY, tmp_path / "authorization.sha256", corrective_identity().run_id)
    selected = select_execution(request, "run")
    assert isinstance(selected, SelectedExecutionV3)
    directory = tmp_path / "normal-parent"
    run = V3MainRun.open(selected, directory, create=True, seed=0)
    try:
        run.execute(tmp_path / "cache", max_units=1, tranche_ceiling_krw=450000,
            provider_factory=lambda _binding: NativeProvider())
    finally:
        run.close()
    selected = select_execution(request, "resume")
    assert isinstance(selected, SelectedExecutionV3)
    reopened = V3MainRun.open(selected, directory, create=False, seed=0)
    try:
        with reopened.private.connect() as connection:
            raw, _ = connection.execute("SELECT raw, sha256 FROM parents WHERE unit_id=?", (unit.unit_id,)).fetchone()
        parent = DurableParentRecordV3.model_validate_json(raw)
        evidence = parent.unit_evidence.evidence
        archive = evidence.runtime_evidence.production_observability_archive
        assert archive is not None
        original_read = reopened.private.read_record
        for field, value in (("trajectory_seed", 1), ("concrete_seed_id", "1"),
                             ("order_key", 9), ("analysis_inclusion", "excluded")):
            row = archive.records[0]
            trial_evidence = row.evidence
            if field == "analysis_inclusion":
                trial_evidence = trial_evidence.model_copy(update={"trial": trial_evidence.trial.model_copy(
                    update={field: value})})
            else:
                trial_evidence = trial_evidence.model_copy(update={field: value})
            changed = archive.model_copy(update={"records": (row.model_copy(update={"evidence": trial_evidence}),
                *archive.records[1:])})
            runtime = evidence.runtime_evidence.model_copy(update={"production_observability_archive": changed})
            forged = parent.model_copy(update={"unit_evidence": parent.unit_evidence.model_copy(update={
                "evidence": evidence.model_copy(update={"runtime_evidence": runtime})})})
            forged_raw = json.dumps(forged.model_dump(mode="json"), sort_keys=True, allow_nan=False).encode()
            with monkeypatch.context() as tamper:
                tamper.setattr(type(reopened.private), "read_record", lambda private, name, **kwargs: (
                    forged_raw if name == f"{unit.unit_id}.parent.json" else original_read(name, **kwargs)))
                with pytest.raises(EntrypointError, match="MAIN_AUTHORIZATION_BINDING_MISMATCH"):
                    reopened._load_parent(unit.unit_id, forged_raw, hashlib.sha256(forged_raw).hexdigest())
    finally:
        reopened.close()


@pytest.mark.parametrize("baseline,failed_stage,failed_ordinal", (
    ("bot_style", "bot_problem_distill", 0),
    ("bot_style", "bot_instantiate_solve", 0),
    ("bot_style", "bot_thought_distill", 0),
    ("bot_style", "bot_problem_distill", 49),
    ("dc_rs", "dc_rs_synthesize", 0),
    ("dc_rs", "dc_rs_generate", 0),
))
def test_early_native_semantic_failure_publishes_and_reopens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deny_main_external: Callable[[], None],
    baseline: str, failed_stage: str, failed_ordinal: int,
) -> None:
    prefix = replace(next(unit for unit in _units() if unit.memory_baseline == baseline), sequence=0)
    suffix = replace(prefix, sequence=1, kind="MEMORY_BEARING", arm="contam",
        prefix_unit_id=prefix.unit_id, execution_template_id=f"game24|{baseline}|contam",
        unit_id=hashlib.sha256(json.dumps([
            "phase13-main-a-disjoint-unit-id-v1", "MEMORY_BEARING", 0,
            "game24", baseline, "contam",
        ], separators=(",", ":")).encode()).hexdigest())
    for path, raw in build_entrypoint_bytes((0,), production_units=(prefix, suffix)).items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    seal_fixture_closure(tmp_path)
    request = SelectionRequest(tmp_path, tmp_path / "package.json", tmp_path / "authorization.json",
        AUTHORITY, tmp_path / "authorization.sha256", corrective_identity().run_id)
    selected = select_execution(request, "run")
    assert isinstance(selected, SelectedExecutionV3)
    calls: list[str] = []

    class Provider(NativeProvider):
        def send_compiled_v3(self, compiled, before_request):
            calls.append(compiled.key.dispatch_id)
            if (compiled.key.parent_id == suffix.unit_id and failed_stage == "dc_rs_synthesize"
                and compiled.key.stage == failed_stage and compiled.key.ordinal == failed_ordinal):
                before_request()
                raise RuntimeError("provider unavailable")
            response = super().send_compiled_v3(compiled, before_request)
            if (compiled.key.parent_id == suffix.unit_id and compiled.key.stage == failed_stage
                and compiled.key.ordinal == failed_ordinal):
                return LLMResponse("invalid", response.raw, response.token_usage, response.latency_ms)
            return response

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(ProductionMainRuntime, "_embedder", lambda _self: _ContractFakeEmbeddingProvider(vector_dimension=1024))
    directory = tmp_path / "early-native"
    run = V3MainRun.open(selected, directory, create=True, seed=0)
    deny_main_external()
    try:
        report = run.execute(tmp_path / "cache", max_units=2, tranche_ceiling_krw=450000,
            provider_factory=lambda _binding: Provider())
        assert report.completed_count == 1
        assert report.terminal_technical_missing_count == 1
        with run.private.connect() as connection:
            raw, sha256 = connection.execute("SELECT raw, sha256 FROM parents WHERE unit_id=?",
                (suffix.unit_id,)).fetchone()
        assert raw is not None
        parent = run._load_parent(suffix.unit_id, raw, sha256)
        if failed_stage == "dc_rs_synthesize":
            assert isinstance(parent, TerminalPartialParent)
            assert parent.terminal_key.stage == failed_stage
        else:
            assert isinstance(parent, DurableParentRecordV3)
            archive = parent.unit_evidence.evidence.runtime_evidence.production_observability_archive
            assert archive is not None
            assert archive.records[-1].terminal_method_call is not None
            assert archive.records[-1].terminal_method_call.stage == failed_stage
            if baseline == "bot_style" and failed_stage == "bot_thought_distill":
                assert isinstance(archive.records[-1].evidence, Phase13TrialEvidence)
                assert archive.records[-1].evidence.target_set.answer_call_id != archive.records[-1].terminal_method_call.call_id
    finally:
        run.close()
    selected = select_execution(request, "resume")
    assert isinstance(selected, SelectedExecutionV3)
    reopened = V3MainRun.open(selected, directory, create=False, seed=0)
    try:
        assert reopened.status().terminal_technical_missing_count == 1
        reopened.execute(tmp_path / "cache", max_units=2, tranche_ceiling_krw=450000,
            provider_factory=lambda _binding: Provider())
        stages = (("bot_problem_distill", "bot_instantiate_solve", "bot_thought_distill")
                  if baseline == "bot_style" else ("dc_rs_synthesize", "dc_rs_generate"))
        assert len(calls) == len(stages) * (failed_ordinal + 1) + stages.index(failed_stage) + 1
    finally:
        reopened.close()
