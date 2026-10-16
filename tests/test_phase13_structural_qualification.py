import pytest
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from memcontam.readiness.phase13_structural_qualification import generate_native_qualification
from .test_phase13_v3_native_parent_reopen import deny_main_external, NativeProvider, _units
from .test_phase13_v3_entrypoint_fixture import build_entrypoint_bytes, seal_fixture_closure, AUTHORITY
from .phase13_corrective_identity import corrective_identity


def test_native_qualification_uses_ordinary_executor_and_frozen_fixture():
    qualification = generate_native_qualification("a" * 64)
    assert len(qualification.fixture_proofs) == 5
    assert qualification.capacity == 8192
    assert qualification.qualification_only and not qualification.scientific_authorization
    assert qualification.real_provider_calls == 0
    resources = tuple(json.loads(raw) for raw in qualification.resource_observations)
    assert {(row["task"], row["arm"]) for row in resources} == {
        (task, arm) for task in ("game24", "math_equation_balancer", "word_sorting")
        for arm in ("clean", "correct", "irrelevant", "contam")}
    assert all(len(row["observations"]) == 50 for row in resources)
    repeated = generate_native_qualification("a" * 64)
    for before, after in zip(qualification.fixture_proofs, repeated.fixture_proofs, strict=True):
        first, second = json.loads(before), json.loads(after)
        assert first == second, (first["baseline"], {key: (first[key], second[key])
            for key in first if first[key] != second[key]})
    assert repeated == qualification


def test_selected_origin_is_client_free_and_bound_to_durable_prefix(tmp_path, monkeypatch, deny_main_external):
    from memcontam.readiness.phase13_v3_entrypoint import SelectionRequest, SelectedExecutionV3, select_execution
    from memcontam.readiness.phase13_main_v3_runner import V3MainRun
    from memcontam.readiness.phase13_main_live_runtime import ProductionMainRuntime
    from memcontam.readiness.phase13_main_preloaded_resources import PreloadedMainResources
    from memcontam.readiness.phase13_main_terminal_partial import TerminalPartialParent
    from memcontam.readiness.phase13_production_runtime_join import production_archive_from_completed_prefix, production_archive_from_ordinary
    from memcontam.evaluation.phase13_seed_response import (
        structural_support_context, StructuralSupportEvidence, validate_structural_support,
        producer_joined_parent, extract_seed_response, SeedResponseError, registered_pair_support,
    )
    from .test_phase13_readiness0_production_dry_run import _ContractFakeEmbeddingProvider
    from memcontam.clients.base import LLMResponse
    from memcontam.readiness import phase13_main_live_runtime

    units = []
    for baseline in ("fh_bounded", "bot_style"):
        prefix = replace(_units()[3], sequence=len(units), memory_baseline=baseline,
            unit_id=hashlib.sha256(json.dumps(["phase13-main-a-disjoint-unit-id-v1", "CLEAN_PREFIX", 0,
                "game24", baseline, "NOT_APPLICABLE"], separators=(",", ":")).encode()).hexdigest(),
            execution_template_id=f"game24|{baseline}|prefix")
        suffix = replace(prefix, sequence=len(units) + 1, kind="MEMORY_BEARING", arm="clean",
            prefix_unit_id=prefix.unit_id, execution_template_id=f"game24|{baseline}|clean",
            unit_id=hashlib.sha256(json.dumps(["phase13-main-a-disjoint-unit-id-v1", "MEMORY_BEARING", 0,
                "game24", baseline, "clean"], separators=(",", ":")).encode()).hexdigest())
        units.extend((prefix, suffix))
    for path, raw in build_entrypoint_bytes((0,), production_units=tuple(units)).items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    seal_fixture_closure(tmp_path, structural_qualification=True)
    request = SelectionRequest(tmp_path, tmp_path / "package.json", tmp_path / "authorization.json",
        AUTHORITY, tmp_path / "authorization.sha256", corrective_identity().run_id)
    selected = select_execution(request, "run")
    assert isinstance(selected, SelectedExecutionV3)
    _assert_selected_domain_coverage(selected)
    units = list(selected.package.production)
    suffix_ids = {units[1].unit_id, units[3].unit_id}
    captured = []
    ordinary_execute = phase13_main_live_runtime.execute_prospective_ordinary

    def capture(run):
        result = ordinary_execute(run)
        captured.append((run, result))
        return result

    class Provider(NativeProvider):
        def send_compiled_v3(self, compiled, before_request):
            if compiled.key.parent_id in suffix_ids and compiled.key.ordinal >= 3:
                before_request()
                raise RuntimeError("synthetic terminal fixture")
            response = super().send_compiled_v3(compiled, before_request)
            if compiled.key.stage == "bot_instantiate_solve":
                payload = json.loads(response.content)
                payload["selected_structure"] = "retrieved-template" if compiled.key.parent_id in suffix_ids else "procedure-based"
                return LLMResponse(json.dumps(payload), response.raw, response.token_usage, response.latency_ms)
            return response

    monkeypatch.setattr(phase13_main_live_runtime, "execute_prospective_ordinary", capture)
    monkeypatch.setattr(ProductionMainRuntime, "_embedder",
        lambda _self: _ContractFakeEmbeddingProvider(vector_dimension=1024))
    # Exercise the real scientific-identity branches with synthetic transport.
    # This is an explicit runtime constructor, not production_provider identity
    # spoofing or a relaxation of producer/scientific-result guards.
    class QualificationRuntime(ProductionMainRuntime):
        def __init__(self, root, cache, *, client, resources, scientific_result):
            super().__init__(root, cache, client=client, resources=resources,
                scientific_result=True)

    monkeypatch.setattr("memcontam.readiness.phase13_main_v3_runner.ProductionMainRuntime", QualificationRuntime)
    runner = V3MainRun.open(selected, tmp_path / "native-run", create=True, seed=0)
    deny_main_external()
    try:
        runner.execute(tmp_path / "cache", max_units=4, tranche_ceiling_krw=450000,
            provider_factory=lambda _binding: Provider())
        from memcontam.readiness.phase13_v3_request import RequestKeyV3
        runner.ledger.reconcile_cost(RequestKeyV3(parent_id=units[1].unit_id,
            stage="full_history_generate", ordinal=3).dispatch_id,
            {"usage": {"input_tokens": 1, "output_tokens": 1}}, "f" * 64)
        runner.execute(tmp_path / "cache", max_units=4, tranche_ceiling_krw=450000,
            provider_factory=lambda _binding: Provider())
        assert len(captured) == 2
        before = {path: path.read_bytes() for path in (tmp_path / "native-run").rglob("*")
                  if path.is_file() and not path.name.endswith("-shm")}

        def forbidden(*_args, **_kwargs):
            pytest.fail("production structural application executed/copied a client or requalified")

        monkeypatch.setattr("memcontam.evaluation.phase13_seed_response.execute_prospective_ordinary", forbidden)
        monkeypatch.setattr("memcontam.experiment.phase13_ordinary_runtime.execute_prospective_ordinary", forbidden)
        monkeypatch.setattr("memcontam.readiness.phase13_structural_qualification.generate_native_qualification", forbidden)
        monkeypatch.setattr(V3MainRun, "open", forbidden)
        monkeypatch.setattr(V3MainRun, "status", forbidden)
        contexts = []
        proofs = []
        for unit, (run, result) in zip((units[1], units[3]), captured, strict=True):
            monkeypatch.setattr(type(run.client), "chat", forbidden)
            monkeypatch.setattr(type(run.client), "__deepcopy__", forbidden, raising=False)
            monkeypatch.setattr(type(run.client), "__copy__", forbidden, raising=False)
            context = structural_support_context(selected, unit, runner.ledger)
            assert context is not None
            proof = StructuralSupportEvidence.model_validate_json(context.qualified_evidence_bytes)
            assert proof.package_sha256 == selected.package_sha256 and context.identity.scientific_result
            assert not proof.conformance_trace and not proof.replay_witness
            validate_structural_support(proof, context)
            contexts.append(context)
            proofs.append(proof)
            identity = run.production_identity
            assert identity is not None
            assert identity == context.identity and identity.scientific_result
            assert not run.allow_test_client and run.validated_resources is not None
            raw_parent = runner.ledger.read_record(f"{unit.unit_id}.parent.json")
            cost_unit = next(row for row in selected.costs.resources.phase4.base.units if row.unit_id == unit.unit_id)
            if json.loads(raw_parent)["schema_version"] == "phase13_main_terminal_partial_parent_v1":
                partial = TerminalPartialParent.model_validate_json(raw_parent)
                result = replace(result, trials=result.trials[:len(partial.archive.records)])
                archive = production_archive_from_completed_prefix(run, result, identity)
                terminal_key = partial.terminal_key
            else:
                from memcontam.readiness.phase13_durable_call_join import scheduled_prefix_keys
                archive = production_archive_from_ordinary(run, result, identity)
                terminal = archive.records[-1].terminal_method_call
                assert terminal is not None
                keys = scheduled_prefix_keys(unit, cost_unit, archive.records)
                assert keys
                terminal_key = keys[-1]
            parent = producer_joined_parent(run, result, identity, archive,
                structural_support_evidence=proof, structural_support_inputs=context)
            response = extract_seed_response(archive, PreloadedMainResources(selected).packet,
                identity.registration_packet_sha256, run.tasks, parent, run=run, result=result, identity=identity,
                structural_support_inputs=context, terminal_ledger=runner.ledger,
                terminal_key=terminal_key, production_unit=unit, cost_unit=cost_unit)
            assert response.parent.structural_support is True and response.exact_mean is None
            assert all(record.scientific_result for record in archive.records)
            for changes in ({"supported": False, "unsupported_reason": "invented"},
                            {"conformance_trace": (("clean", 1, "fake", "a" * 64, "b" * 64, "c" * 64),)},
                            {"fixture_contract_sha256": "f" * 64}, {"candidate_registry_sha256": "f" * 64},
                            {"checkpoint_sha256": "f" * 64}, {"qualification_sha256": "f" * 64},
                            {"branch_checkpoint_sha256": tuple(reversed(proof.branch_checkpoint_sha256))},
                            {"ordered_sample_ids_sha256": "f" * 64}, {"handler_contract_sha256": "f" * 64}):
                from memcontam.readiness.phase13_v3_cost_models import digest
                forged = proof.model_copy(update=changes)
                forged = forged.model_copy(update={"evidence_sha256": hashlib.sha256(json.dumps(
                    forged.model_dump(mode="json", exclude={"evidence_sha256"}),
                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
                raw = forged.model_dump_json().encode()
                with pytest.raises(SeedResponseError, match="STRUCTURAL_QUALIFIED_CONTEXT_MISMATCH"):
                    extract_seed_response(archive, PreloadedMainResources(selected).packet,
                        identity.registration_packet_sha256, run.tasks,
                        parent.model_copy(update={"structural_support": forged.supported,
                            "structural_support_evidence": forged,
                            "structural_support_evidence_sha256": forged.evidence_sha256}),
                        run=run, result=result, identity=identity,
                        structural_support_inputs=replace(context, qualified_evidence_bytes=raw,
                            qualified_evidence_sha256=hashlib.sha256(raw).hexdigest()),
                        terminal_ledger=runner.ledger, terminal_key=terminal_key,
                        production_unit=unit, cost_unit=cost_unit)
                assert context.production_origin is not None
                with pytest.raises(SeedResponseError, match="STRUCTURAL_QUALIFICATION_ORIGIN_REQUIRED"):
                    validate_structural_support(forged, replace(context, qualified_evidence_bytes=raw,
                        qualified_evidence_sha256=hashlib.sha256(raw).hexdigest(),
                        production_origin=replace(context.production_origin, proof_bytes=raw)))
            with pytest.raises(SeedResponseError, match="STRUCTURAL_QUALIFICATION_ORIGIN_REQUIRED"):
                extract_seed_response(archive, PreloadedMainResources(selected).packet,
                    identity.registration_packet_sha256, run.tasks, parent,
                    run=run, result=result, identity=identity,
                    structural_support_inputs=replace(context, production_origin=None),
                    terminal_ledger=runner.ledger, terminal_key=terminal_key,
                    production_unit=unit, cost_unit=cost_unit)
            assert context.production_origin is not None
            for update in ({"capacity": 8191}, {"native_policy": "invented"},
                           {"authorization_sha256": "f" * 64}, {"ordered_task_bytes": []}):
                subject = json.loads(context.production_origin.subject_bytes)
                subject.update(update)
                with pytest.raises(SeedResponseError, match="STRUCTURAL_QUALIFICATION_ORIGIN_REQUIRED"):
                    extract_seed_response(archive, PreloadedMainResources(selected).packet,
                        identity.registration_packet_sha256, run.tasks, parent,
                        run=run, result=result, identity=identity,
                        structural_support_inputs=replace(context, production_origin=replace(
                            context.production_origin, subject_bytes=json.dumps(subject).encode())),
                        terminal_ledger=runner.ledger, terminal_key=terminal_key,
                        production_unit=unit, cost_unit=cost_unit)
        assert registered_pair_support(proofs[0], proofs[1], contexts[0], contexts[1])
        assert {path: path.read_bytes() for path in before} == before
        monkeypatch.setattr("memcontam.evaluation.phase13_seed_response.execute_prospective_ordinary", ordinary_execute)
        monkeypatch.setattr("memcontam.experiment.phase13_ordinary_runtime.execute_prospective_ordinary", ordinary_execute)
        monkeypatch.setattr("memcontam.readiness.phase13_structural_qualification.generate_native_qualification", generate_native_qualification)
        from memcontam.readiness.phase13_structural_qualification import CARRIER, NativeQualification
        from memcontam.readiness.phase13_v3_conformance import ConformanceV3
        from memcontam.readiness.phase13_v3_cost_models import canonical_bytes, digest
        conformance = ConformanceV3.model_validate_json((tmp_path / CARRIER).read_bytes())
        qualified = conformance.structural
        assert qualified is not None
        assert len(qualified.preservation_cases) == 5
        assert len(qualified.covered_cells) == 23
        fake_trace = json.loads(qualified.fixture_proofs[0])
        fake_rows = fake_trace["conformance_trace"]
        fake_rows[0][5] = "f" * 64
        fake_rows[1][4] = "f" * 64
        fake_trace["suffix_contract_sha256"] = hashlib.sha256(json.dumps(fake_rows, separators=(",", ":")).encode()).hexdigest()
        for changes in (
            {"generator_sha256": "f" * 64}, {"fixture_sha256": "f" * 64},
            {"subject_sha256": "f" * 64}, {"native_laws": (("fh_bounded", "invented-policy"),)},
            {"capacity": 8191}, {"native_domain": "invented-domain"}, {"arms": tuple(reversed(qualified.arms))},
            {"fixture_proofs": (json.dumps(fake_trace), *qualified.fixture_proofs[1:])},
            {"boundary_observations": (("invented-success", "SUCCEEDED", "f" * 64),)},
            {"covered_cells": (("game24", "fh_bounded"),)},
            {"preservation_cases": (("dc_rs", "invented", "universal-success"),)},
            {"resource_observations": ()},
        ):
            altered = qualified.model_copy(update=changes)
            altered = altered.model_copy(update={"qualification_sha256": digest(altered, "qualification_sha256")})
            changed = conformance.model_copy(update={"structural": altered})
            changed = changed.model_copy(update={"conformance_hash": digest(changed, "conformance_hash")})
            _rebind_carrier(tmp_path, canonical_bytes(changed))
            with pytest.raises(ValueError, match="STRUCTURAL_QUALIFICATION_BINDING_MISMATCH"):
                select_execution(request, "validate")
        missing = conformance.model_copy(update={"structural": None})
        missing = missing.model_copy(update={"conformance_hash": digest(missing, "conformance_hash")})
        _rebind_carrier(tmp_path, canonical_bytes(missing))
        unqualified = select_execution(request, "validate")
        assert isinstance(unqualified, SelectedExecutionV3)
        try:
            assert structural_support_context(unqualified, unqualified.package.production[1], runner.ledger) is None
        finally:
            unqualified.close()
    finally:
        runner.close()


def _rebind_carrier(root: Path, raw: bytes) -> None:
    from memcontam.readiness.phase13_structural_qualification import CARRIER
    from memcontam.readiness.phase13_v3_entrypoint_models import MainExecutionPackageV3, MainAuthorizationV3
    from memcontam.readiness.phase13_v3_resource_files import FileBinding
    from memcontam.readiness.phase13_v3_source_closure import _rows_hash
    from memcontam.readiness.phase13_v3_cost_models import canonical_bytes, digest
    package = MainExecutionPackageV3.model_validate_json((root / "package.json").read_bytes())
    closure = package.generated_closure
    assert closure is not None
    rows = tuple(FileBinding(path=row.path, size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        if row.path == CARRIER else row for row in closure.rows)
    closure = closure.model_copy(update={"rows": rows, "resource_closure_sha256": _rows_hash(rows)})
    package = package.model_copy(update={"generated_closure": closure, "generated_closure_hash": closure.resource_closure_sha256})
    package = package.model_copy(update={"package_hash": digest(package, "package_hash")})
    authorization = MainAuthorizationV3.model_validate_json((root / "authorization.json").read_bytes())
    authorization = authorization.model_copy(update={"execution_package_sha256": digest(package),
        "execution_package_hash": package.package_hash})
    authorization = authorization.model_copy(update={"authorization_hash": digest(authorization, "authorization_hash")})
    for path, content in ((CARRIER, raw), ("package.json", canonical_bytes(package)),
                          ("authorization.json", canonical_bytes(authorization)),
                          ("authorization.sha256", (digest(authorization) + "\n").encode())):
        (root / path).write_bytes(content)


def _assert_selected_domain_coverage(selected):
    from memcontam.baselines.bot_phase12 import BoTStateV3
    from memcontam.baselines.dynamic_cheatsheet_phase12 import DcRsStateV3
    from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
    from memcontam.baselines.reflexion_phase12 import ReflexionStateV3
    from memcontam.experiment.phase12.branching import BranchSet, build_matched_branches
    from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
    from memcontam.memory.admission import AdmissionContext
    from memcontam.memory.checkpoint_v3 import NATIVE_ENTRY_V1, NativeEntry, NativeState, append_native_entry, serialize_checkpoint
    from memcontam.memory.cards_v3 import canonical_content_hash
    from memcontam.readiness.phase13_main_preloaded_resources import PreloadedMainResources
    from memcontam.readiness.phase13_main_new_mcq_runtime import new_mcq_native_entries
    from memcontam.readiness.phase13_main_live_runtime_support import core_task_name
    from memcontam.readiness.phase13_structural_qualification import COVERED_CELLS, ARMS, qualified_native_domain
    loaded = PreloadedMainResources(selected)
    for task, baseline in COVERED_CELLS:
        unit = replace(selected.package.production[1], task=task, memory_baseline=baseline)
        order = loaded.checkpoint_registry.tasks[task].seeds[0].suffix_sample_ids
        by_id = {row.sample_id: row for row in loaded.tasks(task)}
        tasks = tuple(by_id[sample] for sample in order)
        if baseline == "rag_frozen":
            corpus, indices = loaded.legacy_bundles(task)
            clean = corpus.branches["clean"]
            native = NativeState("rag_frozen", tuple(NativeEntry(entry_id=doc.id,
                semantic_kind="rag_document", schema_version=NATIVE_ENTRY_V1, native_component="corpus",
                content=doc.text, content_hash=canonical_content_hash(doc.text)) for doc in clean.documents),
                {"branch": "clean", "read_only": True, "corpus_id": clean.serialization_id,
                 "index_id": indices.branches["clean"].index_serialization_id})
        else:
            states = {"fh_bounded": FullHistoryStateV3(records=[]), "bot_style": BoTStateV3(entries=[]),
                "reflexion_style": ReflexionStateV3(reflections=[]),
                "dc_rs": DcRsStateV3(archive=[], allow_unparented_strategies=True)}
            native = PHASE13_CORE_BASELINE_REGISTRY[baseline].serialize_state(states[baseline])
        assert isinstance(native, NativeState)
        prefix = serialize_checkpoint(native, checkpoint_index=1)
        if task.startswith("mmlu_pro_"):
            roots = new_mcq_native_entries(core_task_name(task), baseline, loaded.new_mcq_registry)
            branches = (prefix, *(append_native_entry(prefix, roots[arm]) for arm in ARMS[1:]))
        else:
            triplet = next(row for row in loaded.candidate_registry.triplets if row.task == task)
            matched = build_matched_branches(prefix, triplet, loaded.legacy_dc_rs_renderers, AdmissionContext())
            assert isinstance(matched, BranchSet)
            branches = tuple(next(row.checkpoint for row in matched.materialized if row.arm == arm) for arm in ARMS)
        assert qualified_native_domain(selected, unit, prefix, branches, tasks) == "SUPPORTED", (task, baseline)
        assert qualified_native_domain(selected, unit, prefix, branches, tasks[:-1]) is None
        foreign = replace(native, native_state={**native.native_state, "uncovered_policy": True})
        foreign_prefix = serialize_checkpoint(foreign, checkpoint_index=1)
        assert qualified_native_domain(selected, unit, foreign_prefix, branches, tasks) is None
