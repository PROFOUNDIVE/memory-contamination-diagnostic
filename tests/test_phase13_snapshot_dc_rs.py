"""Synthetic native DC-RS contracts through the shared snapshot route."""

from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final, Literal

import pytest

from memcontam.baselines import dynamic_cheatsheet_phase12 as dc
from memcontam.clients.base import LLMResponse
from memcontam.clients.replay import ReplayClient
from memcontam.evaluation.phase13_observability import reconstruct_phase13_trial
from memcontam.evaluation.phase13_observability_models import Phase13TrialEvidence
from memcontam.evaluation.phase13_observability_registration import registered_verifier_result
from memcontam.experiment.phase12.game24_runner import RuntimeIdentities
from memcontam.experiment.phase12.live_branch import LiveArmBranch
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY, RuntimeStateError, RuntimeTrialResult
from memcontam.experiment.phase13_dc_rs_runtime import Phase13DcRsContext
from memcontam.experiment.phase13_ordinary_runtime import ProspectiveOrdinaryRun
from memcontam.experiment.phase13_snapshot_diagnostic import SnapshotBindings, SnapshotError, SnapshotFileBinding, execute_snapshot_unit, snapshot_input_sha256
from memcontam.logging.schema_v3 import ContextEvent
from memcontam.memory.checkpoint_v3 import CheckpointError, NativeEntry, NativeState, deserialize_checkpoint, serialize_checkpoint
from memcontam.memory.cards_v3 import canonical_content_hash
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.memory.stores import MemoryEntry
from memcontam.readiness.phase13_production_runtime_models import ProductionOrdinaryRunIdentity
from memcontam.tasks.base import TaskInstance

from .test_phase13_dc_rs_runtime import _EmbeddingProvider
from .test_phase13_snapshot_bot import capacity_fixture as capacity_fixture


RUNTIME: Final = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
Arm = Literal["clean", "correct", "irrelevant", "contam"]


class SnapshotClient(ReplayClient):
    def chat(self, messages, model, config) -> LLMResponse:
        response = super().chat(messages, model, config)
        content = response.content
        if config["method_stage"] == "dc_rs_synthesize":
            offered = [span.entry_id for span in config["source_spans"]
                       if span.entry_id in {"qa-target", "qa-unrelated"}]
            for alias, entry_id in (("src01", "qa-target"), ("src02", "qa-unrelated")):
                tag = f"<source_ids>{alias}</source_ids>"
                if tag in response.content:
                    content = response.content.replace(tag, f"<source_ids>src{offered.index(entry_id) + 1:02d}</source_ids>")
        return replace(response, content=content, raw={**response.raw, "attempts": 1})


@cache
def _native_embedder() -> BgeM3EmbeddingProvider:
    return BgeM3EmbeddingProvider()


@pytest.fixture(params=[
    TaskInstance(sample_id="qa-meb", task_name="math_equation_balancer",
                 input={"input": "1 ? 1 = 2"}, verifier_spec={"target": "1 + 1 = 2", "target_value": 2}),
    TaskInstance(sample_id="qa-words", task_name="word_sorting",
                 input={"words": ["pear", "apple"]}, verifier_spec={"sorted_words": ["apple", "pear"]}),
], ids=["MEB", "WordSorting"])
def task(request: pytest.FixtureRequest) -> TaskInstance:
    assert isinstance(request.param, TaskInstance)
    return request.param


def _context(task: TaskInstance, sources: str = "src01", arm: Arm = "contam") -> Phase13DcRsContext:
    prior = json.dumps({"input": task.input, "metadata": {}, "sample_id": "qa-prior",
                        "task_name": task.task_name}, sort_keys=True, separators=(",", ":"))
    target = MemoryEntry(
        entry_id="qa-target", content=prior, memory_type="dc_rs_io_pair",
        source_trial_id="qa:trial:1:qa-prior", metadata={"generated_output": "qa-prior-output"},
    )
    intervened = arm != "clean"
    if intervened:
        target = target.model_copy(update={
            "content": json.dumps({"input": task.input, "raw_output": "qa-raw-interaction"},
                                  sort_keys=True, separators=(",", ":")),
            "source_trial_id": None, "metadata": {"render_id": f"synthetic-raw-{arm}", "generated_output": "qa-raw-interaction"},
        })
    decoy = MemoryEntry(entry_id="qa-unrelated", content=prior, memory_type="dc_rs_io_pair",
                        source_trial_id="qa:trial:1:qa-prior", metadata={"generated_output": "qa-unrelated-output"})
    state = dc.DcRsStateV3(archive=[target, decoy], allow_unparented_strategies=True,
                           injected_root_id="qa-target" if intervened else None)
    answer = str(task.verifier_spec.get("target", "apple pear"))
    return Phase13DcRsContext(
        task=task, client=SnapshotClient(responses=[
            "<cheatsheet>qa-derived-guide</cheatsheet>" + (f"<source_ids>{sources}</source_ids>" if sources else ""),
            f"final: {answer}",
        ]), model="replay", verifier=lambda value, row: registered_verifier_result(row, value).is_correct,
        decoding={"temperature": 0.0}, branch=arm,
        identities=RuntimeIdentities("qa", f"qa:{arm}:trial:2:{task.sample_id}", 2, f"dc_rs:{arm}"),
        embedding_provider=_EmbeddingProvider(),
        baseline_configs={"dc_rs": {"embedding_mode": "test_double", "tool_mode": "text_only",
                                    "serialized_cheatsheet_budget_tokens": 8192}},
        initial_states={"dc_rs": state}, expected_intervention=dc._archive_native(target) if intervened else None,
    )


def _shared_fixture(context: Phase13DcRsContext) -> tuple[ProspectiveOrdinaryRun, SnapshotBindings]:
    state = RUNTIME.initial_state(context)
    assert isinstance(state, dc.DcRsStateV3)
    before = RUNTIME.serialize_state(state)
    assert isinstance(before, NativeState)
    assert context.task.task_name in ("math_equation_balancer", "word_sorting")
    assert context.branch != "filter"
    checkpoint = serialize_checkpoint(before, checkpoint_index=0)
    branch = LiveArmBranch(context.branch, "qa-prefix", "qa-source", checkpoint, state,
                           0 if context.branch == "clean" else 1, injected_root_id=state.injected_root_id,
                           candidate_triplet_id=None if context.branch == "clean" else "synthetic-qa",
                           native_render_id=None if context.branch == "clean" else "synthetic-raw")
    run = ProspectiveOrdinaryRun(
        task_name=context.task.task_name, baseline="dc_rs", run_id=context.identities.run_id,
        model="gpt-5.6-luna", client=context.client, allow_test_client=True, verifier=context.verifier,
        decoding=context.decoding, tasks=(context.task,), trajectory_seed=0, arm=context.branch,
        branch=branch, embedding_provider=_native_embedder(), baseline_configs={"dc_rs": {
            key: value for key, value in context.baseline_configs["dc_rs"].items() if key != "embedding_mode"}},
    )
    root = Path(__file__).resolve().parents[1]
    sources = tuple(root / "src/memcontam" / relative for relative in (
        "experiment/phase12/runtime_registry.py", "experiment/phase13_ordinary_runtime.py",
        "experiment/phase13_snapshot_diagnostic.py", "experiment/phase13_dc_rs_runtime.py",
        "experiment/phase13_dc_rs_validation.py", "baselines/dynamic_cheatsheet_phase12.py",
        "baselines/dynamic_cheatsheet_optional.py",
    ))
    bindings = SnapshotBindings(checkpoint.canonical_sha256, snapshot_input_sha256(run),
        tuple(SnapshotFileBinding(path, hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources),
        ProductionOrdinaryRunIdentity(execution_template_id=f"{run.task_name}:dc_rs:{run.arm}",
            trajectory_seed=0, concrete_seed_id="0", scientific_result=False,
            ordered_sample_ids_sha256=hashlib.sha256(json.dumps((context.task.sample_id,), separators=(",", ":")).encode()).hexdigest(),
            registration_packet_sha256=hashlib.sha256((root / "data/phase13/observability/registration_packet_v2.json").read_bytes()).hexdigest()))
    return run, bindings


def _execute(context: Phase13DcRsContext) -> tuple[RuntimeTrialResult, Phase13TrialEvidence]:
    run, bindings = _shared_fixture(context)
    with TemporaryDirectory(prefix="dc-snapshot-") as directory:
        unit = execute_snapshot_unit(run, bindings, Path(directory))
    result = unit.ordinary.trials[0]
    evidence = unit.archive.records[0].evidence
    assert isinstance(evidence, Phase13TrialEvidence)
    assert unit.provenance == "provider_free_qa" and unit.archive.records[0].scientific_result is False
    return result, evidence


@pytest.mark.parametrize("arm", ["clean", "correct", "irrelevant", "contam"])
def test_native_stages_and_writes_when_treatment_is_separate_from_baseline(task: TaskInstance, arm: Arm) -> None:
    context = _context(task, arm=arm)

    result, evidence = _execute(context)

    assert [call.stage for call in result.outcome.method_calls] == ["dc_rs_synthesize", "dc_rs_generate"]
    assert result.outcome.verifier_result is True
    analysis = reconstruct_phase13_trial(evidence)
    assert analysis.theory_exposure.value is (True if arm == "contam" else None)
    assert analysis.theory_exposure.status == ("supported" if arm == "contam" else "not_applicable")
    assert evidence.baseline == "dc_rs" and evidence.trial.branch_id == arm
    assert evidence.target_set.target_entry_ids == (("qa-target",) if arm == "contam" else ())
    assert [entry.native_component for entry in result.native_entries] == ["archive", "strategy"]
    assert [row.writer_stage for row in result.write_envelopes] == ["dc_rs_generate", "dc_rs_synthesize"]
    curator, answer = result.outcome.method_calls
    assert "qa-target" in {span.entry_id for span in curator.source_spans}
    assert result.outcome.answer_call_id == answer.call_id
    assert isinstance(result.context_event, ContextEvent)
    assert result.context_event.final_entry_ids == [result.native_entries[1].entry_id]
    assert answer.source_spans[0].entry_id == result.native_entries[1].entry_id
    assert answer.source_spans[0].parent_call_id == curator.call_id
    span = answer.source_spans[0]
    rendered = answer.messages[span.message_index]["content"][span.start:span.end]
    assert hashlib.sha256(rendered.encode()).hexdigest() == span.rendered_hash
    assert rendered == result.native_entries[1].content
    assert result.native_entries[0].direct_parent_ids == (result.native_entries[1].entry_id,)
    assert RUNTIME.serialize_state(context.initial_states["dc_rs"]) == result.state_before


@pytest.mark.parametrize("case", [
    ("src01", "supported", True), ("src02", "supported", False), ("", "unavailable", None),
])
def test_target_attribution_when_only_explicit_curator_parents_count(
    task: TaskInstance, case: tuple[str, str, bool | None], tmp_path: Path,
) -> None:
    from memcontam.experiment.phase13_snapshot_diagnostic import snapshot_dispatch, snapshot_input_sha256
    from memcontam.readiness.phase13_main_live_dispatch import persist_unit_dispatch
    from memcontam.readiness.phase13_production_observability import ProductionObservabilityArchive
    from .phase13_snapshot_durable_fixtures import ContractTransport
    from .test_phase13_snapshot_durable import _ledger, _reopen

    sources, status, included = case
    context = _context(task, sources)
    assert isinstance(context.client, ReplayClient)
    context.client.responses[0] = context.client.responses[0].replace("qa-derived-guide", "qa-raw-interaction")
    run, bindings = _shared_fixture(context)
    run = replace(run, client=ContractTransport(run.client), decoding={"temperature": 0.0, "top_p": 1.0})
    bindings = replace(bindings, input_sha256=snapshot_input_sha256(run),
        archive_identity=bindings.archive_identity.model_copy(update={"checkpoint_registry_sha256": bindings.state_sha256}))

    snapshot = execute_snapshot_unit(run, bindings, tmp_path)
    result, evidence = snapshot.ordinary.trials[0], snapshot.archive.records[0].evidence
    assert isinstance(evidence, Phase13TrialEvidence)
    unit, output = snapshot_dispatch(snapshot)
    ledger = _ledger(tmp_path, unit)
    ledger.persist_completed(unit.unit_id, persist_unit_dispatch(tmp_path, unit, output))
    child = _reopen(tmp_path)
    assert child.returncode == 0, child.stderr
    archive = ProductionObservabilityArchive.model_validate_json(json.dumps(json.loads(child.stdout)["archive"]))
    assert archive == snapshot.archive
    reopened = archive.records[0].evidence
    assert isinstance(reopened, Phase13TrialEvidence)
    analysis = reconstruct_phase13_trial(reopened)

    assert analysis.target_present_in_store_before_answer.value is True
    assert analysis.target_retrieved.value is True
    for metric in (analysis.target_final_context_included, analysis.theory_exposure):
        assert (metric.status, metric.value) == (status, included)
    strategy = result.native_entries[1]
    curator, answer = result.outcome.method_calls
    assert "qa-target" in {span.entry_id for span in curator.source_spans}
    assert answer.source_spans[0].entry_id == strategy.entry_id
    assert strategy.content not in (answer.raw_response or "")
    assert strategy.direct_parent_ids == {"src01": ("qa-target",), "src02": ("qa-unrelated",), "": ()}[sources]
    assert strategy.lineage_status == ("exact" if sources else "unavailable")
    assert result.outcome.method_calls[-1].source_spans[0].lineage_status == strategy.lineage_status
    assert analysis.operational_use.value is None
    assert isinstance(result.state_after, NativeState)
    checkpoint = serialize_checkpoint(result.state_after)
    restored = RUNTIME.restore_state(deserialize_checkpoint(checkpoint), context)
    assert RUNTIME.serialize_state(restored) == result.state_after


def test_independent_native_writes_when_units_share_only_pre_state(task: TaskInstance) -> None:
    context = _context(task)
    other = replace(context, client=_context(task).client,
                    identities=RuntimeIdentities("qa", f"qa:contam:trial:3:{task.sample_id}", 3, "dc_rs:contam"))
    before = RUNTIME.serialize_state(RUNTIME.initial_state(context))
    assert isinstance(before, NativeState)
    left, right = (RUNTIME.restore_state(before, ctx) for ctx in (context, other))
    assert isinstance(left, dc.DcRsStateV3) and isinstance(right, dc.DcRsStateV3)

    result = RUNTIME.execute_trial(context, left)

    assert left is not right and left.archive is not right.archive
    assert RUNTIME.serialize_state(right) == before
    assert RUNTIME.serialize_state(context.initial_states["dc_rs"]) == before
    second = RUNTIME.execute_trial(other, right)
    assert {row.entry_id for row in result.native_entries}.isdisjoint(row.entry_id for row in second.native_entries)


@pytest.mark.parametrize("sources", ["src99", "qa-target"])
def test_counterfeit_source_alias_refused_before_answer(task: TaskInstance, sources: str) -> None:
    context = _context(task, sources)
    before = RUNTIME.serialize_state(RUNTIME.initial_state(context))

    with pytest.raises(RuntimeStateError, match="INVALID_EXPLICIT_SOURCE_IDS"):
        RUNTIME.execute_trial(context, context.initial_states["dc_rs"])

    assert isinstance(context.client, ReplayClient) and context.client.index == 1
    assert RUNTIME.serialize_state(context.initial_states["dc_rs"]) == before


@pytest.mark.parametrize("field", ["baseline", "family", "content", "parent"])
def test_counterfeit_snapshot_refused_when_family_state_or_parent_changes(task: TaskInstance, field: str) -> None:
    context = _context(task)
    result, _ = _execute(context)
    snapshot = result.state_after
    assert isinstance(snapshot, NativeState)
    strategy = result.native_entries[1]
    mutations = {
        "baseline": replace(snapshot, baseline="bot_style"),
        "family": replace(snapshot, entries=(replace(result.native_entries[0], semantic_kind="reasoning_template"), *snapshot.entries[1:])),
        "content": replace(snapshot, entries=(*snapshot.entries[:-1], replace(strategy, content="counterfeit"))),
        "parent": replace(snapshot, entries=(*snapshot.entries[:-1], replace(strategy, direct_parent_ids=("absent",)))),
    }

    with pytest.raises(RuntimeStateError, match="INVALID_DC_RS_SNAPSHOT"):
        RUNTIME.restore_state(mutations[field], context)


def test_checkpoint_refuses_unknown_promoted_to_exact_empty(task: TaskInstance) -> None:
    result, _ = _execute(_context(task, ""))
    assert isinstance(result.state_after, NativeState)
    checkpoint = serialize_checkpoint(result.state_after)
    strategy = result.native_entries[1]
    forged = replace(checkpoint.state, entries=(*checkpoint.state.entries[:-1], replace(strategy, lineage_status="exact")))

    with pytest.raises(CheckpointError, match="CHECKPOINT_HASH_MISMATCH"):
        deserialize_checkpoint(replace(checkpoint, state=forged))


@pytest.mark.parametrize("status", ["exact", "unavailable"])
def test_parentless_status_preserved_when_checkpoint_reopened(task: TaskInstance, status: Literal["exact", "unavailable"], tmp_path: Path) -> None:
    from memcontam.experiment.phase13_snapshot_diagnostic import snapshot_dispatch, snapshot_input_sha256
    from memcontam.readiness.phase13_main_live_dispatch import persist_unit_dispatch
    from memcontam.readiness.phase13_production_observability import ProductionObservabilityArchive
    from .phase13_snapshot_durable_fixtures import ContractTransport
    from .test_phase13_snapshot_durable import _ledger, _reopen

    context = _context(task, "")
    state = context.initial_states["dc_rs"]
    strategy = NativeEntry(entry_id="qa-parentless", semantic_kind="dynamic_cheatsheet",
                           schema_version="phase12_native_entry_v1", native_component="strategy",
                           content="empty ancestry", content_hash=canonical_content_hash("empty ancestry"),
                           lineage_status=status)
    state.strategies = [strategy]
    snapshot = RUNTIME.serialize_state(state)
    assert isinstance(snapshot, NativeState)

    reopened = RUNTIME.restore_state(deserialize_checkpoint(serialize_checkpoint(snapshot)), context)

    assert isinstance(reopened, dc.DcRsStateV3)
    assert reopened.strategies == [strategy]
    run, bindings = _shared_fixture(replace(context, initial_states={"dc_rs": reopened}))
    run = replace(run, client=ContractTransport(run.client), decoding={"temperature": 0.0, "top_p": 1.0})
    bindings = replace(bindings, input_sha256=snapshot_input_sha256(run),
        archive_identity=bindings.archive_identity.model_copy(update={"checkpoint_registry_sha256": bindings.state_sha256}))
    snapshot = execute_snapshot_unit(run, bindings, tmp_path)
    unit, output = snapshot_dispatch(snapshot)
    ledger = _ledger(tmp_path, unit)
    ledger.persist_completed(unit.unit_id, persist_unit_dispatch(tmp_path, unit, output))

    child = _reopen(tmp_path)

    assert child.returncode == 0, child.stderr
    archive = ProductionObservabilityArchive.model_validate_json(json.dumps(json.loads(child.stdout)["archive"]))
    assert archive == snapshot.archive
    evidence = archive.records[0].evidence
    assert isinstance(evidence, Phase13TrialEvidence)
    assert next(node for node in evidence.lineage if node.entry_id == strategy.entry_id).lineage_status == status
    assert next(node for node in evidence.lineage if node.entry_id == strategy.entry_id).direct_parent_ids == ()


def test_source_binding_refused_when_expected_raw_interaction_changes(task: TaskInstance) -> None:
    context = _context(task)
    assert context.expected_intervention is not None
    forged = replace(context, expected_intervention=replace(context.expected_intervention, render_id="wrong-source"))

    with pytest.raises(RuntimeStateError, match="INVALID_DC_RS_STATE"):
        RUNTIME.initial_state(forged)

    assert isinstance(context.client, ReplayClient) and context.client.index == 0


@pytest.mark.parametrize("field", ["state", "input", "source", "source_hash", "family", "archive"])
def test_shared_route_refuses_missing_binding_before_calls(task: TaskInstance, tmp_path: Path, field: str) -> None:
    run, bindings = _shared_fixture(_context(task))
    mutations = {
        "state": replace(bindings, state_sha256=""),
        "input": replace(bindings, input_sha256=""),
        "source": replace(bindings, source_files=()),
        "source_hash": replace(bindings, source_files=(replace(bindings.source_files[0], sha256="0" * 64), *bindings.source_files[1:])),
        "family": replace(bindings, source_files=tuple(pin for pin in bindings.source_files
                                                       if pin.path.name != "dynamic_cheatsheet_phase12.py")),
        "archive": replace(bindings, archive_identity=bindings.archive_identity.model_copy(update={"scientific_result": True})),
    }

    with pytest.raises(SnapshotError):
        execute_snapshot_unit(run, mutations[field], tmp_path)

    assert isinstance(run.client, ReplayClient) and run.client.index == 0


def test_shared_route_refuses_filter_native_branch(task: TaskInstance, tmp_path: Path) -> None:
    run, bindings = _shared_fixture(_context(task))
    assert run.branch is not None
    forged = replace(run, arm="filter", branch=replace(run.branch, arm="filter"))
    bindings = replace(bindings, input_sha256=snapshot_input_sha256(forged))

    with pytest.raises(SnapshotError, match="SNAPSHOT_NATIVE_BRANCH_REQUIRED"):
        execute_snapshot_unit(forged, bindings, tmp_path)

    assert isinstance(run.client, ReplayClient) and run.client.index == 0


def test_shared_route_refuses_post_state_and_clone_reuse(task: TaskInstance, tmp_path: Path) -> None:
    run, bindings = _shared_fixture(_context(task))
    assert run.branch is not None
    first = execute_snapshot_unit(run, bindings, tmp_path)
    client = _context(task).client
    post = replace(run, run_id="post-state", client=client,
                   branch=replace(run.branch, state=first.ordinary.trials[0].state))

    with pytest.raises(SnapshotError, match="STATE_BINDING"):
        execute_snapshot_unit(post, bindings, tmp_path)
    with pytest.raises(SnapshotError, match="CLONE_ALREADY_CONSUMED"):
        execute_snapshot_unit(replace(run, client=client), bindings, tmp_path)

    assert isinstance(client, ReplayClient) and client.index == 0


def test_shared_route_clones_keep_disjoint_writes(task: TaskInstance, tmp_path: Path) -> None:
    context = _context(task)
    context.initial_states["dc_rs"].archive[:] = context.initial_states["dc_rs"].archive[:1]
    run, bindings = _shared_fixture(context)

    first = execute_snapshot_unit(run, bindings, tmp_path)
    second = execute_snapshot_unit(replace(run, run_id="qa-other", client=_context(task).client), bindings, tmp_path)

    left, right = first.ordinary.trials[0], second.ordinary.trials[0]
    assert left.state_before == right.state_before
    assert left.state is not right.state
    assert {row.entry_id for row in left.native_entries}.isdisjoint(row.entry_id for row in right.native_entries)
    assert {row.call_id for row in left.outcome.method_calls}.isdisjoint(row.call_id for row in right.outcome.method_calls)


def test_shared_route_counterfeit_lineage_refused_before_calls(task: TaskInstance, tmp_path: Path) -> None:
    run, bindings = _shared_fixture(_context(task))
    assert run.branch is not None
    snapshot = run.branch.checkpoint.state
    root = snapshot.entries[0]
    assert isinstance(root, NativeEntry)
    forged = replace(snapshot, entries=(replace(root, direct_parent_ids=("fabricated",)), *snapshot.entries[1:]))
    run = replace(run, branch=replace(run.branch, checkpoint=replace(run.branch.checkpoint, state=forged)))

    with pytest.raises(CheckpointError, match="CHECKPOINT_HASH_MISMATCH"):
        execute_snapshot_unit(run, bindings, tmp_path)

    assert isinstance(run.client, ReplayClient) and run.client.index == 0


@pytest.mark.parametrize("case", [
    ("filter", {}, "DC_RS_FILTER_UNSUPPORTED"),
    ("contam", {"tool_mode": "python_sandbox"}, "DC_RS_TEXT_ONLY_REQUIRED"),
    ("contam", {"embedding_mode": "production"}, "DC_RS_BGE_M3_CONTRACT_REQUIRED"),
])
def test_native_contract_refused_before_client_when_branch_or_mode_unsupported(
    task: TaskInstance, case: tuple[Literal["filter", "contam"], dict[str, str], str],
) -> None:
    branch, settings, code = case
    context = _context(task)
    context = replace(context, branch=branch, baseline_configs={"dc_rs": {**context.baseline_configs["dc_rs"], **settings}})

    with pytest.raises(RuntimeStateError, match=code):
        RUNTIME.execute_trial(context, RUNTIME.initial_state(context))

    assert isinstance(context.client, ReplayClient) and context.client.index == 0
