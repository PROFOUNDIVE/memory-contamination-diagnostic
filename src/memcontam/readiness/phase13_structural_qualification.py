"""Provider-free native qualification carried by the existing MR-P4 conformance.

No execution client is accepted by the production constructor. Qualification is
performed before package hashing; selection authenticates the frozen observations.
Application only checks native preconditions and immutable selected/durable bytes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal
from unittest.mock import patch
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic_core import to_jsonable_python
from pydantic import ValidationError
from memcontam.clients.base import LLMResponse
from memcontam.experiment.phase13_ordinary_runtime import OrdinaryBaseline, REPOSITORY_ROOT
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.memory.checkpoint_v3 import Phase12Checkpoint
from memcontam.tasks.base import TaskInstance

if TYPE_CHECKING:
    from memcontam.experiment.phase12.runtime_registry import RuntimeTrialResult
    from memcontam.evaluation.phase13_seed_response import (
        StructuralSupportInputs, StructuralSupportEvidence, StructuralSupportValidationContext,
        _ProductionStructuralOrigin,
    )
    from .phase13_v3_entrypoint import SelectedExecutionV3
    from .phase13_main_production import ProductionObject
    from .phase13_v3_terminal_ledger import TerminalLedgerV3
    from .phase13_main_preloaded_resources import PreloadedMainResources

from .phase13_cost_policy_models import Sha256
from .phase13_v3_authority_models import FrozenModel
from .phase13_v3_cost_models import canonical_bytes, digest
from .phase13_v3_resource_files import FileBinding

ARMS = ("clean", "correct", "irrelevant", "contam")
CARRIER = "data/phase13/main/mr_p4/corrected_v3/provider_free_conformance_v3.json"
BASELINES: tuple[OrdinaryBaseline, ...] = ("fh_bounded", "bot_style", "reflexion_style", "dc_rs", "rag_frozen")
# These are prospective expectations, not names inferred from observed traces.
NATIVE_LAWS = (
    ("fh_bounded", "chronological-store/pair-atomic-visible-fifo/first-insertion-fit/later-eviction-legal"),
    ("bot_style", "native-envelope/empty/miss/hit/novel-insert/novelty-noop/unbounded-active-capacity"),
    ("reflexion_style", "valid-empty/success-no-write/failure-write/legal-1-3-4-suffix/retention"),
    ("dc_rs", "valid-empty/whole-cheatsheet/retrieve-curate-generate/append/persist/technical-overflow"),
    ("rag_frozen", "task-local-corpus-index/matched-delta/read-only-store/varying-context/reopen"),
)
BOUNDED_ARGUMENT = (
    "Native initialization and matched insertion establish the family invariant. "
    "The registered ordinary transition preserves the native schema and policy, "
    "or terminates technically without conferring structural nonsupport. "
    "Serialization and restore preserve that invariant. Induction applies the same "
    "transition law to each of four matched arms and each of 50 ordered slots. "
    "Fixture traces corroborate the law; they are not the production trajectory. "
    "FH root visibility is checked before the first client call; later lawful "
    "FIFO eviction is not nonsupport. Arbitrary model strings are not guaranteed "
    "to succeed: malformed envelopes, curator overflow and transport failure are technical."
)
NATIVE_DOMAIN = (
    "Current Core registered task/baseline/seed/unit and exact ordered 50 task bytes; "
    "selected authority, governed source, runtime, native serializers and resources; "
    "durably authenticated CLEAN_PREFIX at tau_star-1 and its four matched insertions; "
    "FH-bounded pair-atomic FIFO and DC whole-cheatsheet capacity 8192; "
    "BoT/Reflexion active_capacity=None; selected task-local RAG corpus/index; "
    "native restore/schema and first-insertion preconditions checked before application. "
    "Malformed or uncovered subjects are unavailable; output-dependent semantic, "
    "curator and transport errors are technical, not structural nonsupport."
)
TASK_DOMAINS = ("game24", "math_equation_balancer", "word_sorting",
                "mmlu_pro_engineering", "mmlu_pro_physics")
COVERED_CELLS = tuple((task, baseline) for task in TASK_DOMAINS for baseline in BASELINES
                      if baseline != "rag_frozen" or task in TASK_DOMAINS[:3])
PRESERVATION_CASES = (
    ("fh_bounded",
     "full_history_phase12.FullHistoryPhase12Adapter.execute; runtime_registry._fh_serialize/_fh_restore",
     "Empty or chronological schema-valid transcript records; matched insertion appends one complete pair. "
     "verify_complete_fit checks the first inserted root at 8192 before generation. "
     "Every execute appends one complete query/answer record to stored history; build_context removes only "
     "oldest complete pairs from the visible suffix. Thus later eviction preserves the invariant, not the root. "
     "Malformed answer/provider/verifier errors terminate technically. Records and visibility metadata "
     "roundtrip independently of task content, so the same step applies to every ordered task and all four arms."),
    ("bot_style",
     "bot_phase12.BoTPhase12Adapter.execute; runtime_registry._bot_serialize/_bot_restore",
     "Schema-valid native thought-template envelopes, including empty buffer, active_capacity=None. "
     "Matched insertion appends exactly one envelope. Distill/retrieve/instantiate/thought-distill handles "
     "empty, miss and hit without requiring root selection. Novelty rejection leaves entries unchanged; "
     "admission appends one valid envelope without capacity eviction. Malformed envelopes and failed "
     "calls are technical. Serialization retains template IDs, payloads and competitor IDs. "
     "Task-specific rendering/verifying does not select a different memory update law."),
    ("reflexion_style",
     "reflexion_phase12.ReflexionPhase12Adapter.execute; runtime_registry._reflexion_serialize/_reflexion_restore",
     "Schema-valid reflections, including empty, active_capacity=None. Matched insertion adds one reflection. "
     "Actor success exits without a writer; failure may write then retry, yielding only legal 1/3/4 call "
     "suffixes. No reflection is manufactured after success. Valid writes append under the same retention "
     "policy; malformed writer/actor/verifier/provider outcomes terminate technically. Restore retains "
     "active/evicted entries and root metadata; no task-dependent memory algorithm or future outcome is used."),
    ("dc_rs",
     "dynamic_cheatsheet_phase12.DcRsPhase12Adapter.execute; phase13_dc_rs_runtime.serialize/restore",
     "Validated ordered raw archive plus zero or one complete cheatsheet, text-only, 8192 o200k_base tokens. "
     "Matched insertion appends a raw interaction, leaving the whole cheatsheet unchanged. For every task, "
     "retrieve top-3 raw-query neighbors; curator sees complete prior cheatsheet and complete retrieved "
     "payloads. Accepted whole response is the generation context and persisted next version, never a "
     "concatenated fragment. Append one raw query/answer; move prior version to history. Explicit source "
     "aliases and prior-version edges persist; retrieval alone does not create ancestry. Empty is valid; "
     "malformed/overflow curator, raw-answer overflow and provider errors are technical. restore validates "
     "component equality, content hashes, whole budget and ordinary-history identity before the next step."),
    ("rag_frozen",
     "retrieval_rag_phase12.RagFrozenPhase12Adapter.execute; runtime_registry._rag_serialize/_rag_restore",
     "Selected same-task 24-document clean corpus and four indices with byte-identical clean base and "
     "one matched document delta per nonclean arm. Native corpus/index identities and read_only=True. "
     "For any valid query vector, cosine top-3 selects only this index; final inclusion is an ordered "
     "subset of retrieved IDs. Queries and contexts may vary; execute never mutates corpus/index. "
     "Answer/parser/provider errors are technical, not nonsupport. Serialization retains exact active "
     "document bytes and corpus/index IDs; restore compares those identities. The proof is independent "
     "of retrieval scores, answer correctness and corpus text, but requires the selected resource checks."),
)


class NativeQualification(FrozenModel):
    schema_version: Literal["phase13_native_structural_qualification_v1"] = "phase13_native_structural_qualification_v1"
    subject_sha256: Sha256
    generator_sha256: Sha256
    fixture_sha256: Sha256
    native_laws: tuple[tuple[str, str], ...] = NATIVE_LAWS
    bounded_argument: str = BOUNDED_ARGUMENT
    native_domain: str = NATIVE_DOMAIN
    covered_cells: tuple[tuple[str, str], ...] = COVERED_CELLS
    preservation_cases: tuple[tuple[str, str, str], ...] = PRESERVATION_CASES
    capacity: Literal[8192] = 8192
    horizon: Literal[50] = 50
    arms: tuple[str, ...] = ARMS
    # Proofs refer to qualification fixtures, not measured or future trajectories.
    fixture_proofs: tuple[str, ...]
    native_observations: tuple[str, ...]
    boundary_observations: tuple[tuple[str, str, str], ...]
    resource_observations: tuple[str, ...]
    qualification_only: Literal[True] = True
    real_provider_calls: Literal[0] = 0
    scientific_authorization: Literal[False] = False
    qualification_sha256: Sha256 = "0" * 64


def qualification_subject(authority: object, source: object, runtime: object,
                          resources: tuple[FileBinding, ...]) -> str:
    """Exclude generated artifacts and final package identity: no hash cycle."""
    return _hash({"authority": authority, "governed_source": source,
                   "runtime": runtime, "resources": resources})


def _bytes(value: object) -> bytes:
    return json.dumps(to_jsonable_python(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _hash(value: object) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


class FixtureClient:
    """Fixed public synthetic responses; never a production-client wrapper."""
    def __init__(self, task_name: str = "game24") -> None:
        self.actor_calls = 0
        self.writer_response: str | None = None
        self.task_name = task_name

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        stage = config["method_stage"]
        if stage == "reflexion_generate":
            self.actor_calls += 1
        content = {
            "bot_problem_distill": json.dumps({"key_information": "four numbers", "restrictions": "exact use", "distilled_task": "make 24"}),
            "bot_instantiate_solve": json.dumps({
                "selected_structure": "retrieved-template" if "Set selected_structure to retrieved-template" in messages[-1]["content"] else "procedure-based",
                "solution_trace": "divide", "final_answer": "final: 6/(1-3/4)"}),
            "bot_thought_distill": json.dumps({"description": "arithmetic", "template": "check all numbers", "category": "procedure-based", "explicitly_used_memory_ids": []}),
            "dc_rs_synthesize": "<cheatsheet>Check all numbers.</cheatsheet><source_ids>src01</source_ids>",
            "reflexion_reflect": json.dumps({"mode": "corrective", "failure_class": "incorrect_answer",
                "reflection_text": "Check the arithmetic.", "explicitly_used_memory_ids": []}),
        }.get(stage, "final: 6/(1-3/4)")
        if stage == "reflexion_generate" and self.actor_calls in (2, 4, 5):
            content = "final: 1+3+4+6"
        if stage == "full_history_generate":
            content = "working " * 250 + "\nfinal: 6/(1-3/4)"
        if stage == "bot_thought_distill" and config.get("sample_id") == "structural-v1-01":
            content = json.dumps({"description": "redundant", "template": "check all numbers",
                "category": "procedure-based", "explicitly_used_memory_ids": []})
        if stage == "bot_problem_distill" and config.get("sample_id") == "structural-v1-02":
            content = json.dumps({"key_information": "novel-query", "restrictions": "exact use", "distilled_task": "make 24"})
        if stage == "dc_rs_synthesize" and self.writer_response is not None:
            content = self.writer_response
        elif stage == "dc_rs_synthesize" and "Available archive aliases: src01" not in messages[-1]["content"]:
            content = "<cheatsheet>Check all numbers.</cheatsheet>"
        if stage == "rag_generate" and self.task_name != "game24":
            content = "final: 1+2+3=6" if self.task_name == "math_equation_balancer" else "final: alpha beta"
        return LLMResponse(content, {"replay": True, "attempts": 1},
                           {"prompt_tokens": 1, "completion_tokens": 1}, 0)


class FixtureEmbedder(BgeM3EmbeddingProvider):
    embedding_contract = {"dimension": 1024, "normalized": True,
        "production_identity": "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181", "provider": "test"}

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "structural-qualification-v1-synthetic-embedder"

    @property
    def metadata(self) -> dict[str, object]:
        return {"model_id": self.MODEL_ID, "revision": self.REVISION,
                "vector_dimension": 1024, "normalize_embeddings": True,
                "embedding_library_version": "5.6.0"}

    def encode_document(self, text: str) -> list[float]:
        return [1.0, 0.0] + [0.0] * 1022

    def encode_query(self, text: str) -> list[float]:
        return ([0.0, 1.0] if text == "arithmetic" or "novel-query" in text else [1.0, 0.0]) + [0.0] * 1022


def fixture_inputs(baseline: OrdinaryBaseline) -> StructuralSupportInputs:
    from memcontam.baselines.bot_phase12 import BoTStateV3
    from memcontam.baselines.dynamic_cheatsheet_phase12 import DcRsStateV3
    from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
    from memcontam.baselines.reflexion_phase12 import ReflexionStateV3
    from memcontam.baselines.retrieval_rag_phase12 import RagFrozenStateV3
    from memcontam.contamination.phase12.registry import load_current_candidate_registry
    from memcontam.contamination.phase12.renderers import RendererRegistry
    from memcontam.evaluation.phase13_seed_response import StructuralSupportInputs
    from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
    from memcontam.experiment.phase12.live_branch import build_live_reduced_main_branches
    from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
    from memcontam.experiment.phase13_ordinary_runtime import ProspectiveOrdinaryRun, REPOSITORY_ROOT
    from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
    from memcontam.rag.phase12_corpus import CleanCorpus, build_branch_corpora
    from memcontam.rag.branch_index import build_branch_indices
    from memcontam.tasks.base import TaskInstance
    from memcontam.verifiers.game24 import verify_expression

    tasks = tuple(TaskInstance(sample_id=f"structural-v1-{i:02d}", task_name="game24",
        input={"numbers": [3, 3, 8, 8] if baseline == "rag_frozen" and i % 2 else [1, 3, 4, 6],
               "target": 24}, verifier_spec={"target": 24}) for i in range(50))
    states = {"fh_bounded": FullHistoryStateV3(records=[]), "bot_style": BoTStateV3(entries=[]),
              "reflexion_style": ReflexionStateV3(reflections=[]), "dc_rs": DcRsStateV3(archive=[], allow_unparented_strategies=True)}
    embedder = FixtureEmbedder()
    state: object
    if baseline == "rag_frozen":
        corpus = build_branch_corpora(CleanCorpus.from_documents([
            {"id": "structural-rule", "text": "Follow the registered task procedure."},
            {"id": "structural-check", "text": "Verify the final answer."}], corpus_id="structural-v1"),
            {name: {"id": "unused-" + name, "text": "unused"} for name in ("false", "correct", "irrelevant")})
        indices = build_branch_indices(corpus, embedder, filter_policy=None)
        state = RagFrozenStateV3("clean", corpus.branches["clean"], indices.branches["clean"])
    else:
        state = states[baseline]
    entry = PHASE13_CORE_BASELINE_REGISTRY[baseline]
    native = entry.serialize_state(state)
    if not isinstance(native, NativeState):
        raise ValueError("STRUCTURAL_FIXTURE_NATIVE_REQUIRED")
    prefix = serialize_checkpoint(native, checkpoint_index=0)
    client = FixtureClient()
    context = Game24RuntimeContext(task=tasks[0], client=client, model="gpt-5.6-luna",
        verifier=lambda answer, task: verify_expression(answer, task.input["numbers"]).is_correct,
        decoding={"temperature": 0.0}, branch="clean", embedding_provider=embedder,
        identities=RuntimeIdentities("structural-v1", "structural-v1:trial:1:clean:structural-v1-00", 1),
        baseline_configs={"dc_rs": {"serialized_cheatsheet_budget_tokens": 8192}}, initial_states={baseline: state})
    candidate_raw = (REPOSITORY_ROOT / "data/phase12/registries/candidate_registry_v2.json").read_bytes()
    candidates = load_current_candidate_registry(REPOSITORY_ROOT / "data/phase12/registries/candidate_registry_v2.json")
    renderers = RendererRegistry.governed((REPOSITORY_ROOT / "data/phase13/main/legacy_dc_rs_intervention_registry_v2.json").read_bytes(), candidates, hashlib.sha256(candidate_raw).hexdigest())
    branches = build_live_reduced_main_branches(prefix=prefix, context=context,
        candidate_registry=candidates, registry=PHASE13_CORE_BASELINE_REGISTRY, renderers=renderers)
    run = ProspectiveOrdinaryRun(task_name="game24", baseline=baseline, run_id="structural-v1",
        model=context.model, client=client, allow_test_client=True, verifier=context.verifier,
        decoding=context.decoding, tasks=tasks, trajectory_seed=0, arm="clean", branch=branches.arms["clean"],
        embedding_provider=embedder, initial_states={baseline: state})
    from .phase13_production_runtime_models import ProductionOrdinaryRunIdentity
    identity = ProductionOrdinaryRunIdentity(execution_template_id="structural-v1:" + baseline,
        trajectory_seed=0, concrete_seed_id="0", scientific_result=False,
        ordered_sample_ids_sha256=hashlib.sha256(json.dumps([task.sample_id for task in tasks], separators=(",", ":")).encode()).hexdigest(),
        registration_packet_sha256=hashlib.sha256((REPOSITORY_ROOT / "data/phase13/observability/registration_packet_v2.json").read_bytes()).hexdigest(),
        source_package_manifest_sha256="a" * 64)
    return StructuralSupportInputs(prefix, context, candidates, run, identity,
        tuple(task.model_dump_json().encode() for task in tasks))


def generate_native_qualification(subject_sha256: str, *,
                                  resource_root: Path = REPOSITORY_ROOT) -> NativeQualification:
    from memcontam.evaluation.phase13_seed_response import certify_structural_support
    from memcontam.experiment.phase12.runtime_registry import RuntimeTrialResult
    native_observations: list[str] = []
    def observe(arm: str, slot: int, trial: RuntimeTrialResult) -> None:
        before, after = trial.state_before, trial.state_after
        if before is None or after is None:
            raise ValueError("STRUCTURAL_NATIVE_EXPECTATION_FAILED")
        stages = tuple(call.stage for call in trial.outcome.method_calls)
        baseline = before.baseline
        read_decision = None
        if baseline == "reflexion_style":
            expected = (1 if slot == 1 or slot > 3 else 3 if slot == 2 else 4)
            expected_count = (0 if arm == "clean" else 1) + (1 if slot >= 2 else 0) + (2 if slot >= 3 else 0)
            if (len(stages) != expected or (expected == 1 and before != after)
                or len(after.entries) != expected_count):
                raise ValueError("STRUCTURAL_REFLEXION_EXPECTATION_FAILED")
        elif baseline == "fh_bounded":
            if after.entries[:-1] != before.entries or len(after.entries) != len(before.entries) + 1:
                raise ValueError("STRUCTURAL_FH_STORED_HISTORY_EXPECTATION_FAILED")
            from memcontam.baselines.full_history_phase12 import FullHistoryStateV3, verify_complete_fit
            from .phase13_route_capacity import FH_CAPACITY_CONFIG
            if not isinstance(trial.state, FullHistoryStateV3):
                raise ValueError("STRUCTURAL_FH_NATIVE_EXPECTATION_FAILED")
            fit = verify_complete_fit(fixture_inputs_task(slot), trial.state.records[:-1],
                FH_CAPACITY_CONFIG, requested_fh_mode="bounded")
            retained = tuple(fit.context.post_record_ids)
            original = tuple(fit.context.pre_record_ids)
            if (retained and retained != original[-len(retained):]) or (slot == 50 and not fit.context.removed_record_ids):
                raise ValueError("STRUCTURAL_FH_VISIBLE_FIFO_EXPECTATION_FAILED")
        elif baseline == "rag_frozen":
            if before != after:
                raise ValueError("STRUCTURAL_RAG_EXPECTATION_FAILED")
        elif baseline == "bot_style":
            read_decision = trial.outcome.metadata["retrieval_decision"]["decision"]
            expected_read = "empty_buffer" if not before.entries else "miss" if slot == 3 else "matched"
            if (before.native_state.get("active_capacity") is not None
                or len(after.entries) - len(before.entries) != (0 if slot == 2 else 1)
                or read_decision != expected_read):
                raise ValueError("STRUCTURAL_BOT_EXPECTATION_FAILED")
        elif baseline == "dc_rs":
            old, new = before.native_state["archive"], after.native_state["archive"]
            if (len(new) != len(old) + 1 or new[:-1] != old
                or stages != ("dc_rs_synthesize", "dc_rs_generate")
                or len(after.native_state["strategy_ids"]) != 1):
                raise ValueError("STRUCTURAL_DC_EXPECTATION_FAILED")
            observe_dc_whole_state(trial)
        native_observations.append(_bytes({"family": baseline, "arm": arm, "slot": slot,
            "stages": stages, "read_decision": read_decision, "before": before, "after": after}).decode())
    fixtures = tuple(replace(fixture_inputs(baseline), qualification_observer=observe) for baseline in BASELINES)
    fixture_digest = _hash(tuple((item.run.baseline, item.fixture_task_bytes,
        item.prefix.canonical_sha256) for item in fixtures))
    sequence = 0
    def fixture_uuid() -> UUID:
        nonlocal sequence
        sequence += 1
        return uuid5(NAMESPACE_URL, f"structural-qualification-v1:{sequence}")
    with patch("memcontam.baselines.dynamic_cheatsheet_optional.uuid4", fixture_uuid), patch(
        "memcontam.baselines.reflexion_adapter.uuid4", fixture_uuid):
        proofs = tuple(certify_structural_support(item) for item in fixtures)
    if any(not proof.supported or len(proof.conformance_trace) != 200 for proof in proofs):
        raise ValueError("STRUCTURAL_NATIVE_QUALIFICATION_FAILED")
    observations = tuple(json.dumps(proof.model_dump(mode="json", exclude={
        "replay_witness", "evidence_sha256"}), sort_keys=True, separators=(",", ":")) for proof in proofs)
    result = NativeQualification(subject_sha256=subject_sha256,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixture_sha256=fixture_digest, fixture_proofs=observations, native_observations=tuple(native_observations),
        boundary_observations=qualify_native_boundaries(),
        resource_observations=qualify_rag_resources(resource_root))
    return result.model_copy(update={"qualification_sha256": digest(result, "qualification_sha256")})


def fixture_inputs_task(slot: int) -> TaskInstance:
    from memcontam.tasks.base import TaskInstance
    return TaskInstance(sample_id=f"structural-v1-{slot - 1:02d}", task_name="game24",
        input={"numbers": [1, 3, 4, 6], "target": 24}, verifier_spec={"target": 24})


def observe_dc_whole_state(trial: RuntimeTrialResult) -> None:
    from memcontam.memory.checkpoint_v3 import NativeEntry
    from memcontam.logging.schema import MethodCall
    before, after = trial.state_before, trial.state_after
    if before is None or after is None:
        raise ValueError("STRUCTURAL_DC_WHOLE_EXPECTATION_FAILED")
    old = tuple(entry for entry in before.entries if isinstance(entry, NativeEntry) and entry.native_component == "strategy")
    new = tuple(entry for entry in after.entries if isinstance(entry, NativeEntry) and entry.native_component == "strategy")
    calls = tuple(call for call in trial.outcome.method_calls if isinstance(call, MethodCall))
    expected = "<cheatsheet>Check all numbers.</cheatsheet>" + (
        "<source_ids>src01</source_ids>" if before.native_state["archive"] else "")
    if len(new) != 1 or len(calls) != 2:
        raise ValueError("STRUCTURAL_DC_WHOLE_EXPECTATION_FAILED")
    strategy = new[0]
    curator, generation = calls
    prior_id = old[0].entry_id if old else None
    envelope = next(row for row in trial.write_envelopes if row.native_component == "strategy")
    if (curator.raw_response != expected or strategy.content != expected
        or expected not in generation.messages[-1]["content"]
        or (old and old[0].content not in curator.messages[-1]["content"])
        or envelope.version_predecessor_id != prior_id
        or envelope.direct_parent_ids != strategy.direct_parent_ids
        or len(strategy.direct_parent_ids) != (1 if before.native_state["archive"] else 0)
        or strategy.lineage_status != ("exact" if strategy.direct_parent_ids else "unavailable")
        or after.native_state["strategy_history"] != [*before.native_state["strategy_history"],
            *(entry.to_mapping() for entry in old)]):
        raise ValueError("STRUCTURAL_DC_WHOLE_EXPECTATION_FAILED")


def qualify_native_boundaries() -> tuple[tuple[str, str, str], ...]:
    from memcontam.baselines.prompt_budget import count_text_tokens
    from memcontam.experiment.phase13_ordinary_runtime import execute_prospective_ordinary
    from memcontam.experiment.phase12.runtime_registry import RuntimeStateError
    inputs = fixture_inputs("dc_rs")
    fill = 8192
    exact = ""
    for _ in range(4):
        exact = "<cheatsheet>" + " x" * fill + "</cheatsheet>"
        fill += 8192 - count_text_tokens(exact, "o200k_base")
    if count_text_tokens(exact, "o200k_base") != 8192:
        raise ValueError("STRUCTURAL_EXACT_BUDGET_FIXTURE_INVALID")
    rows = []
    for name, response, expected in (
        ("dc_exact_whole_writer_8192", exact, "SUCCEEDED"),
        ("dc_overflow_is_technical", "<cheatsheet>" + " x" * 8192 + "</cheatsheet>", "DC_RS_WRITER_OUTPUT_BUDGET_EXCEEDED"),
        ("dc_malformed_is_technical", "commentary<cheatsheet>strategy</cheatsheet>", "DC_RS_WRITER_GRAMMAR_INVALID"),
    ):
        client = FixtureClient()
        client.writer_response = response
        run = replace(inputs.run, tasks=inputs.run.tasks[:2], client=client)
        try:
            result = execute_prospective_ordinary(run)
        except RuntimeStateError as error:
            if str(error) != expected:
                raise ValueError("STRUCTURAL_TECHNICAL_EXPECTATION_FAILED") from error
        else:
            if expected != "SUCCEEDED" or len(result.trials) != 2:
                raise ValueError("STRUCTURAL_TECHNICAL_EXPECTATION_FAILED")
        rows.append((name, expected, hashlib.sha256(response.encode()).hexdigest()))
    return (*rows, qualify_fh_exact_boundary())


def qualify_fh_exact_boundary() -> tuple[str, str, str]:
    from memcontam.baselines.full_history import FullHistoryPayload, render_full_history
    from memcontam.baselines.full_history_phase12 import FullHistoryStateV3, verify_complete_fit
    from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
    from memcontam.experiment.phase13_ordinary_runtime import execute_prospective_ordinary
    from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
    from memcontam.memory.stores import MemoryEntry
    from .phase13_route_capacity import FH_CAPACITY_CONFIG
    inputs = fixture_inputs("fh_bounded")
    task = inputs.run.tasks[0]
    fill = 8192
    record = MemoryEntry(entry_id="structural-exact-fh", content="", memory_type="full_history_transcript")
    for _ in range(4):
        record = record.model_copy(update={"content": render_full_history("structural-exact-fh",
            FullHistoryPayload("public synthetic history", " x" * fill))})
        fit = verify_complete_fit(task, [record], FH_CAPACITY_CONFIG, requested_fh_mode="bounded")
        pre, current = fit.context.pre_token_count, fit.context.current_task_token_count
        if pre is None or current is None:
            raise ValueError("STRUCTURAL_FH_EXACT_FIT_EXPECTATION_FAILED")
        fill += 8192 - (pre - current)
    fit = verify_complete_fit(task, [record], FH_CAPACITY_CONFIG, requested_fh_mode="bounded")
    pre, current = fit.context.pre_token_count, fit.context.current_task_token_count
    if pre is None or current is None or not fit.full_fit or pre - current != 8192:
        raise ValueError("STRUCTURAL_FH_EXACT_FIT_EXPECTATION_FAILED")
    runtime = PHASE13_CORE_BASELINE_REGISTRY["fh_bounded"]
    state = FullHistoryStateV3(records=[record])
    native = runtime.serialize_state(state)
    if not isinstance(native, NativeState) or inputs.run.branch is None:
        raise ValueError("STRUCTURAL_FH_EXACT_FIT_EXPECTATION_FAILED")
    checkpoint = serialize_checkpoint(native, checkpoint_index=0)
    branch = replace(inputs.run.branch, state=state, checkpoint=checkpoint,
        prefix_identity=checkpoint.identity.checkpoint_id, source_identity=checkpoint.identity.checkpoint_id)
    result = execute_prospective_ordinary(replace(inputs.run, branch=branch, tasks=inputs.run.tasks[:2]))
    second = result.trials[1]
    history = second.outcome.metadata["full_history_context"]
    if (history["removed_record_ids"] != [record.entry_id] or second.state_after is None
        or len(second.state_after.entries) != 3):
        raise ValueError("STRUCTURAL_FH_LAWFUL_EVICTION_EXPECTATION_FAILED")
    return ("fh_8192_exact_fit_then_pair_atomic_visible_eviction_stored_history_retained", "SUCCEEDED",
            hashlib.sha256(record.content.encode()).hexdigest())


def validate_selected_qualification(selected: SelectedExecutionV3) -> None:
    """Selection-time requalification; application never executes these fixtures."""
    from .phase13_v3_builder_inputs import STATIC_PATHS
    from .phase13_v3_conformance import ConformanceV3
    carrier = next(row for row in selected.resources if row.binding.path == CARRIER)
    if json.loads(carrier.raw).get("structural") is None:
        return
    validate_loaded_qualification_source(selected)
    try:
        conformance = ConformanceV3.model_validate_json(carrier.raw)
    except ValidationError as error:
        raise ValueError("STRUCTURAL_QUALIFICATION_BINDING_MISMATCH") from error
    payload = conformance.structural
    if (payload is None or conformance.identity != selected.package.identity
        or conformance.conformance_hash != digest(conformance, "conformance_hash")):
        raise ValueError("STRUCTURAL_QUALIFICATION_BINDING_MISMATCH")
    by_path = {row.binding.path: row.binding for row in selected.resources}
    subject = qualification_subject(selected.package.authority, selected.package.governed_source,
        selected.package.runtime_identity, tuple(by_path[path] for path in STATIC_PATHS))
    if payload != generate_native_qualification(subject, resource_root=selected.repository_root):
        raise ValueError("STRUCTURAL_QUALIFICATION_BINDING_MISMATCH")


def validate_loaded_qualification_source(selected: SelectedExecutionV3) -> None:
    source = selected.package.governed_source
    if source is None:
        raise ValueError("STRUCTURAL_QUALIFICATION_SOURCE_MISMATCH")
    bound = {row.path: row.sha256 for row in source.rows}
    required = ("src/memcontam/readiness/phase13_structural_qualification.py",
                "src/memcontam/evaluation/phase13_seed_response.py",
                "src/memcontam/readiness/phase13_v3_conformance.py",
                "src/memcontam/readiness/phase13_v3_entrypoint.py",
                "src/memcontam/readiness/phase13_v3_builder.py")
    if any(bound.get(path) != hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest()
           for path in required):
        raise ValueError("STRUCTURAL_QUALIFICATION_SOURCE_MISMATCH")


def production_structural_context(selected: SelectedExecutionV3, unit: ProductionObject,
                                  ledger: TerminalLedgerV3, *, scientific_result: bool = True) -> StructuralSupportValidationContext | None:
    from memcontam.evaluation.phase13_seed_response import (
        SeedResponseError, StructuralSupportEvidence, StructuralSupportValidationContext,
        _ProductionStructuralOrigin, _production_origin_signature, _handler_contract, _native_properties_contract,
    )
    from memcontam.experiment.phase12.branching import BranchSet, build_matched_branches
    from memcontam.memory.admission import AdmissionContext
    from memcontam.memory.checkpoint_v3 import append_native_entry, deserialize_checkpoint
    from .phase13_main_preloaded_resources import PreloadedMainResources
    from .phase13_durable_call_join import validated_prefix_checkpoint
    from .phase13_main_live_runtime_support import core_task_name, production_identity
    from .phase13_main_new_mcq_runtime import new_mcq_native_entries
    from .phase13_v3_conformance import ConformanceV3

    selected.preflight(selected.repository_root)
    if unit not in selected.package.production or unit.kind != "MEMORY_BEARING" or unit.memory_baseline not in BASELINES:
        raise SeedResponseError("STRUCTURAL_INPUT_BINDING_MISMATCH")
    carrier = next(row for row in selected.resources if row.binding.path == CARRIER)
    raw_payload = json.loads(carrier.raw).get("structural")
    payload = None if raw_payload is None else NativeQualification.model_validate_json(json.dumps(raw_payload))
    if payload is None or selected.structural_qualification_sha256 != digest(payload):
        return None
    if (payload.preservation_cases != PRESERVATION_CASES or payload.covered_cells != COVERED_CELLS
        or (unit.task, unit.memory_baseline) not in payload.covered_cells):
        return None
    validate_loaded_qualification_source(selected)
    fixture = next((json.loads(raw) for raw in payload.fixture_proofs
                    if json.loads(raw)["baseline"] == unit.memory_baseline), None)
    if (fixture is None or fixture["handler_contract_sha256"] != _handler_contract(unit.memory_baseline)
        or fixture["native_properties_sha256"] != _native_properties_contract()
        or payload.generator_sha256 != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        return None
    loaded = PreloadedMainResources(selected)
    prefix = validated_prefix_checkpoint(selected, ledger, unit)
    if prefix is None or prefix.state.baseline != unit.memory_baseline:
        raise SeedResponseError("STRUCTURAL_INPUT_BINDING_MISMATCH")
    deserialize_checkpoint(prefix)
    seed = loaded.checkpoint_registry.tasks[unit.task].seeds[unit.seed]
    by_id = {task.sample_id: task for task in loaded.tasks(unit.task)}
    tasks = tuple(by_id[sample] for sample in seed.suffix_sample_ids)
    if (len(tasks) != 50 or prefix.checkpoint_index != seed.tau_star - 1
        or unit.ordered_sample_ids_sha256 != seed.suffix_sample_ids_sha256
        or unit.registration_packet_sha256 != selected.resource_binding("observability_packet").sha256
        or unit.checkpoint_registry_sha256 != selected.resource_binding("common_checkpoint_registry").sha256):
        raise SeedResponseError("STRUCTURAL_INPUT_BINDING_MISMATCH")
    if unit.task in ("mmlu_pro_engineering", "mmlu_pro_physics"):
        roots = new_mcq_native_entries(core_task_name(unit.task), unit.memory_baseline, loaded.new_mcq_registry)
        branches = (prefix, *(append_native_entry(prefix, roots[arm]) for arm in ARMS[1:]))
        registry_sha = selected.resource_binding("main_new_mcq_intervention_registry").sha256
    else:
        triplets = tuple(row for row in loaded.candidate_registry.triplets if row.task == unit.task)
        if len(triplets) != 1:
            raise SeedResponseError("STRUCTURAL_BRANCH_BINDING_MISMATCH")
        matched = build_matched_branches(prefix, triplets[0], loaded.legacy_dc_rs_renderers, AdmissionContext())
        if not isinstance(matched, BranchSet):
            raise SeedResponseError("STRUCTURAL_BRANCH_BINDING_MISMATCH")
        branches = tuple(next(row.checkpoint for row in matched.materialized if row.arm == arm) for arm in ARMS)
        registry_sha = selected.resource_binding("candidate_registry").sha256
    try:
        domain = qualified_native_domain(selected, unit, prefix, branches, tasks)
    except (KeyError, TypeError, ValueError):
        return None
    if domain is None:
        return None
    branch_hashes = (branches[0].canonical_sha256, branches[1].canonical_sha256,
                     branches[2].canonical_sha256, branches[3].canonical_sha256)
    identity = production_identity(unit, scientific_result=scientific_result)
    subject = _bytes({"unit": unit, "prefix": prefix,
        "branches": branches, "ordered_task_bytes": tuple(task.model_dump_json() for task in tasks),
        "package_sha256": selected.package_sha256, "authorization_sha256": selected.authorization_sha256,
        "qualification_sha256": digest(payload), "decoding": {"temperature": 0.0, "top_p": 1.0},
        "capacity": 8192, "native_policy": NATIVE_LAWS, "bounded_argument": BOUNDED_ARGUMENT,
        "preservation_case": next(row for row in payload.preservation_cases if row[0] == unit.memory_baseline),
        "domain_cell": (unit.task, unit.memory_baseline)})
    proof = StructuralSupportEvidence(task=unit.task, baseline=unit.memory_baseline,
        trajectory_seed=unit.seed, concrete_seed_id=str(unit.seed), supported=domain == "SUPPORTED",
        unsupported_reason=None if domain == "SUPPORTED" else domain,
        source_package_manifest_sha256=identity.source_package_manifest_sha256,
        evidence_sha256="0" * 64, checkpoint_id=prefix.identity.checkpoint_id,
        checkpoint_sha256=prefix.canonical_sha256, checkpoint_index=prefix.checkpoint_index,
        ordered_sample_ids_sha256=seed.suffix_sample_ids_sha256,
        execution_template_id=identity.execution_template_id,
        registration_packet_sha256=identity.registration_packet_sha256,
        package_sha256=selected.package_sha256, branch_checkpoint_sha256=branch_hashes,
        candidate_registry_sha256=registry_sha, suffix_contract_sha256=_hash(BOUNDED_ARGUMENT),
        handler_contract_sha256=_handler_contract(unit.memory_baseline),
        fixture_contract_sha256=payload.fixture_sha256, native_properties_sha256=_native_properties_contract(),
        qualification_sha256=digest(payload), subject_contract_sha256=hashlib.sha256(subject).hexdigest())
    proof = proof.model_copy(update={"evidence_sha256": hashlib.sha256(json.dumps(
        proof.model_dump(mode="json", exclude={"evidence_sha256"}),
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    raw = proof.model_dump_json().encode()
    origin_parts = (raw, _bytes(prefix), identity.model_dump_json().encode(), selected.package_sha256.encode(),
                    payload.model_dump_json().encode(), subject)
    origin = _ProductionStructuralOrigin(raw, origin_parts[1], origin_parts[2], selected.package_sha256,
        origin_parts[4], subject, _production_origin_signature(origin_parts))
    return StructuralSupportValidationContext(raw, hashlib.sha256(raw).hexdigest(), identity,
        selected.package_sha256, prefix, payload.fixture_sha256, branch_hashes, registry_sha, origin)


def qualified_native_domain(selected: SelectedExecutionV3, unit: ProductionObject,
                            prefix: Phase12Checkpoint, branches: tuple[Phase12Checkpoint, ...],
                            tasks: tuple[TaskInstance, ...]) -> str | None:
    from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
    from memcontam.experiment.phase12.game24_runner import RuntimeIdentities
    from memcontam.experiment.phase12.live_branch import _inject_root
    from memcontam.baselines.full_history_phase12 import FullHistoryStateV3, verify_complete_fit
    from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, deserialize_checkpoint
    from .phase13_durable_call_join import _RestoreContext
    from .phase13_main_preloaded_resources import PreloadedMainResources
    from .phase13_main_new_mcq_runtime import _baseline_state, _inject_root as inject_mcq
    from .phase13_route_capacity import FH_CAPACITY_CONFIG
    @dataclass(frozen=True, slots=True)
    class RestoreContext(_RestoreContext):
        expected_intervention: NativeEntry | None = None
    loaded = PreloadedMainResources(selected)
    baseline = unit.memory_baseline
    if ((unit.task, baseline) not in COVERED_CELLS or len(tasks) != 50
        or any(task.task_name != unit.task for task in tasks)):
        return None
    from memcontam.tasks.dispatch import render_model_visible_task
    for task in tasks:
        render_model_visible_task(task)
        if baseline == "dc_rs":
            from memcontam.experiment.phase13_dc_rs_validation import validate_task
            validate_task(task, "STRUCTURAL_UNCOVERED_TASK")
    if baseline == "rag_frozen":
        return "SUPPORTED" if selected_rag_domain(loaded, unit.task, prefix, branches) else None
    if baseline is None:
        return None
    run_id = f"main-a-{unit.prefix_unit_id}"
    context = RestoreContext({"dc_rs": {"serialized_cheatsheet_budget_tokens": 8192}}, tasks[0],
        RuntimeIdentities(run_id, f"{run_id}:trial:2:clean:{tasks[0].sample_id}", 2))
    runtime = PHASE13_CORE_BASELINE_REGISTRY[baseline]
    for arm, branch in zip(ARMS, branches, strict=True):
        branch_context = context
        deserialize_checkpoint(branch)
        try:
            state = runtime.restore_state(prefix.state, context)
        except ValueError:
            return None
        if runtime.serialize_state(state) != prefix.state:
            return None
        if arm != "clean":
            root = branch.state.entries[-1]
            if not isinstance(root, NativeEntry):
                return None
            branch_context = replace(context, expected_intervention=root)
            if unit.task in ("mmlu_pro_engineering", "mmlu_pro_physics"):
                inject_mcq(_baseline_state(state), root)
            else:
                _inject_root(state, root)
        native = runtime.serialize_state(state)
        if not isinstance(native, NativeState):
            return None
        reopened = runtime.serialize_state(runtime.restore_state(native, branch_context))
        if reopened != native:
            return None
        if baseline in ("bot_style", "reflexion_style") and native.native_state.get("active_capacity") is not None:
            return None
        if isinstance(state, FullHistoryStateV3) and state.injected_root_id is not None:
            fit = verify_complete_fit(tasks[0], state.records, FH_CAPACITY_CONFIG, requested_fh_mode="bounded")
            if state.injected_root_id not in fit.context.post_record_ids:
                return f"{arm}:1:FH_record_root_visibility"
    return "SUPPORTED"


def rag_resource_domain(corpus: object, index: object) -> bool:
    from .phase13_legacy_rag_models import CorpusBundle, IndexBundle
    from memcontam.contamination.phase12.models import canonical_json_hash
    import math
    if not isinstance(corpus, CorpusBundle) or not isinstance(index, IndexBundle):
        return False
    clean = corpus.branches["clean"]
    base = tuple((doc.id, doc.text) for doc in clean.documents)
    if (corpus.task_id != index.task_id or len(base) != 24
        or len(set(doc_id for doc_id, _ in base)) != 24
        or tuple(doc.document_id for doc in corpus.clean_documents) != tuple(doc_id for doc_id, _ in base)
        or tuple(doc.text for doc in corpus.clean_documents) != tuple(text for _, text in base)):
        return False
    for arm in ARMS:
        branch, native_index = corpus.branches[arm], index.branches[arm]
        documents = tuple((doc.id, doc.text) for doc in branch.documents)
        vectors = native_index.vectors
        if (branch.branch != arm or native_index.branch != arm
            or documents[:24] != base or len(documents) != 24 + (arm != "clean")
            or branch.clean_base_hash != clean.clean_base_hash
            or branch.active_document_ids != tuple(doc_id for doc_id, _ in documents)
            or native_index.documents != branch.documents
            or native_index.corpus_serialization_id != branch.serialization_id
            or set(vectors) != set(branch.active_document_ids)
            or any(len(vector) != 1024 or not all(math.isfinite(v) for v in vector)
                   or abs(sum(v * v for v in vector) - 1.0) > 0.001 for vector in vectors.values())
            or native_index.embedding_contract.get("dimension") != 1024
            or native_index.embedding_contract.get("normalized") is not True
            or native_index.index_artifact_hash != canonical_json_hash({
                "documents": [doc.model_dump(mode="json") for doc in native_index.documents],
                "embedding_contract": native_index.embedding_contract,
                "vectors": {key: list(value) for key, value in vectors.items()}})):
            return False
    return True


def selected_rag_domain(loaded: PreloadedMainResources, task: str,
                        prefix: Phase12Checkpoint, branches: tuple[Phase12Checkpoint, ...]) -> bool:
    from memcontam.memory.checkpoint_v3 import NativeEntry, deserialize_checkpoint
    corpus, index = loaded.legacy_bundles(task)
    if not rag_resource_domain(corpus, index):
        return False
    clean = corpus.branches["clean"]
    native = prefix.state
    if (native.native_state != {"branch": "clean", "corpus_id": clean.serialization_id,
        "index_id": index.branches["clean"].index_serialization_id, "read_only": True}
        or len(branches) != 4):
        return False
    for arm, branch in zip(ARMS, branches, strict=True):
        deserialize_checkpoint(branch)
        if tuple((entry.entry_id, entry.content) for entry in branch.state.entries
                 if isinstance(entry, NativeEntry)) != tuple((doc.id, doc.text) for doc in corpus.branches[arm].documents):
            return False
    return True


def qualify_rag_resources(root: Path) -> tuple[str, ...]:
    from memcontam.baselines.retrieval_rag_phase12 import RagFrozenStateV3
    from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
    from memcontam.experiment.phase13_ordinary_runtime import execute_prospective_ordinary
    from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
    from memcontam.rag.branch_index import BranchIndex
    from memcontam.rag.phase12_corpus import BranchCorpus, Document
    from .phase13_main_resource_contract import RESOURCE_PATHS
    from .phase13_legacy_rag_models import CorpusBundle, IndexBundle
    from memcontam.logging.schema_v3 import RetrievalEvent, ContextEvent
    class QueryEmbedder(FixtureEmbedder):
        def encode_query(self, text: str) -> list[float]:
            first = 'alpha' in text or '1 ?' in text or '[1,3,4,6]' in text.replace(' ', '')
            return ([1.0, 0.0] if first else [0.0, 1.0]) + [0.0] * 1022
    rows = []
    fixture = fixture_inputs("rag_frozen")
    runtime = PHASE13_CORE_BASELINE_REGISTRY["rag_frozen"]
    for task in ("game24", "math_equation_balancer", "word_sorting"):
        corpus_raw = (root / RESOURCE_PATHS["rag_corpus_" + task]).read_bytes()
        index_raw = (root / RESOURCE_PATHS["rag_index_" + task]).read_bytes()
        corpus, indices = CorpusBundle.model_validate_json(corpus_raw), IndexBundle.model_validate_json(index_raw)
        if not rag_resource_domain(corpus, indices):
            raise ValueError("STRUCTURAL_RAG_RESOURCE_DOMAIN_UNCOVERED")
        task_rows = tuple(qualification_task(task, slot) for slot in range(50))
        for arm in ARMS:
            branch, index = corpus.branches[arm], indices.branches[arm]
            documents = tuple(Document.from_mapping(doc.model_dump(mode="json")) for doc in branch.documents)
            state = RagFrozenStateV3(arm, BranchCorpus(arm, documents, branch.active_document_ids, branch.serialization_id),
                BranchIndex(arm, documents, index.embedding_contract, index.vectors,
                    index.index_serialization_id, QueryEmbedder()))
            native = runtime.serialize_state(state)
            if not isinstance(native, NativeState) or fixture.run.branch is None:
                raise ValueError("STRUCTURAL_RAG_RESOURCE_DOMAIN_UNCOVERED")
            checkpoint = serialize_checkpoint(native, checkpoint_index=0)
            live = replace(fixture.run.branch, arm=arm, state=state, checkpoint=checkpoint,
                root_count=0 if arm == "clean" else 1)
            run = replace(fixture.run, model="replay", task_name=task, tasks=task_rows, arm=arm, branch=live,
                client=FixtureClient(task), initial_states={"rag_frozen": state}, verifier=lambda _answer, _task: False)
            trials = tuple(execute_prospective_ordinary(replace(run, tasks=(task_row,))).trials[0]
                           for task_row in task_rows)
            observations = []
            for trial in trials:
                retrieved = trial.retrieval_event
                final = trial.context_event
                if (trial.state_before != native or trial.state_after != native
                    or not isinstance(retrieved, RetrievalEvent) or not isinstance(final, ContextEvent)
                    or len(retrieved.retrieved_entry_ids) != 3
                    or not set(retrieved.retrieved_entry_ids).issubset(branch.active_document_ids)
                    or final.final_entry_ids != retrieved.retrieved_entry_ids):
                    raise ValueError("STRUCTURAL_RAG_SELECTED_EXPECTATION_FAILED")
                reopened = runtime.serialize_state(runtime.restore_state(native,
                    replace(fixture.context, initial_states={"rag_frozen": state})))
                if reopened != native:
                    raise ValueError("STRUCTURAL_RAG_SELECTED_REOPEN_FAILED")
                if trial.outcome.status != "succeeded" and not (
                    trial.outcome.failure_disposition == "provider_call_failed"
                    and all(call.failure_code == "INPUT_ENVELOPE_EXCEEDED" for call in trial.outcome.method_calls)):
                    raise ValueError("STRUCTURAL_RAG_UNEXPECTED_TECHNICAL_OUTCOME")
                observations.append((retrieved.query_hash, tuple(retrieved.retrieved_entry_ids), tuple(final.final_entry_ids)))
            if (len(trials) != 50 or len({row[0] for row in observations}) < 2
                or len({row[1] for row in observations}) < 2):
                raise ValueError("STRUCTURAL_RAG_VARYING_CONTEXT_EXPECTATION_FAILED")
            rows.append(_bytes({"task": task, "arm": arm, "corpus_sha256": hashlib.sha256(corpus_raw).hexdigest(),
                "index_sha256": hashlib.sha256(index_raw).hexdigest(), "native": native,
                "observations": observations, "technical_outcomes": tuple(
                    tuple(call.failure_code for call in trial.outcome.method_calls) for trial in trials)}).decode())
    return tuple(rows)


def qualification_task(task: Literal["game24", "math_equation_balancer", "word_sorting"], slot: int) -> TaskInstance:
    inputs = {"game24": {"numbers": [1, 3, 4, 6] if slot % 2 else [3, 3, 8, 8], "target": 24},
              "math_equation_balancer": {"input": "1 ? 2 ? 3 = 6" if slot % 2 else "2 ? 3 ? 4 = 9"},
              "word_sorting": {"words": ["alpha", "beta"] if slot % 2 else ["gamma", "delta"]}}
    return TaskInstance(sample_id=f"qualification-{task}-{slot:02d}", task_name=task,
                        input=inputs[task], verifier_spec={"target": 24})


def validate_production_context(proof: StructuralSupportEvidence, context: StructuralSupportValidationContext,
                                origin: _ProductionStructuralOrigin) -> None:
    from memcontam.evaluation.phase13_seed_response import SeedResponseError
    if (proof.model_dump_json().encode() != origin.proof_bytes
        or context.qualified_evidence_bytes != origin.proof_bytes
        or _bytes(context.prefix) != origin.prefix_bytes
        or context.identity.model_dump_json().encode() != origin.identity_bytes
        or context.package_sha256 != origin.package_sha256
        or proof.qualification_sha256 != hashlib.sha256(canonical_bytes(
            NativeQualification.model_validate_json(origin.qualification_bytes))).hexdigest()
        or proof.subject_contract_sha256 != hashlib.sha256(origin.subject_bytes).hexdigest()
        or proof.fixture_contract_sha256 != context.fixture_contract_sha256
        or proof.branch_checkpoint_sha256 != context.branch_checkpoint_sha256
        or proof.candidate_registry_sha256 != context.candidate_registry_sha256):
        raise SeedResponseError("STRUCTURAL_QUALIFIED_CONTEXT_MISMATCH")
