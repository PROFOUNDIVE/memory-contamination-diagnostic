import hashlib
import json
from pathlib import Path

import pytest

from memcontam.baselines.bot_phase12 import BoTStateV3
from memcontam.baselines.dynamic_cheatsheet_phase12 import DcRsStateV3
from memcontam.baselines.reflexion_phase12 import ReflexionStateV3
from memcontam.clients.base import LLMResponse
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.contamination.phase12.renderers import RendererRegistry
from memcontam.evaluation.phase13_observability_models import Phase13TrialEvidence
from memcontam.evaluation.phase13_observability_registration import ObservabilityRegistrationPacket
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.live_branch import build_live_reduced_main_branches
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.experiment.phase13_ordinary_runtime import (
    OrdinaryBaseline,
    ProspectiveOrdinaryRun,
    execute_prospective_ordinary,
)
from memcontam.memory.checkpoint_v3 import NativeEntry, NativeState, serialize_checkpoint
from memcontam.memory.embeddings import BgeM3EmbeddingProvider
from memcontam.readiness.phase13_main_live_runtime import ProductionMainRuntime
from memcontam.readiness.phase13_production_observability import validate_production_archive
from memcontam.readiness.phase13_production_runtime_join import (
    ProductionOrdinaryRunIdentity,
    production_archive_from_ordinary,
)
from memcontam.tasks.base import TaskInstance
from memcontam.tasks.word_sorting import build_instance as build_words
from memcontam.verifiers.game24 import verify_expression
from memcontam.verifiers.word_sorting import verify_words

from .test_phase13_native_rendering import _clean_state, _Embedder
from .test_phase13_readiness0_production_dry_run import _ContractFakeEmbeddingProvider


def test_bot_refused_novelty_at_capacity_is_not_reachable_in_current_package(
    tmp_path: Path,
) -> None:
    runtime = ProductionMainRuntime(Path(__file__).resolve().parents[1], tmp_path / "cache")
    state = runtime._initial_states("mmlu_pro_engineering")["bot_style"]
    from memcontam.evaluation.phase13_observability_registration import AUTHORITY_HASHES

    assert runtime._packet.authority_hashes == AUTHORITY_HASHES
    assert isinstance(state, BoTStateV3)
    assert state.active_capacity is None
    assert "bot_style" not in runtime._configs()


class NativeResponses:
    def __init__(self) -> None:
        self.stages: list[str] = []
        self.dc_rs_synthesis_count = 0

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        del messages, model
        stage = config["method_stage"]
        self.stages.append(stage)
        if stage == "dc_rs_synthesize":
            self.dc_rs_synthesis_count += 1
        content = {
            "bot_problem_distill": json.dumps(
                {
                    "key_information": "four numbers",
                    "restrictions": "exact use",
                    "distilled_task": "make 24",
                }
            ),
            "bot_instantiate_solve": json.dumps(
                {
                    "selected_structure": "retrieved-template",
                    "solution_trace": "divide",
                    "final_answer": "final: 6/(1-3/4)",
                }
            ),
            "bot_thought_distill": json.dumps(
                {
                    "description": "arithmetic",
                    "template": "check all numbers",
                    "category": "procedure-based",
                    "explicitly_used_memory_ids": [],
                }
            ),
            "dc_rs_synthesize": (
                "<cheatsheet>ordinary strategy</cheatsheet><source_ids>src01</source_ids>"
                if self.dc_rs_synthesis_count == 1
                else "<cheatsheet>rewritten strategy</cheatsheet>"
            ),
        }.get(stage, "final: 6/(1-3/4)")
        return LLMResponse(
            content,
            {"replay": True, "attempts": 1},
            {"prompt_tokens": 1, "completion_tokens": 1},
            0,
        )


class NoveltyEmbedder(_Embedder):
    def encode_query(self, text: str) -> list[float]:
        return [0.0, 1.0] if text == "arithmetic" else [1.0, 0.0]


def test_nomem_archive_binds_only_the_answer_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from memcontam.experiment import phase13_ordinary_runtime

    monkeypatch.setattr(phase13_ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    task = TaskInstance(
        sample_id="nomem-native-game24",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6], "target": 24},
        verifier_spec={"target": 24},
    )
    run = ProspectiveOrdinaryRun(
        task_name="game24",
        baseline="nomem",
        run_id="native-nomem",
        model="gpt-5.6-luna",
        client=NativeResponses(),
        allow_test_client=True,
        verifier=lambda answer, row: verify_expression(answer, row.input["numbers"]),
        decoding={"temperature": 0.0},
        tasks=(task,),
        trajectory_seed=0,
    )
    result = execute_prospective_ordinary(run)
    raw = Path("data/phase13/observability/registration_packet_v2.json").read_bytes()
    identity = ProductionOrdinaryRunIdentity(
        execution_template_id="game24|nomem",
        trajectory_seed=0,
        concrete_seed_id="0",
        ordered_sample_ids_sha256=hashlib.sha256(
            json.dumps(result.sample_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        registration_packet_sha256=hashlib.sha256(raw).hexdigest(),
        scientific_result=True,
    )
    archive = production_archive_from_ordinary(run, result, identity)
    packet = ObservabilityRegistrationPacket.model_validate_json(raw)

    assert (
        validate_production_archive(
            archive, packet, identity.registration_packet_sha256, frozen_tasks=run.tasks
        ).status
        == "PASS"
    )
    record = archive.records[0]
    forged = archive.model_copy(
        update={
            "records": (
                record.model_copy(
                    update={
                        "method_calls": (
                            record.method_calls[0].model_copy(
                                update={"call_id": "unbound-auxiliary"}
                            ),
                            *record.method_calls,
                        ),
                    }
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="PRODUCTION_CLASSIFIER_JOIN_MISMATCH"):
        validate_production_archive(
            forged, packet, identity.registration_packet_sha256, frozen_tasks=run.tasks
        )


class ReflectionResponses(NativeResponses):
    def __init__(self, *, final_success: bool = False) -> None:
        super().__init__()
        self.final_success = final_success

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        if config["method_stage"] == "reflexion_reflect":
            self.stages.append("reflexion_reflect")
            return LLMResponse(
                json.dumps(
                    {
                        "mode": "corrective",
                        "failure_class": "incorrect_answer",
                        "reflection_text": "Check the arithmetic",
                        "explicitly_used_memory_ids": [],
                    }
                ),
                {"replay": True, "attempts": 1},
                {"prompt_tokens": 1, "completion_tokens": 1},
                0,
            )
        if config["method_stage"] == "reflexion_generate":
            self.stages.append("reflexion_generate")
            content = (
                "final: 6/(1-3/4)"
                if self.final_success and self.stages.count("reflexion_generate") == 2
                else "final: 1+3+4+6"
            )
            return LLMResponse(
                content,
                {"replay": True, "attempts": 1},
                {"prompt_tokens": 1, "completion_tokens": 1},
                0,
            )
        return super().chat(messages, model, config)


@pytest.mark.parametrize("final_success", (False, True))
def test_reflexion_multiple_native_writes_select_final_actor_in_archive(
    monkeypatch: pytest.MonkeyPatch,
    final_success: bool,
) -> None:
    from memcontam.experiment import phase13_ordinary_runtime

    monkeypatch.setattr(phase13_ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    task = TaskInstance(
        sample_id="reflexion-write",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6], "target": 24},
        verifier_spec={"target": 24},
    )
    runtime = PHASE13_CORE_BASELINE_REGISTRY["reflexion_style"]
    state = ReflexionStateV3(reflections=[])
    snapshot = runtime.serialize_state(state)
    assert isinstance(snapshot, NativeState)
    client = ReflectionResponses(final_success=final_success)
    context = Game24RuntimeContext(
        task=task,
        client=client,
        model="gpt-5.6-luna",
        verifier=lambda answer, row: verify_expression(answer, row.input["numbers"]),
        decoding={"temperature": 0.0},
        branch="clean",
        identities=RuntimeIdentities("reflexion", "prefix", 0),
        initial_states={"reflexion_style": state},
    )
    branch = build_live_reduced_main_branches(
        prefix=serialize_checkpoint(snapshot, checkpoint_index=0),
        context=context,
        candidate_registry=load_candidate_registry(
            Path("data/phase12/registries/candidate_registry_v2.json")
        ),
        registry=PHASE13_CORE_BASELINE_REGISTRY,
    ).arms["contam"]
    run = ProspectiveOrdinaryRun(
        task_name="game24",
        baseline="reflexion_style",
        run_id="reflexion-native",
        model="gpt-5.6-luna",
        client=client,
        allow_test_client=True,
        verifier=lambda answer, row: verify_expression(answer, row.input["numbers"]),
        decoding={"temperature": 0.0},
        arm="contam",
        branch=branch,
        tasks=(task,),
        trajectory_seed=0,
    )
    result = execute_prospective_ordinary(run)
    raw = Path("data/phase13/observability/registration_packet_v2.json").read_bytes()
    identity = ProductionOrdinaryRunIdentity(
        execution_template_id="game24:reflexion_style:contam",
        trajectory_seed=0,
        concrete_seed_id="0",
        ordered_sample_ids_sha256=hashlib.sha256(
            json.dumps(result.sample_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        registration_packet_sha256=hashlib.sha256(raw).hexdigest(),
        scientific_result=True,
    )

    archive = production_archive_from_ordinary(run, result, identity)

    assert len(result.trials[0].write_envelopes) == (1 if final_success else 2)
    assert client.stages == (
        ["reflexion_generate", "reflexion_reflect", "reflexion_generate"]
        if final_success
        else ["reflexion_generate", "reflexion_reflect"] * 2
    )
    evidence = archive.records[0].evidence
    assert isinstance(evidence, Phase13TrialEvidence)
    assert evidence.target_set.answer_call_id == result.trials[0].outcome.answer_call_id
    assert (
        validate_production_archive(
            archive,
            ObservabilityRegistrationPacket.model_validate_json(raw),
            identity.registration_packet_sha256,
            frozen_tasks=run.tasks,
        ).status
        == "PASS"
    )


@pytest.mark.parametrize("baseline", ("rag_frozen", "reflexion_style", "dc_rs", "bot_style"))
def test_native_trial_reaches_current_archive_validator(
    baseline: OrdinaryBaseline,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from memcontam.experiment import phase13_ordinary_runtime

    monkeypatch.setattr(phase13_ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    task = TaskInstance(
        sample_id="native-game24",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6], "target": 24},
        verifier_spec={"target": 24},
    )
    tasks = (
        (task, task.model_copy(update={"sample_id": "native-game24-2"}))
        if baseline == "dc_rs"
        else (task,)
    )
    if baseline == "dc_rs":
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    local_embedder = NoveltyEmbedder() if baseline == "bot_style" else _Embedder()
    embedder = BgeM3EmbeddingProvider() if baseline == "dc_rs" else local_embedder
    state = (
        DcRsStateV3(archive=[], allow_unparented_strategies=True)
        if baseline == "dc_rs"
        else _clean_state(baseline, local_embedder)
    )
    runtime = PHASE13_CORE_BASELINE_REGISTRY[baseline]
    snapshot = runtime.serialize_state(state)
    assert isinstance(snapshot, NativeState)
    client = NativeResponses()
    context = Game24RuntimeContext(
        task=task,
        client=client,
        model="gpt-5.6-luna",
        verifier=lambda answer, row: verify_expression(answer, row.input["numbers"]),
        decoding={"temperature": 0.0},
        branch="clean",
        identities=RuntimeIdentities("native", "native:trial:1:native-game24", 1),
        embedding_provider=embedder,
        baseline_configs={
            "fh_bounded": {"context_window_tokens": 10000},
            "dc_rs": {"serialized_cheatsheet_budget_tokens": 8192},
        },
        initial_states={baseline: state},
    )
    registry_raw = Path("data/phase12/registries/candidate_registry_v2.json").read_bytes()
    registry = load_candidate_registry(Path("data/phase12/registries/candidate_registry_v2.json"))
    renderers = (
        RendererRegistry.governed(
            Path("data/phase13/main/legacy_dc_rs_intervention_registry_v2.json").read_bytes(),
            registry,
            hashlib.sha256(registry_raw).hexdigest(),
        )
        if baseline == "dc_rs"
        else None
    )
    prefix = serialize_checkpoint(snapshot, checkpoint_index=0)
    if baseline == "dc_rs":
        checkpoint_file = tmp_path / "dc-prefix-checkpoint.json"
        checkpoint_file.write_bytes(prefix.canonical_bytes)
        saved = checkpoint_file.read_bytes()
        prefix = serialize_checkpoint(
            NativeState.from_mapping(json.loads(saved)), checkpoint_index=prefix.checkpoint_index
        )
        assert prefix.canonical_sha256 == hashlib.sha256(saved).hexdigest()
        runtime.restore_state(prefix.state, context)
    branch = build_live_reduced_main_branches(
        prefix=prefix,
        context=context,
        candidate_registry=registry,
        registry=PHASE13_CORE_BASELINE_REGISTRY,
        renderers=renderers,
    ).arms["contam"]
    run = ProspectiveOrdinaryRun(
        task_name="game24",
        baseline=baseline,
        run_id=f"native-{baseline}",
        model="gpt-5.6-luna",
        client=client,
        allow_test_client=True,
        verifier=lambda answer, row: verify_expression(answer, row.input["numbers"]),
        decoding={"temperature": 0.0},
        arm="contam",
        branch=branch,
        tasks=tasks,
        trajectory_seed=0,
        embedding_provider=embedder,
        baseline_configs={"dc_rs": {"serialized_cheatsheet_budget_tokens": 8192}},
    )
    result = execute_prospective_ordinary(run)
    packet_raw = Path("data/phase13/observability/registration_packet_v2.json").read_bytes()
    packet = ObservabilityRegistrationPacket.model_validate_json(packet_raw)
    identity = ProductionOrdinaryRunIdentity(
        execution_template_id=f"game24:{baseline}:contam",
        trajectory_seed=0,
        concrete_seed_id="0",
        ordered_sample_ids_sha256=hashlib.sha256(
            json.dumps(result.sample_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        registration_packet_sha256=hashlib.sha256(packet_raw).hexdigest(),
        scientific_result=True,
    )

    archive = production_archive_from_ordinary(run, result, identity)

    assert (
        validate_production_archive(
            archive, packet, identity.registration_packet_sha256, frozen_tasks=run.tasks
        ).status
        == "PASS"
    )
    assert archive.records[0].evidence.trial.execution_status == "completed"
    assert client.stages
    if baseline == "bot_style":
        assert client.stages == [
            "bot_problem_distill",
            "bot_instantiate_solve",
            "bot_thought_distill",
        ]
        assert result.trials[0].write_envelopes
        bot_evidence = archive.records[0].evidence
        assert isinstance(bot_evidence, Phase13TrialEvidence)
        assert bot_evidence.new_entry_ids
    if baseline == "dc_rs":
        assert len(archive.records) == 2
        assert result.trials[0].state_after == result.trials[1].state_before
        assert client.stages == ["dc_rs_synthesize", "dc_rs_generate"] * 2
        first = archive.records[0].evidence
        second = archive.records[1].evidence
        assert isinstance(first, Phase13TrialEvidence)
        assert isinstance(second, Phase13TrialEvidence)
        assert first.new_entry_ids
        first_state_after = result.trials[0].state_after
        second_state_after = result.trials[1].state_after
        assert isinstance(first_state_after, NativeState)
        assert isinstance(second_state_after, NativeState)
        first_strategy_ids = tuple(
            entry.entry_id
            for entry in first_state_after.entries
            if isinstance(entry, NativeEntry) and entry.native_component == "strategy"
        )
        second_strategy_ids = tuple(
            entry.entry_id
            for entry in second_state_after.entries
            if isinstance(entry, NativeEntry) and entry.native_component == "strategy"
        )
        assert len(first_strategy_ids) == len(second_strategy_ids) == 1
        assert first_strategy_ids[0] != second_strategy_ids[0]
        assert first_strategy_ids[0] in second.removed_entry_ids
        assert second.target_set.answer_call_spans
        assert {span.lineage_basis for span in second.target_set.answer_call_spans} == {
            "version_edge"
        }
        assert any(node.entry_id == first_strategy_ids[0] for node in second.lineage)


def test_word_sorting_dc_rs_native_branch_consumes_checkpoint_and_validates_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memcontam.experiment import phase13_ordinary_runtime

    monkeypatch.setattr(phase13_ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    row = json.loads(
        Path("data/phase13/main/word_sorting_main_v1.jsonl").read_text().splitlines()[0]
    )
    task = build_words(row)
    tasks = (task, task.model_copy(update={"sample_id": "word-sorting-second"}))
    embedder = _ContractFakeEmbeddingProvider(vector_dimension=1024)
    state = DcRsStateV3(archive=[], allow_unparented_strategies=True)
    runtime = PHASE13_CORE_BASELINE_REGISTRY["dc_rs"]
    snapshot = runtime.serialize_state(state)
    assert isinstance(snapshot, NativeState)
    prefix = serialize_checkpoint(snapshot, checkpoint_index=0)
    checkpoint_file = tmp_path / "words-prefix.json"
    checkpoint_file.write_bytes(prefix.canonical_bytes)
    prefix = serialize_checkpoint(
        NativeState.from_mapping(json.loads(checkpoint_file.read_bytes())),
        checkpoint_index=prefix.checkpoint_index,
    )

    class WordResponses(NativeResponses):
        def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
            if config["method_stage"] == "dc_rs_generate":
                self.stages.append("dc_rs_generate")
                return LLMResponse(
                    "final: syndrome therefrom",
                    {"replay": True, "attempts": 1},
                    {"prompt_tokens": 1, "completion_tokens": 1},
                    0,
                )
            return super().chat(messages, model, config)

    client = WordResponses()
    context = Game24RuntimeContext(
        task=task,
        client=client,
        model="gpt-5.6-luna",
        verifier=lambda answer, item: verify_words(
            answer.split(), item.verifier_spec["sorted_words"]
        ),
        decoding={"temperature": 0.0},
        branch="clean",
        identities=RuntimeIdentities("native-words", f"native-words:trial:1:{task.sample_id}", 1),
        embedding_provider=embedder,
        baseline_configs={"dc_rs": {"serialized_cheatsheet_budget_tokens": 8192}},
        initial_states={"dc_rs": state},
    )
    runtime.restore_state(prefix.state, context)
    registry_path = Path("data/phase12/registries/candidate_registry_v2.json")
    registry = load_candidate_registry(registry_path)
    renderers = RendererRegistry.governed(
        Path("data/phase13/main/legacy_dc_rs_intervention_registry_v2.json").read_bytes(),
        registry,
        hashlib.sha256(registry_path.read_bytes()).hexdigest(),
    )
    branch = build_live_reduced_main_branches(
        prefix=prefix,
        context=context,
        candidate_registry=registry,
        registry=PHASE13_CORE_BASELINE_REGISTRY,
        renderers=renderers,
    ).arms["contam"]
    run = ProspectiveOrdinaryRun(
        task_name="word_sorting",
        baseline="dc_rs",
        run_id="native-words",
        model="gpt-5.6-luna",
        client=client,
        allow_test_client=True,
        verifier=lambda answer, item: verify_words(
            answer.split(), item.verifier_spec["sorted_words"]
        ),
        decoding={"temperature": 0.0},
        arm="contam",
        branch=branch,
        tasks=tasks,
        trajectory_seed=0,
        embedding_provider=embedder,
        baseline_configs={"dc_rs": {"serialized_cheatsheet_budget_tokens": 8192}},
    )
    result = execute_prospective_ordinary(run)
    packet_raw = Path("data/phase13/observability/registration_packet_v2.json").read_bytes()
    identity = ProductionOrdinaryRunIdentity(
        execution_template_id="word_sorting:dc_rs:contam",
        trajectory_seed=0,
        concrete_seed_id="0",
        ordered_sample_ids_sha256=hashlib.sha256(
            json.dumps(result.sample_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        registration_packet_sha256=hashlib.sha256(packet_raw).hexdigest(),
        scientific_result=True,
    )
    archive = production_archive_from_ordinary(run, result, identity)

    assert client.stages == ["dc_rs_synthesize", "dc_rs_generate"] * 2
    assert len(archive.records) == 2
    assert result.trials[0].state_after == result.trials[1].state_before
    first = archive.records[0].evidence
    second = archive.records[1].evidence
    assert isinstance(first, Phase13TrialEvidence)
    assert isinstance(second, Phase13TrialEvidence)
    assert branch.injected_root_id is not None
    assert branch.injected_root_id in first.memory_before_ids
    assert any(node.entry_id == branch.injected_root_id for node in first.lineage)
    assert first.new_entry_ids
    assert set(first.new_entry_ids) <= set(second.memory_before_ids)
    after = result.trials[0].state_after
    assert isinstance(after, NativeState)
    entries = tuple(entry for entry in after.entries if isinstance(entry, NativeEntry))
    assert len(entries) == len(after.entries)
    archive_positions = [
        index for index, entry in enumerate(entries) if entry.native_component == "archive"
    ]
    strategy_positions = [
        index for index, entry in enumerate(entries) if entry.native_component == "strategy"
    ]
    assert archive_positions and strategy_positions
    assert max(archive_positions) < min(strategy_positions)
    archive_ids = {entries[index].entry_id for index in archive_positions}
    assert all(set(entries[index].direct_parent_ids) <= archive_ids for index in strategy_positions)
    assert (
        validate_production_archive(
            archive,
            ObservabilityRegistrationPacket.model_validate_json(packet_raw),
            identity.registration_packet_sha256,
            frozen_tasks=tasks,
        ).status
        == "PASS"
    )
