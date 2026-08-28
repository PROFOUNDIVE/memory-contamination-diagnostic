from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.clients.base import LLMResponse
from memcontam.contamination.phase12.registry import load_candidate_registry
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.experiment.phase12.live_branch import build_live_reduced_main_branches
from memcontam.experiment.phase12.runtime_registry import PHASE13_CORE_BASELINE_REGISTRY
from memcontam.experiment.phase13_ordinary_runtime import (
    OrdinaryArm,
    ProspectiveOrdinaryRun,
    execute_prospective_ordinary,
)
from memcontam.experiment import phase13_ordinary_runtime as ordinary_runtime
from memcontam.evaluation.phase13_observability_registration import (
    ObservabilityRegistrationPacket,
)
from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from memcontam.readiness.phase13_production_observability import validate_production_archive
from memcontam.readiness.phase13_production_runtime_join import (
    ProductionOrdinaryRunIdentity,
    ProductionRuntimeJoinError,
    production_archive_from_ordinary,
)
from memcontam.tasks.base import TaskInstance


ARMS: tuple[OrdinaryArm, ...] = ("clean", "correct", "irrelevant", "contam")


class _Client:
    def __init__(self, fail_at: int | None = None, failure: Exception | None = None) -> None:
        self.configs: list[dict[str, JsonValue]] = []
        self.fail_at = fail_at
        self.failure = failure or TimeoutError("terminal provider failure")

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict[str, JsonValue],
    ) -> LLMResponse:
        del messages, model
        self.configs.append(dict(config))
        if len(self.configs) == self.fail_at:
            raise self.failure
        return LLMResponse(
            content="final: (6 / (1 - 3 / 4))",
            raw={"replay": True, "attempts": 1},
            token_usage={"prompt_tokens": 1, "completion_tokens": 1},
            latency_ms=0,
        )


def test_prospective_ordinary_executes_each_registered_arm_from_native_branch() -> None:
    task = TaskInstance(
        sample_id="game24:arm-test",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6], "target": 24},
        verifier_spec={"target": 24},
    )
    clean_state = FullHistoryStateV3(records=[])
    entry = PHASE13_CORE_BASELINE_REGISTRY["fh_bounded"]
    context = Game24RuntimeContext(
        task=task,
        client=_Client(),
        model="replay",
        verifier=lambda _answer, _task: True,
        decoding={"temperature": 0.0},
        branch="clean",
        identities=RuntimeIdentities("branch-build", "branch-build:trial", 0),
        baseline_configs={"fh_bounded": {"context_window_tokens": 10_000}},
        initial_states={"fh_bounded": clean_state},
    )
    snapshot = entry.serialize_state(clean_state)
    assert isinstance(snapshot, NativeState)
    branches = build_live_reduced_main_branches(
        prefix=serialize_checkpoint(snapshot),
        context=context,
        candidate_registry=load_candidate_registry(
            Path("data/phase12/registries/candidate_registry_v1.json")
        ),
        registry=PHASE13_CORE_BASELINE_REGISTRY,
    )

    results = []
    for arm in ARMS:
        client = _Client()
        branch = branches.arms[arm]
        result = execute_prospective_ordinary(
            ProspectiveOrdinaryRun(
                task_name="game24",
                baseline="fh_bounded",
                arm=arm,
                branch=branch,
                run_id=f"ordinary-{arm}",
                model="replay",
                client=client,
                allow_test_client=True,
                verifier=lambda _answer, _task: True,
                decoding={"temperature": 0.0},
                tasks=(task,),
            )
        )
        assert result.arm == arm
        assert [config["arm"] for config in client.configs] == [arm]
        assert isinstance(branch.state, FullHistoryStateV3)
        assert len(branch.state.records) == branch.root_count
        results.append(result)

    assert len({id(result.trials[0].state) for result in results}) == 4


def test_production_archive_reconstructs_an_ordinary_contamination_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ordinary_runtime, "_validated_common_capacity_tokens", lambda: 8192)
    task = TaskInstance(
        sample_id="game24:production-join",
        task_name="game24",
        input={"numbers": [1, 3, 4, 6], "target": 24},
        verifier_spec={"target": 24},
    )
    clean_state = FullHistoryStateV3(records=[])
    entry = PHASE13_CORE_BASELINE_REGISTRY["fh_bounded"]
    context = Game24RuntimeContext(
        task=task,
        client=_Client(),
        model="gpt-5.6-luna",
        verifier=lambda _answer, _task: True,
        decoding={"temperature": 0.0},
        branch="clean",
        identities=RuntimeIdentities("branch-build", "branch-build:trial", 0),
        baseline_configs={"fh_bounded": {"context_window_tokens": 10_000}},
        initial_states={"fh_bounded": clean_state},
    )
    snapshot = entry.serialize_state(clean_state)
    assert isinstance(snapshot, NativeState)
    checkpoint = NativeState(
        baseline=snapshot.baseline,
        entries=snapshot.entries,
        native_state={**snapshot.native_state, "checkpoint_index": 0},
        schema_version=snapshot.schema_version,
    )
    branch = build_live_reduced_main_branches(
        prefix=serialize_checkpoint(checkpoint),
        context=context,
        candidate_registry=load_candidate_registry(
            Path("data/phase12/registries/candidate_registry_v1.json")
        ),
        registry=PHASE13_CORE_BASELINE_REGISTRY,
    ).arms["contam"]
    run = ProspectiveOrdinaryRun(
        task_name="game24",
        baseline="fh_bounded",
        arm="contam",
        branch=branch,
        run_id="ordinary-production-join",
        model="gpt-5.6-luna",
        client=_Client(),
        allow_test_client=True,
        verifier=lambda _answer, _task: True,
        decoding={"temperature": 0.0},
        tasks=(task,),
        trajectory_seed=1,
    )
    result = execute_prospective_ordinary(run)
    packet_raw = Path("data/phase13/observability/registration_packet_v1.json").read_bytes()
    packet = ObservabilityRegistrationPacket.model_validate_json(packet_raw)
    packet_sha256 = hashlib.sha256(packet_raw).hexdigest()

    with pytest.raises(
        ProductionRuntimeJoinError,
        match="PRODUCTION_TRAJECTORY_SEED_MISMATCH",
    ):
        production_archive_from_ordinary(
            run,
            result,
            ProductionOrdinaryRunIdentity(
                execution_template_id="game24:fh_bounded:contam",
                trajectory_seed=2,
                concrete_seed_id="2",
                ordered_sample_ids_sha256=hashlib.sha256(
                    json.dumps(result.sample_ids, separators=(",", ":")).encode()
                ).hexdigest(),
                registration_packet_sha256=packet_sha256,
                scientific_result=False,
            ),
        )

    with pytest.raises(ProductionRuntimeJoinError, match="PRODUCTION_SAMPLE_ORDER_MISMATCH"):
        production_archive_from_ordinary(
            run,
            result,
            ProductionOrdinaryRunIdentity(
                execution_template_id="game24:fh_bounded:contam",
                trajectory_seed=1,
                concrete_seed_id="1",
                ordered_sample_ids_sha256="0" * 64,
                registration_packet_sha256=packet_sha256,
                scientific_result=False,
            ),
        )

    archive = production_archive_from_ordinary(
        run,
        result,
        ProductionOrdinaryRunIdentity(
            execution_template_id="game24:fh_bounded:contam",
            trajectory_seed=1,
            concrete_seed_id="1",
            ordered_sample_ids_sha256=hashlib.sha256(
                json.dumps(result.sample_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            registration_packet_sha256=packet_sha256,
            scientific_result=False,
        ),
    )
    report = validate_production_archive(archive, packet, packet_sha256)

    assert report.status == "PASS"
    assert report.record_count == 1
    assert archive.records[0].evidence.evidence_scope == "production_runtime"
    assert archive.records[0].evidence.target_set.target_entry_ids == (
        branch.injected_root_id,
    )
    assert archive.records[0].request.retries_after_initial_attempt == 0
    assert len(archive.records[0].evidence.memory_events) == 1

    suffix = tuple(
        task.model_copy(update={"sample_id": f"game24:production-terminal:{index}"})
        for index in range(3)
    )
    provider_failure = TimeoutError("terminal provider failure")
    setattr(provider_failure, "provider_attempts_count", 1)
    setattr(provider_failure, "provider_latency_ms", 23)
    setattr(provider_failure, "provider_status", "incomplete")
    setattr(provider_failure, "provider_incomplete_reason", "max_output_tokens")
    setattr(provider_failure, "provider_usage", {"input_tokens": 7, "output_tokens": 11})
    setattr(provider_failure, "provider_token_usage", {"prompt_tokens": 7, "completion_tokens": 11})
    setattr(provider_failure, "provider_cost_usd", 0.25)
    setattr(provider_failure, "provider_response_id", "resp_incomplete")
    terminal_run = replace(
        run,
        run_id="ordinary-production-terminal",
        client=_Client(fail_at=2, failure=provider_failure),
        tasks=suffix,
    )
    terminal_result = execute_prospective_ordinary(terminal_run)
    terminal_archive = production_archive_from_ordinary(
        terminal_run,
        terminal_result,
        ProductionOrdinaryRunIdentity(
            execution_template_id="game24:fh_bounded:contam",
            trajectory_seed=1,
            concrete_seed_id="1",
            ordered_sample_ids_sha256=hashlib.sha256(
                json.dumps(terminal_result.sample_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            registration_packet_sha256=packet_sha256,
            scientific_result=False,
        ),
    )
    terminal_report = validate_production_archive(terminal_archive, packet, packet_sha256)

    assert terminal_report.record_count == 2
    assert terminal_report.technical_missing_count == 1
    assert terminal_archive.records[-1].evidence.verified_outcome is None
    assert terminal_archive.records[-1].terminal_provider_evidence is not None
    terminal_provider = terminal_archive.records[-1].terminal_provider_evidence
    assert terminal_provider.attempts_count == 1
    assert terminal_provider.latency_ms == 23
    assert terminal_provider.status == "incomplete"
    assert terminal_provider.incomplete_reason == "max_output_tokens"
    assert terminal_provider.usage == {"input_tokens": 7, "output_tokens": 11}
    assert terminal_provider.token_usage == {"prompt_tokens": 7, "completion_tokens": 11}
    assert terminal_provider.cost_usd == 0.25
    assert terminal_provider.response_id == "resp_incomplete"
    assert terminal_archive.records[-1].evidence.trial.analysis_inclusion == "excluded"

    terminal_trial = terminal_result.trials[-1]
    terminal_call = terminal_trial.outcome.method_calls[-1]
    zero_dispatch_call = terminal_call.model_copy(
        update={
            "failure_code": "INPUT_ENVELOPE_EXCEEDED",
            "transport_attempts": 0,
            "latency_ms": None,
            "token_usage": {},
            "provider_status": None,
            "provider_incomplete_reason": None,
            "provider_usage": None,
            "provider_cost_usd": None,
            "provider_response_id": None,
        }
    )
    zero_dispatch_result = replace(
        terminal_result,
        trials=(
            *terminal_result.trials[:-1],
            replace(
                terminal_trial,
                outcome=replace(
                    terminal_trial.outcome,
                    method_calls=(zero_dispatch_call,),
                ),
            ),
        ),
    )
    zero_dispatch_archive = production_archive_from_ordinary(
        terminal_run,
        zero_dispatch_result,
        ProductionOrdinaryRunIdentity(
            execution_template_id="game24:fh_bounded:contam",
            trajectory_seed=1,
            concrete_seed_id="1",
            ordered_sample_ids_sha256=hashlib.sha256(
                json.dumps(zero_dispatch_result.sample_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            registration_packet_sha256=packet_sha256,
            scientific_result=False,
        ),
    )
    zero_dispatch = zero_dispatch_archive.records[-1].terminal_provider_evidence
    assert zero_dispatch is not None
    assert zero_dispatch.trigger_class == "input_envelope_violation"
    assert zero_dispatch.failure_code == "INPUT_ENVELOPE_EXCEEDED"
    assert zero_dispatch.attempts_count == 0
    assert zero_dispatch.latency_ms is None
