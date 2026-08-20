from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from memcontam.baselines.dynamic_cheatsheet_phase12 import (
    DcRsContractError,
    SourceAliasTable,
)
from memcontam.memory.stores import MemoryEntry
from memcontam.tasks.base import TaskInstance


STATUS_PATH = Path("data/phase13/common_capacity_status_v1.json")


def test_source_aliases_are_bounded_reversible_and_collision_safe() -> None:
    table = SourceAliasTable.from_source_ids(("archive:" + "x" * 500, "archive:second"))

    assert table.visible_ids == ("src01", "src02")
    assert table.resolve(("src02", "src01")) == ("archive:second", "archive:" + "x" * 500)
    with pytest.raises(DcRsContractError, match="INVALID_EXPLICIT_SOURCE_IDS"):
        table.resolve(("src01", "src01"))
    with pytest.raises(DcRsContractError, match="INVALID_EXPLICIT_SOURCE_IDS"):
        table.resolve(("src03",))
    with pytest.raises(DcRsContractError, match="INVALID_EXPLICIT_SOURCE_IDS"):
        SourceAliasTable.from_source_ids(("duplicate", "duplicate"))
    with pytest.raises(DcRsContractError, match="INVALID_EXPLICIT_SOURCE_IDS"):
        SourceAliasTable.from_source_ids(tuple(f"source-{index}" for index in range(100)))


def test_approved_token_contract_binds_complete_common_capacity() -> None:
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))

    assert status["status"] == "COMPLETE"
    assert status["reason"] == "COMMON_CAPACITY_MATERIALIZED"
    expected_runtime = {
        "provider": "OpenAI",
        "endpoint": "Responses API",
        "requested_model_id": "gpt-5.6-luna",
        "dated_provider_snapshot": "not_exposed",
        "context_window_tokens": 1_050_000,
        "provider_max_output_tokens": 128_000,
        "answer_max_output_tokens": 4096,
        "writer_max_output_tokens": 8192,
        "service_tier_decision_code": 3,
        "service_tier_provider_literal": "default",
    }
    assert {
        key: status["runtime_contract"][key] for key in expected_runtime
    } == expected_runtime
    assert status["token_contract"]["contract_id"] == "phase13_registered_token_accounting_v1"
    assert status["token_contract"]["encoding"] == "o200k_base"
    assert status["token_contract"]["tiktoken_version"] == "0.13.0"
    assert status["token_contract"]["fixed_prompt_overhead_tokens"] == 0
    assert status["token_contract"]["registered_safety_margin_tokens"] == 0
    assert status["runtime_implementation"] == {
        "openai_sdk_version": "2.46.0",
        "httpx_version": "0.28.1",
        "httpcore_version": "1.0.9",
        "pydantic_version": "2.13.4",
        "client_implementation_sha256": hashlib.sha256(
            Path("src/memcontam/clients/openai_responses.py").read_bytes()
        ).hexdigest(),
    }
    assert status["source_alias_contract"]["visible_format"] == "srcNN"
    assert status["source_alias_contract"]["maximum_visible_alias_tokens"] == 2
    assert status["source_alias_contract"]["runtime_bound"] is True
    implementation = Path("src/memcontam/baselines/dynamic_cheatsheet_phase12.py")
    assert status["source_alias_contract"]["implementation_sha256"] == hashlib.sha256(
        implementation.read_bytes()
    ).hexdigest()
    assert status["token_contract"]["counting_implementation_sha256"] == hashlib.sha256(
        Path("src/memcontam/baselines/prompt_budget.py").read_bytes()
    ).hexdigest()
    assert status["production_builder_hashes"] == {
        "measurement_implementation_sha256": importlib.import_module(
            "memcontam.readiness.phase13_capacity_realization"
        ).measurement_implementation_sha256(Path(".")),
        "dc_rs_runtime_builder_sha256": hashlib.sha256(
            Path("src/memcontam/experiment/phase13_dc_rs_runtime.py").read_bytes()
        ).hexdigest(),
        "dc_rs_prompt_builder_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
        "ordinary_runtime_builder_sha256": hashlib.sha256(
            Path("src/memcontam/experiment/phase13_ordinary_runtime.py").read_bytes()
        ).hexdigest(),
    }
    assert status["registered_writer_io_contract"] == {
        "registered_persisted_raw_answer_ceiling": 8192,
        "capacity_unit": "registered_serialized_tokens",
        "overflow_policy": "fail_closed_without_persistent_archive_admission",
        "writer_response_grammar": "exactly_one_complete_whole_cheatsheet_serialization_v1",
        "F_DC_out_tokens": 0,
    }
    artifact = Path(status["capacity_artifact"]["path"])
    assert status["capacity_artifact"]["sha256"] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    capacity = json.loads(artifact.read_text(encoding="utf-8"))
    validator = importlib.import_module("memcontam.readiness.phase13_capacity_realization")
    validated = validator.validate_common_capacity_artifact(artifact, Path("."))
    assert capacity["production_builder_hashes"]["measurement_implementation_sha256"] == (
        validator.measurement_implementation_sha256(Path("."))
    )
    assert status["validator"]["sha256"] == hashlib.sha256(
        Path(status["validator"]["path"]).read_bytes()
    ).hexdigest()
    assert capacity["B_mem_tokens"] == min(
        capacity["B_FH_feasible"], capacity["B_DC_feasible"]
    )
    assert capacity["L_DC_tokens"] == capacity["B_mem_tokens"] == 8192
    assert validated.B_mem_tokens == 8192
    assert set(capacity["per_task_R_FH"]) == {
        "game24",
        "math_equation_balancer",
        "word_sorting",
        "mmlu_pro_engineering",
        "mmlu_pro_physics",
        "gpqa_diamond",
    }
    assert set(capacity["per_task_I_DC_writer"]) == set(capacity["per_task_R_FH"])
    assert set(capacity["per_task_F_DC_out"]) == set(capacity["per_task_R_FH"])
    assert set(capacity["per_task_F_DC_out"].values()) == {0}
    assert status["completion"] == {
        "capacity_record_materialized": True,
        "runtime_bound": True,
        "remaining_prerequisites": [],
        "next_required_action": None,
    }


def test_capacity_parser_rejects_tampered_materialized_value() -> None:
    module = importlib.import_module("memcontam.readiness.phase13_capacity_realization")
    artifact = Path("data/phase13/common_capacity_v1.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    record = module.parse_common_capacity(artifact.read_bytes())
    payload["B_mem_tokens"] = 8191

    assert record.B_mem_tokens == 8192
    with pytest.raises(module.CapacityRealizationError, match="CAPACITY_FORMULA_MISMATCH"):
        module.parse_common_capacity(json.dumps(payload))


def test_capacity_reserves_are_rebuilt_from_frozen_registries() -> None:
    module = importlib.import_module("memcontam.readiness.phase13_capacity_realization")

    reserves = module.derive_capacity_reserves(Path("."))

    assert reserves.per_task_R_FH == {
        "game24": 4149,
        "math_equation_balancer": 4163,
        "word_sorting": 4215,
        "mmlu_pro_engineering": 5234,
        "mmlu_pro_physics": 4875,
        "gpqa_diamond": 7029,
    }
    assert reserves.per_task_I_DC_writer == {
        "game24": 33032,
        "math_equation_balancer": 33085,
        "word_sorting": 33286,
        "mmlu_pro_engineering": 36561,
        "mmlu_pro_physics": 34972,
        "gpqa_diamond": 37905,
    }


def test_capacity_validator_rejects_self_consistent_but_underived_reserves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("memcontam.readiness.phase13_capacity_realization")
    artifact = Path("data/phase13/common_capacity_v1.json")
    record = module.parse_common_capacity(artifact.read_bytes())
    tampered = record.model_copy(
        update={"per_task_R_FH": {**record.per_task_R_FH, "game24": 4151}}
    )
    monkeypatch.setattr(module, "parse_common_capacity", lambda _raw: tampered)

    with pytest.raises(module.CapacityRealizationError, match="CAPACITY_RESERVE_MISMATCH"):
        module.validate_common_capacity_artifact(artifact, Path("."))


def test_dc_writer_reserve_uses_complete_strategy_and_distinct_prior_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("memcontam.readiness.phase13_capacity_measurement")
    tasks = tuple(
        TaskInstance(
            sample_id=f"game24:{index}",
            task_name="game24",
            input={"numbers": [index, 2, 3, 4]},
        )
        for index in range(1, 6)
    )
    seen: list[tuple[MemoryEntry, tuple[MemoryEntry, ...]]] = []

    def capture(_canonical, prior, pairs):
        assert prior is not None
        seen.append((prior, tuple(pairs)))
        aliases = SourceAliasTable.from_source_ids(tuple(pair.entry_id for pair in pairs))
        return {"role": "user", "content": "writer"}, [], aliases

    monkeypatch.setattr(module, "core_synthesis_message", capture)

    assert module._dc_writer_reserve("game24", tasks) > 0
    assert seen
    assert all(prior.content.startswith("<cheatsheet>") for prior, _pairs in seen)
    assert all(
        module.count_text_tokens(prior.content, module.TOKEN_ENCODING)
        == module.COMMON_VISIBLE_MEMORY_TOKENS
        for prior, _pairs in seen
    )
    assert all(len({pair.content for pair in pairs}) == 3 for _prior, pairs in seen)
