from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from memcontam.baselines.dynamic_cheatsheet_phase12 import (
    DcRsContractError,
    SourceAliasTable,
)


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


def test_approved_token_contract_records_completed_alias_repair_and_narrow_blockers() -> None:
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))

    assert status["status"] == "NOT_READY"
    assert status["reason"] == "REGISTERED_WRITER_IO_BOUNDS_UNFROZEN"
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
        "service_tier_provider_literal": None,
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
        "client_implementation_sha256": (
            "16710d2e27765db224ad9a678076375ac1bba8c684e79edc55a5a52ac0fcfb9f"
        ),
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
        "dc_rs_runtime_builder_sha256": hashlib.sha256(
            Path("src/memcontam/experiment/phase13_dc_rs_runtime.py").read_bytes()
        ).hexdigest(),
        "dc_rs_prompt_builder_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
    }
    assert status["completion"] == {
        "capacity_record_materialized": False,
        "runtime_bound": False,
        "remaining_prerequisites": [
            "registered_raw_answer_token_ceiling",
            "strict_curator_response_grammar",
            "official_provider_contract_source_identity",
            "service_tier_provider_literal",
        ],
        "next_required_action": (
            "freeze the local registered-token answer ceiling, strict curator grammar, "
            "official provider-contract source identity, and literal OpenAI service tier"
        ),
    }
