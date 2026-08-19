from __future__ import annotations

import json
from pathlib import Path


STATUS_PATH = Path("data/phase13/common_capacity_status_v1.json")


def test_approved_token_contract_fails_closed_on_unbounded_writer_framing() -> None:
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))

    assert status["status"] == "NOT_READY"
    assert status["reason"] == "DC_RS_MODEL_VISIBLE_SOURCE_ID_FOOTPRINT_UNBOUNDED"
    expected_runtime = {
        "provider": "OpenAI",
        "endpoint": "Responses API",
        "requested_model_id": "gpt-5.6-luna",
        "dated_provider_snapshot": "not_exposed",
        "context_window_tokens": 1_050_000,
        "provider_max_output_tokens": 128_000,
        "answer_max_output_tokens": 4096,
        "writer_max_output_tokens": 8192,
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
    assert status["completion"]["remaining_prerequisites"] == [
        "bounded_dc_rs_model_visible_source_id_contract",
        "official_provider_contract_source_identity",
        "service_tier",
    ]
    assert status["completion"]["capacity_record_materialized"] is False
    assert status["completion"]["runtime_bound"] is False
