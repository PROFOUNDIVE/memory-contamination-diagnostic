from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from memcontam.clients.base import LLMResponse
from memcontam.readiness.phase13_provider_accounting import (
    OwnedDispatchConfig,
    OwnedProviderAccounting,
    ProviderAccountingError,
    ProviderDispatchFailure,
)


ROOT = Path(__file__).resolve().parents[1]
OWNER = "phase13-h10-execution-owner-v1"
OFFLINE_OWNER = "phase13-offline-compute-owner-v1"
TEMPLATE = "game24-bot_style-contam"


def _attempt(
    attempt_id: str,
    number: int,
    status: str,
    *,
    semantic_call_id: str,
    owner_id: str = OWNER,
) -> dict[str, Any]:
    return {
        "attempt_id": attempt_id,
        "semantic_call_id": semantic_call_id,
        "execution_owner_id": owner_id,
        "attempt_number": number,
        "status": status,
        "input_tokens": 3,
        "output_tokens": 2 if status == "completed" else 0,
        "cost_microusd": 11,
        "latency_ms": 7,
        "storage_bytes": 13,
        "provider_error": "timeout" if status != "completed" else None,
        "raw_evidence": {"wire": attempt_id},
    }


def _totals(attempts: int) -> dict[str, int]:
    return {
        "transport_attempts": attempts,
        "retries": attempts - 1,
        "input_tokens": 3 * attempts,
        "output_tokens": 2,
        "cost_microusd": 11 * attempts,
        "latency_ms": 7 * attempts,
        "storage_bytes": 13 * attempts,
    }


class _TwoAttemptClient:
    def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
        del messages, model
        call_id = config["semantic_call_id"]
        attempts = (
            _attempt("attempt-1", 1, "failed", semantic_call_id=call_id),
            _attempt("attempt-2", 2, "completed", semantic_call_id=call_id),
        )
        return LLMResponse(
            content="final: 24",
            raw={"provider_attempts": attempts, "provider_totals": _totals(2)},
            token_usage={"prompt_tokens": 6, "completion_tokens": 2},
            latency_ms=14,
        )


def _config(owner_id: str = OWNER) -> OwnedDispatchConfig:
    return OwnedDispatchConfig(
        execution_owner_id=owner_id,
        execution_template_id=TEMPLATE,
        provider_config={"temperature": 0},
    )


def test_two_attempt_dispatch_reconciles_to_one_authenticated_execution_owner() -> None:
    accounting = OwnedProviderAccounting.from_authority(_TwoAttemptClient(), ROOT)

    response = accounting.chat([{"role": "user", "content": "solve"}], "model", _config())
    report = accounting.reconcile()

    assert response.content == "final: 24"
    assert len({row.semantic_call_id for row in report.calls}) == 1
    assert report.calls[0].semantic_call_id
    assert report.calls[0].execution_owner_id == OWNER
    assert report.totals.model_dump() == {
        "semantic_calls": 1,
        "dispatches": 1,
        **_totals(2),
    }
    assert tuple((row.operation, row.provider_calls, row.cost_microusd) for row in report.offline) == (
        ("prefix_derivation", 0, 0),
        ("paired_seed_bootstrap", 0, 0),
        ("report_rendering", 0, 0),
    )


class _FailedClient:
    def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
        del messages, model
        call_id = config["semantic_call_id"]
        attempt = _attempt("attempt-failed", 1, "partial", semantic_call_id=call_id)
        raise ProviderDispatchFailure(
            provider_error="connection reset",
            provider_attempts=(attempt,),
            provider_totals={**_totals(1), "output_tokens": 0},
        )


def test_provider_failure_after_reservation_retains_raw_evidence_and_settles() -> None:
    accounting = OwnedProviderAccounting.from_authority(_FailedClient(), ROOT)

    with pytest.raises(ProviderAccountingError) as caught:
        accounting.chat([{"role": "user", "content": "solve"}], "model", _config())

    assert caught.value.code == "PROVIDER_DISPATCH_FAILED"
    report = accounting.reconcile()
    assert report.calls[0].provider_error == "connection reset"
    assert report.calls[0].attempts[0].raw_evidence == {"wire": "attempt-failed"}
    assert report.totals.semantic_calls == 1
    assert report.totals.transport_attempts == 1


def test_offline_owner_is_rejected_before_provider_dispatch() -> None:
    client = _TwoAttemptClient()
    accounting = OwnedProviderAccounting.from_authority(client, ROOT)

    with pytest.raises(ProviderAccountingError) as caught:
        accounting.chat(
            [{"role": "user", "content": "solve"}],
            "model",
            _config(OFFLINE_OWNER),
        )

    assert caught.value.code == "OFFLINE_OWNER_FORBIDDEN"


def test_duplicate_generated_semantic_call_id_is_rejected_before_second_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accounting = OwnedProviderAccounting.from_authority(_TwoAttemptClient(), ROOT)
    monkeypatch.setattr(
        "memcontam.readiness.phase13_provider_accounting.uuid4",
        lambda: "frozen-call-id",
    )
    accounting.chat([{"role": "user", "content": "first"}], "model", _config())

    with pytest.raises(ProviderAccountingError) as caught:
        accounting.chat([{"role": "user", "content": "second"}], "model", _config())

    assert caught.value.code == "DUPLICATE_SEMANTIC_CALL_ID"


Mutation = Callable[[dict[str, Any], str], None]


def _missing_owner(payload: dict[str, Any], call_id: str) -> None:
    del call_id
    del payload["provider_attempts"][0]["execution_owner_id"]


def _unknown_owner(payload: dict[str, Any], call_id: str) -> None:
    del call_id
    payload["provider_attempts"][0]["execution_owner_id"] = "unknown-owner"


def _offline_owner(payload: dict[str, Any], call_id: str) -> None:
    del call_id
    payload["provider_attempts"][0]["execution_owner_id"] = OFFLINE_OWNER


def _duplicate_attempt(payload: dict[str, Any], call_id: str) -> None:
    del call_id
    payload["provider_attempts"].append(dict(payload["provider_attempts"][0]))
    payload["provider_totals"] = _totals(2)


def _duplicate_call_id(payload: dict[str, Any], call_id: str) -> None:
    payload["provider_attempts"][0]["semantic_call_id"] = f"{call_id}-other"


def _omitted_retry(payload: dict[str, Any], call_id: str) -> None:
    payload["provider_attempts"].append(
        _attempt("attempt-3", 3, "completed", semantic_call_id=call_id)
    )
    payload["provider_totals"] = _totals(3)


def _owner_total_mismatch(payload: dict[str, Any], call_id: str) -> None:
    del call_id
    payload["provider_totals"]["input_tokens"] += 1


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (_missing_owner, "PROVIDER_ACCOUNTING_REQUIRED"),
        (_unknown_owner, "UNKNOWN_EXECUTION_OWNER"),
        (_offline_owner, "OFFLINE_OWNER_FORBIDDEN"),
        (_duplicate_attempt, "DUPLICATE_ATTEMPT_ID"),
        (_duplicate_call_id, "SEMANTIC_CALL_ID_MISMATCH"),
        (_omitted_retry, "TRANSPORT_RETRY_SEQUENCE_INVALID"),
        (_owner_total_mismatch, "OWNER_TOTAL_MISMATCH"),
    ],
)
def test_adversarial_provider_accounting_is_rejected(
    mutate: Mutation,
    code: str,
) -> None:
    class _MutatedClient:
        def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
            del messages, model
            call_id = config["semantic_call_id"]
            payload = {
                "provider_attempts": [
                    _attempt("attempt-1", 1, "completed", semantic_call_id=call_id)
                ],
                "provider_totals": _totals(1),
            }
            mutate(payload, call_id)
            return LLMResponse("final: 24", payload, {"prompt_tokens": 3, "completion_tokens": 2})

    accounting = OwnedProviderAccounting.from_authority(_MutatedClient(), ROOT)

    with pytest.raises(ProviderAccountingError) as caught:
        accounting.chat([{"role": "user", "content": "solve"}], "model", _config())

    assert caught.value.code == code
