from memcontam.experiment.phase12.filter_challenge.rootless_local_broker import (
    BrokerRequest,
    _verify_response,
)


def test_historical_meb_probe_uses_rhs_completion_verifier() -> None:
    request = BrokerRequest(
        slot_id="slot-001",
        idempotency_key="i-00000000000000000000000000000000",
        compiler_sha256="3" * 64,
        static_input_sha256="6" * 64,
        predecessor_receipt_sha256=None,
        request=b"{}",
        compiled_input_tokens=1,
        side="control",
        created_at="2026-08-09T12:00:00Z",
        task="math_equation_balancer",
        baseline="full_history",
        probe_id="fv5-cal-meb-001",
        native_stage="answer",
        candidate_class=None,
    )
    specs = {
        "fv5-cal-meb-001": (
            "math_equation_balancer",
            {"target": "2 + 5 = 7", "target_value": 7},
        )
    }

    assert _verify_response(request, "7", specs) is True
