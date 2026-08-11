from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pytest

from memcontam.experiment.phase12.filter_challenge import rootless_local_binding
from memcontam.experiment.phase12.filter_challenge import rootless_local_bootstrap_cli
from memcontam.experiment.phase12.filter_challenge.rootless_local_contract import (
    JsonValue,
    RootlessContractError,
    parse_canonical_object,
)
from memcontam.experiment.phase12.filter_challenge.rootless_local_operator import (
    finalize_preclaim,
)


ROOT = Path(__file__).resolve().parents[1]
PAID_EGRESS_ACK = "I_ACCEPT_LOCAL_ROOTLESS_NON_AUTHORITATIVE_UP_TO_USD_10"


def _arguments(tmp_path: Path, attempt_id: str) -> argparse.Namespace:
    state_home = tmp_path / "state-home"
    state_home.mkdir(mode=0o700)
    return argparse.Namespace(
        state_home=state_home,
        repo_root=ROOT,
        rootless_command="preflight",
        attempt_id=attempt_id,
        plan_source=None,
        plan_descriptor=None,
        review_metadata=None,
        historical_screening_plan=None,
        historical_screening_descriptor=None,
        historical_post_descriptor=None,
        tokenizer_source=None,
        operator_1_label="operator-1",
        operator_2_label="operator-2",
        provider_account_label="provider-1",
        rpm_limit="6",
        tpm_limit="30000",
        paid_egress_ack=PAID_EGRESS_ACK,
    )


def _isolate_local_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = b"synthetic-tokenizer"
    monkeypatch.setattr(
        rootless_local_bootstrap_cli,
        "_preflight_file",
        lambda _arguments, _field: (Path("/synthetic/input"), raw),
    )
    monkeypatch.setattr(rootless_local_bootstrap_cli, "_validate_reviewed_plan", lambda *_args: None)
    monkeypatch.setattr(
        rootless_local_bootstrap_cli,
        "_TOKENIZER_SOURCE_SHA256",
        sha256(raw).hexdigest(),
    )
    monkeypatch.setattr(rootless_local_bootstrap_cli, "validate_rootless_configs", lambda _repo: {})


def test_preflight_observes_bound_authority_on_synthetic_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: passing local gates and synthetic observations for every configured authority.
    arguments = _arguments(tmp_path, "preflight-authority-pass")
    _isolate_local_preflight(monkeypatch)
    observed_roles: list[str] = []

    def observe(source: dict[str, JsonValue]) -> dict[str, JsonValue]:
        role = source["role"]
        assert isinstance(role, str)
        observed_roles.append(role)
        return {"role": role}

    monkeypatch.setattr(rootless_local_binding, "observe_external_authority", observe)

    # When: preflight admits the bound decoding authority.
    rootless_local_bootstrap_cli._preflight(arguments)

    # Then: the production collection predicate observes every authority in binding order.
    authority = parse_canonical_object(
        (ROOT / "configs/phase12/filter_v5_rootless_local/decoding_authority.json").read_bytes()
    )
    sources = authority["ordered_sources"]
    assert isinstance(sources, list)
    assert observed_roles == [source["role"] for source in sources if isinstance(source, dict)]


@pytest.mark.parametrize(
    "acknowledgement",
    (
        "",
        "acknowledged",
        "yes",
        f" {PAID_EGRESS_ACK}",
        f"{PAID_EGRESS_ACK} ",
    ),
)
def test_preflight_rejects_every_nonexact_paid_egress_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    acknowledgement: str,
) -> None:
    # Given: every non-payment gate passes but the operator token is not exact.
    arguments = _arguments(tmp_path, "preflight-paid-egress-token")
    arguments.paid_egress_ack = acknowledgement
    _isolate_local_preflight(monkeypatch)

    # When/Then: preflight blocks before external authority observation or dispatch.
    monkeypatch.setattr(rootless_local_bootstrap_cli, "observe_external_authorities", pytest.fail)
    with pytest.raises(RootlessContractError, match="ROOTLESS_PAID_EGRESS_NOT_ENABLED"):
        rootless_local_bootstrap_cli._preflight(arguments)


def test_preflight_maps_synthetic_hash_drift_to_external_input_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: a bound synthetic collection whose Phase 13 experiment source no longer matches its hash.
    arguments = _arguments(tmp_path, "preflight-authority-drift")
    _isolate_local_preflight(monkeypatch)

    def observe(source: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if source["role"] == "phase13-experiment-design":
            raise RootlessContractError("ROOTLESS_EXTERNAL_AUTHORITY_HASH_MISMATCH")
        return {"role": source["role"]}

    monkeypatch.setattr(rootless_local_binding, "observe_external_authority", observe)

    # When: preflight reaches the mismatched bound authority.
    with pytest.raises(SystemExit) as raised:
        rootless_local_bootstrap_cli.run(arguments)

    # Then: the typed zero-call stop preserves the exact external role and diagnostic.
    status = json.loads(capsys.readouterr().out)
    assert raised.value.code == 65
    assert status["reason_code"] == "ROOTLESS_MISSING_EXTERNAL_INPUT"
    assert status["missing_input_role"] == "ROOTLESS_THEORETICAL_PHASE13_EXPERIMENT_DESIGN"
    assert status["external_authority_diagnostic"] == "ROOTLESS_EXTERNAL_AUTHORITY_HASH_MISMATCH"


def test_preclaim_records_external_authority_hash_drift(tmp_path: Path) -> None:
    # Given: a preflight hash drift with its designated external input role.
    repository = tmp_path / "repository"
    repository.mkdir(mode=0o700)

    # When: the existing preclaim receipt handler finalizes the zero-call stop.
    finalize_preclaim(
        repository,
        tmp_path / "state-home",
        "preclaim-authority-drift",
        "a" * 40,
        "preflight",
        65,
        "2026-08-10T12:00:00Z",
        missing_input_role="ROOTLESS_THEORETICAL_AUTHORITY_AGENTS",
        external_authority_diagnostic="ROOTLESS_EXTERNAL_AUTHORITY_HASH_MISMATCH",
    )

    # Then: the receipt keeps the exact external role and diagnostic, not an admin failure.
    receipt = json.loads(
        (
            repository
            / "runs/phase12-filter-v5-rootless-qa/pre-egress/zero-call-skip.json"
        ).read_bytes()
    )
    assert receipt["reason"] == "ROOTLESS_MISSING_EXTERNAL_INPUT"
    assert receipt["missing_input_role"] == "ROOTLESS_THEORETICAL_AUTHORITY_AGENTS"
    assert receipt["external_authority_diagnostic"] == "ROOTLESS_EXTERNAL_AUTHORITY_HASH_MISMATCH"
    assert receipt["failed_command"] is None
    assert receipt["observed_exit"] is None
