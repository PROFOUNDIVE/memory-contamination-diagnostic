from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from memcontam.experiment.phase12.filter_challenge.bct_archive_storage import _hash
from memcontam.experiment.phase12.filter_challenge.evidence_contract import (
    approval_descriptor_path,
    approved_plan_sha256,
    read_regular_nofollow,
    sha256_regular_nofollow,
)
from memcontam.experiment.phase12.filter_challenge.registry_calibration import (
    BCTAuthorizationV1,
    CalibrationAuthorization,
    LEDGER_ID,
)


class CalibrationAuthorizationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


_Authorization = TypeVar("_Authorization", bound=CalibrationAuthorization)


def load_authorization(path: Path, expected_digest: str, model: type[_Authorization]) -> _Authorization:
    raw = read_regular_nofollow(path, "AUTHORIZATION_INVALID")
    if not hmac.compare_digest(expected_digest, hashlib.sha256(raw).hexdigest()):
        raise CalibrationAuthorizationError("AUTHORIZATION_DIGEST_MISMATCH")
    try:
        return model.model_validate_json(raw)
    except (ValidationError, UnicodeDecodeError) as error:
        raise CalibrationAuthorizationError("AUTHORIZATION_INVALID") from error


def runtime_decoding_sha256(config: Path, freeze: Path) -> str:
    try:
        safe_load = getattr(importlib.import_module("yaml"), "safe_load")
        runtime = safe_load(read_regular_nofollow(config, "AUTHORIZATION_TRUSTED_INPUT_INVALID"))["runtime"]
        freeze_payload = json.loads(read_regular_nofollow(freeze, "AUTHORIZATION_TRUSTED_INPUT_INVALID"))
        provider = runtime["provider"]
        model_id = runtime["model_id"]
        decoding = runtime["decoding"]
    except (AttributeError, ImportError, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise CalibrationAuthorizationError("AUTHORIZATION_TRUSTED_INPUT_INVALID") from error
    if not isinstance(provider, str) or not isinstance(model_id, str) or not isinstance(decoding, dict):
        raise CalibrationAuthorizationError("AUTHORIZATION_TRUSTED_INPUT_INVALID")
    if freeze_payload.get("provider") != provider or freeze_payload.get("model_id") != model_id:
        raise CalibrationAuthorizationError("AUTHORIZATION_TRUSTED_INPUT_MISMATCH")
    return _hash({"provider": provider, "model_id": model_id, "decoding": decoding})


def validate_runtime_authorization(
    authorization: CalibrationAuthorization,
    run_id: str,
    request_path: Path,
    config: Path,
    freeze: Path,
    artifact_root: Path,
    repository_root: Path,
    stage: str,
) -> None:
    if authorization.run_id != run_id or authorization.expires_at <= datetime.now(UTC):
        raise CalibrationAuthorizationError("AUTHORIZATION_RUNTIME_MISMATCH")
    try:
        request = json.loads(read_regular_nofollow(request_path, "AUTHORIZATION_TRUSTED_INPUT_INVALID"))
        head = subprocess.run(["git", "-C", str(repository_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        plan = repository_root / ".omo/plans/phase12-post-filter-v5-calibration-readiness.md"
        expected = {
            "request_sha256": sha256_regular_nofollow(request_path, "AUTHORIZATION_REQUEST_MISMATCH"), "implementation_commit": head,
            "approved_plan_sha256": approved_plan_sha256(plan, approval_descriptor_path(plan)),
            "authority_manifest_sha256": sha256_regular_nofollow(repository_root / "docs/evidence/phase12-filter-v5-bct-v1/authority_transition_manifest.json", "AUTHORIZATION_AUTHORITY_MISMATCH"),
            "freeze_sha256": sha256_regular_nofollow(freeze, "AUTHORIZATION_TRUSTED_INPUT_INVALID"),
            "decoding_sha256": runtime_decoding_sha256(config, freeze),
            "provider": "openai_responses", "model_id": "gpt-4o-2024-11-20", "ledger_id": LEDGER_ID, "artifact_root": str(artifact_root),
        }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, subprocess.SubprocessError, ValueError) as error:
        raise CalibrationAuthorizationError("AUTHORIZATION_TRUSTED_INPUT_INVALID") from error
    expected.update({name: request.get(name) for name in ("maximum_calls", "maximum_input_tokens", "maximum_output_tokens")})
    expected["maximum_wall_seconds"] = request.get("wall_seconds")
    if request.get("stage") != stage or any(getattr(authorization, name) != value or request.get(name) != value for name, value in expected.items()):
        raise CalibrationAuthorizationError("AUTHORIZATION_TRUSTED_INPUT_MISMATCH")
    hard_ceiling_usd = request.get("hard_ceiling_usd")
    if isinstance(hard_ceiling_usd, bool) or not isinstance(hard_ceiling_usd, int):
        raise CalibrationAuthorizationError("AUTHORIZATION_TRUSTED_INPUT_MISMATCH")
    if authorization.hard_ceiling_microusd != hard_ceiling_usd * 1_000_000:
        raise CalibrationAuthorizationError("AUTHORIZATION_TRUSTED_INPUT_MISMATCH")
    if isinstance(authorization, BCTAuthorizationV1):
        try:
            report = json.loads(read_regular_nofollow(repository_root / "docs/evidence/phase12-filter-v5-bct-v1/screening_report.json", "AUTHORIZATION_TRUSTED_INPUT_INVALID"))
            ledger = artifact_root / "budget-ledger.jsonl"
            ledger_head = "0" * 64 if not ledger.exists() else _ledger_head(read_regular_nofollow(ledger, "AUTHORIZATION_TRUSTED_INPUT_INVALID"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise CalibrationAuthorizationError("AUTHORIZATION_TRUSTED_INPUT_INVALID") from error
        if authorization.screening_terminal_seal != report.get("output_seal") or authorization.ledger_head != ledger_head or request.get("screening_terminal_seal") != report.get("output_seal") or request.get("ledger_head") != ledger_head:
            raise CalibrationAuthorizationError("AUTHORIZATION_TRUSTED_INPUT_MISMATCH")


def _ledger_head(raw: bytes) -> str:
    head = "0" * 64
    for sequence, line in enumerate(raw.splitlines(), start=1):
        record = json.loads(line)
        if record.get("sequence") != sequence or record.get("previous_hash") != head or record.get("record_hash") != _hash({key: value for key, value in record.items() if key != "record_hash"}):
            raise CalibrationAuthorizationError("AUTHORIZATION_TRUSTED_INPUT_INVALID")
        head = str(record["record_hash"])
    return head
