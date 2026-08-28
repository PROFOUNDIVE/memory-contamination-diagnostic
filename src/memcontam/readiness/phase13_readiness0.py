from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.readiness.phase13_authority_files import read_regular_nofollow


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Phase13Readiness0Error(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class Authorization(_FrozenModel):
    scope: Literal["MINIMUM_PRODUCTION_FACING_READINESS0_LIVE_PILOT"]
    allow_live_calls: Literal[True]
    authorizes_provider_backed_scientific_calibration: Literal[False]
    authorizes_mr_p5: Literal[False]
    authorizes_mr_p6: Literal[False]
    authorizes_main_a: Literal[False]
    answer_correctness_acceptance_criterion: Literal[False]
    authorization_hash: Sha256


class RuntimeContract(_FrozenModel):
    api: Literal["OpenAI Responses API"]
    model: Literal["gpt-5.6-luna"]
    service_tier: Literal["default"]
    reasoning_mode: Literal["standard"]
    reasoning_effort: Literal["none"]
    reasoning_context: Literal["current_turn"]
    previous_response_id: None
    store: Literal[False]
    tools: tuple[()]
    retries_after_initial_attempt: Literal[0]


class Readiness0Request(_FrozenModel):
    schema_version: Literal["phase13_readiness0_request_v1"]
    status: Literal["BLOCKED_EXTERNAL_DEPENDENCY"]
    scientific_result: Literal[False]
    main_result: Literal[False]
    measured_main_a_trajectory_count: Literal[0]
    authorization: Authorization
    runtime_contract: RuntimeContract
    case_matrix: tuple[Literal["luna_responses_terminal_success"], ...]
    f1c_status: Literal["BLOCKED_ENVIRONMENT_NOT_CONFIGURED"]
    credentials_source: Literal["CURRENT_PROCESS_ENVIRONMENT_ONLY"]
    sibling_dotenv_read: Literal[False]
    external_blockers: tuple[
        Literal["OPENAI_API_KEY_MISSING", "F1C_RUNTIME_ENVIRONMENT_NOT_CONFIGURED"], ...
    ]
    provider_calls_issued: Literal[0]
    request_hash: Sha256


def validate_readiness0_request(path: Path) -> Readiness0Request:
    try:
        request = Readiness0Request.model_validate_json(read_regular_nofollow(path))
    except (OSError, ValidationError) as error:
        raise Phase13Readiness0Error("READINESS0_REQUEST_INVALID") from error
    authorization = request.authorization.model_dump(mode="json", exclude={"authorization_hash"})
    if _canonical_hash(authorization) != request.authorization.authorization_hash:
        raise Phase13Readiness0Error("READINESS0_AUTHORIZATION_HASH_MISMATCH")
    payload = request.model_dump(mode="json", exclude={"request_hash"})
    if _canonical_hash(payload) != request.request_hash:
        raise Phase13Readiness0Error("READINESS0_REQUEST_HASH_MISMATCH")
    if request.case_matrix != ("luna_responses_terminal_success",) or set(
        request.external_blockers
    ) != {"OPENAI_API_KEY_MISSING", "F1C_RUNTIME_ENVIRONMENT_NOT_CONFIGURED"}:
        raise Phase13Readiness0Error("READINESS0_REQUEST_INVALID")
    return request


def _canonical_hash(payload: dict[str, JsonValue]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = ["Phase13Readiness0Error", "Readiness0Request", "validate_readiness0_request"]
