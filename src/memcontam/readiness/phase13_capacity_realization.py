from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from memcontam.readiness.phase13_capacity_measurement import (
    CapacityReserves,
    derive_capacity_reserves,
    measurement_implementation_sha256,
)
from memcontam.readiness.phase13_route_capacity import (
    CommonCapacityAudit,
    materialize_common_capacity,
)


class CapacityRealizationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProviderContract(_FrozenModel):
    path: str
    sha256: str
    context_tokens: int
    provider_max_output_tokens: int
    provider_service_tier: Literal["default"]


class TokenContract(_FrozenModel):
    contract_id: Literal["phase13_registered_token_accounting_v1"]
    encoding: Literal["o200k_base"]
    tiktoken_version: str
    requirements_lock_sha256: str
    counting_implementation_sha256: str
    message_framing_law: str
    special_token_handling: Literal["disallowed_special=()"]
    provider_token_equivalence_claimed: Literal[False]


class WriterIoContract(_FrozenModel):
    registered_persisted_raw_answer_ceiling: Literal[8192]
    overflow_policy: Literal["fail_closed_without_persistent_archive_admission"]
    writer_response_grammar: Literal["exactly_one_complete_whole_cheatsheet_serialization_v1"]
    O_writer_reg: Literal[8192]
    F_DC_out_tokens: Literal[0]


class MeasurementContract(_FrozenModel):
    task_scope: str
    R_FH: str
    I_DC_writer: str
    F_DC_out: str
    outcome_selection_prohibited: Literal[True]


class SourceArtifacts(_FrozenModel):
    legacy_main_registry_manifest_sha256: str
    new_mcq_core_manifest_sha256: str


class BuilderHashes(_FrozenModel):
    measurement_implementation_sha256: str
    openai_client_sha256: str
    full_history_builder_sha256: str
    dc_rs_builder_sha256: str
    dc_rs_runtime_builder_sha256: str
    ordinary_runtime_builder_sha256: str


class CommonCapacityMaterialization(_FrozenModel):
    schema_version: Literal[
        "phase13_common_capacity_v1",
        "phase13_common_capacity_v2",
    ]
    status: Literal["MATERIALIZED"]
    materialized_at_utc: str
    capacity_law_id: Literal["luna_common_visible_memory_capacity_v1"]
    capacity_unit: Literal["registered_serialized_tokens"]
    model_runtime_identity: str
    provider_contract: ProviderContract
    token_contract: TokenContract
    registered_writer_io_contract: WriterIoContract
    measurement_contract: MeasurementContract
    source_artifacts: SourceArtifacts
    production_builder_hashes: BuilderHashes
    per_task_R_FH: dict[str, int]
    per_task_I_DC_writer: dict[str, int]
    per_task_F_DC_out: dict[str, int]
    B_FH_feasible: int
    B_DC_feasible: int
    B_mem_tokens: int
    L_DC_tokens: int
    capacity_law_hash: str


def parse_common_capacity(raw_json: bytes | str) -> CommonCapacityMaterialization:
    try:
        record = CommonCapacityMaterialization.model_validate_json(raw_json)
    except ValidationError as error:
        raise CapacityRealizationError("CAPACITY_ARTIFACT_MALFORMED") from error
    if (
        record.provider_contract.context_tokens != 1_050_000
        or record.provider_contract.provider_max_output_tokens != 128_000
        or record.registered_writer_io_contract.F_DC_out_tokens != 0
        or set(record.per_task_F_DC_out.values()) != {0}
    ):
        raise CapacityRealizationError("CAPACITY_CONTRACT_MISMATCH")
    expected = materialize_common_capacity(
        CommonCapacityAudit(
            model_runtime_identity=record.model_runtime_identity,
            context_contract_id="openai-luna-context-1050000-v1",
            context_tokens=record.provider_contract.context_tokens,
            provider_output_contract_id="openai-luna-output-128000-v1",
            provider_max_output_tokens=record.provider_contract.provider_max_output_tokens,
            tokenizer_encoding_identity=record.token_contract.encoding,
            tokenizer_revision_version=f"tiktoken-{record.token_contract.tiktoken_version}",
            serialization_identity=record.token_contract.contract_id,
            special_token_handling_identity=record.token_contract.special_token_handling,
            message_framing_law_identity="role_lf_content_blank_line_v1",
            token_count_implementation_hash_version=(
                record.token_contract.counting_implementation_sha256
            ),
            per_task_R_FH=record.per_task_R_FH,
            per_task_I_DC_writer=record.per_task_I_DC_writer,
            per_task_F_DC_out=record.per_task_F_DC_out,
            fh_bounded_core_contract_id="fh-bounded-core-v1",
            retention_truncation_rule_id="oldest_first_pair_atomic",
            context_resource_contract_id="luna-common-visible-memory-v1",
        )
    )
    if (
        record.B_FH_feasible != expected.B_FH_feasible
        or record.B_DC_feasible != expected.B_DC_feasible
        or record.B_mem_tokens != expected.B_mem_tokens
        or record.L_DC_tokens != expected.L_DC_tokens
        or record.capacity_law_hash != expected.capacity_law_hash
    ):
        raise CapacityRealizationError("CAPACITY_FORMULA_MISMATCH")
    return record


@lru_cache(maxsize=1)
def validated_common_capacity_tokens() -> int:
    repository_root = Path(__file__).resolve().parents[3]
    return validate_common_capacity_artifact(
        repository_root / "data/phase13/common_capacity_corrected_v2.json",
        repository_root,
    ).B_mem_tokens


def validate_common_capacity_artifact(
    artifact_path: Path,
    repository_root: Path,
) -> CommonCapacityMaterialization:
    record = parse_common_capacity(artifact_path.read_bytes())
    checks = (
        (record.provider_contract.path, record.provider_contract.sha256),
        ("requirements.lock", record.token_contract.requirements_lock_sha256),
        (
            "src/memcontam/baselines/prompt_budget.py",
            record.token_contract.counting_implementation_sha256,
        ),
        (
            "data/phase13/main/main_registry_manifest_v1.json",
            record.source_artifacts.legacy_main_registry_manifest_sha256,
        ),
        (
            "data/phase13/core/materialized/manifest.json",
            record.source_artifacts.new_mcq_core_manifest_sha256,
        ),
        (
            "src/memcontam/clients/openai_responses.py",
            record.production_builder_hashes.openai_client_sha256,
        ),
        (
            "src/memcontam/baselines/full_history_adapter.py",
            record.production_builder_hashes.full_history_builder_sha256,
        ),
        (
            "src/memcontam/baselines/dynamic_cheatsheet_phase12.py",
            record.production_builder_hashes.dc_rs_builder_sha256,
        ),
        (
            "src/memcontam/experiment/phase13_dc_rs_runtime.py",
            record.production_builder_hashes.dc_rs_runtime_builder_sha256,
        ),
        (
            "src/memcontam/experiment/phase13_ordinary_runtime.py",
            record.production_builder_hashes.ordinary_runtime_builder_sha256,
        ),
    )
    if any(
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or hashlib.sha256((repository_root / relative).read_bytes()).hexdigest() != expected
        for relative, expected in checks
    ):
        raise CapacityRealizationError("CAPACITY_ARTIFACT_BINDING_MISMATCH")
    if record.production_builder_hashes.measurement_implementation_sha256 != (
        measurement_implementation_sha256(repository_root)
    ):
        raise CapacityRealizationError("CAPACITY_ARTIFACT_BINDING_MISMATCH")
    try:
        reserves = derive_capacity_reserves(repository_root)
    except (KeyError, OSError, ValueError) as error:
        raise CapacityRealizationError("CAPACITY_SOURCE_REGISTRY_MISMATCH") from error
    if (
        record.per_task_R_FH != reserves.per_task_R_FH
        or record.per_task_I_DC_writer != reserves.per_task_I_DC_writer
    ):
        raise CapacityRealizationError("CAPACITY_RESERVE_MISMATCH")
    return record


__all__ = [
    "CapacityRealizationError",
    "CapacityReserves",
    "CommonCapacityMaterialization",
    "derive_capacity_reserves",
    "measurement_implementation_sha256",
    "parse_common_capacity",
    "validate_common_capacity_artifact",
    "validated_common_capacity_tokens",
]
