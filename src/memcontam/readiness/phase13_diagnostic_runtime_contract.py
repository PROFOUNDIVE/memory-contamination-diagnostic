"""Non-authorizing Diagnostic contract preparation; no client or ledger construction."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Never

from pydantic import Field, TypeAdapter

from . import phase13_v3_request
from .phase13_authority_files import read_regular_nofollow
from .phase13_cost_policy_models import Sha256
from .phase13_live_diagnostic_models import (
    DiagnosticModel,
    DiagnosticObservation,
    LiveDiagnosticBinding,
    LiveDiagnosticError,
)
from .phase13_shared_budget_authority import BudgetAuthority
from .phase13_shared_budget_models import BudgetOperation, FilePin
from .phase13_v3_request import (
    RequestKeyV3,
    RequestMaterialV3,
    Stage,
    compile_request_bytes,
    input_bytes,
)


class DiagnosticAuthorityProjection(DiagnosticModel):
    execution_envelope_id: Literal["CORE_EXECUTION_ENVELOPE_REGISTRY_V4"]
    execution_envelope_sha256: Sha256
    transport_contract_id: Literal["CORE_TRANSPORT_ATTEMPT_CONTRACT_V3"]
    transport_contract_sha256: Sha256
    terminal_failure_contract_id: Literal["CORE_TERMINAL_TECHNICAL_MISSINGNESS_V2"]
    terminal_failure_contract_sha256: Sha256
    evidence: FilePin


class DiagnosticPrefixHistory(DiagnosticModel):
    run_id: str = Field(min_length=1)
    source_run_id: str = Field(min_length=1)
    checkpoint_index: int = Field(ge=1)
    state: FilePin


class DiagnosticRuntimeContract(DiagnosticModel):
    schema_version: Literal["phase13_diagnostic_runtime_contract_v1"]
    provenance: Literal["LIVE_DIAGNOSTIC"]
    main_scope: Literal["NOT_MAIN"]
    qualification: Literal["QA_ONLY_NOT_PROVIDER_QUALIFIED", "LIVE_REQUIRES_ADMISSION"]
    model: Literal["gpt-5.6-luna"]
    service_tier: Literal["default"]
    endpoint: Literal["https://api.openai.com/v1"]
    timeout_seconds: Literal[180]
    attempt_limit: Literal[1]
    retries: Literal[0]
    budget_binding: FilePin
    resource_files: tuple[FilePin, ...] = Field(min_length=1)
    authority: DiagnosticAuthorityProjection
    observations: tuple[DiagnosticObservation, ...] = Field(min_length=1)
    rate_card: FilePin | None = None
    embedding_metadata_sha256: Sha256 | None = None
    common_capacity_tokens: Literal[8192] = 8192
    prefix_histories: tuple[DiagnosticPrefixHistory, ...] = ()


class DiagnosticRuntimeRegistration(DiagnosticModel):
    schema_version: Literal["phase13_diagnostic_runtime_registration_v1"]
    registration_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    authorization_id: str = Field(min_length=1)
    authorization_sha256: Sha256
    scientific_disposition: Literal["NOT_ADOPTED", "DIAGNOSTIC_ONLY"]
    runtime_contract: FilePin


@dataclass(frozen=True, slots=True)
class CompiledDiagnosticOperation:
    request_id: str
    native_call_id: str
    attempt_id: str
    material: RequestMaterialV3
    wire_bytes: bytes
    count_bytes: bytes

    @property
    def wire_sha256(self) -> str:
        return hashlib.sha256(self.wire_bytes).hexdigest()

    @property
    def count_sha256(self) -> str:
        return hashlib.sha256(self.count_bytes).hexdigest()

    @property
    def messages_sha256(self) -> str:
        return hashlib.sha256(input_bytes(self.material)).hexdigest()

    @property
    def native_state_sha256(self) -> str:
        return hashlib.sha256(self.material.native_state).hexdigest()


@dataclass(frozen=True, slots=True)
class DiagnosticExecutionContext:
    binding: LiveDiagnosticBinding
    registration: FilePin
    contract: DiagnosticRuntimeContract

    @classmethod
    def load(cls, binding: LiveDiagnosticBinding, registration_path: Path) -> DiagnosticExecutionContext:
        bound = LiveDiagnosticBinding.model_validate_json(binding.model_dump_json())
        pin = FilePin(path=str(registration_path), sha256=bound.registration_sha256)
        registration = DiagnosticRuntimeRegistration.model_validate_json(_read(pin))
        contract = DiagnosticRuntimeContract.model_validate_json(_read(registration.runtime_contract))
        context = cls(bound, pin, contract)
        context.verify()
        return context

    def verify(self) -> None:
        registration = DiagnosticRuntimeRegistration.model_validate_json(_read(self.registration))
        identity_fields = ("registration_id", "run_id", "source_revision", "authorization_id",
                           "authorization_sha256", "scientific_disposition")
        if (self.registration.sha256 != self.binding.registration_sha256
            or any(getattr(registration, name) != getattr(self.binding, name) for name in identity_fields)
            or self.contract != DiagnosticRuntimeContract.model_validate_json(_read(registration.runtime_contract))
            or self.contract.observations != self.binding.observations):
            raise LiveDiagnosticError("DIAGNOSTIC_RUNTIME_BINDING_MISMATCH")
        pins = {pin.path: pin.sha256 for pin in self.binding.source_files}
        required = (str(Path(__file__).resolve()), str(Path(phase13_v3_request.__file__).resolve()))
        if (any(path not in pins for path in required)
            or pins.get(registration.runtime_contract.path) != registration.runtime_contract.sha256):
            raise LiveDiagnosticError("DIAGNOSTIC_RUNTIME_SOURCE_MISMATCH")
        for source in self.binding.source_files:
            _read(FilePin(path=source.path, sha256=source.sha256))
        for resource in (*self.contract.resource_files, self.contract.authority.evidence,
                         self.contract.budget_binding,
                         *((self.contract.rate_card,) if self.contract.rate_card is not None else ()),
                         *(row.state for row in self.contract.prefix_histories)):
            _read(resource)
        if json.loads(_read(self.contract.authority.evidence)) != self.contract.authority.model_dump(
            mode="json", exclude={"evidence"}
        ):
            raise LiveDiagnosticError("DIAGNOSTIC_AUTHORITY_PROJECTION_MISMATCH")

    def require_live_admission(self) -> Never:
        self.verify()
        raise LiveDiagnosticError("DIAGNOSTIC_LIVE_ADMISSION_UNESTABLISHED")

    def compile(self, request_id: str, material: RequestMaterialV3) -> CompiledDiagnosticOperation:
        self.verify()
        registered = [(observation, request) for observation in self.contract.observations
                      for request in observation.requests if request.request_id == request_id]
        if len(registered) != 1:
            raise LiveDiagnosticError("DIAGNOSTIC_GENERATION_IDENTITY_REQUIRED")
        observation, request = registered[0]
        if request.operation != "generation" or request.native_call_id is None:
            raise LiveDiagnosticError("DIAGNOSTIC_GENERATION_IDENTITY_REQUIRED")
        if (material.temperature != 0.0 or material.top_p != 1.0
            or hashlib.sha256(material.native_state).hexdigest() != observation.state_sha256):
            raise LiveDiagnosticError("DIAGNOSTIC_REQUEST_MATERIAL_MISMATCH")
        stage = TypeAdapter(Stage).validate_python(request.stage, strict=True)
        if not stage.startswith(("bot_", "dc_rs_")):
            raise LiveDiagnosticError("DIAGNOSTIC_GENERATION_IDENTITY_REQUIRED")
        key = RequestKeyV3(parent_id=hashlib.sha256(observation.run_id.encode()).hexdigest(), stage=stage, ordinal=0)
        wire = compile_request_bytes(key, material)
        body = json.loads(wire)
        projection = json.dumps({name: body[name] for name in ("model", "input", "tools", "reasoning")},
                                sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return CompiledDiagnosticOperation(request_id, request.native_call_id,
            hashlib.sha256((self.binding.sha256() + request_id).encode()).hexdigest(), material, wire, projection)

    def price_operations(self, compiled: CompiledDiagnosticOperation, unit_id: str) -> tuple[BudgetOperation, BudgetOperation]:
        if compiled != self.compile(compiled.request_id, compiled.material):
            raise LiveDiagnosticError("DIAGNOSTIC_COMPILED_REQUEST_MISMATCH")
        observation = next(row for row in self.contract.observations
                           if any(request.request_id == compiled.request_id for request in row.requests))
        index = next(index for index, request in enumerate(observation.requests)
                     if request.request_id == compiled.request_id)
        generation = observation.requests[index]
        count = observation.requests[index - 1] if index else None
        if count is None or count.operation != "count" or count.stage != generation.stage:
            raise LiveDiagnosticError("DIAGNOSTIC_COUNT_SCHEDULE_MISMATCH")
        authority = BudgetAuthority.load(Path(self.contract.budget_binding.path))
        operations: list[BudgetOperation] = []
        for request, digest, suffix in ((count, compiled.count_sha256, "/responses/input_tokens"),
                                        (generation, compiled.wire_sha256, "/responses")):
            endpoint = self.contract.endpoint + suffix
            prices = [price for price in authority.pricing.entries if (
                price.request_sha256, price.operation, price.endpoint, price.model, price.service_tier
            ) == (digest, request.operation, endpoint, self.contract.model, self.contract.service_tier)]
            if len(prices) != 1:
                raise LiveDiagnosticError("DIAGNOSTIC_EXACT_PRICE_REQUIRED")
            price = prices[0]
            operation = BudgetOperation(lane_id="D", run_id=observation.run_id,
                execution_binding_sha256=self.binding.sha256(), unit_id=unit_id,
                request_id=request.request_id,
                attempt_id=hashlib.sha256((self.binding.sha256() + request.request_id).encode()).hexdigest(),
                request_sha256=digest, operation=request.operation, endpoint=endpoint,
                model=self.contract.model, service_tier=self.contract.service_tier,
                price_id=price.price_id, bound_usd_micros=price.upper_bound_usd_micros,
                unknown_count_decision=None)
            authority.check_operation(operation)
            operations.append(operation)
        return operations[0], operations[1]


def _read(pin: FilePin) -> bytes:
    path = Path(pin.path)
    raw = read_regular_nofollow(path)
    if path != path.resolve() or hashlib.sha256(raw).hexdigest() != pin.sha256:
        raise LiveDiagnosticError("DIAGNOSTIC_RUNTIME_PIN_MISMATCH")
    return raw
