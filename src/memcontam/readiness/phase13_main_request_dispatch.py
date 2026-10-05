from __future__ import annotations

# noqa: SIZE_OK — provider dispatch markers and failure finalization form one state machine.
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import replace
from tempfile import NamedTemporaryFile
from typing import Protocol, TypeVar

from pydantic import JsonValue

from memcontam.baselines.prompt_budget import count_prompt_tokens
from memcontam.clients.base import LLMClient, LLMResponse
from memcontam.readiness.phase13_authority_files import read_regular_nofollow
from memcontam.readiness.phase13_main_request_recovery import (
    RequestIdentityReceiptV3,
    recover_requests,
    request_lock,
    terminal_parents,
)
from memcontam.readiness.phase13_v3_cost_actual import reconcile_actual
from memcontam.readiness.phase13_v3_count import (
    CountIdentityV3, CountReceiptV3, count_failure_bytes, count_operation,
    count_recovery_gate, read_count_record, receipt_bytes, validate_count_receipt,
)
from memcontam.readiness.phase13_v3_cost_models import CostError, ProviderCostEvidence
from memcontam.readiness.phase13_v3_terminal_ledger import TerminalLedgerV3
from memcontam.readiness.phase13_v3_terminal_models import TerminalEvidenceError, attempt_identity

from .phase13_v3_cost_binding import LiveCosts, TableKey
from .phase13_v3_cost_models import digest
from .phase13_v3_request import (
    STAGES,
    CompiledProviderRequestV3,
    PackageBindingV3,
    ParentTrajectoryV3,
    RequestKeyV3,
    RequestMaterialV3,
    Stage,
    compile_request_bytes,
    input_bytes,
)

ResultT = TypeVar("ResultT")


class CompiledProvider(Protocol):
    def count_identity_v3(self) -> CountIdentityV3: ...

    def count_compiled_v3(self, compiled: CompiledProviderRequestV3,
                         before_count: Callable[[], None]) -> CountReceiptV3: ...

    def send_compiled_v3(self, compiled: CompiledProviderRequestV3,
                         before_request: Callable[[], None]) -> LLMResponse: ...


class DeferredMainClient:
    def __init__(self, factory: Callable[[], LLMClient]) -> None:
        self._factory = factory
        self._client: LLMClient | None = None

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        if self._client is None:
            self._client = self._factory()
        return self._client.chat(messages, model, config)


class DispatchTechnicalFailureV3(RuntimeError):
    def __init__(self, code: str, parent_id: str, realized_cost_krw: int | None,
                 *, evidence_sha256: str | None = None) -> None:
        self.code, self.parent_id, self.realized_cost_krw = code, parent_id, realized_cost_krw
        self.evidence_sha256 = evidence_sha256
        super().__init__(code)


def production_provider(_binding: PackageBindingV3) -> CompiledProvider:
    raise TerminalEvidenceError("MAIN_COUNT_PRICING_NOT_FROZEN")


class ProductionRequestDispatcherV3:
    def __init__(self, ledger: TerminalLedgerV3, binding: PackageBindingV3,
                  parents: tuple[ParentTrajectoryV3, ...], *,
                  provider_factory: Callable[[PackageBindingV3], CompiledProvider] = production_provider,
                  retry_entitlements: frozenset[str] = frozenset()) -> None:
        if (ledger.binding.identity, ledger.binding.package_sha256, ledger.binding.authorization_sha256) != (
            binding.identity, binding.package_sha256, binding.authorization_sha256,
        ):
            raise TerminalEvidenceError("MAIN_AUTHORIZATION_BINDING_MISMATCH")
        self.ledger, self.binding, self.parents = ledger, binding, parents
        self._factory = provider_factory
        if not retry_entitlements <= set(ledger.binding.unit_ids):
            raise TerminalEvidenceError("MAIN_RETRY_ENTITLEMENT_INVALID")
        self._retry_entitlements = retry_entitlements
        self._terminal_parents = terminal_parents(ledger, binding, parents)
        self._terminal_generation = ledger.generation()
        self._compiled: dict[str, CompiledProviderRequestV3] = {}
        self._attempt_counts: dict[str, int] = {}

    @property
    def terminal_parents(self) -> frozenset[str]:
        with request_lock(self.ledger):
            self._refresh_terminal_parents()
            return self._terminal_parents

    @property
    def retry_entitlements(self) -> frozenset[str]:
        return self._retry_entitlements

    def recover(self) -> None:
        with request_lock(self.ledger):
            recover_requests(self.ledger)
            self._terminal_parents = terminal_parents(self.ledger, self.binding, self.parents)
            self._terminal_generation = self.ledger.generation()

    def compiled_request(self, key: RequestKeyV3) -> CompiledProviderRequestV3:
        return self._compiled[key.dispatch_id]

    def dispatch(self, key: RequestKeyV3, compile_material: Callable[[], RequestMaterialV3],
                 parse_result: Callable[[LLMResponse], ResultT]) -> ResultT:
        with request_lock(self.ledger):
            self._refresh_terminal_parents()
            return self._dispatch(key, compile_material, parse_result)

    def _dispatch(self, key: RequestKeyV3, compile_material: Callable[[], RequestMaterialV3],
                   parse_result: Callable[[LLMResponse], ResultT], *, defer_completion: bool = False) -> ResultT:
        if key.parent_id in self._terminal_parents:
            raise DispatchTechnicalFailureV3("MAIN_TRAJECTORY_TERMINAL", key.parent_id, None)
        if key.parent_id not in {parent.parent_id for parent in self.parents}:
            raise TerminalEvidenceError()
        state = self.ledger.state(key.dispatch_id)
        count_recovery_gate(self.ledger)
        retrying = state.kind == "RETRYABLE_ATTEMPT_FAILURE"
        if state.kind not in {"PENDING", "RETRYABLE_ATTEMPT_FAILURE"}:
            raise TerminalEvidenceError("MAIN_RUN_IN_FLIGHT_RECONCILIATION_REQUIRED")
        receipt = RequestIdentityReceiptV3(binding=self.binding, parents=self.parents, key=key)
        self._publish_bytes(key, "identity", receipt.model_dump_json().encode() + b"\n")
        if not retrying:
            self._append(key, "DISPATCH_INTENT")
        material = compile_material()
        request_bytes = compile_request_bytes(key, material)
        count = count_prompt_tokens([message.model_dump() for message in material.messages], "o200k_base")
        compiled = CompiledProviderRequestV3(self.binding, key, material, request_bytes, count)
        self._compiled[key.dispatch_id] = compiled
        self._persist_compiled(compiled)
        if retrying:
            if state.compiled != compiled.evidence or key.dispatch_id not in self._retry_entitlements:
                raise TerminalEvidenceError("MAIN_RETRY_ENTITLEMENT_INVALID")
        else:
            self._append(key, "REQUEST_COMPILED")
        attempt_index = 1 if retrying else 0
        provider = self._factory(self.binding)
        count_method = getattr(provider, "count_compiled_v3", None)
        if count_method is None:
            raise TerminalEvidenceError("MAIN_COUNT_PROVIDER_UNAVAILABLE")
        operation = count_operation(compiled, provider.count_identity_v3())

        def start_count() -> None:
            self._publish_bytes(key, "count-started", receipt_bytes(operation))

        try:
            saved_count = read_count_record(self.ledger, key.dispatch_id, "count-receipt")
            if saved_count is None:
                counted = count_method(compiled, start_count)
                self._publish_bytes(key, "count-receipt", receipt_bytes(counted))
            else:
                counted = CountReceiptV3.model_validate_json(saved_count)
            validate_count_receipt(compiled, counted, provider.count_identity_v3())
        except Exception as error:
            self._publish_bytes(key, "count-failure", count_failure_bytes(
                str(getattr(error, "code", type(error).__name__))))
            if getattr(error, "code", None) == "MAIN_INPUT_ENVELOPE_EXCEEDED":
                for kind in ("INPUT_ENVELOPE_OVERFLOW", "TERMINAL_TECHNICAL_MISSING"):
                    self._append(key, kind, {"failure_code": "MAIN_INPUT_ENVELOPE_EXCEEDED",
                        "transport_attempts": 0, "realized_cost_krw": 0})
                raise DispatchTechnicalFailureV3("MAIN_INPUT_ENVELOPE_EXCEEDED", key.parent_id, None) from error
            raise TerminalEvidenceError("MAIN_COUNT_FAILED_RECONCILIATION_REQUIRED") from error
        def verify_durable_count(started: bool) -> None:
            state = self.ledger.state(key.dispatch_id)
            if state.kind not in ({"ATTEMPT_STARTED"} if started else {
                    "REQUEST_COMPILED", "RETRYABLE_ATTEMPT_FAILURE", "ATTEMPT_STARTED"}):
                raise TerminalEvidenceError("MAIN_COUNT_RECEIPT_REPLAY_REJECTED")
            if read_count_record(self.ledger, key.dispatch_id, "count-started") != receipt_bytes(operation):
                raise TerminalEvidenceError("MAIN_COUNT_RECEIPT_REQUIRED")
            if read_count_record(self.ledger, key.dispatch_id, "count-receipt") != receipt_bytes(counted):
                raise TerminalEvidenceError("MAIN_COUNT_RECEIPT_REQUIRED")
            validate_count_receipt(compiled, counted, provider.count_identity_v3())

        compiled = replace(compiled, count_receipt=counted, verify_durable_count=verify_durable_count)
        self._compiled[key.dispatch_id] = compiled
        while True:
            response: LLMResponse | None = None
            cost = ProviderCostEvidence()
            attempt_ready = False

            def start_attempt(index: int = attempt_index) -> None:
                nonlocal attempt_ready
                verify_durable_count(False)
                self._append(key, "ATTEMPT_STARTED", {"attempt_index": index,
                    "attempt_id": attempt_identity(key.dispatch_id, index)})
                attempt_ready = True

            try:
                response = provider.send_compiled_v3(compiled, start_attempt)
                _validate_received_response(response, key.stage)
                cost = _response_cost(response)
                realized = _realized(cost)
                result = parse_result(response)
                if result is None:
                    raise TerminalEvidenceError("MAIN_SEMANTIC_RESULT_UNAVAILABLE")
            except Exception as error:
                if not attempt_ready:
                    raise
                observed = response or LLMResponse("", {
                    "usage": getattr(error, "provider_usage", None),
                    "authoritative_provider_cost_usd": getattr(error, "authoritative_provider_cost_usd", None),
                }, {}, 0)
                try:
                    cost = _response_cost(observed)
                    realized = _realized(cost)
                except (KeyError, TypeError, ValueError):
                    cost, realized = ProviderCostEvidence(), None
                observation = json.dumps({
                    "response_id": observed.raw.get("response_id", getattr(error, "provider_response_id", None)),
                    "model": observed.raw.get("model", getattr(error, "provider_returned_model", None)),
                    "service_tier": observed.raw.get("service_tier", getattr(error, "provider_service_tier", None)),
                    "status": observed.raw.get("status", getattr(error, "provider_status", None)),
                    "incomplete_reason": observed.raw.get("incomplete_reason", getattr(error, "provider_incomplete_reason", None)),
                    "usage": observed.raw.get("usage"),
                    "provider_cost_usd": observed.raw.get("authoritative_provider_cost_usd"),
                    "failure_type": type(error).__name__,
                    "retry_class": getattr(error, "phase13_retry_class", None),
                }, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                if (attempt_index == 0 and key.dispatch_id in self._retry_entitlements
                        and _eligible_retry(error, response)):
                    self._publish_bytes(key, "retry-observation-0", observation)
                    self._append(key, "RETRYABLE_ATTEMPT_FAILURE", {
                        "attempt_index": 0,
                        "failure_code": str(getattr(error, "phase13_retry_class", None)),
                        "observation_hash": hashlib.sha256(observation).hexdigest(),
                        "cost": cost.model_dump(mode="json"), "realized_cost_krw": None,
                    })
                    attempt_index = 1
                    provider = self._factory(self.binding)
                    continue
                self._publish_bytes(key, "observation", observation)
                self._append(key, "ATTEMPTED_PROVIDER_FAILURE", {
                    "transport_attempts": attempt_index + 1, "cost": cost.model_dump(mode="json"),
                    "realized_cost_krw": realized,
                    "failure_code": str(getattr(error, "code", type(error).__name__)),
                    "observation_hash": hashlib.sha256(observation).hexdigest(),
                })
                raise DispatchTechnicalFailureV3("MAIN_ATTEMPTED_PROVIDER_FAILURE", key.parent_id, realized,
                    evidence_sha256=self.ledger.state(key.dispatch_id).event_hash) from error
            break
        self._attempt_counts[key.dispatch_id] = attempt_index + 1
        if not defer_completion:
            self._append(key, "COMPLETED", {
                "transport_attempts": attempt_index + 1, "cost": cost.model_dump(mode="json"),
                "realized_cost_krw": realized,
                "result_hash": hashlib.sha256(response.content.encode()).hexdigest(),
            })
        return result

    def receive(self, key: RequestKeyV3, compile_material: Callable[[], RequestMaterialV3]) -> LLMResponse:
        with request_lock(self.ledger):
            self._refresh_terminal_parents()
            return self._dispatch(key, compile_material, lambda response: response, defer_completion=True)

    def acknowledge(self, key: RequestKeyV3, response: LLMResponse, *, semantic_success: bool) -> None:
        with request_lock(self.ledger):
            cost = _response_cost(response)
            realized = _realized(cost)
            fields: dict[str, JsonValue] = {
                "transport_attempts": self._attempt_counts.pop(key.dispatch_id, 1),
                "cost": cost.model_dump(mode="json"), "realized_cost_krw": realized,
            }
            if semantic_success and realized is not None:
                self._append(key, "COMPLETED", {**fields, "result_hash": hashlib.sha256(response.content.encode()).hexdigest()})
                return
            raw = json.dumps({"response": response.content, "semantic_success": semantic_success}, sort_keys=True).encode()
            self._publish_bytes(key, "observation", raw)
            self._append(key, "ATTEMPTED_PROVIDER_FAILURE", {**fields,
                "failure_code": "MAIN_SEMANTIC_RESULT_UNAVAILABLE",
                "observation_hash": hashlib.sha256(raw).hexdigest()})
            raise DispatchTechnicalFailureV3("MAIN_ATTEMPTED_PROVIDER_FAILURE", key.parent_id, realized,
                evidence_sha256=self.ledger.state(key.dispatch_id).event_hash)

    def _append(self, key: RequestKeyV3, kind: str, extra: dict[str, JsonValue] | None = None) -> None:
        state = self.ledger.state(key.dispatch_id)
        compiled = self._compiled.get(key.dispatch_id)
        evidence = compiled.evidence if kind == "REQUEST_COMPILED" and compiled is not None else state.compiled
        self.ledger.append({
            "schema_version": "phase13_main_dispatch_evidence_v3", "unit_id": key.dispatch_id,
            "revision": state.revision + 1, "previous_hash": state.event_hash, "kind": kind,
            "compiled": None if evidence is None else evidence.model_dump(mode="json"),
            **(extra or {}),
        })
        self._terminal_generation = self.ledger.generation()
        if kind in {"INPUT_ENVELOPE_OVERFLOW", "TERMINAL_TECHNICAL_MISSING",
                    "ATTEMPTED_PROVIDER_FAILURE", "AMBIGUOUS_ATTEMPT"}:
            failed = {key.parent_id}
            parent = next(row for row in self.parents if row.parent_id == key.parent_id)
            if parent.kind == "CLEAN_PREFIX":
                failed.update(row.parent_id for row in self.parents if row.prefix_parent_id == parent.parent_id)
            self._terminal_parents = frozenset((*self._terminal_parents, *failed))

    def _refresh_terminal_parents(self) -> None:
        generation = self.ledger.generation()
        if generation != self._terminal_generation:
            self._terminal_parents = terminal_parents(self.ledger, self.binding, self.parents)
            self._terminal_generation = generation

    def _persist_compiled(self, compiled: CompiledProviderRequestV3) -> None:
        raw = json.dumps({
            "binding": self.binding.model_dump(mode="json"),
            "key": compiled.key.model_dump(mode="json"),
            "parents": [parent.model_dump(mode="json") for parent in self.parents],
            "compiled": compiled.evidence.model_dump(mode="json"),
            "request_hex": compiled.request_bytes.hex(), "input_hex": input_bytes(compiled.material).hex(),
            "native_state_hex": compiled.native_state.hex(),
        }, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        self._publish_bytes(compiled.key, "compiled", raw)

    def _publish_bytes(self, key: RequestKeyV3, role: str, raw: bytes) -> None:
        if role in {"count-started", "count-receipt", "count-failure"}:
            self.ledger.append_count_record(key.dispatch_id, role, raw)
        if self.ledger.guard is not None:
            self.ledger.guard.publish_record(f"{key.dispatch_id}.{role}.json", raw)
            return
        path = self.ledger.path.parent / f"{key.dispatch_id}.{role}.json"
        with NamedTemporaryFile(dir=path.parent) as temporary:
            temporary.write(raw)
            temporary.flush()
            os.fsync(temporary.fileno())
            try:
                os.link(temporary.name, path)
            except FileExistsError:
                if read_regular_nofollow(path) != raw:
                    raise TerminalEvidenceError() from None
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class CostBoundRequestDispatcherV3:
    def __init__(self, dispatcher: ProductionRequestDispatcherV3,
                 costs: LiveCosts, package_hash: str) -> None:
        if (costs.resources.phase4.policy.authority.identity, digest(costs.package), costs.package.package_hash) != (
            dispatcher.binding.identity, dispatcher.binding.package_sha256, package_hash,
        ):
            raise CostError("MAIN_COST_PROOF_MISMATCH")
        self._dispatcher, self._costs, self._package_hash = dispatcher, costs, package_hash

    def dispatch(self, key: RequestKeyV3, compile_material: Callable[[], RequestMaterialV3],
                 parse_result: Callable[[LLMResponse], ResultT]) -> ResultT:
        self._costs.projected(self._package_hash, TableKey(
            proof_hash=self._costs.package.cost_proof_hash, unit_id=key.parent_id,
        ))
        return self._dispatcher.dispatch(key, compile_material, parse_result)


def _response_cost(response: LLMResponse | None) -> ProviderCostEvidence:
    if response is None:
        return ProviderCostEvidence()
    usage = response.raw.get("usage")
    monetary = response.raw.get("authoritative_provider_cost_usd")
    return ProviderCostEvidence.model_validate({
        "monetary_cost": None if monetary is None else str(monetary),
        "currency": response.raw.get("currency"),
        "usage": None if usage is None else {
        "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
        "cached_input_tokens": usage.get("input_tokens_details", {}).get("cached_tokens", 0),
    }})


def _validate_received_response(response: LLMResponse, stage: Stage) -> None:
    raw = response.raw
    if raw.get("status") == "incomplete":
        raise TerminalEvidenceError("MAIN_PROVIDER_INCOMPLETE")
    if (raw.get("status") != "completed" or raw.get("model") != "gpt-5.6-luna"
            or raw.get("service_tier") != "default"):
        raise TerminalEvidenceError("MAIN_PROVIDER_RESPONSE_CONTRACT_MISMATCH")
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        raise TerminalEvidenceError("MAIN_PROVIDER_USAGE_UNAVAILABLE")
    input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
    if (type(input_tokens) is not int or type(output_tokens) is not int
            or input_tokens < 0 or output_tokens < 0):
        raise TerminalEvidenceError("MAIN_PROVIDER_USAGE_UNAVAILABLE")
    if input_tokens > STAGES[stage][0] or output_tokens > STAGES[stage][1]:
        raise TerminalEvidenceError("MAIN_PROVIDER_ENVELOPE_EXCEEDED")


def _eligible_retry(error: Exception, response: LLMResponse | None) -> bool:
    return (
        response is None
        and getattr(error, "phase13_retry_class", None) in {
            "TIMEOUT_BEFORE_SEMANTIC_PAYLOAD",
            "CONNECTION_FAILURE_BEFORE_SEMANTIC_PAYLOAD",
            "HTTP_429_BEFORE_SEMANTIC_PAYLOAD",
            "REGISTERED_PROVIDER_5XX_BEFORE_SEMANTIC_PAYLOAD",
        }
        and getattr(error, "provider_usage", None) is None
        and getattr(error, "authoritative_provider_cost_usd", None) is None
        and getattr(error, "provider_response_id", None) is None
        and getattr(error, "provider_status", None) is None
        and getattr(error, "provider_failure_acknowledged", False) is True
    )


def _realized(cost: ProviderCostEvidence) -> int | None:
    try:
        return reconcile_actual(cost).realized_krw
    except CostError as error:
        if error.code != "MAIN_TERMINAL_COST_UNKNOWN":
            raise
        return None


__all__ = [
    "CostBoundRequestDispatcherV3",
    "DispatchTechnicalFailureV3",
    "PackageBindingV3",
    "ParentTrajectoryV3",
    "ProductionRequestDispatcherV3",
    "RequestKeyV3",
    "RequestMaterialV3",
]
