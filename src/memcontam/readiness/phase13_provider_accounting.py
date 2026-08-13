from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from memcontam.clients.base import LLMClient, LLMResponse
from memcontam.readiness.phase13_analysis_contract import load_analysis_registry
from memcontam.readiness.phase13_execution_contract import load_execution_registry
from memcontam.readiness.phase13_authority import JsonValue
from memcontam.readiness.phase13_provider_models import (
    AccountingReport,
    AccountingTotals,
    JsonScalar,
    OfflineAccounting,
    OwnedDispatchConfig,
    ProviderAccountingError,
    ProviderDispatchFailure,
    ProviderTotals,
    SettledCall,
    TransportAttempt,
)


class OwnedProviderAccounting:
    def __init__(
        self,
        client: LLMClient,
        root: Path,
    ) -> None:
        execution = load_execution_registry(
            root / "data/phase13/authority/execution_registry_v1.json", root
        )
        analysis = load_analysis_registry(
            root / "data/phase13/authority/analysis_registry_v1.json", root
        )
        self._client = client
        self._execution_owner_id = execution.execution_owner_id
        self._template_ids = frozenset(row.template_id for row in execution.execution_templates)
        self._offline_rows = tuple(
            OfflineAccounting(
                operation=row.operation,
                owner_id=row.owner_id,
                provider_calls=0,
                cost_microusd=0,
            )
            for row in analysis.offline_compute.rows
        )
        self._planned_ids: set[str] = set()
        self._calls: list[SettledCall] = []
        self._dispatches = 0

    @classmethod
    def from_authority(cls, client: LLMClient, root: Path) -> OwnedProviderAccounting:
        return cls(client, root)

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        dispatch: OwnedDispatchConfig,
    ) -> LLMResponse:
        self._validate_dispatch(dispatch)
        semantic_call_id = str(uuid4())
        if semantic_call_id in self._planned_ids:
            raise ProviderAccountingError("DUPLICATE_SEMANTIC_CALL_ID")
        self._planned_ids.add(semantic_call_id)
        self._dispatches += 1
        provider_config = dict(dispatch.provider_config)
        provider_config["execution_owner_id"] = dispatch.execution_owner_id
        provider_config["semantic_call_id"] = semantic_call_id
        provider_config["execution_template_id"] = dispatch.execution_template_id
        try:
            response = self._client.chat(messages, model, provider_config)
        except ProviderDispatchFailure as failure:
            call = self._settle(
                semantic_call_id,
                dispatch,
                failure.provider_attempts,
                failure.provider_totals,
                failure.provider_error,
            )
            self._calls.append(call)
            raise ProviderAccountingError("PROVIDER_DISPATCH_FAILED") from failure
        call = self._settle_from_response(semantic_call_id, dispatch, response)
        self._calls.append(call)
        return response

    def reconcile(self) -> AccountingReport:
        actual_ids = [call.semantic_call_id for call in self._calls]
        if len(set(actual_ids)) != len(actual_ids):
            raise ProviderAccountingError("DUPLICATE_SEMANTIC_CALL_ID")
        if set(actual_ids) != self._planned_ids:
            raise ProviderAccountingError("PLANNED_ACTUAL_SETTLEMENT_MISMATCH")
        totals = self._aggregate(self._calls)
        return AccountingReport(calls=tuple(self._calls), offline=self._offline_rows, totals=totals)

    def _validate_dispatch(self, dispatch: OwnedDispatchConfig) -> None:
        if dispatch.execution_owner_id in {row.owner_id for row in self._offline_rows}:
            raise ProviderAccountingError("OFFLINE_OWNER_FORBIDDEN")
        if dispatch.execution_owner_id != self._execution_owner_id:
            raise ProviderAccountingError("UNKNOWN_EXECUTION_OWNER")
        if dispatch.execution_template_id not in self._template_ids:
            raise ProviderAccountingError("UNKNOWN_EXECUTION_TEMPLATE")

    def _settle_from_response(
        self,
        call_id: str,
        dispatch: OwnedDispatchConfig,
        response: LLMResponse,
    ) -> SettledCall:
        attempts = response.raw.get("provider_attempts")
        totals = response.raw.get("provider_totals")
        if not isinstance(attempts, (list, tuple)) or not isinstance(totals, Mapping):
            raise ProviderAccountingError("PROVIDER_ACCOUNTING_REQUIRED")
        call = self._settle(call_id, dispatch, attempts, totals, None)
        if (
            response.token_usage.get("prompt_tokens") != call.totals.input_tokens
            or response.token_usage.get("completion_tokens") != call.totals.output_tokens
            or response.latency_ms != call.totals.latency_ms
        ):
            raise ProviderAccountingError("OWNER_TOTAL_MISMATCH")
        return call

    def _settle(
        self,
        call_id: str,
        dispatch: OwnedDispatchConfig,
        raw_attempts: tuple[Mapping[str, JsonValue], ...] | list[Mapping[str, JsonValue]],
        raw_totals: Mapping[str, int] | Mapping[str, JsonScalar],
        provider_error: str | None,
    ) -> SettledCall:
        try:
            attempts = tuple(TransportAttempt.model_validate(row) for row in raw_attempts)
            totals = ProviderTotals.model_validate(raw_totals)
        except (ValidationError, TypeError) as error:
            raise ProviderAccountingError("PROVIDER_ACCOUNTING_REQUIRED") from error
        if not attempts:
            raise ProviderAccountingError("PROVIDER_ACCOUNTING_REQUIRED")
        if any(row.semantic_call_id != call_id for row in attempts):
            raise ProviderAccountingError("SEMANTIC_CALL_ID_MISMATCH")
        owners = {row.execution_owner_id for row in attempts}
        if owners & {row.owner_id for row in self._offline_rows}:
            raise ProviderAccountingError("OFFLINE_OWNER_FORBIDDEN")
        if owners != {self._execution_owner_id}:
            raise ProviderAccountingError("UNKNOWN_EXECUTION_OWNER")
        if len({row.attempt_id for row in attempts}) != len(attempts):
            raise ProviderAccountingError("DUPLICATE_ATTEMPT_ID")
        if tuple(row.attempt_number for row in attempts) != tuple(range(1, len(attempts) + 1)):
            raise ProviderAccountingError("TRANSPORT_RETRY_SEQUENCE_INVALID")
        expected = self._sum_attempts(attempts)
        if totals != expected:
            raise ProviderAccountingError("OWNER_TOTAL_MISMATCH")
        return SettledCall(
            semantic_call_id=call_id,
            execution_owner_id=dispatch.execution_owner_id,
            execution_template_id=dispatch.execution_template_id,
            provider_error=provider_error,
            attempts=attempts,
            totals=totals,
        )

    @staticmethod
    def _sum_attempts(attempts: tuple[TransportAttempt, ...]) -> ProviderTotals:
        return ProviderTotals(
            transport_attempts=len(attempts),
            retries=max(0, len(attempts) - 1),
            input_tokens=sum(row.input_tokens for row in attempts),
            output_tokens=sum(row.output_tokens for row in attempts),
            cost_microusd=sum(row.cost_microusd for row in attempts),
            latency_ms=sum(row.latency_ms for row in attempts),
            storage_bytes=sum(row.storage_bytes for row in attempts),
        )

    def _aggregate(self, calls: list[SettledCall]) -> AccountingTotals:
        fields = ProviderTotals.model_fields
        sums = {name: sum(getattr(call.totals, name) for call in calls) for name in fields}
        return AccountingTotals(
            semantic_calls=len(calls),
            dispatches=self._dispatches,
            **sums,
        )
