from __future__ import annotations

from pathlib import Path

from memcontam.clients.base import LLMClient, LLMResponse
from memcontam.readiness.phase13_provider_accounting import (
    OwnedProviderAccounting,
    build_owned_provider_client,
)
from memcontam.readiness.phase13_provider_models import (
    AccountingReport,
    ExecutionTemplateIdentity,
)


class Phase13V2RuntimeError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class Phase13V2ProviderRuntime:
    def __init__(self, client: LLMClient) -> None:
        if not isinstance(client, OwnedProviderAccounting):
            raise Phase13V2RuntimeError("OWNED_PROVIDER_REQUIRED")
        self._client = client

    @classmethod
    def from_provider(
        cls,
        provider: LLMClient,
        root: Path,
        intended_template: ExecutionTemplateIdentity,
    ) -> Phase13V2ProviderRuntime:
        return cls(build_owned_provider_client(provider, root, intended_template))

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: dict,
    ) -> LLMResponse:
        return self._client.chat(messages, model, config)

    def reconcile(self) -> AccountingReport:
        return self._client.reconcile()
