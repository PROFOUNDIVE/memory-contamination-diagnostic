from __future__ import annotations

from dataclasses import dataclass

from memcontam.clients.base import LLMClient

from .phase13_main_checkpoint import CommonCheckpointRegistry
from .phase13_main_request_client import MainRequestClientV3
from .phase13_v3_cost_binding import LiveCosts


@dataclass(frozen=True, slots=True)
class ValidatedOrdinaryResources:
    execution_client: LLMClient
    common_capacity_tokens: int
    checkpoint_bytes: bytes
    checkpoint_sha256: str
    costs: LiveCosts
    request_client: MainRequestClientV3 | None = None

    @property
    def checkpoint_registry(self) -> CommonCheckpointRegistry:
        return CommonCheckpointRegistry.model_validate_json(self.checkpoint_bytes)
