from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from pydantic import TypeAdapter

from memcontam.clients.base import LLMResponse
from memcontam.experiment.phase12.runtime_registry import NoMemRuntimeState, RuntimeTrialResult
from memcontam.memory.checkpoint_v3 import NativeState, serialize_checkpoint
from .phase13_main_request_dispatch import DispatchTechnicalFailureV3, ProductionRequestDispatcherV3
from .phase13_v3_request import MessageV3, RequestKeyV3, RequestMaterialV3, Stage
from .phase13_v3_terminal_models import TerminalEvidenceError
from .phase13_v3_cost_actual import reconcile_actual


StateT = TypeVar("StateT")
_STAGE: TypeAdapter[Stage] = TypeAdapter(Stage)


def native_state_bytes(serialize: Callable[[StateT], StateT], state: StateT) -> bytes:
    snapshot = serialize(state)
    match snapshot:
        case NativeState():
            return serialize_checkpoint(snapshot).canonical_bytes
        case NoMemRuntimeState():
            return b"{}"
        case _:
            raise TerminalEvidenceError("MAIN_NATIVE_STATE_UNAVAILABLE")


class MainRequestClientV3:
    """A response stays ATTEMPT_STARTED until the real baseline acknowledges parsing.

    Reaching the next semantic call acknowledges its predecessor; returning from
    the baseline acknowledges the final call. A crash before either is ambiguous.
    """

    def __init__(self, dispatcher: ProductionRequestDispatcherV3, parent_id: str,
                 preflight: Callable[[], None]) -> None:
        self.dispatcher, self.parent_id, self.preflight = dispatcher, parent_id, preflight
        self._ordinals: dict[Stage, int] = {}
        self._native: Callable[[], bytes] | None = None
        self._pending: tuple[RequestKeyV3, LLMResponse] | None = None
        self._failure: ValueError | RuntimeError | OSError | None = None

    def trial(self, execute: Callable[[], RuntimeTrialResult], native: Callable[[], bytes]) -> RuntimeTrialResult:
        if self._native is not None:
            raise TerminalEvidenceError()
        self._native = native
        try:
            result = execute()
            if self._failure is not None:
                raise self._failure
            self._acknowledge(result.outcome.status == "succeeded")
            return result
        except (ValueError, KeyError, TypeError) as error:
            try:
                self._acknowledge(False)
            except DispatchTechnicalFailureV3:
                raise error from None
            raise
        finally:
            self._native = None

    def _acknowledge(self, success: bool) -> None:
        if self._pending is not None:
            key, response = self._pending
            self._pending = None
            self.dispatcher.acknowledge(key, response, semantic_success=success)

    def realized_cost_krw(self) -> int:
        total = 0
        for stage, count in self._ordinals.items():
            for ordinal in range(count):
                key = RequestKeyV3(parent_id=self.parent_id, stage=stage, ordinal=ordinal)
                cost = self.dispatcher.ledger.state(key.dispatch_id).attempted_cost
                if cost is None:
                    raise TerminalEvidenceError("MAIN_TERMINAL_COST_UNKNOWN")
                total += reconcile_actual(cost).realized_krw
        return total

    def chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        try:
            return self._chat(messages, model, config)
        except (ValueError, RuntimeError, OSError) as error:
            self._failure = error
            raise

    def _chat(self, messages: list[dict[str, str]], model: str, config: dict) -> LLMResponse:
        if self._native is None or model != "gpt-5.6-luna":
            raise TerminalEvidenceError("MAIN_NATIVE_STATE_UNAVAILABLE")
        self.preflight()
        self._acknowledge(True)
        stage = _STAGE.validate_python(config.get("method_stage"), strict=True)
        ordinal = self._ordinals.get(stage, 0)
        key = RequestKeyV3(parent_id=self.parent_id, stage=stage, ordinal=ordinal)
        self._ordinals[stage] = ordinal + 1
        native = self._native
        response = self.dispatcher.receive(key, lambda: RequestMaterialV3(
            messages=tuple(MessageV3.model_validate(message) for message in messages),
            native_state=native(), temperature=config.get("temperature", 0.0), top_p=config.get("top_p", 1.0),
        ))
        self._pending = key, response
        return response
