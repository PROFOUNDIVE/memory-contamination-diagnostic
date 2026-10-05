from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from memcontam.clients.base import LLMResponse
from memcontam.readiness.phase13_main_v3_runner import V3MainRun
from memcontam.readiness.phase13_v3_entrypoint import SelectedExecutionV3, SelectionRequest, select_execution
from memcontam.readiness.phase13_v3_request import CompiledProviderRequestV3, PackageBindingV3
from memcontam.readiness.phase13_v3_count import CountReceiptV3
from .phase13_count_fake import CountedProvider


class FakeProvider(CountedProvider):
    def __init__(self) -> None:
        self.constructors = 0
        self.requests: list[str] = []
        self.count_requests: list[str] = []

    def factory(self, binding: PackageBindingV3) -> FakeProvider:
        self.constructors += 1
        return self

    def count_compiled_v3(
        self, compiled: CompiledProviderRequestV3, before_count: Callable[[], None],
    ) -> CountReceiptV3:
        receipt = super().count_compiled_v3(compiled, before_count)
        self.count_requests.append(compiled.key.dispatch_id)
        return receipt

    def send_compiled_v3(self, compiled: CompiledProviderRequestV3, before_request: Callable[[], None]) -> LLMResponse:
        before_request()
        self.requests.append(compiled.key.dispatch_id)
        return LLMResponse(
            "final: 0",
            {
                "usage": {"input_tokens": 1, "output_tokens": 0},
                "attempts": 1,
                "authoritative_provider_cost_usd": "0.0000002",
                "currency": "USD",
                "status": "completed",
                "response_id": f"fake-{compiled.key.dispatch_id}",
                "model": "gpt-5.6-luna",
                "service_tier": "default",
            },
            {"prompt_tokens": 1, "completion_tokens": 0},
            0,
        )


def open_run(request: SelectionRequest, *, create: bool, seed: int = 0) -> V3MainRun:
    selected = select_execution(request, "run" if create else "resume")
    assert isinstance(selected, SelectedExecutionV3)
    return V3MainRun.open(selected, request.repository_root / "safety-run", create=create, seed=seed)


def launcher(root: Path, *, execute: bool = False) -> None:
    from .phase13_corrective_identity import corrective_identity
    from unittest.mock import patch
    import memcontam.readiness.phase13_main_request_dispatch as dispatch
    import memcontam.readiness.phase13_main_live_runtime as runtime

    request = SelectionRequest(root, root / "package.json", root / "authorization.json", root / "authority",
                               root / "authorization.sha256", corrective_identity().run_id)
    try:
        run = open_run(request, create=False)
    except ValueError as error:
        print(str(error), flush=True)
        return
    try:
        if execute:
            fake = FakeProvider()
            original = runtime.validate_production_archive

            def barrier(*args, **kwargs):
                result = original(*args, **kwargs)
                print(f"COMPLETED {fake.constructors} {len(fake.requests)}", flush=True)
                input()
                return result

            def deny(*args, **kwargs):
                raise AssertionError("REAL_PROVIDER_OR_NETWORK_FORBIDDEN")

            with patch.object(dispatch, "count_prompt_tokens", return_value=1), \
                 patch.object(runtime, "validate_production_archive", side_effect=barrier), \
                  patch.object(dispatch, "production_provider", side_effect=deny), \
                  patch("socket.socket.connect", side_effect=deny), patch("socket.create_connection", side_effect=deny), \
                  patch("httpx.Client.send", side_effect=deny), patch("httpx.AsyncClient.send", side_effect=deny), \
                  patch("openai.OpenAI.__init__", side_effect=deny), patch("openai.AsyncOpenAI.__init__", side_effect=deny):
                run.execute(root / "cache", max_units=1, tranche_ceiling_krw=450000, provider_factory=fake.factory)
        else:
            print("OWNED", flush=True)
            input()
    finally:
        run.close()
