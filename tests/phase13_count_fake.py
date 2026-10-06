from collections.abc import Callable

from memcontam.readiness.phase13_v3_count import CountIdentityV3, CountReceiptV3, count_operation
from memcontam.readiness.phase13_v3_cost_models import CountPricingV1
from memcontam.readiness.phase13_v3_request import CompiledProviderRequestV3


def fake_count_pricing(operations: int, rate: str = "0.000000001") -> CountPricingV1:
    return CountPricingV1(
        endpoint="https://fake.invalid/v1/responses/input_tokens",
        deployment_sha256="1" * 64,
        billing_evidence_sha256="2" * 64,
        compatibility_evidence_sha256="3" * 64,
        maximum_usd_per_operation=rate,
        maximum_count_operations=operations,
    )


class CountedProvider:
    """Offline-only provider count with an explicit synthetic USD charge."""

    def count_identity_v3(self) -> CountIdentityV3:
        return CountIdentityV3(base_url="https://fake.invalid/v1", sdk_version="fake",
            account_sha256="1" * 64, runtime_sha256="2" * 64,
            source_sha256="3" * 64, schema_sha256="4" * 64)

    def count_compiled_v3(self, compiled: CompiledProviderRequestV3,
                          before_count: Callable[[], None]) -> CountReceiptV3:
        before_count()
        return CountReceiptV3(operation=count_operation(compiled, self.count_identity_v3()),
            object="response.input_tokens", input_tokens=1, monetary_cost_usd="0.001")
