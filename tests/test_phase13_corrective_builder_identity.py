from pathlib import Path

import pytest

from memcontam.readiness.phase13_v3_builder import build_mr_p4, validate_mr_p4
from memcontam.readiness.phase13_v3_conformance import ConformanceV3
from memcontam.readiness.phase13_v3_cost_models import ActivatedPolicyV3

from .phase13_corrective_identity import corrective_identity
from .test_phase13_v3_artifact_builder import builder_source as builder_source, synthetic_pricing
from .test_phase13_v3_entrypoint_integration import deny_external as deny_external


@pytest.mark.parametrize("generation", ["disposable-alpha", "disposable-beta"])
def test_builder_binds_explicit_generation_without_authorization(
    builder_source: tuple[Path, str, Path], tmp_path: Path, generation: str, deny_external,
) -> None:
    repository, commit, authority = builder_source
    identity = corrective_identity(generation)
    result = build_mr_p4(repository, authority, tmp_path, governed_source_commit=commit,
                         identity=identity, count_pricing=synthetic_pricing(repository, authority, identity))
    assert result.identity == result.authority.identity == identity
    assert validate_mr_p4(repository, authority, tmp_path) == result
    conformance = ConformanceV3.model_validate_json(
        (tmp_path / "mr_p4/corrected_v3/provider_free_conformance_v3.json").read_bytes())
    policy = ActivatedPolicyV3.model_validate_json(
        (tmp_path / "cost_envelope_v3/activated_policy_v3.json").read_bytes())
    assert conformance.identity == policy.authority.identity == identity
    assert conformance.real_provider_calls == 0
    assert not (tmp_path / "mr_p6").exists()
