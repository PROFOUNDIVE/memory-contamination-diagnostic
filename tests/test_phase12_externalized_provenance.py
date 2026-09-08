from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError
import pytest
import yaml

from memcontam.readiness import phase13_v3_builder as builders
from memcontam.readiness.phase13_v3_cost_models import canonical_bytes, digest
from memcontam.readiness.phase13_v3_entrypoint_models import ExecutionResourceV3
from .test_phase13_v3_artifact_builder import builder_source as builder_source, staged as staged


REGISTRY_PATH = "tests/fixtures/phase12_externalized_provenance_registry_v1.json"
EXPECTED = (
    ("References/Theoretical Artifacts/AGENTS.md", 10499,
     "362f3ba6c51dec7ebfd61b68a9c908e64ef84858e93f796bf5de6b40fb70cd46"),
    (".sisyphus/plans/BASELINE-FIDELITY-V2_source-contract_remediation.md", None,
     "5a5afe7f0d5fa171ff9d0b279fdd5875ee6885e718043cdfff3e59c449428e0f"),
    (".omo/approvals/phase12-post-filter-v5-calibration-readiness.plan.sha256", 65,
     "7b878988972b5bc3c1a2ba24785b978cc26b973e1e44e8059ff8d3133227842e"),
    (".omo/plans/phase12-filter-v5-screening-bct-execution.md", 144691,
     "9270d31770eb97e732602cfe85a250111208afeae293b0a20ab618baadb43317"),
    (".omo/approvals/phase12-filter-v5-screening-bct-execution.plan.sha256", 65,
     "92c6d30f026a10f47067e5467c0e9e0abc35b653385f4f08ad7d301838e06160"),
    (".omo/evidence/phase12-post-filter-v5-calibration-readiness/task-3-screening-stage-result.json", 246,
     "583d1bd5a579af84b00ded45e67b66f491940237c4e708027d9da827b4bbb8f7"),
)


class LegacyIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    path: str
    size_bytes: int | None
    sha256: str
    role: Literal["legacy_phase12_provenance_only"]
    availability: Literal["externalized_exact_revision_unavailable"]
    current_authorization_member: Literal[False]
    surviving_copy_path: None
    replacement_path: None


class LegacyRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    schema_version: Literal["phase12_externalized_provenance_registry_v1"]
    authority: Literal[False]
    scientific_input: Literal[False]
    runtime_input: Literal[False]
    replacement_bytes: Literal[False]
    superseding_current_gate: Literal["phase13-main-a-corrected-authorized-execution-v3"]
    records: tuple[LegacyIdentity, ...]


def parse_legacy_registry(raw: bytes) -> LegacyRegistry:
    registry = LegacyRegistry.model_validate_json(raw)
    assert tuple((row.path, row.size_bytes, row.sha256) for row in registry.records) == EXPECTED
    return registry


def legacy_registry() -> LegacyRegistry:
    root = Path(__file__).resolve().parents[1]
    return parse_legacy_registry((root / REGISTRY_PATH).read_bytes())


def synthetic_legacy_methods_inputs(
    temporary_root: Path, config_source: Path,
) -> tuple[Path, Path, Path]:
    registry = legacy_registry()
    approved = b"# Synthetic legacy contract fixture\n- [ ] 1. Freeze strict inventory\n"
    approved_digest = sha256(approved).hexdigest()
    plan = temporary_root / ".omo/plans/phase12-post-filter-v5-calibration-readiness.md"
    descriptor = temporary_root / ".omo/approvals/phase12-post-filter-v5-calibration-readiness.plan.sha256"
    plan.parent.mkdir(parents=True)
    descriptor.parent.mkdir(parents=True)
    plan.write_bytes(approved.replace(b"[ ]", b"[x]", 1))
    descriptor.write_text(approved_digest + "\n", encoding="ascii")
    config = temporary_root / "configs/phase12/filter_v5_bct_calibration.yaml"
    config.parent.mkdir(parents=True)
    payload = TypeAdapter(dict[str, JsonValue]).validate_python(
        yaml.safe_load(config_source.read_bytes()), strict=True,
    )
    payload["approved_plan_sha256"] = approved_digest
    config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    assert sha256(descriptor.read_bytes()).hexdigest() != registry.records[2].sha256
    return plan, descriptor, config


@pytest.mark.parametrize("index", range(6))
def test_exact_externalized_identity_has_no_replacement_or_authority(index: int) -> None:
    registry = legacy_registry()
    row = registry.records[index]
    assert (row.path, row.size_bytes, row.sha256) == EXPECTED[index]
    assert row.surviving_copy_path is row.replacement_path is None
    assert row.current_authorization_member is False


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("field", ("sha256", "size_bytes", "availability", "current_authorization_member"))
def test_each_externalized_identity_mutation_is_rejected(index: int, field: str) -> None:
    payload = legacy_registry().model_dump(mode="json")
    changes = {"sha256": "0" * 64, "size_bytes": -1, "availability": "available", "current_authorization_member": True}
    payload["records"][index][field] = changes[field]

    with pytest.raises((ValidationError, AssertionError)):
        parse_legacy_registry(json.dumps(payload).encode())


@pytest.mark.parametrize("field", ("authority", "scientific_input", "runtime_input", "replacement_bytes"))
def test_registry_cannot_be_promoted_to_current_authority(field: str) -> None:
    payload = legacy_registry().model_dump(mode="json")
    payload[field] = True

    with pytest.raises(ValidationError):
        parse_legacy_registry(json.dumps(payload).encode())


def test_externalized_registry_is_absent_from_actual_v3_package_and_authorization(
    staged: tuple[ModuleType, Path, Path, Path],
) -> None:
    _, root, output, authority = staged
    registry = legacy_registry()
    package = builders.build_mr_p5(root, authority, output)
    authorization = builders.build_mr_p6(root, authority, output)

    for model in (package, authorization):
        serialized = model.model_dump_json()
        assert REGISTRY_PATH not in serialized
        for row in registry.records:
            assert row.sha256 not in serialized
    assert builders.validate_mr_p6(root, authority, output) == authorization


def test_self_rehashed_package_cannot_add_registry_as_runtime_resource(
    staged: tuple[ModuleType, Path, Path, Path],
) -> None:
    _, root, output, authority = staged
    registry = legacy_registry()
    target = root / REGISTRY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_bytes(registry))
    package = builders.build_mr_p5(root, authority, output)
    extra = ExecutionResourceV3(role="externalized_legacy", path=REGISTRY_PATH,
                               size=target.stat().st_size, sha256=digest(registry))
    forged = package.model_copy(update={"resources": (*package.resources, extra)})
    forged = forged.model_copy(update={"package_hash": digest(forged, "package_hash")})
    (output / "mr_p5/execution_package_v3.json").write_bytes(canonical_bytes(forged))

    with pytest.raises(ValueError, match="MAIN_COST_PROOF_MISMATCH"):
        builders.validate_mr_p5(root, authority, output)
