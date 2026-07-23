from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from memcontam.experiment.phase12.contracts import (
    BehaviorTestRegistry,
    BehaviorTestRow,
    EmbeddingRuntimeContract,
    FidelityCertificate,
    MetricRegistry,
    canonical_json_hash,
)


REQUIRED_TEST_IDS = (
    "MFT-01",
    "MFT-02",
    "MFT-03",
    "MFT-04",
    "INV-01",
    "INV-02",
    "INV-03",
    "DIR-01",
    "DIR-02",
    "DIR-03",
    "DIR-04",
)
_REGISTRY_ID = "phase12-behavior-registry-v1"
_SCHEMA_VERSION = "phase12-behavior-registry-v1"
_SOURCE_DESIGN_HASH = "984fe2881690d93a8ccced87abf03de4bf0012158462cf07ed23505414073eb0"
_INV03_REGISTRY_ID = "inv03-equivalence-registry-v1"
_INV03_REGISTRY_HASH = "ffdb247dc187d462208dbe9f7a4ead8bfa27def24a3052baedca77c50aa2e620"


class BehaviorRegistryError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class BehaviorRegistryBundle:
    behavior_tests: BehaviorTestRegistry
    metric_registry: MetricRegistry
    embedding_runtime_contract: EmbeddingRuntimeContract
    fidelity_certificate: FidelityCertificate

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(
            {
                "behavior_tests": self.behavior_tests.model_dump(mode="json"),
                "metric_registry": self.metric_registry.model_dump(mode="json"),
                "embedding_runtime_contract": self.embedding_runtime_contract.model_dump(
                    mode="json"
                ),
                "fidelity_certificate": self.fidelity_certificate.model_dump(mode="json"),
            }
        )


@dataclass(frozen=True)
class RegistryValidationReport:
    valid: bool
    test_ids: tuple[str, ...]
    bundle_hash: str


def load_behavior_registry_bundle(root: Path) -> BehaviorRegistryBundle:
    rows = tuple(_load_rows(root / "behavior_tests.jsonl"))
    try:
        behavior_tests = BehaviorTestRegistry(
            registry_id=_REGISTRY_ID,
            schema_version=_SCHEMA_VERSION,
            required_test_ids=REQUIRED_TEST_IDS,
            rows=rows,
            source_experiment_design_sha256=_SOURCE_DESIGN_HASH,
            frozen_after_pilot_b=True,
            artifact_hash=_registry_hash(rows),
        )
        metric_registry = MetricRegistry.model_validate(_load_yaml(root / "metric_registry.yaml"))
        embedding_runtime_contract = EmbeddingRuntimeContract.model_validate(
            _load_yaml(root / "embedding_runtime_contract.yaml")
        )
        fidelity_certificate = FidelityCertificate.model_validate(
            _load_json(root / "fidelity_certificate.json")
        )
    except (ValidationError, TypeError, ValueError) as exc:
        raise BehaviorRegistryError("REGISTRY_SCHEMA_INVALID") from exc
    bundle = BehaviorRegistryBundle(
        behavior_tests=behavior_tests,
        metric_registry=metric_registry,
        embedding_runtime_contract=embedding_runtime_contract,
        fidelity_certificate=fidelity_certificate,
    )
    validate_registry_bundle(bundle)
    return bundle


def validate_registry_bundle(bundle: BehaviorRegistryBundle) -> RegistryValidationReport:
    registry = bundle.behavior_tests
    row_ids = tuple(row.test_id for row in registry.rows)
    if len(set(row_ids)) != len(row_ids):
        raise BehaviorRegistryError("REGISTRY_DUPLICATE_ID")
    if set(row_ids) != set(REQUIRED_TEST_IDS) or len(row_ids) != len(REQUIRED_TEST_IDS):
        raise BehaviorRegistryError("BEHAVIOR_REGISTRY_INCOMPLETE")
    if registry.required_test_ids != REQUIRED_TEST_IDS:
        raise BehaviorRegistryError("BEHAVIOR_REGISTRY_INCOMPLETE")
    if registry.artifact_hash != _registry_hash(registry.rows):
        raise BehaviorRegistryError("REGISTRY_HASH_MISMATCH")
    for row in registry.rows:
        if row.row_hash != canonical_json_hash(row.model_dump(mode="json", exclude={"row_hash"})):
            raise BehaviorRegistryError("REGISTRY_HASH_MISMATCH")
        if _is_outcome_tuned(row.auxiliary_contract_refs):
            raise BehaviorRegistryError("OUTCOME_TUNED_REGISTRY")
    inv03 = next(row for row in registry.rows if row.test_id == "INV-03")
    if dict(inv03.auxiliary_contract_refs) != {
        "inv03_equivalence_registry_id": _INV03_REGISTRY_ID,
        "inv03_equivalence_registry_hash": _INV03_REGISTRY_HASH,
    }:
        raise BehaviorRegistryError("INV03_EQUIVALENCE_CONTRACT_MISMATCH")
    if bundle.fidelity_certificate.overall_status != "blocked":
        raise BehaviorRegistryError("F1C_STATUS_MISMATCH")
    return RegistryValidationReport(valid=True, test_ids=row_ids, bundle_hash=bundle.artifact_hash)


def _load_rows(path: Path) -> list[BehaviorTestRow]:
    try:
        payloads = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise BehaviorRegistryError("REGISTRY_READ_ERROR") from exc
    try:
        rows = [BehaviorTestRow.model_validate(payload) for payload in payloads]
    except ValidationError as exc:
        raise BehaviorRegistryError("REGISTRY_SCHEMA_INVALID") from exc
    return rows


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BehaviorRegistryError("REGISTRY_READ_ERROR") from exc
    if not isinstance(payload, dict):
        raise BehaviorRegistryError("REGISTRY_SCHEMA_INVALID")
    return payload


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BehaviorRegistryError("REGISTRY_READ_ERROR") from exc
    if not isinstance(payload, dict):
        raise BehaviorRegistryError("REGISTRY_SCHEMA_INVALID")
    return payload


def _registry_hash(rows: tuple[BehaviorTestRow, ...]) -> str:
    return canonical_json_hash(
        {
            "registry_id": _REGISTRY_ID,
            "schema_version": _SCHEMA_VERSION,
            "required_test_ids": REQUIRED_TEST_IDS,
            "rows": [row.model_dump(mode="json") for row in rows],
            "source_experiment_design_sha256": _SOURCE_DESIGN_HASH,
            "frozen_after_pilot_b": True,
        }
    )


def _is_outcome_tuned(refs: Mapping[str, str]) -> bool:
    return any(
        "outcome" in key.lower() or "outcome" in value.lower() for key, value in refs.items()
    )
