from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from memcontam.experiment.phase12.filter_challenge.contracts import (
    OperationalProbeSuite,
    ProbeInventoryManifest,
)
from memcontam.experiment.phase12.filter_challenge.registry_common import (
    RegistryValidationError,
    StringTuple,
    stable_hash,
    validate_ids,
)


class ProbeInventoryRegistry(ProbeInventoryManifest):
    evidence_layer: Literal["build"]
    scientific_result: Literal[False]
    fixture_only: Literal[True]
    probe_ids: StringTuple
    inventory_frozen: Literal[False]

    @property
    def registry_id(self) -> str:
        return self.calibration_probe_inventory_id

    def stable_hash(self) -> str:
        return stable_hash(self, "calibration_probe_inventory_manifest_hash")

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        validate_ids(self.probe_ids, "PROBE_IDS_EMPTY", "PROBE_IDS_DUPLICATE")
        if self.calibration_probe_inventory_manifest_hash != self.stable_hash():
            raise RegistryValidationError("HASH_MISMATCH")
        return self


class OperationalSuiteRegistry(OperationalProbeSuite):
    evidence_layer: Literal["build"]
    scientific_result: Literal[False]
    fixture_only: Literal[True]
    suite_frozen: Literal[False]
    suite_ids: StringTuple

    @property
    def registry_id(self) -> str:
        return self.operational_probe_suite_id

    def stable_hash(self) -> str:
        return stable_hash(self, "operational_probe_suite_manifest_hash")

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        validate_ids(self.suite_ids, "SUITE_IDS_EMPTY", "SUITE_IDS_DUPLICATE")
        if self.operational_probe_suite_manifest_hash != self.stable_hash():
            raise RegistryValidationError("HASH_MISMATCH")
        return self
