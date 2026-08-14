from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from memcontam.readiness.phase13_authority import Identifier, Sha256
from memcontam.readiness.phase13_authority_files import AuthorityFileError, read_regular_nofollow


class Phase13ProvenanceError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProvenanceArtifact(_StrictModel):
    role: Identifier
    path: Annotated[str, Field(min_length=1)]
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def _relative_path(cls, value: str) -> str:
        if value.startswith("/") or any(part in {".", ".."} for part in value.split("/")):
            raise Phase13ProvenanceError("ARTIFACT_PATH_INVALID")
        return value


class ProvenanceManifest(_StrictModel):
    schema_version: Annotated[str, Field(pattern=r"^phase13_provenance_manifest_v[0-9]+$")]
    bundle_id: Identifier
    artifacts: tuple[ProvenanceArtifact, ...]
    manifest_hash: Sha256

    @field_validator("artifacts", mode="before")
    @classmethod
    def _artifacts(cls, value: list[dict[str, str]]) -> tuple[dict[str, str], ...]:
        return tuple(value)

    @model_validator(mode="after")
    def _closed_manifest(self) -> ProvenanceManifest:
        roles = tuple(row.role for row in self.artifacts)
        paths = tuple(row.path for row in self.artifacts)
        if len(roles) != len(set(roles)) or len(paths) != len(set(paths)):
            raise Phase13ProvenanceError("DUPLICATE_ARTIFACT")
        payload = self.model_dump(mode="json", exclude={"manifest_hash"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(canonical).hexdigest() != self.manifest_hash:
            raise Phase13ProvenanceError("MANIFEST_HASH_MISMATCH")
        return self


class ProvenanceSeal(_StrictModel):
    schema_version: Annotated[str, Field(pattern=r"^phase13_provenance_seal_v[0-9]+$")]
    bundle_id: Identifier
    manifest_hash: Sha256


@dataclass(frozen=True, slots=True)
class ProvenanceReport:
    bundle_id: str
    artifact_count: int
    manifest_hash: str


def validate_provenance_bundle(
    root: Path,
    manifest_path: Path,
    seal_path: Path,
) -> ProvenanceReport:
    try:
        manifest = ProvenanceManifest.model_validate_json(read_regular_nofollow(manifest_path))
        seal = ProvenanceSeal.model_validate_json(read_regular_nofollow(seal_path))
    except AuthorityFileError as error:
        raise Phase13ProvenanceError(error.code) from error
    except ValidationError as error:
        message = str(error)
        for code in (
            "ARTIFACT_PATH_INVALID",
            "DUPLICATE_ARTIFACT",
            "MANIFEST_HASH_MISMATCH",
        ):
            if code in message:
                raise Phase13ProvenanceError(code) from error
        raise Phase13ProvenanceError("PROVENANCE_SCHEMA_INVALID") from error
    if (seal.bundle_id, seal.manifest_hash) != (manifest.bundle_id, manifest.manifest_hash):
        raise Phase13ProvenanceError("SEAL_MISMATCH")
    for artifact in manifest.artifacts:
        try:
            raw = read_regular_nofollow(root / artifact.path)
        except AuthorityFileError as error:
            raise Phase13ProvenanceError(error.code) from error
        if hashlib.sha256(raw).hexdigest() != artifact.sha256:
            raise Phase13ProvenanceError("ARTIFACT_HASH_MISMATCH")
    return ProvenanceReport(manifest.bundle_id, len(manifest.artifacts), manifest.manifest_hash)
