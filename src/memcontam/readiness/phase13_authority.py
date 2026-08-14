from __future__ import annotations

import hashlib
import json
from typing import Annotated, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
ParameterValue: TypeAlias = bool | int | float | str | tuple[str, ...]


class Phase13AuthorityError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArtifactRef(_StrictModel):
    path: Annotated[str, Field(min_length=1)]
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def _relative_path(cls, value: str) -> str:
        if value.startswith("/") or any(part in {".", ".."} for part in value.split("/")):
            raise Phase13AuthorityError("MALFORMED_REFERENCE")
        return value


class AuthorityRef(_StrictModel):
    role: Identifier
    artifact: ArtifactRef


class RegistryRef(_StrictModel):
    kind: Identifier
    registry_id: Identifier
    artifact: ArtifactRef


class Phase13AuthorityRequirements(_StrictModel):
    schema_version: Annotated[str, Field(pattern=r"^phase13_authority_requirements_v[0-9]+$")]
    authority_hashes: dict[Identifier, Sha256]
    registry_kinds: tuple[Identifier, ...]
    parameter_names: tuple[Identifier, ...]

    @model_validator(mode="after")
    def _unique_requirements(self) -> Phase13AuthorityRequirements:
        if len(self.registry_kinds) != len(set(self.registry_kinds)):
            raise Phase13AuthorityError("DUPLICATE_REGISTRY_KIND")
        if len(self.parameter_names) != len(set(self.parameter_names)):
            raise Phase13AuthorityError("DUPLICATE_PARAMETER_NAME")
        return self


class Phase13AuthorityFreeze(_StrictModel):
    schema_version: Annotated[str, Field(pattern=r"^phase13_authority_freeze_v[0-9]+$")]
    freeze_id: Identifier
    authorities: tuple[AuthorityRef, ...]
    registries: tuple[RegistryRef, ...]
    parameters: dict[Identifier, ParameterValue]
    closure_hash: Sha256

    @model_validator(mode="after")
    def _internally_closed(self) -> Phase13AuthorityFreeze:
        roles = tuple(row.role for row in self.authorities)
        kinds = tuple(row.kind for row in self.registries)
        registry_ids = tuple(row.registry_id for row in self.registries)
        paths = tuple(row.artifact.path for row in self.authorities) + tuple(
            row.artifact.path for row in self.registries
        )
        if len(roles) != len(set(roles)):
            raise Phase13AuthorityError("DUPLICATE_AUTHORITY_ROLE")
        if len(kinds) != len(set(kinds)):
            raise Phase13AuthorityError("DUPLICATE_REGISTRY_KIND")
        if len(registry_ids) != len(set(registry_ids)):
            raise Phase13AuthorityError("DUPLICATE_REGISTRY_ID")
        if len(paths) != len(set(paths)):
            raise Phase13AuthorityError("DUPLICATE_ARTIFACT_PATH")
        payload = self.model_dump(mode="json", exclude={"closure_hash"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(canonical).hexdigest() != self.closure_hash:
            raise Phase13AuthorityError("CLOSURE_HASH_MISMATCH")
        return self


def parse_authority_requirements(raw_json: bytes | str) -> Phase13AuthorityRequirements:
    try:
        return Phase13AuthorityRequirements.model_validate_json(raw_json)
    except ValidationError as error:
        raise Phase13AuthorityError("MALFORMED_REQUIREMENTS") from error


def parse_authority_freeze(
    raw_json: bytes | str,
    requirements: Phase13AuthorityRequirements,
) -> Phase13AuthorityFreeze:
    try:
        freeze = Phase13AuthorityFreeze.model_validate_json(raw_json)
    except ValidationError as error:
        message = str(error)
        for code in (
            "CLOSURE_HASH_MISMATCH",
            "DUPLICATE_AUTHORITY_ROLE",
            "DUPLICATE_REGISTRY_KIND",
            "DUPLICATE_REGISTRY_ID",
            "DUPLICATE_ARTIFACT_PATH",
            "MALFORMED_REFERENCE",
        ):
            if code in message:
                raise Phase13AuthorityError(code) from error
        raise Phase13AuthorityError("MALFORMED_FREEZE") from error
    authority_hashes = {row.role: row.artifact.sha256 for row in freeze.authorities}
    if authority_hashes != requirements.authority_hashes:
        raise Phase13AuthorityError("AUTHORITY_SET_MISMATCH")
    if set(row.kind for row in freeze.registries) != set(requirements.registry_kinds):
        raise Phase13AuthorityError("REGISTRY_SET_MISMATCH")
    if set(freeze.parameters) != set(requirements.parameter_names):
        raise Phase13AuthorityError("PARAMETER_SET_MISMATCH")
    return freeze
