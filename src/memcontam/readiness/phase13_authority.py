from __future__ import annotations

import hashlib
import json
from typing import Annotated, Final, Literal, TypeAlias, assert_never

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
ArtifactPath = Annotated[str, Field(min_length=1)]
AuthorityRole = Literal["theory", "baseline", "protocol", "experiment_design"]
RegistryKind = Literal["calibration_v2", "execution", "analysis"]
JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)

EXPECTED_AUTHORITY_HASHES: Final[dict[AuthorityRole, str]] = {
    "theory": "34f63f37a49e92607c78ced038c4c70b4c9d5e3fa8fc57d6e97de1ee79db59a8",
    "baseline": "c28f0e2b00db6a2731f64933ccc67c5ea5a163d6233c526e6b473e540f988204",
    "protocol": "06d23e29dff6c607bc2035c5641fbb696fb5c09dd86f2ce190a99c6baa57eefc",
    "experiment_design": "6b8ab4e414c86dbcb4afc9c2781b13f9312e8ba2834d20473d261f264e6e1acf",
}
EXPECTED_REGISTRIES: Final[frozenset[RegistryKind]] = frozenset(
    {"calibration_v2", "execution", "analysis"}
)


class Phase13AuthorityError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ArtifactRef(_StrictModel):
    kind: Literal["artifact"]
    artifact_id: Identifier
    path: ArtifactPath
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        if value.startswith("/") or ".." in value.split("/"):
            raise Phase13AuthorityError("MALFORMED_REFERENCE")
        return value


class AuthorityRef(_StrictModel):
    kind: Literal["authority"]
    authority_role: AuthorityRole
    artifact: ArtifactRef


class RegistryRef(_StrictModel):
    kind: Literal["registry"]
    registry_kind: RegistryKind
    registry_id: Identifier
    artifact: ArtifactRef


class ScientificDesignClassification(_StrictModel):
    kind: Literal["scientific_design"]
    class_code: Literal["A"]
    H_primary: Annotated[int, Field(gt=0)]
    primary_analysis_window_id: Identifier


class ExecutionClassification(_StrictModel):
    kind: Literal["execution"]
    class_code: Literal["B"]
    H_run: Annotated[int, Field(gt=0)]


class InferenceClassification(_StrictModel):
    kind: Literal["inference"]
    class_code: Literal["C"]
    estimator_id: Identifier


class PlanningClassification(_StrictModel):
    kind: Literal["planning"]
    class_code: Literal["D"]
    calibration_seed_count_per_task: Annotated[int, Field(gt=0)]


class ReproducibilityClassification(_StrictModel):
    kind: Literal["reproducibility"]
    class_code: Literal["E"]
    bootstrap_replicates: Annotated[int, Field(gt=0)]
    bootstrap_rng_seed: Annotated[int, Field(ge=0)]
    serialization_version: Identifier


ParameterClassification = Annotated[
    ScientificDesignClassification
    | ExecutionClassification
    | InferenceClassification
    | PlanningClassification
    | ReproducibilityClassification,
    Field(discriminator="kind"),
]


def _contains_bare_h(value: JsonValue) -> bool:
    match value:
        case dict() as mapping:
            return "H" in mapping or any(_contains_bare_h(item) for item in mapping.values())
        case list() as items:
            return any(_contains_bare_h(item) for item in items)
        case str() | int() | float() | bool() | None:
            return False
        case unreachable:
            assert_never(unreachable)


class Phase13AuthorityFreeze(_StrictModel):
    schema_version: Literal["phase13-authority-freeze-v1"]
    closure_id: Identifier
    authorities: tuple[AuthorityRef, ...]
    parameter_classifications: tuple[ParameterClassification, ...]
    registries: tuple[RegistryRef, ...]
    closure_hash: Sha256

    @field_validator("authorities", "parameter_classifications", "registries", mode="before")
    @classmethod
    def _parse_tuple(cls, value: list[JsonValue]) -> tuple[JsonValue, ...]:
        return tuple(value)

    @model_validator(mode="before")
    @classmethod
    def _reject_bare_h(cls, value: JsonValue) -> JsonValue:
        if _contains_bare_h(value):
            raise Phase13AuthorityError("BARE_H_PROHIBITED")
        return value

    @model_validator(mode="after")
    def _validate_closure(self) -> Phase13AuthorityFreeze:
        authority_roles = [reference.authority_role for reference in self.authorities]
        if len(authority_roles) != len(set(authority_roles)) or set(authority_roles) != set(
            EXPECTED_AUTHORITY_HASHES
        ):
            raise Phase13AuthorityError("FULL_CLOSURE_REQUIRED")
        if any(
            reference.artifact.sha256 != EXPECTED_AUTHORITY_HASHES[reference.authority_role]
            for reference in self.authorities
        ):
            raise Phase13AuthorityError("AUTHORITY_HASH_DRIFT")

        registry_kinds = [reference.registry_kind for reference in self.registries]
        if len(registry_kinds) != len(set(registry_kinds)):
            raise Phase13AuthorityError("DUPLICATE_REGISTRY")
        if set(registry_kinds) != EXPECTED_REGISTRIES:
            raise Phase13AuthorityError("FULL_CLOSURE_REQUIRED")

        classes: set[str] = set()
        for classification in self.parameter_classifications:
            match classification:
                case ScientificDesignClassification():
                    classes.add("A")
                case ExecutionClassification():
                    classes.add("B")
                case InferenceClassification():
                    classes.add("C")
                case PlanningClassification():
                    classes.add("D")
                case ReproducibilityClassification():
                    classes.add("E")
                case unreachable:
                    assert_never(unreachable)
        if classes != {"A", "B", "C", "D", "E"} or len(self.parameter_classifications) != 5:
            raise Phase13AuthorityError("FULL_CLOSURE_REQUIRED")

        payload = self.model_dump(mode="json", exclude={"closure_hash"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(canonical).hexdigest() != self.closure_hash:
            raise Phase13AuthorityError("CLOSURE_HASH_MISMATCH")
        return self


def _validation_code(error: ValidationError) -> str:
    issue = error.errors()[0]
    location = tuple(str(item) for item in issue["loc"])
    message = str(issue["msg"])
    for code in (
        "BARE_H_PROHIBITED",
        "FULL_CLOSURE_REQUIRED",
        "AUTHORITY_HASH_DRIFT",
        "DUPLICATE_REGISTRY",
        "CLOSURE_HASH_MISMATCH",
        "MALFORMED_REFERENCE",
    ):
        if code in message:
            return code
    if "registry_kind" in location:
        return "UNKNOWN_REGISTRY"
    if "class_code" in location:
        return "WRONG_PARAMETER_CLASS"
    if "reproducibility" in location:
        return "MISSING_E_SETTINGS"
    if "scientific_design" in location and "H_primary" in location:
        return "MISSING_H_PRIMARY"
    if location == ("closure_hash",):
        return "FULL_CLOSURE_REQUIRED"
    if "artifact" in location or "sha256" in location:
        return "MALFORMED_REFERENCE"
    return "MALFORMED_CLOSURE"


def parse_phase13_authority_freeze(raw_json: bytes | str) -> Phase13AuthorityFreeze:
    try:
        return Phase13AuthorityFreeze.model_validate_json(raw_json)
    except ValidationError as error:
        raise Phase13AuthorityError(_validation_code(error)) from error
