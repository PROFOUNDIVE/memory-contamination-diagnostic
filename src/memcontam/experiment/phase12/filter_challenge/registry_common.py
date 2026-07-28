from __future__ import annotations

import hashlib
import json
from typing import Annotated, TypeAlias, TypeVar

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


class RegistryValidationError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class StrictRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


_Item = TypeVar("_Item")


def parse_tuple(value: list[_Item] | tuple[_Item, ...]) -> tuple[_Item, ...]:
    match value:
        case list() | tuple():
            return tuple(value)
        case _:
            raise RegistryValidationError("SEQUENCE_REQUIRED")


def stable_hash(model: BaseModel, own_hash_field: str) -> str:
    payload = model.model_dump(mode="json")
    del payload[own_hash_field]
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_ids(ids: tuple[str, ...], empty_code: str, duplicate_code: str) -> None:
    if not ids:
        raise RegistryValidationError(empty_code)
    if len(set(ids)) != len(ids):
        raise RegistryValidationError(duplicate_code)


StringTuple: TypeAlias = Annotated[tuple[str, ...], BeforeValidator(parse_tuple)]
NonNegativeInt: TypeAlias = Annotated[int, Field(ge=0)]
PositiveInt: TypeAlias = Annotated[int, Field(gt=0)]
UnitInterval: TypeAlias = Annotated[float, Field(ge=0, le=1)]
