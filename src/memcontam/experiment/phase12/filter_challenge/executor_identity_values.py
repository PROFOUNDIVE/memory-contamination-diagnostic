from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import singledispatch
from hashlib import sha256
from types import FunctionType, MethodType
from typing import Any


def held_fixed_config(config: Mapping[str, Any], excluded: frozenset[str]) -> tuple:
    return tuple(
        sorted(
            (key, canonical_identity_value(value))
            for key, value in config.items()
            if key not in excluded
        )
    )


def selected_config(config: Mapping[str, Any], keys: tuple[str, ...]) -> tuple:
    return tuple(
        (key, canonical_identity_value(config[key])) for key in keys if key in config
    )


def service_contract_identity(value: Any) -> Any:
    if value is None:
        return None
    metadata = getattr(value, "metadata", None)
    embedding_contract = getattr(value, "embedding_contract", None)
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "metadata": canonical_identity_value(metadata) if isinstance(metadata, Mapping) else None,
        "embedding_contract": (
            canonical_identity_value(embedding_contract)
            if isinstance(embedding_contract, Mapping)
            else None
        ),
    }


@singledispatch
def canonical_identity_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(
            (field.name, canonical_identity_value(getattr(value, field.name)))
            for field in fields(value)
        )
    return (
        "instance",
        type(value).__module__,
        type(value).__qualname__,
    )


@canonical_identity_value.register(type(None))
def _canonical_none(value: None) -> None:
    return value


@canonical_identity_value.register(bool)
@canonical_identity_value.register(int)
@canonical_identity_value.register(float)
@canonical_identity_value.register(str)
def _canonical_scalar(value: bool | int | float | str) -> bool | int | float | str:
    return value


@canonical_identity_value.register(Mapping)
def _canonical_mapping(value: Mapping[Any, Any]) -> tuple:
    return tuple(
        sorted(
            (str(key), canonical_identity_value(item)) for key, item in value.items()
        )
    )


@canonical_identity_value.register(list)
@canonical_identity_value.register(tuple)
def _canonical_sequence(value: list[Any] | tuple[Any, ...]) -> tuple:
    return tuple(canonical_identity_value(item) for item in value)


@canonical_identity_value.register(set)
@canonical_identity_value.register(frozenset)
def _canonical_set(value: set[Any] | frozenset[Any]) -> tuple:
    return tuple(sorted((canonical_identity_value(item) for item in value), key=repr))


@canonical_identity_value.register(FunctionType)
@canonical_identity_value.register(MethodType)
def _canonical_callable(value: FunctionType | MethodType) -> tuple:
    code = value.__code__
    closure = tuple(
        canonical_identity_value(cell.cell_contents) for cell in (value.__closure__ or ())
    )
    return (
        "callable",
        value.__module__,
        value.__qualname__,
        code.co_filename,
        code.co_firstlineno,
        sha256(code.co_code).hexdigest(),
        canonical_identity_value(code.co_consts),
        canonical_identity_value(value.__defaults__),
        canonical_identity_value(getattr(value, "__self__", None)),
        closure,
    )
