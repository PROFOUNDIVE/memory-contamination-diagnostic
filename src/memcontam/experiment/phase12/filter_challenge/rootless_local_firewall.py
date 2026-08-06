from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Final

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.experiment.phase12.filter_challenge.rootless_local_models import (
    ROOTLESS_PROFILE,
    RootlessLocalReceipt,
)


ROOTLESS_PROFILE_FORBIDDEN: Final = "ROOTLESS_PROFILE_FORBIDDEN"
_YAML_ROOTLESS_PROFILE: Final = re.compile(
    rf"(?m)^profile[ \t]*:[ \t]*[\"']?{ROOTLESS_PROFILE}[\"']?[ \t]*(?:#.*)?\r?$"
)


class _JsonObject(dict[str, JsonValue]):
    def __init__(self, items: list[tuple[str, JsonValue]]) -> None:
        super().__init__(items)
        self.has_rootless_profile = any(
            key == "profile" and item == ROOTLESS_PROFILE for key, item in items
        )


def _object_pairs(items: list[tuple[str, JsonValue]]) -> _JsonObject:
    return _JsonObject(items)


def has_forbidden_rootless_profile(
    value: RootlessLocalReceipt | Mapping[str, JsonValue] | bytes | str,
) -> bool:
    if isinstance(value, RootlessLocalReceipt):
        return True
    if isinstance(value, Mapping):
        return value.get("profile") == ROOTLESS_PROFILE
    try:
        decoded: JsonValue | _JsonObject = json.loads(value, object_pairs_hook=_object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError):
        try:
            text = value.decode("utf-8") if isinstance(value, bytes) else value
        except UnicodeDecodeError:
            return False
        return _YAML_ROOTLESS_PROFILE.search(text) is not None
    return isinstance(decoded, _JsonObject) and decoded.has_rootless_profile


def has_forbidden_rootless_aggregate_profile(value: bytes | str) -> bool:
    try:
        decoded: JsonValue | _JsonObject = json.loads(value, object_pairs_hook=_object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    records = decoded
    if isinstance(decoded, _JsonObject):
        records = decoded.get("rows", decoded.get("aggregates", decoded.get("claims")))
    return isinstance(records, list) and any(
        isinstance(record, _JsonObject) and record.has_rootless_profile for record in records
    )


__all__ = (
    "ROOTLESS_PROFILE_FORBIDDEN",
    "has_forbidden_rootless_aggregate_profile",
    "has_forbidden_rootless_profile",
)
