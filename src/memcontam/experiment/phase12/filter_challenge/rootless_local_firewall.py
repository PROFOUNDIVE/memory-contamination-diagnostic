from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final

from memcontam.experiment.phase12.filter_challenge.mft_state_models import JsonValue
from memcontam.experiment.phase12.filter_challenge.rootless_local_models import ROOTLESS_PROFILE


ROOTLESS_PROFILE_FORBIDDEN: Final = "ROOTLESS_PROFILE_FORBIDDEN"


class _JsonObject(dict[str, JsonValue]):
    def __init__(self, items: list[tuple[str, JsonValue]]) -> None:
        super().__init__(items)
        self.has_rootless_profile = any(
            key == "profile" and item == ROOTLESS_PROFILE for key, item in items
        )


def _object_pairs(items: list[tuple[str, JsonValue]]) -> _JsonObject:
    return _JsonObject(items)


def has_forbidden_rootless_profile(value: Mapping[str, JsonValue] | bytes | str) -> bool:
    if isinstance(value, Mapping):
        return value.get("profile") == ROOTLESS_PROFILE
    try:
        decoded: JsonValue | _JsonObject = json.loads(value, object_pairs_hook=_object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(decoded, _JsonObject) and decoded.has_rootless_profile


__all__ = ("ROOTLESS_PROFILE_FORBIDDEN", "has_forbidden_rootless_profile")
