from __future__ import annotations

import unicodedata
from typing import TypeAlias, assert_never

JsonValue: TypeAlias = (
    None | bool | int | str | list["JsonValue"] | dict[str, "JsonValue"]
)


class CanonicalJsonError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def canonical_json_bytes(value: JsonValue) -> bytes:
    return _serialize(value).encode("utf-8")


def _serialize(value: JsonValue) -> str:
    match value:
        case None:
            return "null"
        case bool() as boolean:
            return "true" if boolean else "false"
        case int() as integer:
            return str(integer)
        case str() as text:
            return _serialize_string(text)
        case list() as items:
            return "[" + ",".join(_serialize(item) for item in items) + "]"
        case dict() as members:
            normalized = [(unicodedata.normalize("NFC", key), item) for key, item in members.items()]
            if len({key for key, _ in normalized}) != len(normalized):
                raise CanonicalJsonError("CANONICAL_JSON_NFC_KEY_COLLISION")
            normalized.sort(key=lambda pair: pair[0].encode("utf-8"))
            return "{" + ",".join(
                f"{_serialize_string(key)}:{_serialize(item)}" for key, item in normalized
            ) + "}"
        case unreachable:
            assert_never(unreachable)


def _serialize_string(value: str) -> str:
    escaped: list[str] = []
    for character in unicodedata.normalize("NFC", value):
        if character == '"':
            escaped.append('\\"')
        elif character == "\\":
            escaped.append("\\\\")
        elif character == "\b":
            escaped.append("\\b")
        elif character == "\t":
            escaped.append("\\t")
        elif character == "\n":
            escaped.append("\\n")
        elif character == "\f":
            escaped.append("\\f")
        elif character == "\r":
            escaped.append("\\r")
        elif ord(character) < 0x20:
            escaped.append(f"\\u{ord(character):04x}")
        else:
            escaped.append(character)
    return '"' + "".join(escaped) + '"'


__all__ = ["CanonicalJsonError", "JsonValue", "canonical_json_bytes"]
