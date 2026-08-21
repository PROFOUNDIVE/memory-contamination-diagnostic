from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping

from pydantic import JsonValue

UNICODE_VERSION: Final = "15.1.0"
_CASE_FOLDING_DATA_SHA256: Final = (
    "4e55acfdc32825a22e87670e9056a3bf94ad7c5400065778e9e10f8314372bcf"
)
_CASE_FOLDING_SEMANTIC_SHA256: Final = (
    "661466e49c100e00238e2bde53b9b6895cc82ff63dbeb5f2a7dace01c779b0fb"
)
_WHITE_SPACE_DATA_SHA256: Final = (
    "05672956317b6296bc2ec3d6cef1f6452b57ff4f2efc6dc55b0a19277d5fcfd1"
)
_WHITE_SPACE: Final = frozenset(
    "\u0009\u000a\u000b\u000c\u000d\u0020\u0085\u00a0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)
_CONFORMANCE_VECTORS: Final = (
    ("Ａ\u00a0B", "a b", ("a", "b")),
    ("İ", "i̇", ("i̇",)),
    ("́A—٢", "́a—٢", ("a", "٢")),
    (" ◌́ Á １２ ", "◌́ á 12", ("á", "12")),
)
_unicode_data = importlib.import_module("unicodedata2")


class UnicodeContractError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class UnicodeProvenance:
    unicode_data_version: str
    unicode_data_manifest_hash: str
    case_folding_data_sha256: str
    case_folding_semantic_sha256: str
    white_space_data_sha256: str
    executable_source_sha256: str
    conformance_vectors_sha256: str
    conformance_vector_count: int


def mcq_normalize(text: str) -> str:
    version = getattr(_unicode_data, "unidata_version", None)
    if version != UNICODE_VERSION:
        raise UnicodeContractError("NEW_MCQ_UNICODE_VERSION_MISMATCH")
    normalize = getattr(_unicode_data, "normalize", None)
    if not callable(normalize):
        raise UnicodeContractError("NEW_MCQ_UNICODE_DATA_UNAVAILABLE")
    normalized = normalize("NFKC", text)
    if not isinstance(normalized, str):
        raise UnicodeContractError("NEW_MCQ_UNICODE_DATA_UNAVAILABLE")
    folded = _casefold_15_1(normalized)
    spaced = "".join(" " if character in _WHITE_SPACE else character for character in folded)
    return " ".join(part for part in spaced.split(" ") if part)


def mcq_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    current: list[str] = []
    for character in mcq_normalize(text):
        category = _unicode_data.category(character)
        if category[0] in {"L", "N"} or (category[0] == "M" and current):
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def unicode_provenance() -> UnicodeProvenance:
    if any(
        mcq_normalize(text) != normalized or mcq_tokens(text) != tokens
        for text, normalized, tokens in _CONFORMANCE_VECTORS
    ):
        raise UnicodeContractError("NEW_MCQ_UNICODE_CONFORMANCE_MISMATCH")
    module_file = getattr(_unicode_data, "__file__", None)
    if not isinstance(module_file, str):
        raise UnicodeContractError("NEW_MCQ_UNICODE_DATA_UNAVAILABLE")
    module_path = Path(module_file)
    module_bytes = module_path.read_bytes()
    white_space_code_points: list[JsonValue] = [
        ord(character) for character in sorted(_WHITE_SPACE)
    ]
    manifest: dict[str, JsonValue] = {
        "unicode_data_version": UNICODE_VERSION,
        "module_filename": module_path.name,
        "module_sha256": hashlib.sha256(module_bytes).hexdigest(),
        "module_size": len(module_bytes),
        "case_folding_data_sha256": _CASE_FOLDING_DATA_SHA256,
        "case_folding_semantic_sha256": _casefold_semantic_sha256(),
        "white_space_data_sha256": _WHITE_SPACE_DATA_SHA256,
        "white_space_code_points": white_space_code_points,
    }
    vectors: list[JsonValue] = [
        {"input": text, "normalized": normalized, "tokens": list(tokens)}
        for text, normalized, tokens in _CONFORMANCE_VECTORS
    ]
    return UnicodeProvenance(
        unicode_data_version=UNICODE_VERSION,
        unicode_data_manifest_hash=_json_hash(manifest),
        case_folding_data_sha256=_CASE_FOLDING_DATA_SHA256,
        case_folding_semantic_sha256=_CASE_FOLDING_SEMANTIC_SHA256,
        white_space_data_sha256=_WHITE_SPACE_DATA_SHA256,
        executable_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        conformance_vectors_sha256=_json_hash(vectors),
        conformance_vector_count=len(vectors),
    )


@cache
def _casefold_table() -> Mapping[int, str]:
    table = {
        code_point: folded
        for code_point in range(0x110000)
        if (folded := chr(code_point).casefold()) != chr(code_point)
    }
    payload: list[JsonValue] = [
        [code_point, [ord(character) for character in table[code_point]]]
        for code_point in sorted(table)
    ]
    if _json_hash(payload) != _CASE_FOLDING_SEMANTIC_SHA256:
        raise UnicodeContractError("NEW_MCQ_UNICODE_CASE_FOLDING_MISMATCH")
    return MappingProxyType(table)


def _casefold_15_1(text: str) -> str:
    table = _casefold_table()
    return "".join(table.get(ord(character), character) for character in text)


def _casefold_semantic_sha256() -> str:
    _casefold_table()
    return _CASE_FOLDING_SEMANTIC_SHA256


def _json_hash(value: JsonValue) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "UNICODE_VERSION",
    "UnicodeContractError",
    "UnicodeProvenance",
    "mcq_normalize",
    "mcq_tokens",
    "unicode_provenance",
]
