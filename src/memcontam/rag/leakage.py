from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from memcontam.rag.phase12_corpus import BranchCorpusSet


class RagContractError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class LeakageReport:
    codes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.codes

    def require_clean(self) -> None:
        if self.codes:
            raise RagContractError(self.codes[0])


def audit_leakage(corpus: BranchCorpusSet, split_manifest: Mapping[str, Any]) -> LeakageReport:
    codes = []
    if split_manifest.get("corpus_hash") != corpus.clean.content_hash:
        codes.append("STALE_CORPUS_MANIFEST")
    splits = split_manifest.get("splits", {})
    if not isinstance(splits, Mapping):
        codes.append("CORPUS_LEAKAGE")
    elif _has_cross_split_duplicate(splits):
        codes.append("CORPUS_LEAKAGE")
    return LeakageReport(tuple(codes))


def validate_rag_frozen_inputs(
    corpus: BranchCorpusSet,
    split_manifest: Mapping[str, Any],
    *,
    affinity_band: str = "mid",
    rag_mode: str = "frozen",
    input_surfaces: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    audit_leakage(corpus, split_manifest).require_clean()
    if affinity_band == "extreme":
        raise RagContractError("AFFINITY_BAND_FORBIDDEN")
    if rag_mode not in {"frozen", "rag_frozen"}:
        raise RagContractError("RAG_MODE_MISMATCH")
    for surface, value in (input_surfaces or {}).items():
        if _contains_inv03_metadata(value):
            code = {
                "embedding": "INV03_METADATA_REACHED_EMBEDDING",
                "renderer": "INV03_METADATA_REACHED_RENDERER",
                "filter": "INV03_METADATA_REACHED_FILTER",
                "verifier": "INV03_METADATA_REACHED_VERIFIER",
                "ranking": "INV03_RANKING_CHANGED",
            }.get(surface, "INV03_METADATA_LEAKAGE")
            raise RagContractError(code)


def _has_cross_split_duplicate(splits: Mapping[str, Any]) -> bool:
    seen: set[str] = set()
    for records in splits.values():
        if not isinstance(records, list):
            return True
        fingerprints = {_fingerprint(record) for record in records if isinstance(record, Mapping)}
        if len(fingerprints) != len(records) or seen & fingerprints:
            return True
        seen.update(fingerprints)
    return False


def _fingerprint(record: Mapping[str, Any]) -> str:
    text = record.get("text", record.get("content"))
    if not isinstance(text, str):
        return ""
    return " ".join(text.casefold().split())


def _contains_inv03_metadata(value: Any) -> bool:
    if isinstance(value, Mapping):
        return "inv03_metadata" in value or any(
            _contains_inv03_metadata(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_inv03_metadata(item) for item in value)
    return False
