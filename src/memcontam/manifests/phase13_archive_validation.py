from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from memcontam.manifests.phase13_archive_authority import (
    ArchiveAuthorityError,
    load_archive_projection,
)
from memcontam.manifests.phase13_archive_models import Phase13Archive
from memcontam.manifests.phase13_archive_reconstruction import ReconstructionError, reconstruct_archive
from memcontam.manifests.phase13_archive_sources import SourceValidationError, validate_archive_sources


class Phase13ArchiveError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class Phase13ArchiveReport:
    archive_valid: bool
    resolved_edges: int
    claim_ids: tuple[str, ...]
    errors: tuple[Phase13ArchiveError, ...] = ()

    @property
    def reason_code(self) -> str | None:
        return None if not self.errors else self.errors[0].code

    def to_dict(self) -> dict[str, bool | int | str | None | list[str]]:
        return {
            "archive_valid": self.archive_valid,
            "resolved_edges": self.resolved_edges,
            "claim_ids": list(self.claim_ids),
            "reason_code": self.reason_code,
        }


def validate_phase13_archive(root: Path) -> Phase13ArchiveReport:
    try:
        archive = _read_archive(root)
        projection = load_archive_projection(archive.authorities)
        sources = validate_archive_sources(archive, projection)
        edges = reconstruct_archive(archive, projection, sources)
    except (
        Phase13ArchiveError,
        ArchiveAuthorityError,
        SourceValidationError,
        ReconstructionError,
    ) as error:
        wrapped = error if isinstance(error, Phase13ArchiveError) else Phase13ArchiveError(error.code)
        return Phase13ArchiveReport(False, 0, (), (wrapped,))
    return Phase13ArchiveReport(True, edges, tuple(row.claim_id for row in archive.claims))


def _read_archive(root: Path) -> Phase13Archive:
    try:
        return Phase13Archive.model_validate_json((root / "phase13_archive.json").read_bytes())
    except (OSError, ValidationError) as error:
        raise Phase13ArchiveError("PHASE13_ARCHIVE_SCHEMA_INVALID") from error


__all__ = ("Phase13ArchiveError", "Phase13ArchiveReport", "validate_phase13_archive")
