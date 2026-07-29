from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


FinalVerifierMode = Literal["plan-compliance", "code-quality", "integration", "scope", "terminal"]


class FinalVerifierError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)

@dataclass(frozen=True, slots=True)
class FinalVerifierRequest:
    mode: FinalVerifierMode
    repository_root: Path
    plan: Path
    expected_plan_sha256: str
    evidence_root: Path
    validation_summary: Path
    output: Path
    approval_paths: tuple[Path, ...]
    base_commit: str | None = None
    execution_prerequisites: Path | None = None
    fixture_root: Path | None = None
    scratch_root: Path | None = None
    search_config: Path | None = None
    source_repository_root: Path | None = None
