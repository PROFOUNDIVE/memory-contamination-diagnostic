from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from .support.docs_contracts import ProhibitedClaimError, reject_overclaims

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
STATUS = ROOT / "docs" / "historical" / "phase12-filter-v5-build-status.md"
EVIDENCE = ROOT / ".sisyphus" / "evidence" / "phase12-filter-v5-build-v1"
MANIFEST = EVIDENCE / "implementation_manifest.json"


FORBIDDEN_OVERCLAIMS: Final[tuple[tuple[str, str], ...]] = (
    ("This reports scientific results.", "scientific results"),
    ("This establishes benchmark evidence.", "benchmark evidence"),
    ("This establishes manuscript evidence.", "manuscript evidence"),
    ("This establishes causal effects.", "establishes causal effects"),
    ("The software is production ready.", "is production ready"),
    ("A paid provider run completed.", "paid provider run completed"),
    ("Pilot-A evidence is established.", "Pilot-A evidence is established"),
    ("Pilot-B evidence is established.", "Pilot-B evidence is established"),
    ("Main evidence is established.", "Main evidence is established"),
    ("Provider authorization is present.", "Provider authorization is present"),
    ("The canonical patch is complete.", "canonical patch is complete"),
    ("This is a complete upstream reproduction.", "complete upstream reproduction"),
    ("The descendant HEAD is certified.", "descendant HEAD is certified"),
)

BOUNDED_NON_CLAIMS: Final[tuple[str, ...]] = (
    "No scientific results are reported.",
    "No benchmark evidence is reported.",
    "No manuscript evidence is reported.",
    "No causal effects are established.",
    "Production readiness is not claimed.",
    "No paid provider run completed.",
    "No Pilot-A evidence is reported.",
    "No Pilot-B evidence is reported.",
    "No Main evidence is reported.",
    "No provider authorization is claimed.",
    "The canonical patch is not complete.",
    "This is not a complete upstream reproduction.",
    "The descendant HEAD is not certified.",
)


@contextmanager
def _passthrough_context() -> Iterator[None]:
    yield


@pytest.mark.parametrize(("claim", "fragment"), FORBIDDEN_OVERCLAIMS)
def test_reject_overclaims_rejects_affirmative_forbidden_claims(
    claim: str, fragment: str
) -> None:
    with pytest.raises(ProhibitedClaimError) as caught:
        reject_overclaims(claim)
    assert caught.value.claim == fragment


def test_reject_overclaims_does_not_let_an_earlier_negation_excuse_a_later_claim() -> None:
    mixed_claim = "This is not a prototype, and benchmark evidence is established."

    with pytest.raises(ProhibitedClaimError) as caught:
        reject_overclaims(mixed_claim)
    assert caught.value.claim == "benchmark evidence"


def test_prohibited_claim_error_survives_stdlib_contextmanager() -> None:
    with pytest.raises(ProhibitedClaimError) as caught:
        with _passthrough_context():
            reject_overclaims("This reports scientific results.")
    assert caught.value.claim == "scientific results"


def test_contextmanager_passes_through_normal_exception() -> None:
    with pytest.raises(ZeroDivisionError):
        with _passthrough_context():
            _ = 1 / 0


@pytest.mark.parametrize("claim", BOUNDED_NON_CLAIMS)
def test_reject_overclaims_accepts_bounded_negative_claims(claim: str) -> None:
    reject_overclaims(claim)


@pytest.mark.parametrize("path", (README, STATUS), ids=("README.md", "status.md"))
def test_entrypoint_docs_reject_overclaims_directly(path: Path) -> None:
    reject_overclaims(path.read_text(encoding="utf-8"))


def test_readme_routes_filter_evidence_to_history() -> None:
    text = README.read_text(encoding="utf-8")
    assert "docs/historical/README.md" in text
    assert "docs/historical/phase12-filter-v5-build-status.md" not in text


def test_sealed_evidence_inventory_and_report_hashes_match_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    reports = manifest["reports"]
    expected_files = {"implementation_manifest.json", *reports}

    assert len(expected_files) == 9
    assert {path.name for path in EVIDENCE.iterdir()} == expected_files
    for filename, expected_digest in reports.items():
        assert hashlib.sha256((EVIDENCE / filename).read_bytes()).hexdigest() == expected_digest
