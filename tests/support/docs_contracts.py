import re
from dataclasses import dataclass
from pathlib import Path


_CONTRACT_LINE = re.compile(r"^- `(?P<kind>[a-z_]+):(?P<value>[^`]+)`$", re.MULTILINE)
_PROHIBITED_CLAIMS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bexact\s+(?:source|upstream(?:-method)?)\s+(?:reproduction|replica)\b",
        r"\b(?:phase-12\s+)?scientific\s+(?:result|results|finding|findings|outcome|outcomes)\b",
        r"\bf1c\s+(?:passes|passed)\s+in\s+this\s+checkout\b",
        r"\bretrieval\s+(?:is|equals|establishes|proves|counts\s+as)\s+(?:theory\s+)?exposure\b",
        r"\bexposure\s+(?:is|equals|establishes|proves|counts\s+as)\s+(?:operational\s+)?use\b",
        r"\b(?:text|text-only)\s+(?:and|with)\s+(?:code|python)\s+evidence\s+(?:is|are|was|were)\s+(?:pooled|combined|merged)\b",
        r"\bpaid\s+provider\s+(?:run|calls?)\s+(?:completed|performed|made)\b",
        r"\b(?:benchmark|manuscript)(?:-quality)?\s+(?:result|results|evidence|claim|claims)\b",
    )
)
_NEGATION = re.compile(
    r"\b(?:not|no|never|isn't|aren't|cannot|can't|doesn't|don't)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class DocumentedContractSet:
    entries: frozenset[tuple[str, str]]

    def values(self, kind: str) -> frozenset[str]:
        return frozenset(value for entry_kind, value in self.entries if entry_kind == kind)


def extract_documented_contracts(path: Path) -> DocumentedContractSet:
    text = path.read_text(encoding="utf-8")
    return DocumentedContractSet(
        frozenset((match["kind"], match["value"]) for match in _CONTRACT_LINE.finditer(text))
    )


def reject_overclaims(text: str) -> None:
    for pattern in _PROHIBITED_CLAIMS:
        for match in pattern.finditer(text):
            clause_start = max(text.rfind(separator, 0, match.start()) for separator in ".;")
            if not _NEGATION.search(text[clause_start + 1 : match.start()]):
                raise ValueError(f"PROHIBITED_PHASE12_CLAIM: {match.group(0)}")
