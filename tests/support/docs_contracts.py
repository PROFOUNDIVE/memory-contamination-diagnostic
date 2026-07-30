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
        r"\b(?:establish(?:es|ed)?|report(?:s|ed)?|show(?:s|ed)?|demonstrat(?:es|ed)?|prove(?:s|d)?)\s+causal\s+effects?\b",
        r"\bcausal\s+effects?\s+(?:is|are|was|were)\s+(?:established|reported|shown|demonstrated|proven)\b",
        r"\b(?:is|are|was|were)\s+production[- ]ready\b",
        r"\bproduction\s+readiness\s+(?:is|are|was|were)\s+(?:claimed|present|established|confirmed|demonstrated|achieved)\b",
        r"\bpaid[- ]provider\s+(?:execution|run|calls?)\s+(?:(?:is|are|was|were)\s+)?(?:complete(?:d)?|performed|made|executed)\b",
        r"\b(?:pilot-[ab]|main)\s+evidence\s+(?:(?:is|are|was|were)\s+)?(?:established|reported|present|confirmed|available)\b",
        r"\bprovider\s+authorization\s+(?:(?:is|are|was|were)\s+)?(?:claimed|present|available|granted|confirmed|established)\b",
        r"\bcanonical[- ]patch\s+(?:is|are|was|were)\s+(?:complete|finished|applied)\b",
        r"\bcomplete\s+upstream(?:-method)?\s+(?:reproduction|replica)\b",
        r"\bdescendant\s+head\s+(?:is|are|was|were)\s+(?:certified|verified|validated)\b",
        r"\b(?:benchmark|manuscript)(?:-quality)?\s+(?:result|results|evidence|claim|claims)\b",
    )
)
_NEGATION = re.compile(
    r"\b(?:not|no|never|isn't|aren't|cannot|can't|doesn't|don't)\b", re.IGNORECASE
)
_ASSERTION_BOUNDARY = re.compile(r"[.!?:;]|\b(?:and|but|however|yet|while)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class DocumentedContractSet:
    entries: frozenset[tuple[str, str]]

    def values(self, kind: str) -> frozenset[str]:
        return frozenset(value for entry_kind, value in self.entries if entry_kind == kind)


class ProhibitedClaimError(ValueError):
    __slots__ = ("claim",)

    def __init__(self, claim: str) -> None:
        self.claim = claim
        super().__init__(f"PROHIBITED_PHASE12_CLAIM: {claim}")

    def __str__(self) -> str:
        return str(self.args[0])


def extract_documented_contracts(path: Path) -> DocumentedContractSet:
    text = path.read_text(encoding="utf-8")
    return DocumentedContractSet(
        frozenset((match["kind"], match["value"]) for match in _CONTRACT_LINE.finditer(text))
    )


def reject_overclaims(text: str) -> None:
    for pattern in _PROHIBITED_CLAIMS:
        for match in pattern.finditer(text):
            segment_start = max(
                (boundary.end() for boundary in _ASSERTION_BOUNDARY.finditer(text, 0, match.start())),
                default=0,
            )
            if not _NEGATION.search(text[segment_start : match.start()]):
                raise ProhibitedClaimError(match.group(0))
