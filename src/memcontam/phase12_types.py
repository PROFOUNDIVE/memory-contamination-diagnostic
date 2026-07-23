from collections.abc import Mapping
from typing import Literal

RunFamily = Literal[
    "readiness",
    "pilot_a",
    "pilot_b",
    "behavioral",
    "main_a",
    "main_b",
    "main_c",
    "sequential",
    "extension",
    "exploratory_code",
]

CanonicalRunFamily = Literal["readiness", "pilot_a", "pilot_b", "main", "exploratory_code"]
CANONICAL_RUN_FAMILY_MEMBERS: Mapping[CanonicalRunFamily, tuple[RunFamily, ...]] = {
    "readiness": ("readiness",),
    "pilot_a": ("pilot_a",),
    "pilot_b": ("pilot_b",),
    "main": ("main_a", "main_b", "main_c"),
    "exploratory_code": ("exploratory_code",),
}
