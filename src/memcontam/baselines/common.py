from __future__ import annotations


def parse_final_answer(response: str) -> str:
    """Extract the optional ``final:`` envelope from a model response."""
    stripped = response.strip()
    if stripped.lower().startswith("final:"):
        return stripped.split(":", 1)[1].strip()
    return stripped
