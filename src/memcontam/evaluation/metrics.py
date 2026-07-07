from __future__ import annotations


def rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total
