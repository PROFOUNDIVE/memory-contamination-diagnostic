from __future__ import annotations


def require_declared_parent_support(
    memory_support_ids: tuple[str, ...], declared_parent_ids: tuple[str, ...]
) -> None:
    unsupported = set(memory_support_ids).difference(declared_parent_ids)
    if unsupported:
        raise ValueError("memory card support requires a declared parent")
