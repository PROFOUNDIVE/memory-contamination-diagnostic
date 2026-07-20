from __future__ import annotations


def require_declared_parent_support(
    parent_card_ids: tuple[str, ...], declared_support_ids: tuple[str, ...]
) -> None:
    unsupported = set(parent_card_ids).difference(declared_support_ids)
    if unsupported:
        raise ValueError("memory card parent requires declared support")
