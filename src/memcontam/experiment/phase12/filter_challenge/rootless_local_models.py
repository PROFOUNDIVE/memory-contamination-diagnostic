from __future__ import annotations

from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict


ROOTLESS_PROFILE: Final = "local_rootless_non_authoritative"
ROOTLESS_RECEIPT_SCHEMA_VERSION: Final = "rootless_local_receipt_v1"
ROOTLESS_RECEIPT_KIND: Final = "rootless_local_receipt"
ROOTLESS_TERMINAL: Final = "LOCAL_ROOTLESS_BCT_REVIEW_REQUIRED"

RootlessProfile: TypeAlias = Literal["local_rootless_non_authoritative"]
RootlessTerminal: TypeAlias = Literal["LOCAL_ROOTLESS_BCT_REVIEW_REQUIRED"]


class RootlessLocalReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["rootless_local_receipt_v1"]
    profile: RootlessProfile
    kind: Literal["rootless_local_receipt"]
    terminal: RootlessTerminal


__all__ = (
    "ROOTLESS_PROFILE",
    "ROOTLESS_RECEIPT_KIND",
    "ROOTLESS_RECEIPT_SCHEMA_VERSION",
    "ROOTLESS_TERMINAL",
    "RootlessLocalReceipt",
    "RootlessProfile",
    "RootlessTerminal",
)
