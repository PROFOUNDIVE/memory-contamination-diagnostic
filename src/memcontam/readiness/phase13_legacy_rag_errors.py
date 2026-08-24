from __future__ import annotations

class LegacyRagValidationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)

    def __str__(self) -> str:
        return self.code


def fail_validation(code: str) -> None:
    raise LegacyRagValidationError(code)


__all__ = ["LegacyRagValidationError", "fail_validation"]
