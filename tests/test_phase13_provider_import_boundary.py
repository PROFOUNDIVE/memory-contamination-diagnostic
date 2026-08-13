from __future__ import annotations

import builtins
import sys

import pytest


def test_root_deterministic_cli_never_imports_provider_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = builtins.__import__

    def reject_provider_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name in {
            "memcontam.clients.factory",
            "memcontam.clients.openai_responses",
        }:
            raise AssertionError(f"provider import forbidden: {name}")
        return imported(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_provider_import)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "memcontam",
            "phase13",
            "validate-calibration-v2",
            "--config",
            "configs/phase13/pre_main_calibration_v2.yaml",
        ],
    )

    from memcontam.cli import main

    main()
