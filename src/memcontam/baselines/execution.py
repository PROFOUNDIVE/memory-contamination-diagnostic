from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def execute_baseline(executor: Any, *args: Any, **kwargs: Any) -> Any:
    execute = getattr(executor, "execute", None)
    if callable(execute):
        return execute(*args, **kwargs)
    run = getattr(executor, "run", None)
    if callable(run):
        return run(*args, **kwargs)
    raise TypeError("baseline executor must define execute() or run()")


def assert_prompt_bytes(
    fixture_path: str | Path,
    *,
    stage: str,
    messages: list[dict[str, str]],
    replacements: Mapping[str, str] | None = None,
) -> None:
    path = Path(fixture_path)
    fixture = json.loads(path.read_text(encoding="utf-8"))
    fixture_stage = fixture.get("stage")
    if fixture_stage != stage:
        raise AssertionError(f"{path}: expected stage {fixture_stage!r}, got {stage!r}")
    expected_messages = _replace_fixture_values(fixture.get("messages"), replacements or {})
    if expected_messages != messages:
        raise AssertionError(f"{path}: prompt messages do not match the committed fixture")


def _replace_fixture_values(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        return value
    if isinstance(value, list):
        return [_replace_fixture_values(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_fixture_values(item, replacements) for key, item in value.items()}
    return value
