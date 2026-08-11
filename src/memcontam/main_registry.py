from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias, assert_never

from pydantic import BaseModel, ConfigDict

Task: TypeAlias = Literal["game24", "math_equation_balancer", "word_sorting"]
SourceValue: TypeAlias = str | int


class MainRegistryError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class VerifierSpec(_FrozenModel):
    target: str
    target_value: int


class Game24MainRow(_FrozenModel):
    sample_id: str
    numbers: tuple[int, int, int, int]
    target: int
    canonical_signature: str
    source_row: int


class MebMainRow(_FrozenModel):
    sample_id: str
    input: str
    verifier_spec: VerifierSpec
    canonical_signature: str
    source_row: int


class WordSortingMainRow(_FrozenModel):
    sample_id: str
    words: tuple[str, ...]
    sorted_words: tuple[str, ...]
    canonical_signature: str
    source_row: int


MainRow: TypeAlias = Game24MainRow | MebMainRow | WordSortingMainRow


class MainExclusion(_FrozenModel):
    source_row: int
    canonical_signature: str
    reason: Literal["duplicate_canonical_signature", "registered_non_main_signature"]


class FrozenTaskPool(_FrozenModel):
    task: Task
    source_count: int
    rows: tuple[MainRow, ...]
    exclusions: tuple[MainExclusion, ...]


def freeze_task_pool(
    *,
    task: Task,
    rows: Sequence[Mapping[str, SourceValue]],
    excluded_signatures: frozenset[str],
) -> FrozenTaskPool:
    accepted: list[MainRow] = []
    exclusions: list[MainExclusion] = []
    seen: set[str] = set()
    for source_row, row in enumerate(rows, start=1):
        parsed = _parse_source_row(task, row, source_row, len(accepted) + 1)
        signature = parsed.canonical_signature
        if signature in excluded_signatures:
            exclusions.append(MainExclusion(
                source_row=source_row,
                canonical_signature=signature,
                reason="registered_non_main_signature",
            ))
            continue
        if signature in seen:
            exclusions.append(MainExclusion(
                source_row=source_row,
                canonical_signature=signature,
                reason="duplicate_canonical_signature",
            ))
            continue
        seen.add(signature)
        accepted.append(parsed)
    return FrozenTaskPool(
        task=task,
        source_count=len(rows),
        rows=tuple(accepted),
        exclusions=tuple(exclusions),
    )


def _parse_source_row(
    task: Task,
    row: Mapping[str, SourceValue],
    source_row: int,
    registry_row: int,
) -> MainRow:
    match task:
        case "game24":
            return _game24_row(row, source_row, registry_row)
        case "math_equation_balancer":
            return _meb_row(row, source_row, registry_row)
        case "word_sorting":
            return _word_sorting_row(row, source_row, registry_row)
        case unreachable:
            assert_never(unreachable)


def _game24_row(
    row: Mapping[str, SourceValue], source_row: int, registry_row: int
) -> Game24MainRow:
    raw_input, raw_target = row.get("input"), row.get("target")
    if not isinstance(raw_input, str) or not isinstance(raw_target, (str, int)):
        raise MainRegistryError("GAME24_SOURCE_ROW_INVALID")
    try:
        numbers = tuple(int(value) for value in raw_input.split())
        target = int(raw_target)
    except ValueError as error:
        raise MainRegistryError("GAME24_SOURCE_ROW_INVALID") from error
    if len(numbers) != 4:
        raise MainRegistryError("GAME24_SOURCE_ROW_INVALID")
    typed_numbers = (numbers[0], numbers[1], numbers[2], numbers[3])
    signature = ",".join(str(value) for value in sorted(typed_numbers))
    return Game24MainRow(
        sample_id=f"phase13_main_game24_{registry_row:04d}",
        numbers=typed_numbers,
        target=target,
        canonical_signature=signature,
        source_row=source_row,
    )


def _meb_row(
    row: Mapping[str, SourceValue], source_row: int, registry_row: int
) -> MebMainRow:
    raw_input, target, target_value = row.get("input"), row.get("target"), row.get("target_value")
    if not isinstance(raw_input, str) or not isinstance(target, str) or not isinstance(target_value, int):
        raise MainRegistryError("MEB_SOURCE_ROW_INVALID")
    signature = ",".join(re.findall(r"-?\d+", raw_input))
    if not signature:
        raise MainRegistryError("MEB_SOURCE_ROW_INVALID")
    return MebMainRow(
        sample_id=f"phase13_main_meb_{registry_row:04d}",
        input=raw_input,
        verifier_spec=VerifierSpec(target=target, target_value=target_value),
        canonical_signature=signature,
        source_row=source_row,
    )


def _word_sorting_row(
    row: Mapping[str, SourceValue], source_row: int, registry_row: int
) -> WordSortingMainRow:
    raw_input, target = row.get("input"), row.get("target")
    if not isinstance(raw_input, str) or not isinstance(target, str) or "List:" not in raw_input:
        raise MainRegistryError("WORD_SORTING_SOURCE_ROW_INVALID")
    words = tuple(raw_input.split("List:", 1)[1].strip().split())
    sorted_words = tuple(target.split())
    if not words or sorted(words) != list(sorted_words):
        raise MainRegistryError("WORD_SORTING_SOURCE_ROW_INVALID")
    return WordSortingMainRow(
        sample_id=f"phase13_main_word_sorting_{registry_row:04d}",
        words=words,
        sorted_words=sorted_words,
        canonical_signature="|".join(sorted(words)),
        source_row=source_row,
    )
