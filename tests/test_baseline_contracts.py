from __future__ import annotations

import importlib
import importlib.util

import pytest
from pydantic import TypeAdapter, ValidationError


def _contracts():
    assert importlib.util.find_spec("memcontam.baselines.contracts"), (
        "BASELINE-FIDELITY-V1 requires memcontam.baselines.contracts"
    )
    return importlib.import_module("memcontam.baselines.contracts")


def test_shared_non_empty_str_and_exact_semantic_stages_are_public_contracts() -> None:
    contracts = _contracts()

    assert hasattr(contracts, "NonEmptyStr")
    assert getattr(contracts, "SEMANTIC_STAGES") == (
        "no_memory_generate",
        "full_history_generate",
        "rag_generate",
        "bot_problem_distill",
        "bot_instantiate_solve",
        "bot_thought_distill",
        "reflexion_generate",
        "reflexion_reflect",
    )
    with pytest.raises(ValidationError):
        TypeAdapter(contracts.NonEmptyStr).validate_python("   ")


def test_canonical_task_json_is_compact_and_key_sorted() -> None:
    dispatch = importlib.import_module("memcontam.tasks.dispatch")

    assert dispatch.canonical_task_json({"z": [2, 1], "a": "value"}) == (
        '{"a":"value","z":[2,1]}'
    )
