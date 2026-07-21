from __future__ import annotations

import importlib.util
import inspect

import pytest

import memcontam.cli as cli
import memcontam.memory.corpus as corpus


def _config(run: dict[str, object], arms: list[str]) -> dict[str, object]:
    return {
        "run": {"mode": "faithful", **run},
        "models": ["model"],
        "tasks": [{"name": "game24", "sample_path": "unused.jsonl", "limit": 1}],
        "baselines": ["full_history"],
        "arms": arms,
        "logging": {"output_dir": "runs"},
    }


@pytest.mark.parametrize(
    ("run", "arms", "message"),
    [
        (
            {"stage": "pilot", "execution_class": "live", "provider": "openai_compatible"},
            ["contaminated"],
            "clean arm",
        ),
        (
            {"stage": "main", "execution_class": "live", "provider": "openai_compatible"},
            ["contaminated_filter"],
            "clean arm",
        ),
        (
            {
                "stage": "replay",
                "execution_class": "offline_contract_replay",
                "provider": "replay",
                "scientific_result": True,
                "scientific_gate_id": "approved-gate",
            },
            ["clean"],
            "replay.*scientific",
        ),
        (
            {
                "stage": "pilot",
                "execution_class": "live",
                "provider": "openai_compatible",
                "scientific_result": True,
                "scientific_gate_id": None,
            },
            ["clean"],
            "scientific_gate_id",
        ),
        (
            {"stage": "replay", "execution_class": "live", "provider": "openai_compatible"},
            ["clean"],
            "unsupported provider configuration",
        ),
        (
            {
                "stage": "pilot",
                "execution_class": "live",
                "provider": "openai_compatible",
                "scientific_result": True,
                "scientific_gate_id": "approved-gate",
            },
            ["clean"],
            "accepted scientific gate",
        ),
    ],
)
def test_protocol_validation_fails_closed(
    run: dict[str, object], arms: list[str], message: str
) -> None:
    with pytest.raises(SystemExit, match=message):
        cli._validate_run_config(_config(run, arms))


def test_protocol_validation_accepts_clean_non_scientific_live_run() -> None:
    cli._validate_run_config(
        _config(
            {
                "stage": "pilot",
                "execution_class": "live",
                "provider": "openai_compatible",
            },
            ["clean"],
        )
    )


def test_protocol_validation_rejects_test_double_embedding_outside_offline_replay() -> None:
    config = _config(
        {
            "stage": "pilot",
            "execution_class": "live",
            "provider": "openai_compatible",
        },
        ["clean"],
    )
    config["embedding"] = {"mode": "test_double"}

    with pytest.raises(SystemExit, match="test_double"):
        cli._validate_run_config(config)


def test_oracle_qa_has_a_dedicated_test_only_namespace() -> None:
    spec = importlib.util.find_spec("memcontam.memory.oracle_qa")

    assert spec is not None
    assert spec.loader is not None


def test_production_paths_do_not_call_the_test_only_filter() -> None:
    assert "drop_known_contaminated" not in inspect.getsource(cli)
    assert "drop_known_contaminated" not in inspect.getsource(corpus)
