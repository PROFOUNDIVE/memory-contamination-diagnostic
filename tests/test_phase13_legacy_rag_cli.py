from __future__ import annotations

import argparse
from pathlib import Path

from memcontam.readiness import phase13_legacy_rag_materialize
from memcontam.readiness.phase13_cli import add_parser, run
from memcontam.readiness.phase13_legacy_rag_models import (
    LegacyRagMaterializationReport,
    TaskStatus,
)


def test_phase13_parser_registers_legacy_rag_commands() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_parser(subparsers)

    cases = (
        ("audit-legacy-rag", ["--evaluation-root", "main", "--output", "opaque.json"]),
        (
            "materialize-legacy-rag",
            [
                "--repository-root", ".", "--opaque-exclusion", "opaque.json",
                "--cache-root", "cache", "--output", "legacy",
            ],
        ),
        (
            "validate-legacy-rag",
            [
                "--repository-root", ".", "--root", "legacy",
                "--expected-manifest-sha256", "0" * 64,
            ],
        ),
    )
    for command, options in cases:
        args = parser.parse_args(["phase13", command, *options])
        assert args.phase13_command == command


def test_materialize_cli_defers_readiness_gate_to_loaded_package(monkeypatch) -> None:
    import memcontam.memory.embeddings as embeddings

    report = LegacyRagMaterializationReport(
        package_status="TRACK2_LEGACY_RAG_MATERIALIZATION_COMPLETE",
        tasks={
            task: TaskStatus(status="TRACK2_LEGACY_RAG_MATERIALIZATION_COMPLETE")
            for task in ("game24", "math_equation_balancer", "word_sorting")
        },
    )
    monkeypatch.setattr(embeddings, "BgeM3EmbeddingProvider", lambda **_kwargs: None)
    monkeypatch.setattr(
        phase13_legacy_rag_materialize,
        "materialize_legacy_rag_package",
        lambda _request: report,
    )

    run(
        argparse.Namespace(
            phase13_command="materialize-legacy-rag",
            repository_root=Path("."),
            opaque_exclusion=Path("opaque.json"),
            cache_root=Path("cache"),
            output=Path("legacy"),
        )
    )
