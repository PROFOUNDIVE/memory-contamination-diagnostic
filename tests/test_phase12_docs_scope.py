from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import get_args

import pytest

from memcontam.experiment.phase12 import cli as phase12_cli
from memcontam.experiment.phase12 import contracts
from memcontam.experiment.phase12 import outcomes
from memcontam.logging import schema_v3


ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    ROOT / "docs" / "phase12-implementation-contract.md",
    ROOT / "docs" / "logging-v3-phase12.md",
    ROOT / "docs" / "phase12-operator-runbook.md",
)
AUDITED_HEAD = "830b89c8c169ffa9cdea472887fdae134dbae7cf"
DESIGN_SHA256 = "984fe2881690d93a8ccced87abf03de4bf0012158462cf07ed23505414073eb0"


def _support_module():
    path = ROOT / "tests" / "support" / "docs_contracts.py"
    spec = importlib.util.spec_from_file_location("phase12_docs_contracts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _phase12_commands() -> frozenset[str]:
    parser = argparse.ArgumentParser()
    phase12_cli.add_parser(parser.add_subparsers())
    top_level = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    phase12 = top_level.choices["phase12"]
    commands = next(
        action for action in phase12._actions if isinstance(action, argparse._SubParsersAction)
    )
    return frozenset(f"phase12 {name}" for name in commands.choices)


def test_docs_match_exported_contracts() -> None:
    support = _support_module()
    assert all(path.is_file() for path in DOCS)

    combined = "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    documented_contracts = [support.extract_documented_contracts(path) for path in DOCS]
    documented = support.DocumentedContractSet(
        frozenset().union(*(contract.entries for contract in documented_contracts))
    )

    for value in (
        AUDITED_HEAD,
        DESIGN_SHA256,
        "logging_v3",
        "phase12",
        "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181",
        "python scripts/verify_bge_m3_fidelity.py",
        "missing_cached_bge_m3",
        "RouteSelectionManifest",
        "SeedAllocationManifest",
        "SelectedPackageResourceManifest",
        "ExploratoryActivationManifest",
        "BFV2",
        "P12I",
        "model_behavior",
        "NoMem",
    ):
        assert value in combined
    expected = {
        "command": _phase12_commands(),
        "protocol_index": frozenset(get_args(contracts.ProtocolIndex)),
        "experimental_arm": frozenset(get_args(contracts.ExperimentalArm)),
        "route_candidate": frozenset(get_args(contracts.RouteCandidateId)),
        "rag_mode": frozenset(get_args(contracts.RagMode)),
        "fidelity_label": frozenset(get_args(contracts.FidelityLabel)),
        "tool_mode": frozenset(get_args(contracts.ToolMode)),
        "evidence_layer": frozenset(get_args(contracts.EvidenceLayer)),
        "run_family": frozenset(get_args(contracts.RunFamily)),
        "execution_status": frozenset(get_args(outcomes.ExecutionStatus)),
        "failure_class": frozenset(get_args(outcomes.FailureClass)),
        "analysis_inclusion": frozenset(get_args(outcomes.AnalysisInclusion)),
        "parse_status": frozenset(get_args(outcomes.ParseStatus)),
        "execution_key": frozenset(("branch_free_prefix", "memory_arm", "nomem_singleton")),
        "metadata_kind": frozenset(
            (
                "pre_route",
                "selected_route",
                "exploratory_code_non_scientific",
                "exploratory_code_scientific",
            )
        ),
        "trial_kind": frozenset(("branch_free_prefix", "memory_branch", "nomem_singleton")),
        "record_type": frozenset(schema_v3._EVENT_TYPES),
        "sensitivity_kind": frozenset(
            ("base", "timing", "horizon", "affinity", "fh_budget", "embedding", "behavior")
        ),
    }
    for kind, values in expected.items():
        assert documented.values(kind) == values, f"documented {kind} values differ from exports"


def test_docs_reject_exact_source_or_scientific_overclaim() -> None:
    support = _support_module()
    for path in DOCS:
        support.reject_overclaims(path.read_text(encoding="utf-8"))

    for prohibited in (
        "Phase-12 is an exact source reproduction.",
        "These are Phase-12 scientific results.",
        "Retrieval is exposure.",
        "Exposure counts as operational use.",
        "Text and code evidence are pooled.",
        "This reports benchmark results.",
        "A paid provider run completed.",
    ):
        with pytest.raises(ValueError, match="PROHIBITED_PHASE12_CLAIM"):
            support.reject_overclaims(prohibited)

    for bounded in (
        "Phase-12 is not an exact source reproduction.",
        "This document does not report scientific results.",
        "Retrieval is not exposure, and exposure is not operational use.",
        "Text and code evidence aren't pooled.",
    ):
        support.reject_overclaims(bounded)
