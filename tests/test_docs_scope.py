from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
README = ROOT / "README.md"
STATUS = DOCS / "phase12-filter-v5-build-status.md"
V2_AUTHORITY = DOCS / "baseline-fidelity-v2.md"
V2_EVIDENCE = DOCS / "baseline-fidelity-v2-evidence.md"
CONTRACT_CONFIG = ROOT / "configs" / "logging_contract_replay.yaml"
PHASE11_CONFIG = ROOT / "configs" / "logging_contract_phase11_replay.yaml"
FULL_MATRIX_CONFIG = ROOT / "configs" / "full_matrix.yaml"
EVIDENCE = ROOT / ".sisyphus" / "evidence" / "phase12-filter-v5-build-v1"
MANIFEST = EVIDENCE / "implementation_manifest.json"

EXPECTED_DOCS = frozenset(
    {
        "baseline-fidelity-v2-evidence.md", "baseline-fidelity-v2.md",
        "bge-m3-cache-setup.md", "logging-v3-phase12.md",
        "phase12-filter-v5-build-status.md", "phase12-implementation-contract.md",
        "phase12-operator-runbook.md",
    }
)
AUTHORITY_SHA256 = {
    "baseline-fidelity-v2.md": "2af5daa75616473731dfc31d572d0999fcb27b8ff520378d2717eba9103fa51d",
    "baseline-fidelity-v2-evidence.md": "09bdaed245e7560bebbf05d72fd8dc8ad71ed782b06e9397b9234e3c835c59f4",
    "bge-m3-cache-setup.md": "a86fa73fad0781110a675645fb545427b73d443a6e25dd1d59a221036d073a4a",
    "logging-v3-phase12.md": "27fd535389c6323a7f2a13ba5d70fb17376c737ed588b86e67d831e5ce8740e9",
    "phase12-implementation-contract.md": "72ad0713ed473d3b6afdbe6d68b2bd6cfacac1e5513beb897c21e0db9dce0f89",
    "phase12-operator-runbook.md": "b207d119d34ba1f25ffa5203871ab54b926d3028a1cf9ce05bcf28fcb1f1523c",
}
AUTHORITY_BLOB_SHA1 = {
    "bge-m3-cache-setup.md": "d2d1e7b2d2405e77c1708ae2a6af808a0316d825",
    "phase12-operator-runbook.md": "ec33322d39cef8e9dfecaa2ebcda029e0dc61927",
}
V2_HEADINGS = {
    V2_AUTHORITY: (
        "## Authority and Claim Boundary", "## Exact Method Claims",
        "## V1 and V2 No-Pooling Rule", "## Fidelity Gate Status",
        "## Prompt and Provider Versions", "## Canonical Reproduction Commands",
        "## Unresolved Non-Claims",
    ),
    V2_EVIDENCE: (
        "## Evidence Provenance", "## Resource Usage",
        "## Artifact Hash Manifest", "## Seal Status",
    ),
}
MFT_IDS = (
    "MFT-FV5-01-PAIR-MATCH", "MFT-FV5-02-EXPOSURE-REQUIRED",
    "MFT-FV5-03-TRISTATE", "MFT-FV5-04-FAIL-OPEN",
    "MFT-FV5-05-ROUTE-INVARIANCE", "MFT-FV5-06-SCRIPTED-CORRECT",
    "MFT-FV5-07-SCRIPTED-IRRELEVANT", "MFT-FV5-08-NO-WRITEBACK",
    "MFT-FV5-09-CONTAM-SHADOW-SHARE", "MFT-FV5-10-PARSER-BOUNDARY",
    "MFT-FV5-11-CONTROL-CACHE", "MFT-FV5-12-PROBE-KEY-INVARIANCE",
    "MFT-FV5-13-ANSWER-CALL-PROVENANCE", "MFT-FV5-14-ACTIVATION-DOMAIN",
    "MFT-FV5-15-ELIGIBILITY-STATES", "MFT-FV5-16-COVERAGE-NOT-ESTIMABLE",
)
BCT_IDS = (
    "BCT-FV5-01-CERTIFIED-FALSE", "BCT-FV5-02-CORRECT", "BCT-FV5-03-IRRELEVANT",
    "BCT-FV5-04-ORDINARY-FALSE",
)
BLOCKERS = (
    "SEARCH_CONFIG_PENDING_FREEZE", "SCIENTIFIC_INVENTORY_PENDING_FREEZE",
    "CANONICAL_PATCHES_PENDING", "PROVIDER_CONFIG_DISABLED", "PROVIDER_AUTHORIZATION_ABSENT",
)


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()


def test_documentation_inventory_is_exact() -> None:
    assert {path.name for path in DOCS.glob("*.md")} == EXPECTED_DOCS


def test_immutable_authority_artifacts_have_sealed_hashes() -> None:
    for name, expected in AUTHORITY_SHA256.items():
        path = DOCS / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        if name in AUTHORITY_BLOB_SHA1:
            assert _git_blob_sha1(path) == AUTHORITY_BLOB_SHA1[name]


def test_baseline_fidelity_v2_authority_has_each_required_heading_once() -> None:
    for path, headings in V2_HEADINGS.items():
        text = path.read_text(encoding="utf-8")
        assert [heading for heading in headings if text.count(heading) != 1] == []


def test_baseline_fidelity_v2_uses_exact_bounded_method_claims() -> None:
    claims = (
        "one-call no-persistent-memory baseline",
        "context-bounded full-history with full append-only store",
        "training-free dense retrieval with black-box input-layer augmentation",
        "deterministic paper-aligned BoT-style proxy",
        "failure-gated verbal-reflection adaptation with one same-sample retry",
        "adapted optional DC-RS appendix comparator",
    )
    text = V2_AUTHORITY.read_text(encoding="utf-8")
    assert all(claim in text for claim in claims)


def test_baseline_fidelity_v2_authority_reports_current_pass_status() -> None:
    text = V2_AUTHORITY.read_text(encoding="utf-8")
    assert "Overall V2 certification for the current compatible closeout tuple: **PASS**." in text
    assert "F1A structural integration replay: **PASS**." in text
    assert "F1B source-contract replay: **PASS**." in text
    assert "F1C pinned real-retriever and mocked-live boundary: **PASS**." in text


def test_v2_evidence_historical_snapshot_has_exact_sealed_rows() -> None:
    evidence = V2_EVIDENCE.read_text(encoding="utf-8")
    expected_paths = frozenset(
        {
            "configs/baseline_fidelity_v2_structural_replay.yaml",
            "configs/baseline_fidelity_v2_source_contract_replay.yaml",
            "configs/baseline_fidelity_v2_bge_smoke.yaml",
            "data/replay/baseline_fidelity_v2_source_contract.yaml",
            "data/memory/baseline_fidelity_v2_contract_corpus.jsonl",
            "data/memory/baseline_fidelity_v2_contract_corpus.manifest.json",
            "scripts/inspect_baseline_fidelity_v2.py",
            "scripts/verify_bge_m3_fidelity.py",
            "scripts/report_baseline_resource_usage.py",
        }
    )
    rows = re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$", evidence, re.MULTILINE)
    assert len(rows) == len(expected_paths)
    assert {path for path, _ in rows} == expected_paths
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for _, digest in rows)
    assert dict(rows)["scripts/verify_bge_m3_fidelity.py"] == (
        "e4a2c2c92e6b6e4fe1dea6e8d6d6439403f28f6e25565aad4cc8098fc5dd2123"
    )


def test_readme_is_current_authority_index() -> None:
    text = README.read_text(encoding="utf-8")
    assert set(re.findall(r"\]\((docs/[^)#]+)\)", text)) == {
        f"docs/{name}" for name in EXPECTED_DOCS
    }
    assert "## Documentation Authorities" in text
    assert "Baseline-Fidelity-V2 authority and evidence are the sole current status authority" in text
    assert "frozen Phase-12 contract snapshot" in text
    assert "not a current BFV2 status source" in text


def test_filter_v5_status_matches_sealed_build_evidence() -> None:
    assert STATUS.is_file()
    status = STATUS.read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mft = json.loads((EVIDENCE / "mft_fv5_report.json").read_text(encoding="utf-8"))["report"]
    bct = json.loads((EVIDENCE / "bct_readiness_report.json").read_text(encoding="utf-8"))["report"]
    assert (
        hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
        == "dd964902513ddcfebe10f482191310f4e57e931eb66adebfb3343def21e07571"
    )
    assert f"| Implementation commit | `{manifest['header']['implementation_commit']}` |" in status
    assert "| Evidence-recording commit | `b814b0100f66a19a7111f8f06755e550e8704a52` |" in status
    assert (
        manifest["header"]["plan_sha256"]
        == "65f4c45b5db702af0f60a5296d116bc1ed64ac7440b447c676b069a8e204c12b"
    )
    assert mft["ordered_test_ids"] == list(MFT_IDS)
    assert [(item["test_id"], item["count"]) for item in mft["execution_counts"]] == [
        (test_id, 1) for test_id in MFT_IDS
    ]
    results = mft["state_report"]["results"] + mft["safety_report"]["cases"]
    assert [item["test_id"] for item in results] == list(MFT_IDS)
    assert all(item["status"] == "pass" for item in results)
    assert (
        mft["provider_calls_issued"]
        == mft["state_report"]["provider_calls_issued"]
        == mft["safety_report"]["provider_calls_issued"]
        == 0
    )
    assert mft["all_passed"] is True
    assert f"All {len(mft['ordered_test_ids'])} deterministic MFT gates passed" in status
    assert f"Provider calls: `{mft['provider_calls_issued']}`" in status
    assert bct["software_interface_status"] == "ready"
    assert bct["execution_status"] == "blocked"
    assert [(item["test_id"], item["status"]) for item in bct["family_statuses"]] == [
        (test_id, "not_executed") for test_id in BCT_IDS
    ]
    assert bct["canonical_patch_status"] == "pending_before_provider_backed_pilot_b"
    assert tuple(bct["blocking_reason_codes"]) == BLOCKERS
    assert bct["provider_calls_issued"] == 0
    assert f"| Software interface | `{bct['software_interface_status']}` |" in status
    assert f"| BCT execution | `{bct['execution_status']}` |" in status
    for item in bct["family_statuses"]:
        assert f"| `{item['test_id']}` | `{item['status']}` |" in status
    assert f"| Canonical patch | `{bct['canonical_patch_status']}` |" in status
    assert "does not certify the descendant HEAD" in status
    for phrase in (
        "contract-invalid direct-write containment",
        "not semantic truth detection or general contamination mitigation",
        "remains active for Pilot-A/current runtime",
        "separate from additive Filter-Challenge-v1 build evidence",
    ):
        assert phrase in status
    assert all(value in status for value in (*MFT_IDS, *BCT_IDS, *BLOCKERS))


def test_logging_contract_replay_config_is_offline_replay_only() -> None:
    config = yaml.safe_load(CONTRACT_CONFIG.read_text(encoding="utf-8"))
    assert config["run"]["mode"] == "faithful"
    assert config["run"]["stage"] == "replay"
    assert config["run"]["provider"] == "replay"
    assert config["logging"]["schema_version"] == "logging_v1"
    assert config["embedding"]["offline_fallback"] is True
    assert config["live_smoke"]["enabled"] is False


def test_full_matrix_validate_config_rejects_todo_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    import memcontam.cli as cli

    monkeypatch.chdir(ROOT)
    with pytest.raises(SystemExit, match="unresolved task limits"):
        cli.validate_config(FULL_MATRIX_CONFIG)


def test_full_matrix_carries_phase11_keys_but_keeps_placeholders() -> None:
    config = yaml.safe_load(FULL_MATRIX_CONFIG.read_text(encoding="utf-8"))
    assert config["logging"]["schema_version"] == "logging_v2"
    assert config["run"]["contract_level"] == "phase11"
    assert config["evaluation"]["evaluation_law_id"] == "phase11_full_matrix_online_v1"
    assert config["target_contamination_set"] == {
        "target_set_id": "controlled_injected_derived_v1",
        "definition_version": "phase11_v1",
        "included_classes": ["injected", "derived"],
        "require_exact_lineage": True,
    }
    assert [task["limit"] for task in config["tasks"]] == ["TODO", "TODO", "TODO"]


def test_phase11_config_remains_historical_logging_v2_replay() -> None:
    config = yaml.safe_load(PHASE11_CONFIG.read_text(encoding="utf-8"))
    assert config["run"]["mode"] == "faithful"
    assert config["run"]["stage"] == "replay"
    assert config["run"]["contract_level"] == "phase11"
    assert config["run"]["provider"] == "replay"
    assert config["logging"]["schema_version"] == "logging_v2"
    assert config["live_smoke"]["enabled"] is False
