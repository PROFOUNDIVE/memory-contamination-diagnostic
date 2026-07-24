import hashlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
G0_DOC = ROOT / "docs" / "g0-baseline-fidelity-gate-v0.4.md"
README = ROOT / "README.md"
V05_DOC = ROOT / "docs" / "g0-baseline-fidelity-gate-v0.5.md"
FOLLOWUP_DOC = ROOT / "docs" / "g0-dc-rs-reflexion-fidelity-followup.md"
LOGGING_RELEASE_V072_DOC = ROOT / "docs" / "logging-audit-remediation-v0.7.2.md"
PHASE11_CONTRACT_DOC = ROOT / "docs" / "logging-contract-v2-phase11.md"
PHASE11_REPORT_DOC = ROOT / "docs" / "logging-audit-remediation-phase11.md"
CONTRACT_CONFIG = ROOT / "configs" / "logging_contract_replay.yaml"
PHASE11_CONFIG = ROOT / "configs" / "logging_contract_phase11_replay.yaml"
FULL_MATRIX_CONFIG = ROOT / "configs" / "full_matrix.yaml"
V1_AUTHORITY = ROOT / "docs" / "baseline-fidelity-v1.md"
V2_AUTHORITY = ROOT / "docs" / "baseline-fidelity-v2.md"
V2_EVIDENCE = ROOT / "docs" / "baseline-fidelity-v2-evidence.md"
HISTORICAL_BASELINE_REPORTS = (
    V1_AUTHORITY,
    G0_DOC,
    V05_DOC,
    ROOT / "docs" / "g0-baseline-fidelity-gate-v0.6.md",
)
SUPERSESSION_NOTICE = "This historical report cannot support a Baseline-Fidelity-V2 fidelity claim."

V2_AUTHORITY_PHRASES = (
    "sole authority for Baseline-Fidelity-V2",
    "Overall V2 certification: **BLOCKED**",
    "F1A structural integration replay: **PASS**",
    "F1B source-contract replay: **PASS**",
    "F1C pinned real-retriever and mocked-live boundary: **BLOCKED**",
    "missing_cached_bge_m3",
    "QA and fidelity evidence, not benchmark or manuscript-quality evidence",
)
V2_EVIDENCE_PHRASES = (
    ".sisyphus/evidence/baseline-fidelity-v2/evidence_manifest.json",
    "semantic calls",
    "transport retries",
    "prompt tokens",
    "completion tokens",
    "latency ms",
    "retrievals",
    "memory writes",
    "configs/baseline_fidelity_v2_structural_replay.yaml",
    "configs/baseline_fidelity_v2_source_contract_replay.yaml",
    "configs/baseline_fidelity_v2_bge_smoke.yaml",
)
V2_METADATA_PHRASES = (
    "prompt_version: baseline_fidelity_v2",
    "memory_policy_version: baseline_fidelity_v2",
    "baseline_fidelity_v2_structural_fixture",
    "baseline_fidelity_v2_source_contract_fixture",
    "mocked_openai_compatible_v1",
    "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181",
    "python -m pytest -q tests/test_baseline_fidelity_replay.py",
    "python -m pytest -q tests/test_baseline_source_contract_replay.py",
    "python scripts/verify_bge_m3_fidelity.py",
    "--stage replay --contract phase11",
)
V2_HEADINGS = (
    (V2_AUTHORITY, (
        "## Authority and Claim Boundary",
        "## Exact Method Claims",
        "## V1 and V2 No-Pooling Rule",
        "## Fidelity Gate Status",
        "## Prompt and Provider Versions",
        "## Canonical Reproduction Commands",
        "## Unresolved Non-Claims",
    )),
    (V2_EVIDENCE, (
        "## Evidence Provenance",
        "## Resource Usage",
        "## Artifact Hash Manifest",
        "## Seal Status",
    )),
)
FOLLOWUP_CLAIMS = (
    "Faithful adapted DC-RS optional appendix comparator: top-3 cosine retrieval over prior same-identity input/output pairs, label-free pre-answer cheatsheet synthesis, then memory-conditioned generation, with native method-call costs logged.",
    "Faithful adapted Reflexion control flow: failed trajectory plus sanitized evaluator feedback produces linguistic reflection, latest-three reflection memory conditions a same-sample retry, stopping on success or attempt limit; no weight updates.",
)
OVERCLAIMS = (
    "full reproduction",
    "benchmark improvement",
    "main DC baseline",
    "benchmark evidence",
    "manuscript evidence",
)


def _missing(text: str, phrases: tuple[str, ...]) -> list[str]:
    return [phrase for phrase in phrases if phrase not in text]


def _present(text: str, phrases: tuple[str, ...]) -> list[str]:
    return [phrase for phrase in phrases if phrase in text]


def _affirmative_exact_reproduction(text: str) -> bool:
    return "exact reproduction" in text.replace("not an exact reproduction", "")


def test_baseline_fidelity_v2_docs_are_the_only_current_authority() -> None:
    authority = V2_AUTHORITY.read_text(encoding="utf-8")
    evidence = V2_EVIDENCE.read_text(encoding="utf-8")
    assert not (missing := _missing(authority, V2_AUTHORITY_PHRASES)), missing
    assert not (missing := _missing(authority, V2_METADATA_PHRASES)), missing
    assert not (missing := _missing(evidence, V2_EVIDENCE_PHRASES)), missing


def test_baseline_fidelity_v2_authority_has_each_required_heading_once() -> None:
    for path, headings in V2_HEADINGS:
        text = path.read_text(encoding="utf-8")
        assert not (duplicates := [heading for heading in headings if text.count(heading) != 1]), duplicates


def test_baseline_fidelity_v2_uses_exact_bounded_method_claims() -> None:
    claims = (
        "one-call no-persistent-memory baseline",
        "context-bounded full-history with full append-only store",
        "training-free dense retrieval with black-box input-layer augmentation",
        "deterministic paper-aligned BoT-style proxy",
        "failure-gated verbal-reflection adaptation with one same-sample retry",
        "adapted optional DC-RS appendix comparator",
    )
    assert not (missing := _missing(V2_AUTHORITY.read_text(encoding="utf-8"), claims)), missing


@pytest.mark.parametrize("path", HISTORICAL_BASELINE_REPORTS)
def test_historical_baseline_reports_are_explicitly_superseded_for_v2(path: Path) -> None:
    required = (SUPERSESSION_NOTICE, "docs/baseline-fidelity-v2.md", "docs/baseline-fidelity-v2-evidence.md")
    assert not (missing := _missing(path.read_text(encoding="utf-8"), required)), missing


def test_v2_evidence_hashes_match_committed_artifacts() -> None:
    evidence = V2_EVIDENCE.read_text(encoding="utf-8")
    paths = (
        "configs/baseline_fidelity_v2_structural_replay.yaml",
        "configs/baseline_fidelity_v2_source_contract_replay.yaml",
        "configs/baseline_fidelity_v2_bge_smoke.yaml",
        "data/replay/baseline_fidelity_v2_source_contract.yaml",
        "data/memory/baseline_fidelity_v2_contract_corpus.jsonl",
        "data/memory/baseline_fidelity_v2_contract_corpus.manifest.json",
        "scripts/inspect_baseline_fidelity_v2.py",
        "scripts/verify_bge_m3_fidelity.py",
        "scripts/report_baseline_resource_usage.py",
    )
    for relative_path in paths:
        digest = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert f"| `{relative_path}` | `{digest}` |" in evidence


def test_readme_v2_and_v09_repository_contracts() -> None:
    text = README.read_text(encoding="utf-8")
    required = (
        "docs/baseline-fidelity-v2.md",
        "docs/baseline-fidelity-v2-evidence.md",
        "F1A and F1B pass",
        "missing_cached_bge_m3",
        "`v0.8` is a repository research-artifact tag",
        "It is not an overall V2 certification because F1C remains blocked.",
        "## v0.9 Phase-12 Repository Contract Refactor",
        "Phase-12 plan execution completed as repository-contract work",
        "removed bootstrap scaffolding and deduplicated tests/docs",
        "`v0.9` is a repository research-artifact tag, not scientific, benchmark, or manuscript-quality evidence.",
        "F1C remains `BLOCKED` (`missing_cached_bge_m3`).",
        "P12I may pass, but scientific admission remains false.",
        "Text/code evidence are not pooled.",
        "| `v0.9` |",
    )
    assert not (missing := _missing(text, required)), missing
    assert not (present := _present(text, ("F1C pass",))), present


def test_g0_doc_contract() -> None:
    text = G0_DOC.read_text(encoding="utf-8")
    required = (
        "partial G0 pass for `retrieval_rag` and `bot_style` only",
        "no_memory",
        "full_history",
        "reflexion_style",
        "dynamic_cheatsheet_optional",
        "expel_optional",
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "bot_problem_distill",
        "bot_instantiate_solve",
        "bot_thought_distill",
        "bot_novelty_decide",
        "data/memory/catalog_v1.jsonl",
        "(run_id, task_name, baseline, arm, backbone)",
        "replay fixtures",
        "live runs must use the identical stage structure",
    )
    assert not (missing := _missing(text, required)), missing
    assert not (present := _present(text, ("exact reproduction", "all baselines pass G0", "full method reproduction"))), present


def test_readme_baseline_fidelity_contract() -> None:
    text = README.read_text(encoding="utf-8")
    required = (
        "not a complete reproduction",
        "configs/g0_rag_bot_faithful_replay.yaml",
        "scripts/inspect_g0_rag_bot_fidelity.py",
        "g0_rag_bot_faithful_replay",
    )
    assert not (missing := _missing(text, required)), missing
    assert not (present := _present(text, ("all baselines pass G0", "exact reproduction"))), present


def test_v05_doc_contract() -> None:
    assert V05_DOC.is_file(), "v0.5 release report must exist"
    text = V05_DOC.read_text(encoding="utf-8")
    required = (
        "3 tasks × 3 baselines × 3 arms × 2 models = 162 trials",
        "full_history_generate=54",
        "reflexion_generate=54",
        "reflexion_reflect=6",
        "dynamic_cheatsheet_generate=54",
        "dynamic_cheatsheet_curate=54",
        "54 per baseline",
        "18 per baseline/arm",
        "222 total method calls",
        "6 Reflexion reflected trials (game24_pilot_001)",
        "6 DC preserved_missing_tag rows (game24_pilot_001)",
        "faithful append-only full-history",
        "Reflexion-style verbal memory proxy / faithful adapted control flow",
        "faithful adapted DC-Cu optional appendix comparator",
        "not an exact reproduction",
        "https://github.com/noahshinn/reflexion",
        "218cf0ef1df84b05ce379dd4a8e47f17766733a0",
        "https://github.com/suzgunmirac/dynamic-cheatsheet",
        "5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9",
    )
    assert not (missing := _missing(text, required)), missing
    assert text.count("MIT") >= 2
    assert not (present := _present(text, ("all baselines pass G0", "benchmark result", "backbone-independent", "admission-control proof"))), present
    assert not _affirmative_exact_reproduction(text)


def test_readme_v05_contract_and_package_version() -> None:
    text = README.read_text(encoding="utf-8")
    required = (
        "## v0.5",
        "full_history",
        "reflexion_style",
        "dynamic_cheatsheet_optional",
        "configs/g0_fh_reflexion_dc_faithful_replay.yaml",
        "scripts/inspect_g0_fh_reflexion_dc_fidelity.py",
        "g0_fh_reflexion_dc_faithful_replay",
        "docs/g0-baseline-fidelity-gate-v0.5.md",
        "docs/g0-baseline-fidelity-gate-v0.4.md",
        "fidelity/QA artifact",
        "not benchmark or manuscript-quality evidence",
    )
    assert not (missing := _missing(text, required)), missing
    assert 'version = "0.1.0"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_followup_doc_contract() -> None:
    assert FOLLOWUP_DOC.is_file(), "v0.5+ follow-up report must exist"
    text = FOLLOWUP_DOC.read_text(encoding="utf-8")
    required = (
        "# G0 DC-RS and Reflexion Same-Sample Retry Fidelity Follow-up",
        "post-`v0.5` follow-up",
        "optional appendix comparator",
        "same-sample retry",
        "https://aclanthology.org/2026.eacl-long.333/",
        "https://github.com/suzgunmirac/dynamic-cheatsheet",
        "https://arxiv.org/abs/2303.11366",
        "https://github.com/noahshinn/reflexion",
        "| Must Preserve | Safely Adapted | Omitted |",
        "top-3 cosine",
        "same-identity",
        "no weight updates",
        "configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml",
        "data/replay/g0_dc_rs_reflexion_fidelity_followup_v1.yaml",
        "g0_dc_rs_reflexion_fidelity_followup_replay",
        "scripts/inspect_g0_dc_rs_reflexion_fidelity.py",
        "108 trial rows",
        "174 native method calls",
        '"dc_rs_calls": 108',
        '"reflexion_calls": 66',
        '"method_calls": 174',
        '"trials": 108',
        *FOLLOWUP_CLAIMS,
    )
    assert not (missing := _missing(text, required)), missing
    assert not (present := _present(text, OVERCLAIMS)), present


def test_logging_contract_replay_config_is_offline_replay_only() -> None:
    config = yaml.safe_load(CONTRACT_CONFIG.read_text(encoding="utf-8"))
    assert config["run"]["mode"] == "faithful"
    assert config["run"]["stage"] == "replay"
    assert config["run"]["provider"] == "replay"
    assert config["logging"]["schema_version"] == "logging_v1"
    assert config["embedding"]["offline_fallback"] is True
    assert config["live_smoke"]["enabled"] is False


def test_full_matrix_validate_config_rejects_todo_limits(monkeypatch) -> None:
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


def test_readme_followup_contract() -> None:
    text = README.read_text(encoding="utf-8")
    required = (
        "## v0.5+ DC-RS and Reflexion Same-Sample Retry Follow-up",
        "`v0.5` remains the historical full G0 baseline-fidelity pass",
        "configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml",
        "scripts/inspect_g0_dc_rs_reflexion_fidelity.py",
        "g0_dc_rs_reflexion_fidelity_followup_replay",
        "docs/g0-dc-rs-reflexion-fidelity-followup.md",
        *FOLLOWUP_CLAIMS,
    )
    assert not (missing := _missing(text, required)), missing


@pytest.mark.parametrize("path", (README, FOLLOWUP_DOC))
def test_readme_and_followup_forbid_overclaim_phrases(path: Path) -> None:
    assert not (present := _present(path.read_text(encoding="utf-8"), OVERCLAIMS)), present


def test_phase11_docs_and_readme_contract() -> None:
    assert PHASE11_CONTRACT_DOC.is_file()
    assert PHASE11_REPORT_DOC.is_file()
    contract = PHASE11_CONTRACT_DOC.read_text(encoding="utf-8")
    report = PHASE11_REPORT_DOC.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    required = (
        "configs/logging_contract_phase11_replay.yaml",
        "--contract phase11",
        "evaluation_law_id",
        "target_set_id",
        "B-star",
        "LineageEdge",
        "not_evaluable",
        "No API-connected pilot was run.",
        "not a full PROV-DM model",
        "causal use",
        "retrievable memory",
    )
    assert not (missing := _missing(contract, required)), missing
    report_required = (
        "report template",
        "Status:** `TEMPLATE`",
        "[ ] python -m pytest tests/test_phase11_logging_contract_gate.py -q",
        "No API-connected pilot was run.",
    )
    assert not (missing := _missing(report, report_required)), missing
    readme_required = (
        "docs/logging-contract-v2-phase11.md",
        "docs/logging-audit-remediation-phase11.md",
        "configs/logging_contract_phase11_replay.yaml",
        "## v0.7.2 Phase-11 `logging_v2` Contract Release and Status",
        "docs/logging-audit-remediation-v0.7.2.md",
    )
    assert not (missing := _missing(readme, readme_required)), missing
    forbidden = (
        "v1 is Phase-11 evidence",
        "v1 artifacts are Phase-11 evidence",
        "replay gate is a pilot",
        "replay gate is the main run",
        "replay gate is a benchmark",
        "approximate lineage is exact derivation",
        "exposure is causal use",
        "complete PROV-DM implementation",
    )
    for text in (contract, report, readme):
        assert not (present := _present(text, forbidden)), present
    assert "| `phase11` |" not in readme
    assert "## Phase-11 `logging_v2` Contract Status" not in readme
    assert readme.count("configs/logging_contract_phase11_replay.yaml") >= 1


def test_phase11_config_and_release_doc_contract() -> None:
    config = yaml.safe_load(PHASE11_CONFIG.read_text(encoding="utf-8"))
    assert config["logging"]["schema_version"] == "logging_v2"
    assert config["run"]["contract_level"] == "phase11"
    assert LOGGING_RELEASE_V072_DOC.is_file()
    required = (
        "Tag:** `v0.7.2`",
        "configs/logging_contract_phase11_replay.yaml",
        "--contract phase11",
        "No API-connected pilot was run.",
        "not a live pilot, main run, benchmark result, or manuscript result",
    )
    assert not (missing := _missing(LOGGING_RELEASE_V072_DOC.read_text(encoding="utf-8"), required)), missing
