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


def test_baseline_fidelity_v2_docs_are_the_only_current_authority() -> None:
    authority = V2_AUTHORITY.read_text(encoding="utf-8")
    evidence = V2_EVIDENCE.read_text(encoding="utf-8")

    assert "sole authority for Baseline-Fidelity-V2" in authority
    assert "Overall V2 certification: **BLOCKED**" in authority
    assert "F1A structural integration replay: **PASS**" in authority
    assert "F1B source-contract replay: **PASS**" in authority
    assert "F1C pinned real-retriever and mocked-live boundary: **BLOCKED**" in authority
    assert "missing_cached_bge_m3" in authority
    assert "QA and fidelity evidence, not benchmark or manuscript-quality evidence" in authority
    assert ".sisyphus/evidence/baseline-fidelity-v2/evidence_manifest.json" in evidence


def test_baseline_fidelity_v2_authority_has_each_required_heading_once() -> None:
    authority = V2_AUTHORITY.read_text(encoding="utf-8")
    evidence = V2_EVIDENCE.read_text(encoding="utf-8")
    authority_headings = [
        "## Authority and Claim Boundary",
        "## Exact Method Claims",
        "## V1 and V2 No-Pooling Rule",
        "## Fidelity Gate Status",
        "## Prompt and Provider Versions",
        "## Canonical Reproduction Commands",
        "## Unresolved Non-Claims",
    ]
    evidence_headings = [
        "## Evidence Provenance",
        "## Resource Usage",
        "## Artifact Hash Manifest",
        "## Seal Status",
    ]
    for heading in authority_headings:
        assert authority.count(heading) == 1, f"V2 authority must contain {heading!r} once"
    for heading in evidence_headings:
        assert evidence.count(heading) == 1, f"V2 evidence must contain {heading!r} once"


def test_baseline_fidelity_v2_uses_exact_bounded_method_claims() -> None:
    text = V2_AUTHORITY.read_text(encoding="utf-8")
    claims = [
        "one-call no-persistent-memory baseline",
        "context-bounded full-history with full append-only store",
        "training-free dense retrieval with black-box input-layer augmentation",
        "deterministic paper-aligned BoT-style proxy",
        "failure-gated verbal-reflection adaptation with one same-sample retry",
        "adapted optional DC-RS appendix comparator",
    ]
    for claim in claims:
        assert claim in text, f"V2 authority must use the exact bounded claim {claim!r}"


def test_historical_baseline_reports_are_explicitly_superseded_for_v2() -> None:
    for path in HISTORICAL_BASELINE_REPORTS:
        text = path.read_text(encoding="utf-8")
        assert SUPERSESSION_NOTICE in text, f"{path.name} must reject V2 evidence use"
        assert "docs/baseline-fidelity-v2.md" in text
        assert "docs/baseline-fidelity-v2-evidence.md" in text


def test_v2_docs_record_versions_commands_resources_and_non_claims() -> None:
    text = V2_AUTHORITY.read_text(encoding="utf-8")
    evidence = V2_EVIDENCE.read_text(encoding="utf-8")
    for phrase in [
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
    ]:
        assert phrase in text, f"V2 authority must contain {phrase!r}"
    for phrase in [
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
    ]:
        assert phrase in evidence, f"V2 evidence must contain {phrase!r}"


def test_v2_evidence_hashes_match_committed_artifacts() -> None:
    evidence = V2_EVIDENCE.read_text(encoding="utf-8")
    paths = [
        "configs/baseline_fidelity_v2_structural_replay.yaml",
        "configs/baseline_fidelity_v2_source_contract_replay.yaml",
        "configs/baseline_fidelity_v2_bge_smoke.yaml",
        "data/replay/baseline_fidelity_v2_source_contract.yaml",
        "data/memory/baseline_fidelity_v2_contract_corpus.jsonl",
        "data/memory/baseline_fidelity_v2_contract_corpus.manifest.json",
        "scripts/inspect_baseline_fidelity_v2.py",
        "scripts/verify_bge_m3_fidelity.py",
        "scripts/report_baseline_resource_usage.py",
    ]
    for relative_path in paths:
        digest = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert f"| `{relative_path}` | `{digest}` |" in evidence


def test_readme_points_to_v2_authority_without_claiming_f1c_pass() -> None:
    text = README.read_text(encoding="utf-8")
    assert "docs/baseline-fidelity-v2.md" in text
    assert "docs/baseline-fidelity-v2-evidence.md" in text
    assert "F1A and F1B pass" in text
    assert "missing_cached_bge_m3" in text
    assert "F1C pass" not in text
    assert "`v0.8` is a repository research-artifact tag" in text
    assert "It is not an overall V2 certification because F1C remains blocked." in text


def test_g0_doc_states_partial_rag_bot_scope() -> None:
    text = G0_DOC.read_text(encoding="utf-8")
    assert "partial G0 pass for `retrieval_rag` and `bot_style` only" in text, (
        "G0 doc must state the implementation plan targets a partial RAG + BoT pass"
    )

    out_of_scope = [
        "no_memory",
        "full_history",
        "reflexion_style",
        "dynamic_cheatsheet_optional",
        "expel_optional",
    ]
    for baseline in out_of_scope:
        assert baseline in text, (
            f"G0 doc must explicitly label {baseline} as out of implementation scope"
        )

    misleading = [
        "exact reproduction",
        "all baselines pass G0",
        "full method reproduction",
    ]
    for phrase in misleading:
        assert phrase not in text, f"G0 doc must not contain misleading claim: {phrase!r}"


def test_g0_doc_names_pinned_checkpoint_revision() -> None:
    text = G0_DOC.read_text(encoding="utf-8")
    assert "1110a243fdf4706b3f48f1d95db1a4f5529b4d41" in text, (
        "G0 doc must name the pinned sentence-transformers checkpoint revision"
    )


def test_g0_doc_lists_bot_stages() -> None:
    text = G0_DOC.read_text(encoding="utf-8")
    for stage in [
        "bot_problem_distill",
        "bot_instantiate_solve",
        "bot_thought_distill",
        "bot_novelty_decide",
    ]:
        assert stage in text, f"G0 doc must list BoT stage {stage}"


def test_g0_doc_names_versioned_corpus() -> None:
    text = G0_DOC.read_text(encoding="utf-8")
    assert "data/memory/catalog_v1.jsonl" in text, (
        "G0 doc must name the versioned legal corpus path"
    )


def test_g0_doc_states_persistence_key() -> None:
    text = G0_DOC.read_text(encoding="utf-8")
    assert "(run_id, task_name, baseline, arm, backbone)" in text, (
        "G0 doc must state the BoT meta-buffer persistence key tuple"
    )


def test_g0_doc_states_replay_only_llm_boundary() -> None:
    text = G0_DOC.read_text(encoding="utf-8")
    assert "replay fixtures" in text, (
        "G0 doc must state that external LLM responses are replay fixtures in this gate"
    )
    assert "live runs must use the identical stage structure" in text, (
        "G0 doc must state the temporary LLM boundary for the gate"
    )


def test_readme_does_not_overstate_baseline_fidelity() -> None:
    text = README.read_text(encoding="utf-8")
    assert "not a complete reproduction" in text, (
        "README must retain the no-complete-reproduction caveat"
    )
    assert "all baselines pass G0" not in text, "README must not claim all baselines pass G0"
    assert "exact reproduction" not in text, "README must not claim exact reproduction"


def test_readme_points_to_v0_4_config_and_inspector() -> None:
    text = README.read_text(encoding="utf-8")
    assert "configs/g0_rag_bot_faithful_replay.yaml" in text, "README must point to the v0.4 config"
    assert "scripts/inspect_g0_rag_bot_fidelity.py" in text, (
        "README must point to the v0.4 fidelity inspector"
    )
    assert "g0_rag_bot_faithful_replay" in text, "README must reference the v0.4 canonical run id"


def test_v05_doc_exists() -> None:
    assert V05_DOC.is_file(), "v0.5 release report must exist"


def test_v05_doc_states_exact_contract() -> None:
    text = V05_DOC.read_text(encoding="utf-8")
    assert "3 tasks × 3 baselines × 3 arms × 2 models = 162 trials" in text, (
        "v0.5 doc must state the exact 162-row matrix contract"
    )

    counts = [
        ("full_history_generate=54", "full_history_generate"),
        ("reflexion_generate=54", "reflexion_generate"),
        ("reflexion_reflect=6", "reflexion_reflect"),
        ("dynamic_cheatsheet_generate=54", "dynamic_cheatsheet_generate"),
        ("dynamic_cheatsheet_curate=54", "dynamic_cheatsheet_curate"),
    ]
    for expected, label in counts:
        assert expected in text, f"v0.5 doc must report stage count {label}"

    assert "54 per baseline" in text, "v0.5 doc must state 54 trials per baseline"
    assert "18 per baseline/arm" in text, "v0.5 doc must state 18 trials per baseline/arm"
    assert "222 total method calls" in text, "v0.5 doc must state the 222-call total"
    assert "6 Reflexion reflected trials (game24_pilot_001)" in text, (
        "v0.5 doc must anchor the six Reflexion reflections to game24_pilot_001"
    )
    assert "6 DC preserved_missing_tag rows (game24_pilot_001)" in text, (
        "v0.5 doc must anchor the six DC fallback rows to game24_pilot_001"
    )


def test_v05_doc_contains_approved_baseline_labels() -> None:
    text = V05_DOC.read_text(encoding="utf-8")
    labels = [
        "faithful append-only full-history",
        "Reflexion-style verbal memory proxy / faithful adapted control flow",
        "faithful adapted DC-Cu optional appendix comparator",
    ]
    for label in labels:
        assert label in text, f"v0.5 doc must use approved baseline label: {label}"


def test_v05_doc_contains_limitation_language() -> None:
    text = V05_DOC.read_text(encoding="utf-8")
    assert "not an exact reproduction" in text, "v0.5 doc must contain explicit limitation language"


def test_v05_doc_contains_official_sources_and_shas() -> None:
    text = V05_DOC.read_text(encoding="utf-8")
    assert "https://github.com/noahshinn/reflexion" in text, (
        "v0.5 doc must cite the Reflexion repository"
    )
    assert "218cf0ef1df84b05ce379dd4a8e47f17766733a0" in text, (
        "v0.5 doc must pin the full Reflexion SHA"
    )
    assert "https://github.com/suzgunmirac/dynamic-cheatsheet" in text, (
        "v0.5 doc must cite the Dynamic Cheatsheet repository"
    )
    assert "5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9" in text, (
        "v0.5 doc must pin the full Dynamic Cheatsheet SHA"
    )
    assert text.count("MIT") >= 2, (
        "v0.5 doc must include MIT license attribution for both official sources"
    )


def _affirmative_exact_reproduction(text: str) -> bool:
    return "exact reproduction" in text.replace("not an exact reproduction", "")


def test_v05_doc_does_not_contain_forbidden_overclaims() -> None:
    text = V05_DOC.read_text(encoding="utf-8")
    forbidden = [
        "all baselines pass G0",
        "benchmark result",
        "backbone-independent",
        "admission-control proof",
    ]
    for phrase in forbidden:
        assert phrase not in text, f"v0.5 doc must not contain forbidden overclaim: {phrase!r}"
    assert not _affirmative_exact_reproduction(text), (
        "v0.5 doc must not make an affirmative exact-reproduction claim"
    )


def test_readme_contains_v05_section() -> None:
    text = README.read_text(encoding="utf-8")
    assert "## v0.5" in text, "README must contain a top-level v0.5 section"
    for baseline in ["full_history", "reflexion_style", "dynamic_cheatsheet_optional"]:
        assert baseline in text, f"README v0.5 section must name {baseline}"
    assert "configs/g0_fh_reflexion_dc_faithful_replay.yaml" in text, (
        "README must point to the v0.5 config"
    )
    assert "scripts/inspect_g0_fh_reflexion_dc_fidelity.py" in text, (
        "README must point to the v0.5 inspector"
    )
    assert "g0_fh_reflexion_dc_faithful_replay" in text, (
        "README must reference the v0.5 canonical run id"
    )
    assert "docs/g0-baseline-fidelity-gate-v0.5.md" in text, "README must link to the v0.5 report"
    assert "docs/g0-baseline-fidelity-gate-v0.4.md" in text, "README must keep the v0.4 report link"
    assert "fidelity/QA artifact" in text, (
        "README must state that replay output is a fidelity/QA artifact"
    )
    assert "not benchmark or manuscript-quality evidence" in text, (
        "README must deny benchmark/manuscript-evidence status"
    )


def test_pyproject_version_remains_0_1_0() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.1.0"' in text, "pyproject.toml must remain at version 0.1.0"


def test_followup_doc_exists() -> None:
    assert FOLLOWUP_DOC.is_file(), "v0.5+ follow-up report must exist"


def test_followup_doc_contains_section_title_and_scope() -> None:
    text = FOLLOWUP_DOC.read_text(encoding="utf-8")
    assert "# G0 DC-RS and Reflexion Same-Sample Retry Fidelity Follow-up" in text, (
        "Follow-up doc must use the approved title"
    )
    assert "post-`v0.5` follow-up" in text, "Follow-up doc must identify itself as post-v0.5"
    assert "optional appendix comparator" in text, (
        "Follow-up doc must label DC-RS as optional appendix comparator"
    )
    assert "same-sample retry" in text, "Follow-up doc must mention same-sample retry"


def test_followup_doc_contains_official_sources() -> None:
    text = FOLLOWUP_DOC.read_text(encoding="utf-8")
    assert "https://aclanthology.org/2026.eacl-long.333/" in text, (
        "Follow-up doc must cite the Dynamic Cheatsheet paper"
    )
    assert "https://github.com/suzgunmirac/dynamic-cheatsheet" in text, (
        "Follow-up doc must cite the Dynamic Cheatsheet repository"
    )
    assert "https://arxiv.org/abs/2303.11366" in text, "Follow-up doc must cite the Reflexion paper"
    assert "https://github.com/noahshinn/reflexion" in text, (
        "Follow-up doc must cite the Reflexion repository"
    )


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


def test_followup_doc_contains_adaptation_table() -> None:
    text = FOLLOWUP_DOC.read_text(encoding="utf-8")
    assert "| Must Preserve | Safely Adapted | Omitted |" in text, (
        "Follow-up doc must contain a three-column adaptation table"
    )
    assert "top-3 cosine" in text, "Follow-up doc must mention top-3 cosine retrieval"
    assert "same-identity" in text, "Follow-up doc must mention same-identity retrieval"
    assert "no weight updates" in text, "Follow-up doc must state no weight updates"


def test_followup_doc_contains_exact_artifacts_and_counts() -> None:
    text = FOLLOWUP_DOC.read_text(encoding="utf-8")
    assert "configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml" in text, (
        "Follow-up doc must name the exact config"
    )
    assert "data/replay/g0_dc_rs_reflexion_fidelity_followup_v1.yaml" in text, (
        "Follow-up doc must name the exact fixture"
    )
    assert "g0_dc_rs_reflexion_fidelity_followup_replay" in text, (
        "Follow-up doc must name the exact run id"
    )
    assert "scripts/inspect_g0_dc_rs_reflexion_fidelity.py" in text, (
        "Follow-up doc must name the exact inspector"
    )
    assert "108 trial rows" in text, "Follow-up doc must state 108 trial rows"
    assert "174 native method calls" in text, "Follow-up doc must state 174 native method calls"
    assert '"dc_rs_calls": 108' in text, "Follow-up doc must record dc_rs_calls 108"
    assert '"reflexion_calls": 66' in text, "Follow-up doc must record reflexion_calls 66"
    assert '"method_calls": 174' in text, "Follow-up doc must record method_calls 174"
    assert '"trials": 108' in text, "Follow-up doc must record trials 108"


def test_followup_doc_contains_bounded_claim_phrases() -> None:
    text = FOLLOWUP_DOC.read_text(encoding="utf-8")
    assert (
        "Faithful adapted DC-RS optional appendix comparator: top-3 cosine retrieval over prior same-identity input/output pairs, label-free pre-answer cheatsheet synthesis, then memory-conditioned generation, with native method-call costs logged."
        in text
    ), "Follow-up doc must contain the bounded DC-RS claim verbatim"
    assert (
        "Faithful adapted Reflexion control flow: failed trajectory plus sanitized evaluator feedback produces linguistic reflection, latest-three reflection memory conditions a same-sample retry, stopping on success or attempt limit; no weight updates."
        in text
    ), "Follow-up doc must contain the bounded Reflexion claim verbatim"


def test_followup_doc_does_not_contain_forbidden_overclaims() -> None:
    text = FOLLOWUP_DOC.read_text(encoding="utf-8")
    forbidden = [
        "full reproduction",
        "benchmark improvement",
        "main DC baseline",
        "benchmark evidence",
        "manuscript evidence",
    ]
    for phrase in forbidden:
        assert phrase not in text, f"Follow-up doc must not contain forbidden overclaim: {phrase!r}"


def test_readme_contains_followup_section() -> None:
    text = README.read_text(encoding="utf-8")
    assert "## v0.5+ DC-RS and Reflexion Same-Sample Retry Follow-up" in text, (
        "README must contain the follow-up section above v0.5"
    )
    assert "`v0.5` remains the historical full G0 baseline-fidelity pass" in text, (
        "README must state that v0.5 remains the historical full pass"
    )
    assert "configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml" in text, (
        "README must point to the follow-up config"
    )
    assert "scripts/inspect_g0_dc_rs_reflexion_fidelity.py" in text, (
        "README must point to the follow-up inspector"
    )
    assert "g0_dc_rs_reflexion_fidelity_followup_replay" in text, (
        "README must reference the follow-up canonical run id"
    )
    assert "docs/g0-dc-rs-reflexion-fidelity-followup.md" in text, (
        "README must link to the follow-up report"
    )


def test_readme_contains_followup_bounded_claim_phrases() -> None:
    text = README.read_text(encoding="utf-8")
    assert (
        "Faithful adapted DC-RS optional appendix comparator: top-3 cosine retrieval over prior same-identity input/output pairs, label-free pre-answer cheatsheet synthesis, then memory-conditioned generation, with native method-call costs logged."
        in text
    ), "README must contain the bounded DC-RS claim verbatim"
    assert (
        "Faithful adapted Reflexion control flow: failed trajectory plus sanitized evaluator feedback produces linguistic reflection, latest-three reflection memory conditions a same-sample retry, stopping on success or attempt limit; no weight updates."
        in text
    ), "README must contain the bounded Reflexion claim verbatim"


def test_readme_and_followup_forbid_overclaim_phrases() -> None:
    for path in (README, FOLLOWUP_DOC):
        text = path.read_text(encoding="utf-8")
        for phrase in [
            "full reproduction",
            "benchmark improvement",
            "main DC baseline",
            "benchmark evidence",
            "manuscript evidence",
        ]:
            assert phrase not in text, f"{path.name} must not contain forbidden phrase: {phrase!r}"


def test_phase11_docs_exist_and_name_executable_contract() -> None:
    assert PHASE11_CONTRACT_DOC.is_file()
    assert PHASE11_REPORT_DOC.is_file()
    text = PHASE11_CONTRACT_DOC.read_text(encoding="utf-8")
    for phrase in [
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
    ]:
        assert phrase in text, f"Phase-11 contract doc must contain {phrase!r}"


def test_phase11_docs_forbid_scope_overclaims() -> None:
    texts = [
        PHASE11_CONTRACT_DOC.read_text(encoding="utf-8"),
        PHASE11_REPORT_DOC.read_text(encoding="utf-8"),
        README.read_text(encoding="utf-8"),
    ]
    forbidden = [
        "v1 is Phase-11 evidence",
        "v1 artifacts are Phase-11 evidence",
        "replay gate is a pilot",
        "replay gate is the main run",
        "replay gate is a benchmark",
        "approximate lineage is exact derivation",
        "exposure is causal use",
        "complete PROV-DM implementation",
    ]
    for text in texts:
        for phrase in forbidden:
            assert phrase not in text, f"docs must not contain overclaim: {phrase!r}"


def test_phase11_report_is_unfilled_template() -> None:
    text = PHASE11_REPORT_DOC.read_text(encoding="utf-8")
    assert "report template" in text
    assert "Status:** `TEMPLATE`" in text
    assert "[ ] python -m pytest tests/test_phase11_logging_contract_gate.py -q" in text
    assert "No API-connected pilot was run." in text


def test_phase11_config_and_readme_links_are_current() -> None:
    config = yaml.safe_load(PHASE11_CONFIG.read_text(encoding="utf-8"))
    assert config["logging"]["schema_version"] == "logging_v2"
    assert config["run"]["contract_level"] == "phase11"
    readme = README.read_text(encoding="utf-8")
    for path in [
        "docs/logging-contract-v2-phase11.md",
        "docs/logging-audit-remediation-phase11.md",
        "configs/logging_contract_phase11_replay.yaml",
    ]:
        assert path in readme


def test_readme_merges_phase11_release_and_status_sections() -> None:
    readme = README.read_text(encoding="utf-8")
    assert "## v0.7.2 Phase-11 `logging_v2` Contract Release and Status" in readme
    assert "| `phase11` |" not in readme
    assert "## Phase-11 `logging_v2` Contract Status" not in readme
    assert readme.count("configs/logging_contract_phase11_replay.yaml") >= 1


def test_v072_release_doc_exists_and_stays_within_scope() -> None:
    assert LOGGING_RELEASE_V072_DOC.is_file()
    text = LOGGING_RELEASE_V072_DOC.read_text(encoding="utf-8")
    for phrase in [
        "Tag:** `v0.7.2`",
        "configs/logging_contract_phase11_replay.yaml",
        "--contract phase11",
        "No API-connected pilot was run.",
        "not a live pilot, main run, benchmark result, or manuscript result",
    ]:
        assert phrase in text


def test_phase11_readme_links_include_v072_release_doc() -> None:
    readme = README.read_text(encoding="utf-8")
    for path in [
        "docs/logging-audit-remediation-v0.7.2.md",
        "docs/logging-contract-v2-phase11.md",
        "docs/logging-audit-remediation-phase11.md",
    ]:
        assert path in readme
