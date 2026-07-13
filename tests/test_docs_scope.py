from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
G0_DOC = ROOT / "docs" / "g0-baseline-fidelity-gate-v0.4.md"
README = ROOT / "README.md"
V05_DOC = ROOT / "docs" / "g0-baseline-fidelity-gate-v0.5.md"


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
        assert phrase not in text, (
            f"G0 doc must not contain misleading claim: {phrase!r}"
        )


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
    assert "not full reproduction" in text, (
        "README must retain the no-full-reproduction caveat"
    )
    assert "all baselines pass G0" not in text, (
        "README must not claim all baselines pass G0"
    )
    assert "exact reproduction" not in text, (
        "README must not claim exact reproduction"
    )


def test_readme_points_to_v0_4_config_and_inspector() -> None:
    text = README.read_text(encoding="utf-8")
    assert "configs/g0_rag_bot_faithful_replay.yaml" in text, (
        "README must point to the v0.4 config"
    )
    assert "scripts/inspect_g0_rag_bot_fidelity.py" in text, (
        "README must point to the v0.4 fidelity inspector"
    )
    assert "g0_rag_bot_faithful_replay" in text, (
        "README must reference the v0.4 canonical run id"
    )



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
    assert "not an exact reproduction" in text, (
        "v0.5 doc must contain explicit limitation language"
    )


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
        assert phrase not in text, (
            f"v0.5 doc must not contain forbidden overclaim: {phrase!r}"
        )
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
    assert "docs/g0-baseline-fidelity-gate-v0.5.md" in text, (
        "README must link to the v0.5 report"
    )
    assert "docs/g0-baseline-fidelity-gate-v0.4.md" in text, (
        "README must keep the v0.4 report link"
    )
    assert "fidelity/QA artifact" in text, (
        "README must state that replay output is a fidelity/QA artifact"
    )
    assert "not benchmark/manuscript evidence" in text, (
        "README must deny benchmark/manuscript-evidence status"
    )


def test_pyproject_version_remains_0_1_0() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.1.0"' in text, "pyproject.toml must remain at version 0.1.0"
