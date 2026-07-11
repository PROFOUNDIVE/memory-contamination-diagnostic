from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
G0_DOC = ROOT / "docs" / "g0-baseline-fidelity-gate-v0.4.md"
README = ROOT / "README.md"


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
