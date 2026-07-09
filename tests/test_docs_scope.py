from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
G0_DOC = ROOT / "docs" / "g0-baseline-fidelity-gate-v0.2.md"
README = ROOT / "README.md"


def test_g0_doc_states_partial_rag_bot_scope() -> None:
    text = G0_DOC.read_text(encoding="utf-8")
    assert "partial G0 pass: RAG + BoT only" in text, (
        "G0 doc must state the implementation plan targets a partial RAG + BoT pass"
    )

    out_of_scope = [
        "no_memory",
        "full_history",
        "reflexion_style",
        "Dynamic Cheatsheet",
        "ExpeL",
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
