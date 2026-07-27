from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from memcontam.clients.config import ProviderConfig
from memcontam.memory.cards_v3 import MEMORY_CARD_V3, MemoryCardEnvelopeV3, canonical_content_hash
from memcontam.readiness.pilot_a_scientific_records import ROW_NAMES, record_suffix


ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC_CONFIG = ROOT / "configs" / "phase12" / "pilot_a_game24_scientific.yaml"


def _envelope(entry_id: str, content: str) -> MemoryCardEnvelopeV3:
    trial_id = f"trial-{entry_id}"
    return MemoryCardEnvelopeV3(
        entry_id=entry_id,
        baseline="fh_bounded",
        semantic_kind="full_history_transcript",
        schema_version=MEMORY_CARD_V3,
        writer_id="fh_appender",
        writer_event_id=f"event-{entry_id}",
        writer_stage="full_history_generate",
        created_trial_id=trial_id,
        source_trial_ids=(trial_id,),
        source_outcome=None,
        trial_support_ids=(trial_id,),
        memory_support_ids=(),
        direct_parent_ids=(),
        version_predecessor_id=None,
        order_key=1,
        native_component="history",
        content=content,
        content_hash=canonical_content_hash(content),
    )


def test_scientific_filter_context_uses_recorded_prefix_write_evidence() -> None:
    scientific = importlib.import_module("memcontam.readiness.pilot_a_scientific")
    recorded = _envelope("recorded-prefix-write", "recorded content")

    context = scientific._admission_context("fh_bounded", (SimpleNamespace(write_envelopes=(recorded,)),))

    assert context.evidence_envelopes == (recorded,)
    assert context.active_envelopes == (recorded,)
    assert context.writer_event_ids == {recorded.writer_event_id}
    assert context.trial_record_ids == set(recorded.trial_support_ids)


def test_scientific_config_binds_filter_v4_claim_boundary() -> None:
    config = yaml.safe_load(SCIENTIFIC_CONFIG.read_text(encoding="utf-8"))

    assert config["filter_policy_version"] == "operational-evidence-filter-v4"
    assert config["filter_interpretation"] == "contract_invalid_direct_write_containment"
    assert config["filter_claim_status"] == "operational_secondary"


def test_scientific_prefix_has_no_silent_reflexion_bootstrap_and_persists_each_seed(
    monkeypatch,
) -> None:  # noqa: ANN001
    scientific = importlib.import_module("memcontam.readiness.pilot_a_scientific")
    launch_tests = importlib.import_module("tests.test_phase12_pilot_a_launch")
    config = yaml.safe_load(SCIENTIFIC_CONFIG.read_text(encoding="utf-8"))
    base = launch_tests._runtime_context(launch_tests._Client(), "run", "model")
    base.initial_states["reflexion_style"].reflections.clear()
    task_ids = {
        task_id
        for seed in config["trajectory_seeds"]
        for task_id in seed["ordered_prefix_task_ids"]
    }
    instances = {
        task_id: base.task.model_copy(update={"sample_id": task_id}) for task_id in task_ids
    }
    persisted: list[bool] = []

    monkeypatch.setattr(scientific, "_instances", lambda _config: instances)
    monkeypatch.setattr(scientific, "load_candidate_registry", lambda _path: SimpleNamespace())
    monkeypatch.setattr(scientific, "record_prefix", lambda *_args: None)

    def blocked_prefix(*, contexts, **_kwargs):  # noqa: ANN001
        assert all(
            not context.initial_states["reflexion_style"].reflections for context in contexts
        )
        return SimpleNamespace(selection=SimpleNamespace(blocked=True))

    monkeypatch.setattr(scientific, "run_live_clean_prefix", blocked_prefix)
    scientific._run_seeds(
        config,
        "run",
        launch_tests._Client(),
        ProviderConfig.from_run_config(config),
        lambda *_args: base,
        rows={name: [] for name in scientific.ROW_NAMES},
        seal_progress=lambda: persisted.append(True),
    )

    assert persisted == [True, True]


def test_suffix_log_records_the_actual_filter_partition_decision(
    monkeypatch,
) -> None:  # noqa: ANN001
    records = importlib.import_module("memcontam.readiness.pilot_a_scientific_records")
    monkeypatch.setattr(records, "_record_suffix_trial", lambda *_args: None)
    rows = {name: [] for name in ROW_NAMES}
    clean_trial = SimpleNamespace(baseline="fh_bounded", arm="clean", suffix_id="task-1")
    filter_trial = SimpleNamespace(baseline="fh_bounded", arm="filter", suffix_id="task-1")
    branches = {
        "fh_bounded": SimpleNamespace(
            arms={
                "clean": SimpleNamespace(
                    checkpoint=SimpleNamespace(identity=SimpleNamespace(checkpoint_id="checkpoint"))
                ),
                "filter": SimpleNamespace(
                    injected_root_id="direct-write",
                    filter_state=SimpleNamespace(
                        decisions=(
                            SimpleNamespace(
                                entry_id="direct-write", state="active", reason=None
                            ),
                        )
                    ),
                ),
            },
            events=(),
        )
    }

    record_suffix(
        rows,
        0,
        {"fh_bounded": SimpleNamespace(trials=(clean_trial, filter_trial))},
        (),
        branches,
        None,
    )

    assert rows["admission_events"] == [
        {
            "decision": "active",
            "entry_id": "direct-write",
            "event_id": "seed:0:memory_branch:fh_bounded:filter:task-1:admission",
            "policy_version": "operational-evidence-filter-v4",
            "reason": None,
            "record_type": "admission_event",
            "schema_version": "logging_v3",
            "trial_id": "seed:0:memory_branch:fh_bounded:filter:task-1",
        }
    ]


def test_rag_filter_keeps_the_quarantined_document_for_audit_but_not_the_index() -> None:
    from memcontam.rag.branch_index import build_branch_indices
    from memcontam.rag.phase12_corpus import CleanCorpus, build_branch_corpora
    from tests.test_phase12_rag_corpus import FixtureEmbedder

    fixture = json.loads(
        (ROOT / "tests" / "fixtures" / "phase12" / "FX-RAG-001.json").read_text(encoding="utf-8")
    )
    corpora = build_branch_corpora(
        CleanCorpus.from_documents(fixture["clean_corpus"], corpus_id="game24-clean-v1"),
        fixture["triplet_documents"],
    )
    indices = build_branch_indices(corpora, FixtureEmbedder(fixture), filter_policy=None)

    assert tuple(document.document_id for document in corpora.branches["filter"].documents) == (
        "doc-clean-a",
        "doc-clean-b",
        "doc-false",
    )
    assert tuple(document.document_id for document in indices.branches["filter"].documents) == (
        "doc-clean-a",
        "doc-clean-b",
        "doc-false",
    )


def test_filter_v4_mft_covers_each_baseline_route_and_content_class(tmp_path: Path) -> None:
    module = importlib.import_module("memcontam.experiment.phase12.filter_mft")
    output = tmp_path / "filter-v4-mft.json"

    report = module.write_filter_mft_report(output)

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["schema_version"] == "phase12_filter_mft_v4"
    assert report["evidence_layer"] == "build_calibration_only"
    assert report["scientific_result"] is False
    assert report["excluded_policy_fields"] == ["audit_label", "candidate_role", "treatment_arm"]
    assert {
        (case["baseline"], case["route_valid"], case["content_class"])
        for case in report["cases"]
    } == {
        (baseline, route_valid, content_class)
        for baseline in ("fh_bounded", "rag_frozen", "bot_style", "reflexion_style")
        for route_valid in (False, True)
        for content_class in ("correct", "false")
    }
    assert all(
        case["observed_state"] == ("active" if case["route_valid"] else "quarantine")
        for case in report["cases"]
    )
    assert all(
        {
            "case_id",
            "route_class",
            "expected_decision",
            "actual_decision",
            "reason_code",
            "passed",
        }.issubset(case)
        and case["passed"]
        for case in report["cases"]
    )
