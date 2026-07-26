from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

import memcontam.cli as cli
from memcontam.baselines.bot_phase12 import BoTStateV3
from memcontam.baselines.full_history_phase12 import FullHistoryStateV3
from memcontam.baselines.reflexion_phase12 import ReflexionStateV3
from memcontam.baselines.retrieval_rag_phase12 import RagFrozenStateV3
from memcontam.clients.base import LLMResponse
from memcontam.clients.base import LLMClient
from memcontam.clients.cost_guard import CostGuard
from memcontam.experiment.phase12.game24_runner import Game24RuntimeContext, RuntimeIdentities
from memcontam.memory.checkpoint_v3 import NativeEntry
from memcontam.memory.stores import MemoryEntry
from memcontam.rag.branch_index import build_branch_indices
from memcontam.rag.phase12_corpus import CleanCorpus, build_branch_corpora
from memcontam.tasks.game24 import build_instance
from memcontam.verifiers.game24 import verify_expression


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase12" / "pilot_a_game24_minimal.yaml"


def _module():
    return importlib.import_module("memcontam.readiness.pilot_a_launch")


def _run_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["memcontam", *args])
    cli.main()


def _write_evidence(root: Path, name: str, payload: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(payload), encoding="utf-8")


def _passing_evidence(root: Path) -> None:
    for name in ("t5-f1c.json", "t5-micro-retrieval.json", "t6-invariants.json", "t6-archive.json"):
        _write_evidence(root, name, {"overall": "pass"})
    _write_evidence(
        root,
        "t7-plumbing.json",
        {
            "arm": "Clean",
            "baselines": ["nomem", "fh_bounded", "rag_frozen", "bot_style", "reflexion_style"],
            "instance_count": 1,
            "overall": "pass",
            "scientific_result": False,
        },
    )


class _Client:
    def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
        del messages, model
        answer = "final: 6 / (1 - 3 / 4)"
        responses = {
            "no_memory_generate": answer,
            "full_history_generate": answer,
            "rag_generate": answer,
            "bot_problem_distill": json.dumps(
                {
                    "key_information": "numbers = [1, 3, 4, 6], target = 24",
                    "restrictions": "Use every number exactly once.",
                    "distilled_task": "Construct an expression equal to 24.",
                }
            ),
            "bot_instantiate_solve": json.dumps(
                {
                    "selected_structure": "retrieved-template",
                    "solution_trace": "Use rational intermediate values.",
                    "final_answer": answer,
                }
            ),
            "bot_thought_distill": json.dumps(
                {
                    "description": "Use rational intermediate values.",
                    "template": "Build exact fractions before combining values.",
                    "category": "procedure-based",
                    "explicitly_used_memory_ids": [],
                }
            ),
            "reflexion_generate": answer,
        }
        return LLMResponse(content=responses[config["method_stage"]], raw={}, token_usage={}, latency_ms=0)


class _Embedder:
    embedding_contract = {
        "dimension": 2,
        "normalized": True,
        "production_identity": "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181",
        "provider": "test",
    }

    def encode_document(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]

    def encode_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


class _MalformedBoTClient(_Client):
    def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
        if config["method_stage"] == "bot_thought_distill":
            return LLMResponse(content="not json", raw={}, token_usage={}, latency_ms=0)
        return super().chat(messages, model, config)


class _MalformedBoTSolveClient(_Client):
    def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
        if config["method_stage"] == "bot_instantiate_solve":
            return LLMResponse(content="not json", raw={}, token_usage={}, latency_ms=0)
        return super().chat(messages, model, config)


def _runtime_context(client: LLMClient, run_id: str, model: str) -> Game24RuntimeContext:
    embedder = _Embedder()
    corpora = build_branch_corpora(
        CleanCorpus.from_documents(
            [
                {"id": "clean-a", "text": "Use rational intermediate values."},
                {"id": "clean-b", "text": "Check arithmetic exactly."},
                {"id": "clean-c", "text": "Use all four values exactly once."},
            ],
            corpus_id="game24-clean",
        ),
        {
            "false": {"id": "false", "text": "Use only integer intermediates."},
            "correct": {"id": "correct", "text": "Fractions are valid."},
            "irrelevant": {"id": "irrelevant", "text": "Sort alphabetically."},
        },
    )
    indices = build_branch_indices(corpora, embedder, filter_policy=None)
    bot_entries: list[MemoryEntry | NativeEntry] = [
        MemoryEntry(
            entry_id="bot-a",
            content="Use rational intermediate values.",
            memory_type="thought_template",
            metadata={"description": "Use rational intermediate values.", "category": "procedure-based"},
        ),
        MemoryEntry(
            entry_id="bot-b",
            content="Check arithmetic exactly.",
            memory_type="thought_template",
            metadata={"description": "Check arithmetic exactly.", "category": "procedure-based"},
        ),
    ]
    task = build_instance({"sample_id": "game24-build-1", "numbers": [1, 3, 4, 6]})
    return Game24RuntimeContext(
        task=task,
        client=client,
        model=model,
        verifier=lambda answer, seen_task: verify_expression(
            answer, seen_task.input["numbers"], seen_task.verifier_spec["target"]
        ),
        decoding={"temperature": 0.0, "top_p": 1.0, "max_output_tokens": 2048},
        branch="clean",
        identities=RuntimeIdentities(run_id, f"{run_id}:build-1", 1),
        embedding_provider=embedder,
        baseline_configs={"fh_bounded": {"context_window_tokens": 10_000}},
        initial_states={
            "fh_bounded": FullHistoryStateV3(records=[]),
            "rag_frozen": RagFrozenStateV3("clean", corpora.branches["clean"], indices.branches["clean"]),
            "bot_style": BoTStateV3(
                entries=bot_entries, clean_competitor_ids=("bot-a", "bot-b"), active_capacity=3
            ),
            "reflexion_style": ReflexionStateV3(reflections=[], active_capacity=3),
        },
    )


def test_cost_preview_is_safe_for_the_frozen_plumbing_plan_and_fails_above_ceiling(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()

    _run_cli(monkeypatch, "phase12", "cost-preview", "--config", str(CONFIG))

    preview = json.loads(capsys.readouterr().out)
    assert preview["safe"] is True
    assert preview["scientific_result"] is False
    assert preview["projected_max_cost_usd"] < 5

    with pytest.raises(module.PilotALaunchError, match="PROJECTED_COST_EXCEEDS_CEILING"):
        module.cost_preview(
            CONFIG,
            cost_guard=CostGuard(input_per_million_usd=100_000, output_per_million_usd=100_000),
        )


@pytest.mark.parametrize(
    ("arm", "instances", "allow_live_calls", "scientific_result", "code"),
    [
        ("Contam", 1, True, False, "CLEAN_ARM_REQUIRED"),
        ("Filter", 1, True, False, "CLEAN_ARM_REQUIRED"),
        ("Clean", 2, True, False, "ONE_INSTANCE_REQUIRED"),
        ("Clean", 1, False, False, "LIVE_CALL_AUTHORIZATION_REQUIRED"),
        ("Clean", 1, True, True, "PLUMBING_SCIENTIFIC_RESULT_FORBIDDEN"),
    ],
)
def test_plumbing_guards_reject_before_constructing_a_provider(
    arm: str, instances: int, allow_live_calls: bool, scientific_result: bool, code: str
) -> None:
    module = _module()
    constructed = False

    def unexpected_factory() -> object:
        nonlocal constructed
        constructed = True
        raise AssertionError("provider construction must not be reached")

    with pytest.raises(module.PilotALaunchError, match=code):
        module.run_plumbing(
            CONFIG,
            arm=arm,
            instances=instances,
            allow_live_calls=allow_live_calls,
            scientific_result=scientific_result,
            client_factory=unexpected_factory,
        )

    assert constructed is False


def test_mocked_clean_plumbing_writes_validated_non_scientific_archive(tmp_path: Path) -> None:
    module = _module()
    evidence_root = tmp_path / "evidence"

    report = module.run_plumbing(
        CONFIG,
        arm="Clean",
        instances=1,
        allow_live_calls=True,
        scientific_result=False,
        run_id="plumbing",
        artifact_root=tmp_path,
        evidence_root=evidence_root,
        client_factory=_Client,
        context_factory=_runtime_context,
    )

    assert report["overall"] == "pass"
    assert report["baselines"] == ["nomem", "fh_bounded", "rag_frozen", "bot_style", "reflexion_style"]
    assert report["scientific_result"] is False
    assert report["unresolved_references"] == 0
    assert report["hash_mismatches"] == 0
    assert (evidence_root / "t7-plumbing.json").is_file()


def test_mocked_clean_plumbing_logs_a_bot_parse_failure_without_aborting(tmp_path: Path) -> None:
    module = _module()

    report = module.run_plumbing(
        CONFIG,
        arm="Clean",
        instances=1,
        allow_live_calls=True,
        scientific_result=False,
        run_id="plumbing-failure",
        artifact_root=tmp_path,
        evidence_root=tmp_path / "evidence",
        client_factory=_MalformedBoTClient,
        context_factory=_runtime_context,
    )

    assert report["overall"] == "pass"
    failures = (tmp_path / "runs" / "plumbing-failure" / "failures.jsonl").read_text(encoding="utf-8")
    assert "bot_style" in failures


def test_mocked_clean_plumbing_accepts_a_prefix_of_bot_calls_after_parse_failure(tmp_path: Path) -> None:
    module = _module()

    report = module.run_plumbing(
        CONFIG,
        arm="Clean",
        instances=1,
        allow_live_calls=True,
        scientific_result=False,
        run_id="plumbing-solve-failure",
        artifact_root=tmp_path,
        evidence_root=tmp_path / "evidence",
        client_factory=_MalformedBoTSolveClient,
        context_factory=_runtime_context,
    )

    assert report["overall"] == "pass"


def test_clean_plumbing_archive_cli_mode_writes_a_canonical_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    module.run_plumbing(
        CONFIG,
        arm="Clean",
        instances=1,
        allow_live_calls=True,
        scientific_result=False,
        run_id="plumbing-cli",
        artifact_root=tmp_path,
        evidence_root=tmp_path / "evidence",
        client_factory=_Client,
        context_factory=_runtime_context,
    )
    output = tmp_path / "t7-plumbing.json"

    _run_cli(
        monkeypatch,
        "phase12",
        "validate-archive",
        "--run-dir",
        str(tmp_path / "runs" / "plumbing-cli"),
        "--mode",
        "clean-plumbing",
        "--output",
        str(output),
    )

    report = json.loads(capsys.readouterr().out)
    assert report["overall"] == "pass"
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert output.read_text(encoding="utf-8") == json.dumps(
        report, sort_keys=True, separators=(",", ":")
    ) + "\n"


def test_admission_only_requires_all_t5_t6_t7_evidence_and_never_starts_a_run(tmp_path: Path) -> None:
    module = _module()
    evidence_root = tmp_path / "evidence"
    _passing_evidence(evidence_root)

    admitted = module.evaluate_pilot_a_admission(CONFIG, evidence_root=evidence_root)

    assert admitted == {
        "admitted": True,
        "reason_code": None,
        "scientific_result": False,
    }
    (evidence_root / "t7-plumbing.json").unlink()

    blocked = module.evaluate_pilot_a_admission(CONFIG, evidence_root=evidence_root)

    assert blocked == {
        "admitted": False,
        "reason_code": "T7_PLUMBING_EVIDENCE_REQUIRED",
        "scientific_result": False,
    }


def test_pilot_a_cli_admission_only_reads_evidence_without_starting_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    evidence_root = tmp_path / "evidence"
    _passing_evidence(evidence_root)
    monkeypatch.setattr(module, "DEFAULT_EVIDENCE_ROOT", evidence_root)

    _run_cli(monkeypatch, "phase12", "pilot-a", "--config", str(CONFIG), "--admission-only")

    assert json.loads(capsys.readouterr().out)["admitted"] is True


def test_blocked_handoff_records_real_hashes_without_faking_plumbing(tmp_path: Path) -> None:
    module = _module()
    evidence_root = tmp_path / "evidence"
    for name in ("t5-f1c.json", "t6-invariants.json"):
        _write_evidence(evidence_root, name, {"overall": "pass"})

    handoff = module.write_handoff(
        CONFIG, evidence_root=evidence_root, destination=evidence_root / "t7-handoff.json"
    )

    assert handoff["status"] == "BLOCKED_AWAITING_HUMAN_AUTHORIZATION"
    assert handoff["scientific_result"] is False
    assert handoff["f1c_evidence"]["sha256"]
    assert handoff["invariant_report"]["sha256"]
    assert handoff["plumbing_archive"] is None
