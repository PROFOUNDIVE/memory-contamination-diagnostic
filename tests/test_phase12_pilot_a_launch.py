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
from memcontam.clients.cost_guard import CostGuard, CostLimitExceeded
from memcontam.experiment.phase12.game24_runner import (
    Game24RuntimeContext,
    RuntimeIdentities,
    run_clean_game24_trial,
)
from memcontam.memory.checkpoint_v3 import NATIVE_ENTRY_V1, NativeEntry
from memcontam.memory.cards_v3 import canonical_content_hash
from memcontam.memory.stores import MemoryEntry
from memcontam.rag.branch_index import build_branch_indices
from memcontam.rag.phase12_corpus import CleanCorpus, build_branch_corpora
from memcontam.tasks.game24 import build_instance
from memcontam.verifiers.game24 import verify_expression


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase12" / "pilot_a_game24_minimal.yaml"
SCIENTIFIC_CONFIG = ROOT / "configs" / "phase12" / "pilot_a_game24_scientific.yaml"
CHECKLIST = ROOT / "docs" / "phase12-pilot-a-operator-checklist.md"
LEGACY_PLUMBING_ARCHIVE = ROOT / "runs" / "runs" / "phase12-pilot-a-plumbing-r2"


def _module():
    return importlib.import_module("memcontam.readiness.pilot_a_launch")


def _run_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["memcontam", *args])
    cli.main()


def _write_evidence(root: Path, name: str, payload: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(payload), encoding="utf-8")


def _rewrite_archive_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "public_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, record in manifest["artifacts"].items():
        path = run_dir / filename
        record["count"] = len(path.read_text(encoding="utf-8").splitlines()) if filename.endswith(".jsonl") else 1
        record["sha256"] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _passing_evidence(root: Path) -> None:
    artifacts = {
        "filter-v4-mft.json": {
            "cases": [{"passed": True}],
            "policy_version": "operational-evidence-filter-v4",
            "scientific_result": False,
        },
        "phase12-filter-v4-f1c.json": {"overall": "pass"},
        "phase12-filter-v4-archive.json": {"overall": "pass"},
        "phase12-filter-v4-invariants.json": {"overall": "pass"},
    }
    evidence: dict[str, dict[str, str]] = {}
    for name, payload in artifacts.items():
        _write_evidence(root, name, payload)
        path = root / name
        evidence[name] = {
            "path": str(path),
            "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            "status": "pass",
        }
    _write_evidence(
        root,
        "pilot_a_readiness_manifest_phase12_filter_v4.json",
        {
            "config_hashes": {
                "scientific_pilot_a": __import__("hashlib").sha256(
                    SCIENTIFIC_CONFIG.read_bytes()
                ).hexdigest()
            },
            "evidence": evidence,
            "filter_policy": {
                "claim_status": "operational_secondary",
                "interpretation": "contract_invalid_direct_write_containment",
                "version": "operational-evidence-filter-v4",
            },
            "implementation_commit": _module()._implementation_commit(),
        },
    )


def _partial_scientific_rows(row_names: tuple[str, ...]) -> dict[str, list[dict[str, object]]]:
    rows: dict[str, list[dict[str, object]]] = {name: [] for name in row_names}
    trial_id = "seed:0:branch_free_prefix:fh_bounded:clean:game24-pilot-1"
    rows["trials"].append({"trial_id": trial_id, "trial_kind": "branch_free_prefix"})
    rows["calls"].append(
        {"trial_id": trial_id, "call_id": f"{trial_id}:call:1", "cost_usd": 0.01, "retry_count": 1}
    )
    rows["retrieval_events"].append({"trial_id": trial_id})
    rows["context_events"].append({"trial_id": trial_id})
    rows["eligibility_events"].append(
        {
            "trial_id": trial_id,
            "seed": 0,
            "condition_id": "fh_bounded-clean",
            "baseline": "fh_bounded",
            "baseline_family": "full_history",
            "checkpoint_id": "pilot-a-t1",
            "checkpoint_index": 1,
            "horizon": 1,
            "eligible": False,
            "reason_codes": ["TEST_ONLY_BLOCK"],
        }
    )
    rows["seed_status"].append(
        {
            "seed": 0,
            "eligible": False,
            "status": "blocked",
            "reason": "joint_checkpoint_blocked",
            "selected_checkpoint": None,
            "fallback_checkpoint_used": False,
            "joint_eligible_indices": [],
        }
    )
    return rows


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
            "reflexion_reflect": json.dumps(
                {
                    "mode": "corrective",
                    "failure_class": "incorrect_answer",
                    "reflection_text": "Check the arithmetic against the current four numbers.",
                    "explicitly_used_memory_ids": [],
                }
            ),
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


class _ContractSensitiveClient(_Client):
    def chat(self, messages, model, config) -> LLMResponse:  # noqa: ANN001
        stage = config["method_stage"]
        if stage in {"no_memory_generate", "full_history_generate", "rag_generate"}:
            prompt = "\n".join(message["content"] for message in messages)
            content = "final: 6 / (1 - 3 / 4)" if "final: <answer>" in prompt else "6 / (1 - 3 / 4)"
            return LLMResponse(content=content, raw={}, token_usage={}, latency_ms=0)
        if stage == "bot_instantiate_solve":
            prompt = "\n".join(message["content"] for message in messages)
            selected_structure = (
                "retrieved-template"
                if "Set selected_structure to retrieved-template" in prompt
                else "procedure-based"
            )
            final_answer = (
                "final: 6 / (1 - 3 / 4)"
                if 'final_answer must be "final: <answer>"' in prompt
                else "6 / (1 - 3 / 4)"
            )
            return LLMResponse(
                content=json.dumps(
                    {
                        "selected_structure": selected_structure,
                        "solution_trace": "Use rational intermediate values.",
                        "final_answer": final_answer,
                    }
                ),
                raw={},
                token_usage={},
                latency_ms=0,
            )
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
            "reflexion_style": ReflexionStateV3(
                reflections=[
                    NativeEntry(
                        entry_id="reflexion-clean-a",
                        content="Check the final arithmetic and use all four numbers once.",
                        semantic_kind="verbal_reflection",
                        schema_version=NATIVE_ENTRY_V1,
                        native_component="reflections",
                        content_hash=canonical_content_hash(
                            "Check the final arithmetic and use all four numbers once."
                        ),
                    )
                ],
                active_capacity=3,
            ),
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


def test_operator_checklist_uses_the_artifact_root_run_path() -> None:
    checklist = CHECKLIST.read_text(encoding="utf-8")

    assert '"${MEMCONTAM_ARTIFACT_ROOT}/runs/phase12-pilot-a-plumbing"' in checklist
    assert "runs/runs/phase12-pilot-a-plumbing" not in checklist


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


def test_plumbing_archive_retains_failed_answer_prompt_and_raw_response(tmp_path: Path) -> None:
    module = _module()
    module.run_plumbing(
        CONFIG,
        arm="Clean",
        instances=1,
        allow_live_calls=True,
        scientific_result=False,
        run_id="plumbing-failure-provenance",
        artifact_root=tmp_path,
        evidence_root=tmp_path / "evidence",
        client_factory=_MalformedBoTSolveClient,
        context_factory=_runtime_context,
    )
    run_dir = tmp_path / "runs" / "plumbing-failure-provenance"
    calls = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text().splitlines()]
    failures = [
        json.loads(line) for line in (run_dir / "failures.jsonl").read_text().splitlines()
    ]
    answer_call = next(row for row in calls if row["call_id"] == "bot_style:2")
    failure = next(row for row in failures if row["baseline"] == "bot_style")

    assert answer_call["messages"]
    assert answer_call["response_text"] == "not json"
    assert answer_call["response_text_sha256"] == __import__("hashlib").sha256(b"not json").hexdigest()
    assert failure["call_ids"] == ["bot_style:1", "bot_style:2"]
    assert failure["answer_call_id"] == "bot_style:2"
    assert failure["raw_response_hash"] == answer_call["response_text_sha256"]
    assert failure["error_type"] == "BaselineOutputError"
    assert failure["parser_contract"] == "bot_solve_json_v1"


def test_live_answer_prompts_state_the_contract_consumed_by_their_parsers() -> None:
    context = _runtime_context(_ContractSensitiveClient(), "prompt-contract", "gpt-test")

    outcomes = run_clean_game24_trial(context)

    assert {baseline: result.outcome.status for baseline, result in outcomes.items()} == {
        "nomem": "succeeded",
        "fh_bounded": "succeeded",
        "rag_frozen": "succeeded",
        "bot_style": "succeeded",
        "reflexion_style": "succeeded",
    }


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("trial_calls", "TRIAL_CALL_REFERENCE_FAILED"),
        ("call_trial", "CALL_TRIAL_REFERENCE_FAILED"),
        ("answer_call", "ANSWER_CALL_REFERENCE_FAILED"),
        ("retrieval_trial", "EVENT_TRIAL_REFERENCE_FAILED"),
        ("context_trial", "EVENT_TRIAL_REFERENCE_FAILED"),
        ("live_provider_calls", "OPERATIONS_RECONCILIATION_FAILED"),
        ("retry_total", "OPERATIONS_RECONCILIATION_FAILED"),
        ("cost_total", "OPERATIONS_RECONCILIATION_FAILED"),
    ],
)
def test_plumbing_archive_reconciles_references_and_operation_totals(
    tmp_path: Path, mutation: str, reason_code: str
) -> None:
    module = _module()
    module.run_plumbing(
        CONFIG,
        arm="Clean",
        instances=1,
        allow_live_calls=True,
        scientific_result=False,
        run_id="plumbing-reconcile",
        artifact_root=tmp_path,
        evidence_root=tmp_path / "evidence",
        client_factory=_Client,
        context_factory=_runtime_context,
    )
    run_dir = tmp_path / "runs" / "plumbing-reconcile"
    paths = {
        "trials": run_dir / "trials.jsonl",
        "calls": run_dir / "calls.jsonl",
        "retrieval": run_dir / "retrieval_events.jsonl",
        "context": run_dir / "context_events.jsonl",
        "ledger": run_dir / "decision_ledger.json",
    }
    if mutation in {"trial_calls", "answer_call"}:
        trials = [json.loads(line) for line in paths["trials"].read_text(encoding="utf-8").splitlines()]
        trials[0]["calls" if mutation == "trial_calls" else "answer_call_id"] = "missing:call"
        paths["trials"].write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in trials),
            encoding="utf-8",
        )
    elif mutation == "call_trial":
        calls = [json.loads(line) for line in paths["calls"].read_text(encoding="utf-8").splitlines()]
        calls[0]["trial_id"] = "missing:trial"
        paths["calls"].write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in calls),
            encoding="utf-8",
        )
    elif mutation in {"retrieval_trial", "context_trial"}:
        path = paths["retrieval" if mutation == "retrieval_trial" else "context"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["trial_id"] = "missing:trial"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
    else:
        ledger = json.loads(paths["ledger"].read_text(encoding="utf-8"))
        ledger[mutation] = 1.0 if mutation == "cost_total" else 999
        paths["ledger"].write_text(json.dumps(ledger, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    _rewrite_archive_manifest(run_dir)

    report = module.validate_plumbing_archive(run_dir)

    assert report["overall"] == "fail"
    assert report["reason_code"] == reason_code


@pytest.mark.parametrize(
    ("answer_call_id", "expected_overall", "expected_reason_code"),
    [
        ("unknown:call:2", "pass", None),
        ("unknown:call:4", "fail", "ANSWER_CALL_REFERENCE_FAILED"),
    ],
)
def test_plumbing_archive_resolves_legacy_answer_calls_only_when_declared_for_same_trial(
    tmp_path: Path, answer_call_id: str, expected_overall: str, expected_reason_code: str | None
) -> None:
    module = _module()
    module.run_plumbing(
        CONFIG,
        arm="Clean",
        instances=1,
        allow_live_calls=True,
        scientific_result=False,
        run_id="plumbing-legacy-answer-call",
        artifact_root=tmp_path,
        evidence_root=tmp_path / "evidence",
        client_factory=_Client,
        context_factory=_runtime_context,
    )
    run_dir = tmp_path / "runs" / "plumbing-legacy-answer-call"
    trials_path = run_dir / "trials.jsonl"
    trials = [json.loads(line) for line in trials_path.read_text(encoding="utf-8").splitlines()]
    trials[3]["answer_call_id"] = answer_call_id
    trials_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in trials),
        encoding="utf-8",
    )
    _rewrite_archive_manifest(run_dir)

    report = module.validate_plumbing_archive(run_dir)

    assert report["overall"] == expected_overall
    assert report["reason_code"] == expected_reason_code


def test_preserved_legacy_plumbing_archive_validates_without_provider_access() -> None:
    report = _module().validate_plumbing_archive(LEGACY_PLUMBING_ARCHIVE)

    assert report["overall"] == "pass"
    assert report["unresolved_references"] == 0


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

    admitted = module.evaluate_pilot_a_admission(SCIENTIFIC_CONFIG, evidence_root=evidence_root)

    assert admitted["admitted"] is True
    assert admitted["reason_code"] is None
    assert admitted["filter_policy_version"] == "operational-evidence-filter-v4"
    assert admitted["filter_interpretation"] == "contract_invalid_direct_write_containment"
    assert admitted["implementation_commit"] == module._implementation_commit()
    assert admitted["config_sha256"] == __import__("hashlib").sha256(
        SCIENTIFIC_CONFIG.read_bytes()
    ).hexdigest()
    (evidence_root / "phase12-filter-v4-f1c.json").unlink()

    blocked = module.evaluate_pilot_a_admission(SCIENTIFIC_CONFIG, evidence_root=evidence_root)

    assert blocked["admitted"] is False
    assert blocked["reason_code"] == "FILTER_V4_EVIDENCE_HASH_MISMATCH"


def test_pilot_a_cli_admission_only_reads_evidence_without_starting_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    evidence_root = tmp_path / "evidence"
    _passing_evidence(evidence_root)
    monkeypatch.setattr(module, "DEFAULT_EVIDENCE_ROOT", evidence_root)

    _run_cli(
        monkeypatch,
        "phase12",
        "pilot-a",
        "--config",
        str(SCIENTIFIC_CONFIG),
        "--admission-only",
    )

    assert json.loads(capsys.readouterr().out)["admitted"] is True


def test_pilot_a_cli_dispatches_the_scientific_runner_after_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    evidence_root = tmp_path / "evidence"
    _passing_evidence(evidence_root)
    monkeypatch.setattr(module, "DEFAULT_EVIDENCE_ROOT", evidence_root)
    monkeypatch.setattr(
        module,
        "run_scientific_pilot_a",
        lambda _config, *, allow_live_calls, parent_run_id, root_attempt_run_id: {
            "overall": "pass",
            "parent_run_id": parent_run_id,
            "root_attempt_run_id": root_attempt_run_id,
            "scientific_result": allow_live_calls,
        },
    )

    _run_cli(
        monkeypatch,
        "phase12",
        "pilot-a",
        "--config",
        str(SCIENTIFIC_CONFIG),
        "--allow-live-calls",
        "--parent-run-id",
        "pilot-a-attempt-1",
        "--root-attempt-run-id",
        "pilot-a-root-attempt",
    )

    assert json.loads(capsys.readouterr().out) == {
        "overall": "pass",
        "parent_run_id": "pilot-a-attempt-1",
        "root_attempt_run_id": "pilot-a-root-attempt",
        "scientific_result": True,
    }


def test_mocked_scientific_pilot_a_executes_two_seed_live_archive(tmp_path: Path) -> None:
    module = _module()
    evidence_root = tmp_path / "evidence"
    _passing_evidence(evidence_root)

    report = module.run_scientific_pilot_a(
        SCIENTIFIC_CONFIG,
        allow_live_calls=True,
        artifact_root=tmp_path,
        evidence_root=evidence_root,
        client_factory=_Client,
        context_factory=_runtime_context,
        run_id="pilot-a-scientific",
        parent_run_id="pilot-a-attempt-1",
        root_attempt_run_id="pilot-a-root-attempt",
    )

    run_dir = tmp_path / "runs" / "pilot-a-scientific"
    trials = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text().splitlines()]
    calls = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text().splitlines()]
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert report["overall"] == "pass"
    assert report["scientific_result"] is True
    assert report["trajectory_seeds"] == [0, 1]
    assert report["eligible_seeds"] == [0]
    assert report["live_provider_calls"] > 0
    assert {row["baseline"] for row in trials} == set(module.PLUMBING_BASELINES)
    assert {row["arm"] for row in trials if row["trial_kind"] == "memory_branch"} == {
        "clean",
        "contam",
        "filter",
    }
    assert all(call["latency_ms"] == 0 for call in calls)
    assert run["parent_run_id"] == "pilot-a-attempt-1"
    assert run["root_attempt_run_id"] == "pilot-a-root-attempt"
    assert run["requested_scientific_result"] is True
    assert run["result_eligible"] is True
    assert module.validate_scientific_pilot_a_archive(run_dir)["overall"] == "pass"
    ledger = json.loads((run_dir / "decision_ledger.json").read_text(encoding="utf-8"))
    assert ledger["prefix"]["completed_trials"] == sum(
        row["trial_kind"] == "branch_free_prefix" for row in trials
    )
    assert ledger["eligibility"]
    assert ledger["joint"]
    assert {
        "baseline",
        "baseline_family",
        "checkpoint_id",
        "checkpoint_index",
        "condition_id",
        "eligible",
        "horizon",
        "reason_codes",
        "seed",
    } <= set(ledger["eligibility"][0])
    assert {
        "baseline_eligible",
        "fallback_checkpoint_used",
        "joint_eligible_indices",
        "not_estimable",
        "primary_intersection",
        "selected_checkpoint",
    } <= set(ledger["joint"][0])
    assert all(row["fallback_checkpoint_used"] is False for row in ledger["joint"])


def test_scientific_pilot_a_writes_sidecars_before_constructing_the_provider_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("memcontam.readiness.pilot_a_scientific")
    run_dir = tmp_path / "runs" / "pilot-a-sidecars-first"
    rows = _partial_scientific_rows(module.ROW_NAMES)

    def client_factory() -> _Client:
        assert all(
            (run_dir / filename).is_file()
            for filename in (
                "run.json",
                "resolved_config.json",
                "provider_profile.json",
                "decision_ledger.json",
            )
        )
        return _Client()

    monkeypatch.setattr(module, "_run_seeds", lambda *_args, **_kwargs: rows)

    report = module.run_scientific_pilot_a(
        SCIENTIFIC_CONFIG,
        allow_live_calls=True,
        artifact_root=tmp_path,
        client_factory=client_factory,
        run_id="pilot-a-sidecars-first",
    )

    assert report["overall"] == "pass"


def test_scientific_pilot_a_stops_when_joint_checkpoint_eligibility_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("memcontam.readiness.pilot_a_scientific")
    rows = _partial_scientific_rows(module.ROW_NAMES)
    rows["seed_status"] = [
        {"seed": 0, "eligible": False, "fallback_checkpoint_used": False}
    ]
    monkeypatch.setattr(module, "_run_seeds", lambda *_args, **_kwargs: rows)

    with pytest.raises(module.ScientificPilotAError, match="JOINT_CHECKPOINT_ELIGIBILITY_EMPTY"):
        module.run_scientific_pilot_a(
            SCIENTIFIC_CONFIG,
            allow_live_calls=True,
            artifact_root=tmp_path,
            client_factory=_Client,
            run_id="pilot-a-empty-eligibility",
        )

    run_dir = tmp_path / "runs" / "pilot-a-empty-eligibility"
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "invalidated"
    assert run["status_reason"] == "joint_checkpoint_eligibility_empty"
    assert module.validate_scientific_pilot_a_archive(run_dir)["overall"] == "pass"


@pytest.mark.parametrize(
    ("error", "status", "reason"),
    [
        (RuntimeError("provider unavailable"), "interrupted", "provider_failure"),
        (CostLimitExceeded("cost ceiling reached"), "blocked", "cost_limit_exceeded"),
        (KeyboardInterrupt(), "interrupted", "interrupted"),
    ],
)
def test_scientific_pilot_a_seals_partial_archive_before_propagating_terminal_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    status: str,
    reason: str,
) -> None:
    module = importlib.import_module("memcontam.readiness.pilot_a_scientific")
    run_id = f"pilot-a-{reason}"

    def stop_after_progress(*_args, **kwargs) -> None:  # noqa: ANN001
        rows = kwargs["rows"]
        rows.update(_partial_scientific_rows(module.ROW_NAMES))
        raise error

    monkeypatch.setattr(module, "_run_seeds", stop_after_progress)

    with pytest.raises(type(error)):
        module.run_scientific_pilot_a(
            SCIENTIFIC_CONFIG,
            allow_live_calls=True,
            artifact_root=tmp_path,
            client_factory=_Client,
            run_id=run_id,
        )

    run_dir = tmp_path / "runs" / run_id
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    ledger = json.loads((run_dir / "decision_ledger.json").read_text(encoding="utf-8"))
    failures = [json.loads(line) for line in (run_dir / "failures.jsonl").read_text().splitlines()]
    assert run["status"] == status
    assert run["status_reason"] == reason
    assert run["requested_scientific_result"] is True
    assert run["scientific_result"] is False
    assert run["result_eligible"] is False
    assert ledger["live_provider_calls"] == 1
    assert ledger["cost_total"] == 0.01
    assert ledger["retry_total"] == 1
    assert ledger["prefix"]["completed_trials"] == 1
    assert ledger["eligibility"]
    assert ledger["joint"][0]["fallback_checkpoint_used"] is False
    assert failures[-1]["provenance"] == "scientific_pilot_a_runner"
    assert (run_dir / "seed_status.jsonl").is_file()
    assert (run_dir / "public_artifact_manifest.json").is_file()
    assert (run_dir / "archive_seal.json").is_file()
    assert module.validate_scientific_pilot_a_archive(run_dir)["overall"] == "pass"


def test_scientific_pilot_a_allows_non_completed_terminal_status_when_no_seed_is_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("memcontam.readiness.pilot_a_scientific")
    rows = _partial_scientific_rows(module.ROW_NAMES)
    rows["seed_status"].append(
        {
            "seed": 1,
            "eligible": False,
            "status": "blocked",
            "reason": "blocked by test",
            "selected_checkpoint": None,
            "fallback_checkpoint_used": False,
            "joint_eligible_indices": [],
        }
    )
    monkeypatch.setattr(module, "_run_seeds", lambda *_args, **_kwargs: rows)

    report = module.run_scientific_pilot_a(
        SCIENTIFIC_CONFIG,
        allow_live_calls=True,
        artifact_root=tmp_path,
        client_factory=_Client,
        run_id="pilot-a-empty-eligibility",
    )

    run = json.loads((tmp_path / "runs" / "pilot-a-empty-eligibility" / "run.json").read_text())

    assert report["overall"] == "pass"
    assert run["status"] == "blocked"
    assert run["status_reason"] == "all_joint_checkpoint_blocked"
    assert run["requested_scientific_result"] is True
    assert run["scientific_result"] is False
    assert run["result_eligible"] is False
    assert module.validate_scientific_pilot_a_archive(tmp_path / "runs" / "pilot-a-empty-eligibility")["overall"] == "pass"


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
