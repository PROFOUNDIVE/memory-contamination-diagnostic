# memory-contamination-diagnostic

Controlled memory-contamination diagnostic harness for reasoning-memory systems.

All claims below are for **faithful adapted baselines**, not complete reproductions of the official methods. Replay evidence is a fidelity/QA artifact, not benchmark or manuscript-quality evidence.

---

## Baseline Fidelity V2 Authority

[`docs/baseline-fidelity-v2.md`](docs/baseline-fidelity-v2.md) is the sole authority for
Baseline-Fidelity-V2. [`docs/baseline-fidelity-v2-evidence.md`](docs/baseline-fidelity-v2-evidence.md)
records its evidence provenance, resource usage, and artifact hashes. Older V1 and G0
reports are historical and can't support a V2 fidelity claim.

F1A and F1B pass as offline, non-scientific QA gates. The F1C gate is implemented with
pinned BGE-M3 cache-only semantics and mocked OpenAI-compatible answer dispatch. This
checkout lacks the required BGE-M3 revision, so the verifier reports
`missing_cached_bge_m3` and overall V2 certification remains blocked. Replay evidence
isn't benchmark, manuscript-quality, causal, or production contamination evidence.
No V2 release or tag is claimed.

## Historical Release Roadmap

| Tag | Scope | Baselines | Key features |
|---|---|---|---|
| `v0.7.2` | Phase-11 `logging_v2` release tag for offline contract QA | Existing strict logging baselines | Typed evaluation law, fixed target set, exact direct-edge lineage, answer-call exposure, pair/checkpoint joins |
| `phase11` | Phase-11 `logging_v2` operator contract and offline replay QA workflow | Existing strict logging baselines | Fixed evaluation law, fixed target set, exact direct-edge lineage, answer-call exposure, pair/checkpoint joins |
| `v0.7` | Strict logging audit remediation over the locked offline 39-row gate | `no_memory`, `full_history`, `retrieval_rag`, `reflexion_style`, `bot_style` | `run.json` plus typed split streams, answer-call source-span exposure, durable failures, stage-gated aggregation |
| `v0.6` | Stricter Reflexion retry fidelity over the locked 3-task pilot set | `dynamic_cheatsheet_rs_optional`, `reflexion_style` | Retry actor sees only latest-three reflections + current input; failed trajectory stays inside `reflexion_reflect` |
| `v0.5` | Full G0 native-memory pass over the locked 3-task pilot set | `full_history`, `reflexion_style`, `dynamic_cheatsheet_optional` | Append-only full history, failure-gated reflection, latest-three reflection window, generate/curate DC-Cu loop |
| `v0.5+` | DC-RS appendix comparator + same-sample retry follow-up (historical; superseded by `v0.6`) | `dynamic_cheatsheet_rs_optional`, `reflexion_style` | Top-3 cosine DC-RS synthesis, same-sample retry bounded by `max_attempts: 2` |
| `v0.4` | Partial G0 pass for retrieval + template-memory baselines | `retrieval_rag`, `bot_style` | Pinned encoder, versioned corpus, exact top-k retrieval, five-stage BoT meta-buffer |
| `v0.3` | Scaffold release of `retrieval_rag` and `bot_style` | `retrieval_rag`, `bot_style` | Hash-projection retriever, in-memory meta-buffer |
| `v0.2` | Multitask replay QA demo | `no_memory`, `retrieval_rag`, `reflexion_style`, `bot_style` | 3 tasks, replay fixtures, contamination lineage, shallow aggregate metrics |
| `v0.1` | Initial scaffold | — | Early technical notes |

For detailed reports see the [Documentation](#documentation) section.

---

## v0.7 Strict Logging Audit Remediation

`v0.7` closes the strict logging audit blockers with a replay-only 39-row contract gate. New strict runs write `run.json`, canonical `trials.jsonl`, and four typed streams: `calls.jsonl`, `failures.jsonl`, `filter_events.jsonl`, and `memory_events.jsonl`. Exposure is derived from the explicit answer call's source spans, not from retrieval or memory-presence proxies.

Verification commands:

```bash
python -m memcontam.cli validate-config configs/logging_contract_replay.yaml
python -m pytest tests/test_logging_contract_gate.py tests/test_task_verifiers.py tests/test_cli_run.py tests/test_contamination_catalog.py tests/test_openai_compatible_client.py tests/test_aggregate.py -q
python -m pytest tests/test_logging_schema.py tests/test_logging_writer.py tests/test_method_calls.py tests/test_cli_run.py tests/test_aggregate.py tests/test_logging_contract_gate.py -q
python -m ruff check src tests scripts

RUN_ID="logging-contract-replay-$(date -u +%Y%m%dT%H%M%SZ)"
python -m memcontam.cli run configs/logging_contract_replay.yaml --run-id "$RUN_ID"
python -m memcontam.cli aggregate "runs/$RUN_ID" --stage replay
```

This offline gate is not an API-connected pilot, main run, or benchmark result. No API-connected pilot was run; main readiness requires later evidence and an explicit decision. See [`docs/logging-audit-remediation-v0.7.md`](docs/logging-audit-remediation-v0.7.md) for the release report and [`docs/logging-contract-v1.md`](docs/logging-contract-v1.md) for operator rules.

## v0.7.2 Phase-11 `logging_v2` Contract Release

`v0.7.2` is the release tag for the Phase-11 `logging_v2` contract implementation and its offline replay QA workflow. It adds typed evaluation-law metadata, fixed target-set joins, exact direct-edge lineage rules, answer-call-only exposure accounting, and deterministic pair/checkpoint aggregation semantics on top of the strict logging surface.

Verification commands:

```bash
python -m memcontam.cli validate-config configs/logging_contract_phase11_replay.yaml
python -m pytest tests/test_phase11_logging_contract_gate.py -q
python -m pytest tests/test_docs_scope.py -q
python -m ruff check src tests scripts

RUN_ID="phase11-logging-contract-replay-$(date -u +%Y%m%dT%H%M%SZ)"
python -m memcontam.cli run configs/logging_contract_phase11_replay.yaml --run-id "$RUN_ID"
python -m memcontam.cli aggregate "runs/$RUN_ID" --stage replay --contract phase11
```

This is offline contract QA, not a pilot, main run, benchmark, manuscript result, or empirical result. No API-connected pilot was run. See [`docs/logging-audit-remediation-v0.7.2.md`](docs/logging-audit-remediation-v0.7.2.md) for the release report, [`docs/logging-contract-v2-phase11.md`](docs/logging-contract-v2-phase11.md) for the operator contract, and [`docs/logging-audit-remediation-phase11.md`](docs/logging-audit-remediation-phase11.md) for the unfilled operator review template.

## Phase-11 `logging_v2` Contract Status

The repository contains the Phase-11 contract implementation and an offline
replay QA workflow. The contract is documented in
[`docs/logging-contract-v2-phase11.md`](docs/logging-contract-v2-phase11.md).
The accompanying report at
[`docs/logging-audit-remediation-phase11.md`](docs/logging-audit-remediation-phase11.md)
is a template, not a claim that a release review has passed. The historical
[`docs/logging-contract-v1.md`](docs/logging-contract-v1.md) remains the v1
operator contract and is not Phase-11 evidence.

Phase-11 verification uses the versioned replay config and explicit contract
flag:

```bash
python -m memcontam.cli validate-config configs/logging_contract_phase11_replay.yaml
python -m pytest tests/test_phase11_logging_contract_gate.py -q
python -m pytest tests/test_docs_scope.py -q
python -m ruff check src tests scripts

RUN_ID="phase11-logging-contract-replay-$(date -u +%Y%m%dT%H%M%SZ)"
python -m memcontam.cli run configs/logging_contract_phase11_replay.yaml --run-id "$RUN_ID"
python -m memcontam.cli aggregate "runs/$RUN_ID" --stage replay --contract phase11
```

This is offline contract QA, not a pilot, main run, benchmark, manuscript
result, or empirical result. No API-connected pilot was run. The workflow
does not claim causal use, complete PROV-DM, or automatic migration of legacy
or `logging_v1` artifacts.

---

## Strict Offline Logging Contract Operator Rules

`configs/logging_contract_replay.yaml` is a strict offline replay gate for the locked 39-row logging matrix. It checks cross-stream joins, answer-call source spans, filter and memory lineage, failure continuation, and redaction of provider raw payload sentinels. It requires no API credentials.

Verification commands:

```bash
python -m memcontam.cli validate-config configs/logging_contract_replay.yaml
python -m pytest tests/test_logging_contract_gate.py tests/test_task_verifiers.py tests/test_cli_run.py tests/test_contamination_catalog.py tests/test_openai_compatible_client.py tests/test_aggregate.py -q
python -m ruff check src tests scripts

# Optional replay-only CLI contract check; use a new UTC-suffixed run ID.
RUN_ID="logging-contract-replay-$(date -u +%Y%m%dT%H%M%SZ)"
python -m memcontam.cli run configs/logging_contract_replay.yaml --run-id "$RUN_ID"
python -m memcontam.cli aggregate "runs/$RUN_ID" --stage replay
```

This offline gate is not an API-connected pilot, main run, or benchmark result. No API-connected pilot was run; main readiness requires later evidence and an explicit decision. See [`docs/logging-contract-v1.md`](docs/logging-contract-v1.md) for the operator contract.

---

## v0.6 Stricter Reflexion Retry Fidelity

`v0.6` tightens the Reflexion same-sample retry prompt to the stricter Option A boundary: the retry actor no longer receives the failed raw response, parsed answer, or verifier feedback directly. It conditions only on the latest three ordered reflections from the current identity plus the current task input. The reflection stage still sees the failed trajectory and sanitized evaluator feedback so it can write a mitigation memory, but that information reaches the retry exclusively through the reflection memory.

- Faithful adapted DC-RS optional appendix comparator: top-3 cosine retrieval over prior same-identity input/output pairs, label-free pre-answer cheatsheet synthesis, then memory-conditioned generation, with native method-call costs logged.
- Faithful adapted Reflexion control flow: failed trajectory plus sanitized evaluator feedback produces linguistic reflection, latest-three reflection memory conditions a same-sample retry, stopping on success or attempt limit; no weight updates.

The canonical replay evidence run is `g0_dc_rs_reflexion_fidelity_followup_replay` using `configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml` and inspected by `scripts/inspect_g0_dc_rs_reflexion_fidelity.py`. It emits 108 trial rows and 174 native method calls (DC-RS 108, Reflexion 66). External LLM responses are replay fixtures, so the verification requires no API keys or live model access.

Verification commands:

```bash
python -m memcontam.cli validate-config configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml
python -m pytest tests/test_dc_rs_faithful.py tests/test_reflexion_faithful.py tests/test_cli_run.py tests/test_replay_client.py tests/test_replay_fixtures.py tests/test_contamination_catalog.py tests/test_logging_schema.py tests/test_aggregate.py tests/test_docs_scope.py -q
python -m pytest tests/test_task_verifiers.py tests/test_cli_run.py tests/test_contamination_catalog.py tests/test_openai_compatible_client.py tests/test_aggregate.py tests/test_docs_scope.py -q
python -m ruff check src tests scripts
python -m memcontam.cli run configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml --run-id g0_dc_rs_reflexion_fidelity_followup_replay
python -m memcontam.cli aggregate runs/g0_dc_rs_reflexion_fidelity_followup_replay
python scripts/inspect_g0_dc_rs_reflexion_fidelity.py runs/g0_dc_rs_reflexion_fidelity_followup_replay
```

See [`docs/g0-baseline-fidelity-gate-v0.6.md`](docs/g0-baseline-fidelity-gate-v0.6.md) for the full v0.6 report. The prior follow-up report is at [`docs/g0-dc-rs-reflexion-fidelity-followup.md`](docs/g0-dc-rs-reflexion-fidelity-followup.md).

---

## v0.5+ DC-RS and Reflexion Same-Sample Retry Follow-up

This post-`v0.5` follow-up records source-bounded fidelity claims after the Task 8 replay evidence exists. `v0.5` remains the historical full G0 baseline-fidelity pass; this follow-up tightens Reflexion to the source-required same-sample retry semantics and adds DC-RS as an optional appendix comparator.

- Faithful adapted DC-RS optional appendix comparator: top-3 cosine retrieval over prior same-identity input/output pairs, label-free pre-answer cheatsheet synthesis, then memory-conditioned generation, with native method-call costs logged.
- Faithful adapted Reflexion control flow: failed trajectory plus sanitized evaluator feedback produces linguistic reflection, latest-three reflection memory conditions a same-sample retry, stopping on success or attempt limit; no weight updates.

The canonical replay evidence run is `g0_dc_rs_reflexion_fidelity_followup_replay` using `configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml` and inspected by `scripts/inspect_g0_dc_rs_reflexion_fidelity.py`. It emits 108 trial rows and 174 native method calls (DC-RS 108, Reflexion 66). External LLM responses are replay fixtures, so the verification requires no API keys or live model access. The replay output is a fidelity/QA artifact, not benchmark or manuscript-quality evidence.

Verification commands:

```bash
python -m memcontam.cli validate-config configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml
python -m pytest tests/test_dc_rs_faithful.py tests/test_reflexion_faithful.py tests/test_cli_run.py tests/test_replay_client.py tests/test_replay_fixtures.py tests/test_contamination_catalog.py tests/test_logging_schema.py tests/test_aggregate.py tests/test_docs_scope.py -q
python -m pytest tests/test_task_verifiers.py tests/test_cli_run.py tests/test_contamination_catalog.py tests/test_openai_compatible_client.py tests/test_aggregate.py tests/test_docs_scope.py -q
python -m ruff check src tests scripts
python -m memcontam.cli run configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml --run-id g0_dc_rs_reflexion_fidelity_followup_replay
python -m memcontam.cli aggregate runs/g0_dc_rs_reflexion_fidelity_followup_replay
python scripts/inspect_g0_dc_rs_reflexion_fidelity.py runs/g0_dc_rs_reflexion_fidelity_followup_replay
```

See [`docs/g0-dc-rs-reflexion-fidelity-followup.md`](docs/g0-dc-rs-reflexion-fidelity-followup.md) for the full follow-up report.

---

## v0.5 Faithful Full-History, Reflexion, and Dynamic Cheatsheet Baselines

`v0.5` is the historical full G0 baseline-fidelity pass for `full_history`, `reflexion_style`, and `dynamic_cheatsheet_optional` over the locked 3-task pilot set.

- `full_history` is a faithful append-only full-history baseline. Each trial makes one `full_history_generate` call and appends one sanitized transcript entry. Prior memory is rendered verbatim; no retrieval, summarization, or truncation is used in this gate.
- `reflexion_style` is a Reflexion-style verbal memory proxy / faithful adapted control flow. It calls `reflexion_generate` on every trial and `reflexion_reflect` only after verifier failure, then appends a non-empty reflection. The actor reads the latest three ordered reflections from the current identity. This is not a complete reproduction of the official Reflexion agent.
- `dynamic_cheatsheet_optional` is a faithful adapted DC-Cu optional appendix comparator. Each trial calls `dynamic_cheatsheet_generate` then `dynamic_cheatsheet_curate`. A parsed non-empty `<cheatsheet>` block replaces the cheatsheet; missing or empty tags preserve prior state. Code-execution, provider-tool, and retrieval paths are removed. DC is not a new main baseline.

The matrix is `3 tasks × 3 baselines × 3 arms × 2 models = 162 trials`. The canonical replay evidence run is `g0_fh_reflexion_dc_faithful_replay` using `configs/g0_fh_reflexion_dc_faithful_replay.yaml` and inspected by `scripts/inspect_g0_fh_reflexion_dc_fidelity.py`.

External LLM responses are replay fixtures in this gate, so the verification requires no API keys or live model access. The replay output is a fidelity/QA artifact, not benchmark or manuscript-quality evidence. Live runs must keep the same stage structure.

`v0.5` is the intended repository research-artifact Git tag; `pyproject.toml` remains at version `0.1.0` and no package is published.

Verification commands:

```bash
python -m memcontam.cli validate-config configs/g0_fh_reflexion_dc_faithful_replay.yaml
python -m pytest tests/test_full_history_faithful.py tests/test_reflexion_faithful.py tests/test_dynamic_cheatsheet_faithful.py tests/test_cli_run.py tests/test_contamination_catalog.py tests/test_logging_schema.py tests/test_replay_client.py tests/test_replay_fixtures.py tests/test_aggregate.py tests/test_docs_scope.py -q
python -m pytest tests/test_task_verifiers.py tests/test_cli_run.py tests/test_contamination_catalog.py tests/test_openai_compatible_client.py tests/test_aggregate.py tests/test_docs_scope.py -q
python -m ruff check src tests scripts
python -m memcontam.cli run configs/g0_fh_reflexion_dc_faithful_replay.yaml --run-id g0_fh_reflexion_dc_faithful_replay
python -m memcontam.cli aggregate runs/g0_fh_reflexion_dc_faithful_replay
python scripts/inspect_g0_fh_reflexion_dc_fidelity.py runs/g0_fh_reflexion_dc_faithful_replay
```

See [`docs/g0-baseline-fidelity-gate-v0.5.md`](docs/g0-baseline-fidelity-gate-v0.5.md) for the full v0.5 report and [`docs/g0-baseline-fidelity-gate-v0.4.md`](docs/g0-baseline-fidelity-gate-v0.4.md) for the prior RAG/BoT report.

---

## v0.4 Faithful Adapted RAG/BoT Baselines

`v0.4` is the historical partial G0 baseline-fidelity pass for `retrieval_rag` and `bot_style`.

- `retrieval_rag` is wired to the pinned learned encoder `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, the versioned corpus `data/memory/catalog_v1.jsonl`, exact top-k retrieval, and full provenance records that are aligned with the prompt and trial metadata. The default production path requires that pinned checkpoint in the local cache; the canonical offline replay config explicitly sets `embedding.offline_fallback: true` to substitute deterministic fake embeddings for network-free QA. RAG is read-only and emits no memory write events.
- `bot_style` runs the five reference-aligned stages (`bot_problem_distill`, `bot_instantiate_solve`, `bot_thought_distill`, `bot_novelty_decide`) and uses the same configured embedding provider for top-1 template retrieval, then persists the meta-buffer keyed by `(run_id, task_name, baseline, arm, backbone)`, accepting verified-success templates only with novelty-gated insertion.

External LLM responses are replay fixtures in this gate, so the verification requires no API keys or live model access. Live runs must keep the same stage structure.

The remaining baselines (`no_memory`, `full_history`, `reflexion_style`, `dynamic_cheatsheet_optional`, `expel_optional`) are explicitly out of scope for this slice and are not claimed to pass G0. This is not a benchmark result or a full paper reproduction.

Verification commands:

```bash
python -m memcontam.cli validate-config configs/g0_rag_bot_faithful_replay.yaml
python -m pytest tests/test_task_verifiers.py tests/test_cli_run.py tests/test_contamination_catalog.py tests/test_openai_compatible_client.py tests/test_aggregate.py tests/test_docs_scope.py -q
python -m memcontam.cli run configs/g0_rag_bot_faithful_replay.yaml --run-id g0_rag_bot_faithful_replay
python -m memcontam.cli aggregate runs/g0_rag_bot_faithful_replay
python scripts/inspect_g0_rag_bot_fidelity.py runs/g0_rag_bot_faithful_replay
```

See [`docs/g0-baseline-fidelity-gate-v0.4.md`](docs/g0-baseline-fidelity-gate-v0.4.md) for the full v0.4 report, [`docs/g0-baseline-fidelity-gate-v0.3.md`](docs/g0-baseline-fidelity-gate-v0.3.md) for the previous scaffold release, and [`docs/g0-baseline-fidelity-gate-v0.2.md`](docs/g0-baseline-fidelity-gate-v0.2.md) for the pre-implementation gap analysis.

---

## v0.3 Partial G0 Fidelity Gate (historical scaffold release)

`v0.3` tagged the first partial G0 baseline-fidelity pass for `retrieval_rag` and `bot_style`. It used a deterministic hash-projection retriever and an in-memory meta-buffer without pinned model revisions or versioned corpus hashing. The v0.3 config and inspector are preserved for historical comparison, but v0.4 is the current implementation report.

- `retrieval_rag` used deterministic sentence-embedding retrieval with logged cosine scores and provenance.
- `bot_style` implemented distill → retrieve → instantiate → update-buffer, with a `memory_write_event` recording lineage.

These two baselines were raised from prompt-label proxies to faithful adapted baselines. The remaining baselines (`no_memory`, `full_history`, `reflexion_style`, `dynamic_cheatsheet_optional`, `expel_optional`) were explicitly out of scope for this slice and are not claimed to pass G0.

This was not a benchmark result or a full paper reproduction. Historical verification commands:

```bash
python -m memcontam.cli validate-config configs/pilot_game24.yaml
python -m pytest tests/test_cli_run.py tests/test_aggregate.py tests/test_logging_schema.py tests/test_docs_scope.py -q
python -m memcontam.cli run configs/pilot_game24.yaml --run-id g0_rag_bot_gate_replay
python -m memcontam.cli aggregate runs/g0_rag_bot_gate_replay
python .sisyphus/evidence/inspect_g0_replay.py
```

---

## v0.2 Multitask Replay Gate

`v0.2` is a replay-only QA demo across three locked pilot tasks: Game24, Math Equation Balancer, and WordSorting. It validates the logging contract, multitask builder/verifier dispatch, optional live-smoke wiring, repeated-failure tracking, local proxy baselines, contamination lineage fields, and shallow aggregate metrics without API keys or network access in replay mode.

This release is not a benchmark result or a full paper reproduction. The proxy baseline claim is:

> retrieval-only RAG lower-bound, Reflexion-style verbal memory proxy, BoT-style thought-template proxy; not a complete reproduction.

The current G0 fidelity work is scoped to RAG + BoT only; `no_memory`, `full_history`, `reflexion_style`, Dynamic Cheatsheet, and ExpeL are not claimed to pass G0 in this plan. The canonical replay evidence run is `g0_rag_bot_faithful_replay` (config `configs/g0_rag_bot_faithful_replay.yaml`), inspected by `scripts/inspect_g0_rag_bot_fidelity.py`. See `docs/g0-baseline-fidelity-gate-v0.2.md` for the full implementation result and verification commands.

---

## Quick Start

```bash
python -m memcontam.cli validate-config configs/pilot_multitask_replay.yaml
python -m pytest tests/test_task_verifiers.py tests/test_cli_run.py tests/test_contamination_catalog.py tests/test_openai_compatible_client.py tests/test_aggregate.py -q
python -m memcontam.cli run configs/pilot_multitask_replay.yaml --run-id pilot_multitask_replay_qa
python -m memcontam.cli aggregate runs/pilot_multitask_replay_qa
```

The bundled config emits 90 replay trial rows:

```text
3 tasks x 3 samples x 5 baselines x 1 arm x 2 replay model labels
```

---

## Implemented In v0.2

- Multitask CLI dispatch for Game24, Math Equation Balancer, and WordSorting.
- Validated task builders and answer verifiers for the two new tasks.
- Optional live-smoke path via `OpenAICompatibleClient`.
- Per-identity repeated-failure tracking (`task_name`, `sample_id`, `baseline`, `arm`, `backbone`).
- Expanded contamination catalog entries for the three-task pilot set.
- Metric-ready `TrialLog` schema with reproducibility metadata.
- Controlled contamination exposure and filter-decision evidence.
- Local `retrieval_rag`, `reflexion_style`, and `bot_style` proxy baselines.
- Shallow aggregate JSON output with unsupported metrics marked as `not_computed`.
- Tests for verifiers, CLI replay/proxy behavior, catalog constraints, live-smoke wiring, and aggregate metrics.

---

## Documentation

- Baseline-Fidelity-V2 authority: [`docs/baseline-fidelity-v2.md`](docs/baseline-fidelity-v2.md)
- Baseline-Fidelity-V2 evidence provenance: [`docs/baseline-fidelity-v2-evidence.md`](docs/baseline-fidelity-v2-evidence.md)
- Historical Baseline-Fidelity-V1 authority: [`docs/baseline-fidelity-v1.md`](docs/baseline-fidelity-v1.md)
- v0.7.2 Phase-11 logging release report: [`docs/logging-audit-remediation-v0.7.2.md`](docs/logging-audit-remediation-v0.7.2.md)
- v0.7 logging audit remediation report: [`docs/logging-audit-remediation-v0.7.md`](docs/logging-audit-remediation-v0.7.md)
- v0.7 strict offline logging operator contract: [`docs/logging-contract-v1.md`](docs/logging-contract-v1.md)
- Phase-11 `logging_v2` operator contract: [`docs/logging-contract-v2-phase11.md`](docs/logging-contract-v2-phase11.md)
- Phase-11 remediation report template: [`docs/logging-audit-remediation-phase11.md`](docs/logging-audit-remediation-phase11.md)
- Historical v0.6 stricter Reflexion retry fidelity report: [`docs/g0-baseline-fidelity-gate-v0.6.md`](docs/g0-baseline-fidelity-gate-v0.6.md)
- v0.5+ DC-RS and Reflexion same-sample retry follow-up report: [`docs/g0-dc-rs-reflexion-fidelity-followup.md`](docs/g0-dc-rs-reflexion-fidelity-followup.md)
- Historical v0.5 G0 full pass report: [`docs/g0-baseline-fidelity-gate-v0.5.md`](docs/g0-baseline-fidelity-gate-v0.5.md)
- Historical v0.4 G0 partial pass report: [`docs/g0-baseline-fidelity-gate-v0.4.md`](docs/g0-baseline-fidelity-gate-v0.4.md)
- v0.3 G0 partial pass report: [`docs/g0-baseline-fidelity-gate-v0.3.md`](docs/g0-baseline-fidelity-gate-v0.3.md)
- v0.2 G0 gap analysis: [`docs/g0-baseline-fidelity-gate-v0.2.md`](docs/g0-baseline-fidelity-gate-v0.2.md)
- v0.2 technical notes: [`docs/replay-qa-demo-v0.2.md`](docs/replay-qa-demo-v0.2.md)
- v0.1 technical notes: [`docs/replay-qa-demo-v0.1.md`](docs/replay-qa-demo-v0.1.md)
