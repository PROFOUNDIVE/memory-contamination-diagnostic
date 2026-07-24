# memory-contamination-diagnostic

Controlled memory-contamination diagnostic harness for reasoning-memory systems.

All claims concern **faithful adapted baselines**, not a complete reproduction of official methods. Replay output is a fidelity/QA artifact, not benchmark or manuscript-quality evidence.

## Baseline Fidelity V2 Authority

[`docs/baseline-fidelity-v2.md`](docs/baseline-fidelity-v2.md) is the sole authority for Baseline-Fidelity-V2; [`docs/baseline-fidelity-v2-evidence.md`](docs/baseline-fidelity-v2-evidence.md) records provenance, resource usage, and artifact hashes. Older V1 and G0 reports are historical and cannot support a V2 fidelity claim.

F1A and F1B pass as offline, non-scientific QA gates. F1C uses pinned BGE-M3 cache-only semantics with mocked OpenAI-compatible answer dispatch. This checkout lacks the required revision, so the verifier reports `missing_cached_bge_m3` and overall V2 certification remains blocked. Replay evidence is not benchmark, manuscript-quality, causal, or production contamination evidence. `v0.8` is a repository research-artifact tag for the completed V2 source-contract remediation. It is not an overall V2 certification because F1C remains blocked.

## Phase-12 Repository Contract Status

Phase-12 support is a repository contract, not a scientific result. It adds `logging_v3`, five Methods-facing arms (`Clean`, `Correct`, `Irrelevant`, `Contam`, `Filter`), branch-free clean prefixes, matched suffixes, P12I build-layer replay checks, canonical configs, manifests, archive validation, and separate exploratory Python-sandbox governance.

The committed workflow remains offline and non-scientific until the exact F1C cache gate and required external governance artifacts are available. P12I may pass as build-layer readiness while scientific admission remains false. Text-only evidence and exploratory code evidence are not pooled.

- [`docs/phase12-implementation-contract.md`](docs/phase12-implementation-contract.md)
- [`docs/logging-v3-phase12.md`](docs/logging-v3-phase12.md)
- [`docs/phase12-operator-runbook.md`](docs/phase12-operator-runbook.md)

## v0.9 Phase-12 Repository Contract Refactor

Phase-12 plan execution completed as repository-contract work; this refactor removed bootstrap scaffolding and deduplicated tests/docs. `v0.9` is a repository research-artifact tag, not scientific, benchmark, or manuscript-quality evidence. F1C remains `BLOCKED` (`missing_cached_bge_m3`). P12I may pass, but scientific admission remains false. Text/code evidence are not pooled.

## Historical Release Roadmap

| Tag | Scope | Key features |
|---|---|---|
| `v0.9` | Phase-12 repository-contract refactor | Completed plan execution; bootstrap-scaffolding removal; deduplicated tests/docs; scientific admission remains blocked |
| `v0.8` | Baseline-Fidelity-V2 source-contract remediation | F1A/F1B offline QA pass; V2 source-contract artifacts; blocked F1C cache gate preserved |
| `v0.7.2` | Phase-11 `logging_v2` offline contract QA | Typed evaluation law, fixed target set, exact direct-edge lineage, answer-call exposure, pair/checkpoint joins |
| `v0.7` | Strict logging audit remediation | `run.json`, typed streams, answer-call source spans, durable failures, stage-gated aggregation |
| `v0.6` | Stricter Reflexion retry fidelity | Latest-three reflection window; failed trajectory remains inside `reflexion_reflect` |
| `v0.5` | Full G0 native-memory pass | `full_history`, `reflexion_style`, `dynamic_cheatsheet_optional` |
| `v0.5+` | DC-RS appendix comparator and same-sample retry follow-up | `dynamic_cheatsheet_rs_optional`, `reflexion_style`, `max_attempts: 2` |
| `v0.4` | Partial G0 pass | `retrieval_rag`, `bot_style`, pinned encoder, versioned corpus, BoT meta-buffer |
| `v0.3` | Historical scaffold release | Hash-projection retriever and in-memory meta-buffer |
| `v0.2` | Multitask replay QA demo | Three tasks, replay fixtures, contamination lineage, aggregate metrics |

## Verification

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

## v0.7 Strict Logging Audit Remediation

`v0.7` is a replay-only 39-row contract gate. Strict runs write `run.json`, canonical `trials.jsonl`, and typed `calls.jsonl`, `failures.jsonl`, `filter_events.jsonl`, and `memory_events.jsonl`; exposure is derived from the explicit answer call's source spans. This offline gate is not an API-connected pilot, main run, or benchmark result. No API-connected pilot was run. See [`docs/logging-audit-remediation-v0.7.md`](docs/logging-audit-remediation-v0.7.md) and [`docs/logging-contract-v1.md`](docs/logging-contract-v1.md).

## v0.8 Baseline-Fidelity-V2 Source-Contract Remediation

`v0.8` records the completed Baseline-Fidelity-V2 source-contract remediation. F1A structural integration replay and F1B source-contract replay pass as offline QA gates; F1C remains blocked until the exact pinned BGE-M3 cache is available and `scripts/verify_bge_m3_fidelity.py` passes. See [`docs/baseline-fidelity-v2.md`](docs/baseline-fidelity-v2.md) and [`docs/baseline-fidelity-v2-evidence.md`](docs/baseline-fidelity-v2-evidence.md).

## v0.7.2 Phase-11 `logging_v2` Contract Release and Status

`v0.7.2` is the Phase-11 `logging_v2` contract implementation and offline replay QA workflow: typed evaluation-law metadata, fixed target-set joins, exact direct-edge lineage, answer-call-only exposure accounting, and deterministic pair/checkpoint aggregation. [`docs/logging-contract-v2-phase11.md`](docs/logging-contract-v2-phase11.md) is the contract; [`docs/logging-audit-remediation-phase11.md`](docs/logging-audit-remediation-phase11.md) is the unfilled review template; [`docs/logging-audit-remediation-v0.7.2.md`](docs/logging-audit-remediation-v0.7.2.md) is the release report. The historical [`docs/logging-contract-v1.md`](docs/logging-contract-v1.md) is not Phase-11 evidence. Use `configs/logging_contract_phase11_replay.yaml` for the offline contract QA workflow.

## v0.6 Stricter Reflexion Retry Fidelity

`v0.6` limits the retry actor to the latest three ordered reflections from the current identity plus the current task input. The reflection stage receives the failed trajectory and sanitized evaluator feedback; the retry receives that information only through reflection memory. See [`docs/g0-baseline-fidelity-gate-v0.6.md`](docs/g0-baseline-fidelity-gate-v0.6.md).

## v0.5+ DC-RS and Reflexion Same-Sample Retry Follow-up

This post-`v0.5` follow-up is source-bounded: `v0.5` remains the historical full G0 baseline-fidelity pass, while the follow-up tightens Reflexion to same-sample retry semantics and adds DC-RS as an optional appendix comparator.

- Faithful adapted DC-RS optional appendix comparator: top-3 cosine retrieval over prior same-identity input/output pairs, label-free pre-answer cheatsheet synthesis, then memory-conditioned generation, with native method-call costs logged.
- Faithful adapted Reflexion control flow: failed trajectory plus sanitized evaluator feedback produces linguistic reflection, latest-three reflection memory conditions a same-sample retry, stopping on success or attempt limit; no weight updates.

The canonical replay evidence run is `g0_dc_rs_reflexion_fidelity_followup_replay` using `configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml` and `scripts/inspect_g0_dc_rs_reflexion_fidelity.py`. It emits 108 trial rows and 174 native method calls. See [`docs/g0-dc-rs-reflexion-fidelity-followup.md`](docs/g0-dc-rs-reflexion-fidelity-followup.md).

## v0.5 Faithful Full-History, Reflexion, and Dynamic Cheatsheet Baselines

`v0.5` is the historical full G0 baseline-fidelity pass for `full_history`, `reflexion_style`, and `dynamic_cheatsheet_optional` over the locked three-task pilot set. The canonical replay evidence run is `g0_fh_reflexion_dc_faithful_replay` using `configs/g0_fh_reflexion_dc_faithful_replay.yaml` and `scripts/inspect_g0_fh_reflexion_dc_fidelity.py`. See [`docs/g0-baseline-fidelity-gate-v0.5.md`](docs/g0-baseline-fidelity-gate-v0.5.md) and [`docs/g0-baseline-fidelity-gate-v0.4.md`](docs/g0-baseline-fidelity-gate-v0.4.md).

## v0.4 Faithful Adapted RAG/BoT Baselines

`v0.4` is the historical partial G0 baseline-fidelity pass for `retrieval_rag` and `bot_style`. The canonical replay evidence run is `g0_rag_bot_faithful_replay` using `configs/g0_rag_bot_faithful_replay.yaml` and `scripts/inspect_g0_rag_bot_fidelity.py`. Remaining baselines, including `expel_optional`, are out of scope for this slice and are not claimed to pass G0. See [`docs/g0-baseline-fidelity-gate-v0.4.md`](docs/g0-baseline-fidelity-gate-v0.4.md).

## v0.3 Partial G0 Fidelity Gate (historical scaffold release)

`v0.3` used a deterministic hash-projection retriever and in-memory meta-buffer; its config and inspector remain historical comparison artifacts. See [`docs/g0-baseline-fidelity-gate-v0.3.md`](docs/g0-baseline-fidelity-gate-v0.3.md).

## v0.2 Multitask Replay Gate

`v0.2` is a replay-only QA demo for Game24, Math Equation Balancer, and WordSorting. Its proxy-baseline claim is: retrieval-only RAG lower-bound, Reflexion-style verbal memory proxy, BoT-style thought-template proxy; not a complete reproduction. See [`docs/g0-baseline-fidelity-gate-v0.2.md`](docs/g0-baseline-fidelity-gate-v0.2.md).

## Documentation

- Phase-12: [`docs/phase12-implementation-contract.md`](docs/phase12-implementation-contract.md), [`docs/logging-v3-phase12.md`](docs/logging-v3-phase12.md), [`docs/phase12-operator-runbook.md`](docs/phase12-operator-runbook.md)
- Baseline fidelity: [`docs/baseline-fidelity-v2.md`](docs/baseline-fidelity-v2.md), [`docs/baseline-fidelity-v2-evidence.md`](docs/baseline-fidelity-v2-evidence.md), [`docs/baseline-fidelity-v1.md`](docs/baseline-fidelity-v1.md)
- Logging: [`docs/logging-audit-remediation-v0.7.2.md`](docs/logging-audit-remediation-v0.7.2.md), [`docs/logging-audit-remediation-v0.7.md`](docs/logging-audit-remediation-v0.7.md), [`docs/logging-contract-v1.md`](docs/logging-contract-v1.md), [`docs/logging-contract-v2-phase11.md`](docs/logging-contract-v2-phase11.md), [`docs/logging-audit-remediation-phase11.md`](docs/logging-audit-remediation-phase11.md)
- Historical G0: [`docs/g0-baseline-fidelity-gate-v0.6.md`](docs/g0-baseline-fidelity-gate-v0.6.md), [`docs/g0-dc-rs-reflexion-fidelity-followup.md`](docs/g0-dc-rs-reflexion-fidelity-followup.md), [`docs/g0-baseline-fidelity-gate-v0.5.md`](docs/g0-baseline-fidelity-gate-v0.5.md), [`docs/g0-baseline-fidelity-gate-v0.4.md`](docs/g0-baseline-fidelity-gate-v0.4.md), [`docs/g0-baseline-fidelity-gate-v0.3.md`](docs/g0-baseline-fidelity-gate-v0.3.md), [`docs/g0-baseline-fidelity-gate-v0.2.md`](docs/g0-baseline-fidelity-gate-v0.2.md)
- Technical notes: [`docs/replay-qa-demo-v0.2.md`](docs/replay-qa-demo-v0.2.md), [`docs/replay-qa-demo-v0.1.md`](docs/replay-qa-demo-v0.1.md)
