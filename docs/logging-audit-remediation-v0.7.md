# Logging Audit Remediation — v0.7

**Tag:** `v0.7`  
**Repo:** `memory-contamination-diagnostic`  
**Scope:** strict logging remediation for the locked offline 39-row replay contract gate  
**Evidence run:** `logging-contract-replay-*` replay-only runs from `configs/logging_contract_replay.yaml`

> For operator-facing artifact rules, see [`docs/logging-contract-v1.md`](logging-contract-v1.md). For the preceding baseline-fidelity release, see [`docs/g0-baseline-fidelity-gate-v0.6.md`](g0-baseline-fidelity-gate-v0.6.md).

---

## TL;DR

`v0.7` closes the strict logging audit blockers by making new replay evidence self-describing, joinable, stage-gated, and failure-durable. The canonical trial row remains `trials.jsonl`; strict runs now also write `run.json`, `calls.jsonl`, `failures.jsonl`, `filter_events.jsonl`, and `memory_events.jsonl` through one writer.

Exposure is derived only from render-time source spans on the explicit answer-producing provider call. Retrieval membership, memory availability, string search, similarity, or legacy proxy fields are not enough to claim supported exposure.

Replay evidence is a fidelity/QA artifact, not benchmark or manuscript-quality evidence. This offline gate is not an API-connected pilot, main run, or benchmark result. No API-connected pilot was run; main readiness requires later evidence and an explicit decision.

---

## Exact Scope

The offline gate uses one locked sample from each pilot task and the five main baselines:

- Tasks: `game24`, `math_equation_balancer`, `word_sorting`
- Baselines: `no_memory`, `full_history`, `retrieval_rag`, `reflexion_style`, `bot_style`
- Arms: `no_memory × clean` only; memory baselines across `clean`, `contaminated`, and `contaminated_filter`
- Model label: `replay_logging_contract`

This yields `3 no_memory rows + 36 memory-baseline rows = 39 strict replay trials`. Optional Dynamic Cheatsheet/DC-RS comparators remain outside this denominator.

---

## Remediation Points

- `run.json` records authoritative run metadata, stage, schema version, provider, model snapshots, sample/order hashes, counts, and final status.
- `calls.jsonl` is the prompt/response/telemetry/source-span authority. `trials.jsonl.prompt_messages` must equal the messages from `answer_call_id`.
- `failures.jsonl` records provider, parser, verifier, and runner failure origins without logging exception messages, headers, credentials, or raw SDK payloads.
- `filter_events.jsonl` records item-level apply decisions and post-answer outcomes for contaminated-filter rows.
- `memory_events.jsonl` records normalized mutation evidence and before/after snapshot hashes for memory-writing baselines.
- Strict aggregation requires an explicit matching `--stage`; legacy artifacts require `--allow-legacy` and cannot be relabeled as strict evidence.

---

## Verification Commands

Run from the repository root:

```bash
python -m memcontam.cli validate-config configs/logging_contract_replay.yaml
python -m pytest tests/test_logging_contract_gate.py tests/test_task_verifiers.py tests/test_cli_run.py tests/test_contamination_catalog.py tests/test_openai_compatible_client.py tests/test_aggregate.py -q
python -m pytest tests/test_logging_schema.py tests/test_logging_writer.py tests/test_method_calls.py tests/test_cli_run.py tests/test_aggregate.py tests/test_logging_contract_gate.py -q
python -m ruff check src tests scripts

RUN_ID="logging-contract-replay-$(date -u +%Y%m%dT%H%M%SZ)"
python -m memcontam.cli run configs/logging_contract_replay.yaml --run-id "$RUN_ID"
python -m memcontam.cli aggregate "runs/$RUN_ID" --stage replay
```

Expected results:

- Config validation succeeds without API credentials.
- The contract test emits and validates exactly 39 strict successful replay rows.
- Focused logging tests and the historical replay/aggregate suite pass.
- Ruff reports no errors.
- Strict aggregate reports `stage=replay`, `n_trials=39`, and no legacy/main/pilot claim.

---

## Limitations And Non-Claims

- This is an offline logging contract gate, not a benchmark, pilot, main run, or manuscript result.
- It does not claim API-connected readiness or main-run GO.
- It does not rewrite, backfill, or relabel historical `runs/*` artifacts.
- Legacy rows remain readable but are visibly ineligible for strict Phase-10 evidence unless the strict streams and answer-call source spans exist.
- Live runs must keep the same stage structure and require separate provider snapshots and evidence.

---

## Artifact File List

- `configs/logging_contract_replay.yaml`
- `docs/logging-contract-v1.md`
- `src/memcontam/logging/schema.py`
- `src/memcontam/logging/writer.py`
- `src/memcontam/logging/provenance.py`
- `src/memcontam/clients/recording.py`
- `src/memcontam/evaluation/aggregate.py`
- `src/memcontam/cli.py`
- `tests/test_logging_schema.py`
- `tests/test_logging_writer.py`
- `tests/test_method_calls.py`
- `tests/test_aggregate.py`
- `tests/test_logging_contract_gate.py`
- `README.md`
