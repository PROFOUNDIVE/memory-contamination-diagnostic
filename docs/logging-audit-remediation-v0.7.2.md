# Logging Audit Remediation — v0.7.2

**Tag:** `v0.7.2`  
**Repo:** `memory-contamination-diagnostic`  
**Scope:** Phase-11 `logging_v2` offline replay contract release over the existing strict logging baselines  
**Evidence workflow:** `configs/logging_contract_phase11_replay.yaml` with replay-only runs aggregated via `--contract phase11`

> For the Phase-11 operator contract, see [`docs/logging-contract-v2-phase11.md`](logging-contract-v2-phase11.md). For the preceding strict logging release, see [`docs/logging-audit-remediation-v0.7.md`](logging-audit-remediation-v0.7.md). The operator-facing Phase-11 checklist template remains [`docs/logging-audit-remediation-phase11.md`](logging-audit-remediation-phase11.md).

---

## TL;DR

`v0.7.2` packages the Phase-11 `logging_v2` contract release for offline replay QA. The strict logging surface now carries typed evaluation-law metadata, a fixed controlled target set, exact direct-edge lineage for supported derivation claims, answer-call-only exposure accounting, and deterministic pair/checkpoint identifiers for Phase-11 aggregation.

This release keeps the existing split-stream logging shape (`run.json`, `trials.jsonl`, `calls.jsonl`, `failures.jsonl`, `filter_events.jsonl`, and `memory_events.jsonl`) while tightening what those rows are allowed to claim. Approximate lineage remains auditable but not exact; answer-prompt exposure remains a prompt-support record, not a causal-use claim.

Replay evidence is a fidelity/QA artifact, not benchmark or manuscript-quality evidence. This offline contract release is not an API-connected pilot, main run, benchmark result, or empirical result. No API-connected pilot was run.

---

## Release status

`v0.7.2` is the release tag for the committed Phase-11 `logging_v2` contract implementation. The existing `v0.7` report remains the historical Phase-10 strict logging remediation release, while the `phase11` report template stays as the operator review checklist to fill only after a concrete replay run and artifact inspection have been completed.

The `pyproject.toml` package version remains `0.1.0`; no Python package is published. This repository tag identifies a source-controlled research artifact and its offline contract workflow only.

---

## Exact scope

Phase-11 covers the strict replay QA path for the existing logging baselines and adds contract-level structure rather than a new benchmark matrix:

- Contract: `logging_v2`, `contract_level=phase11`
- Evaluation metadata: `evaluation_law_id`, `regime`, `task_law_id`, `inference_law_id`, optional `checkpoint_policy_id`
- Fixed target set: `target_set_id`, definition version, included contamination classes, exact-lineage requirement
- Exposure basis: exact source spans on the explicit answer-producing call only
- Pairing basis: `trajectory_pair_id`, `checkpoint_index`, and deterministic `pair_id`
- Regimes: `online` for memory-writing replay, `frozen` for read-only checkpoint replay

The replay workflow remains offline: provider `replay`, embedding `offline_fallback: true`, and `live_smoke.enabled=false`.

---

## Release points

- `src/memcontam/logging/schema.py` and `src/memcontam/logging/provenance.py` define the canonical Phase-11 lineage, target-set, and exposure rules.
- `src/memcontam/logging/writer.py` rejects rows whose answer spans, target membership, evaluation-law identity, or frozen/online update context disagree with the manifest.
- `src/memcontam/cli.py` wires Phase-11 run metadata, deterministic pairing fields, and frozen-run safety checks into replay execution.
- `src/memcontam/evaluation/aggregate.py` requires contract-aware aggregation and validates exact-edge consistency, target membership, and pair completeness under `--contract phase11`.
- Memory-writing baselines (`full_history`, `reflexion_style`, `dynamic_cheatsheet_optional`, `bot_style`) now preserve Phase-11 parent/root lineage on emitted descendants instead of relying on legacy binary contamination labels alone.

---

## Verification commands

Run from the repository root:

```bash
python -m memcontam.cli validate-config configs/logging_contract_phase11_replay.yaml
python -m pytest tests/test_phase11_logging_contract_gate.py -q
python -m pytest tests/test_docs_scope.py -q
python -m ruff check src tests scripts

RUN_ID="phase11-logging-contract-replay-$(date -u +%Y%m%dT%H%M%SZ)"
python -m memcontam.cli run configs/logging_contract_phase11_replay.yaml --run-id "$RUN_ID"
python -m memcontam.cli aggregate "runs/$RUN_ID" --stage replay --contract phase11
```

Expected results:

- Config validation succeeds without provider secrets.
- The focused Phase-11 contract gate and docs-scope tests pass.
- Ruff reports no errors.
- The replay run writes a strict `logging_v2` manifest plus the five strict JSONL streams.
- Aggregate succeeds only when `--contract phase11` matches the run manifest and pairing/target checks pass.

---

## Limitations and non-claims

- This is an offline contract release, not a live pilot, main run, benchmark result, or manuscript result.
- It does not claim complete PROV-DM coverage; only the direct evidence needed for this diagnostic is modeled canonically.
- It does not upgrade approximate lineage into exact derivation.
- It does not claim answer-prompt exposure is causal use, causal effect, or an intervention result.
- It does not relabel or backfill historical `logging_v1` or legacy `runs/*` artifacts as Phase-11 evidence.
- Generated caches and analysis sidecars are non-authoritative artifacts and must stay outside retrievable agent memory.

---

## Artifact file list

- `configs/logging_contract_phase11_replay.yaml`
- `docs/logging-contract-v2-phase11.md`
- `docs/logging-audit-remediation-phase11.md`
- `docs/logging-audit-remediation-v0.7.2.md`
- `src/memcontam/logging/schema.py`
- `src/memcontam/logging/provenance.py`
- `src/memcontam/logging/writer.py`
- `src/memcontam/cli.py`
- `src/memcontam/evaluation/aggregate.py`
- `src/memcontam/baselines/bot_runtime.py`
- `src/memcontam/baselines/dynamic_cheatsheet_optional.py`
- `src/memcontam/baselines/full_history.py`
- `src/memcontam/baselines/reflexion_style.py`
- `tests/test_logging_schema.py`
- `tests/test_logging_writer.py`
- `tests/test_provenance_storage_scaling.py`
- `tests/test_cli_run.py`
- `tests/test_aggregate.py`
- `tests/test_phase11_logging_contract_gate.py`
- `tests/test_bot_style.py`
- `tests/test_dc_rs_faithful.py`
- `tests/test_full_history_faithful.py`
- `tests/test_reflexion_faithful.py`
- `tests/test_docs_scope.py`
- `README.md`
