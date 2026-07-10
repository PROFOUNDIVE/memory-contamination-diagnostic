# G0 Baseline Fidelity Gate — v0.3 Partial Pass

**Tag:** `v0.3`  
**Repo:** `memory-contamination-diagnostic`  
**Scope:** Partial G0 pass for `retrieval_rag` and `bot_style` only  
**Evidence run:** `runs/g0_rag_bot_gate_replay/trials.jsonl`

> For the pre-implementation gap analysis, see [`docs/g0-baseline-fidelity-gate-v0.2.md`](g0-baseline-fidelity-gate-v0.2.md).

---

## What v0.3 claims

`v0.3` raises only two baselines from prompt-label proxies to faithful *adapted* baselines inside the replay harness:

- **`retrieval_rag`** — deterministic sentence-embedding retrieval with cosine scores and provenance records; prompt includes retrieved IDs, text, and scores.
- **`bot_style`** — problem distillation → semantic template retrieval → template instantiation → post-solution buffer update, with a logged `memory_write_event`.

All other baselines remain explicitly out of scope for this G0 slice:

- `no_memory`
- `full_history`
- `reflexion_style`
- `dynamic_cheatsheet_optional`
- `expel_optional`

This is an **adapted baseline gate**, not a full paper reproduction.

---

## Key implementation points

### RAG

- `src/memcontam/memory/retrieval.py` exposes `retrieve_records(query, entries, k=3)`.
- Retrieval uses a lightweight deterministic sentence embedding (hash projection) so replay stays deterministic and offline.
- `src/memcontam/baselines/retrieval_rag.py` logs retrieval provenance in the prompt and trial metadata.
- RAG remains **read-only**; no memory mutation.

### BoT

- `src/memcontam/baselines/bot_style.py` mirrors the official `meta_distiller_prompt` flow: distill the task, retrieve a high-level thought template, instantiate it for the current task, then solve.
- After a trial, the runner (`src/memcontam/cli.py:_bot_memory_writeback`) distills the raw response into a new `thought_template` `MemoryEntry`, appends it to the meta-buffer, and emits a `memory_write_event` with lineage.
- The meta-buffer starts from scratch; no hand-written seed templates.

---

## Verification commands

Run from the repository root after any baseline or runner change:

```bash
python -m memcontam.cli validate-config configs/pilot_game24.yaml
python -m pytest tests/test_cli_run.py tests/test_aggregate.py tests/test_logging_schema.py tests/test_docs_scope.py -q
python -m memcontam.cli run configs/pilot_game24.yaml --run-id g0_rag_bot_gate_replay
python -m memcontam.cli aggregate runs/g0_rag_bot_gate_replay
python .sisyphus/evidence/inspect_g0_replay.py
```

Expected results:

- Config validates.
- 49 focused tests pass.
- Replay writes `runs/g0_rag_bot_gate_replay/trials.jsonl`.
- Aggregate emits valid JSON.
- Inspection script reports RAG/BoT evidence present.

---

## Backward compatibility

The original v0.2 multitask demo still works without API keys:

```bash
python -m memcontam.cli validate-config configs/pilot_multitask_replay.yaml
python -m pytest tests/test_task_verifiers.py tests/test_cli_run.py tests/test_contamination_catalog.py tests/test_openai_compatible_client.py tests/test_aggregate.py -q
python -m memcontam.cli run configs/pilot_multitask_replay.yaml --run-id pilot_multitask_replay_qa
python -m memcontam.cli aggregate runs/pilot_multitask_replay_qa
```

---

## Files changed for this slice

- `src/memcontam/memory/retrieval.py`
- `src/memcontam/baselines/retrieval_rag.py`
- `src/memcontam/baselines/bot_style.py`
- `src/memcontam/cli.py`
- `tests/test_logging_schema.py`
- `tests/test_aggregate.py`
- `tests/test_cli_run.py`
- `tests/test_docs_scope.py`
- `docs/g0-baseline-fidelity-gate-v0.3.md`
- `README.md`

---

## Known limitations

- Only Game24 is used for the canonical replay evidence run.
- Sentence embeddings are deterministic hash projections, not a learned model.
- Optional baselines are not implemented.
