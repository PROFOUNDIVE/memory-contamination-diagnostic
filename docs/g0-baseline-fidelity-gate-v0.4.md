# G0 Baseline Fidelity Gate — v0.4 Faithful Adapted RAG/BoT Baselines

**Tag:** `v0.4`  
**Repo:** `memory-contamination-diagnostic`  
**Scope:** partial G0 pass for `retrieval_rag` and `bot_style` only  
**Evidence run:** `runs/g0_rag_bot_faithful_replay/trials.jsonl`

> For the previous implementation report, see [`docs/g0-baseline-fidelity-gate-v0.3.md`](g0-baseline-fidelity-gate-v0.3.md).  
> For the pre-implementation gap analysis, see [`docs/g0-baseline-fidelity-gate-v0.2.md`](g0-baseline-fidelity-gate-v0.2.md).

---

## What v0.4 claims

`v0.4` raises the same two baselines as v0.3 from prompt-label proxies to faithful *adapted* baselines, but now pins the retriever model, corpus, and BoT meta-buffer persistence so the gate is reproducible rather than relying on transient replay fixtures.

- **`retrieval_rag`** — deterministic sentence-embedding retrieval with cosine scores, full provenance records, and a versioned legal corpus; the default production path uses the pinned learned encoder, while the canonical offline replay config explicitly sets `embedding.offline_fallback: true` to substitute deterministic fake embeddings for network-free QA.
- **`bot_style`** — reference-aligned five-stage reasoning (`bot_problem_distill`, `bot_instantiate_solve`, `bot_thought_distill`, `bot_novelty_decide`) with the same configured provider for top-1 template retrieval and persistent meta-buffer updates keyed by `(run_id, task_name, baseline, arm, backbone)`.

All other baselines remain explicitly out of scope for this G0 slice:

- `no_memory`
- `full_history`
- `reflexion_style`
- `dynamic_cheatsheet_optional`
- `expel_optional`

This is an **adapted baseline gate**, not a full paper reproduction or benchmark result.

---

## Key implementation points

### RAG mechanism

- `src/memcontam/cli.py` loads the pinned learned encoder `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` from the configured local cache by default and lets missing-checkpoint failures propagate; only configs with `embedding.offline_fallback: true` use deterministic fake embeddings.
- `src/memcontam/memory/embeddings.py` defines the pinned encoder provider and deterministic fake fallback provider.
- `src/memcontam/memory/corpus.py` reads the versioned legal corpus `data/memory/catalog_v1.jsonl` and hashes it for reproducibility.
- `src/memcontam/memory/retrieval.py` performs exact top-k retrieval against the loaded corpus and returns records with `document_id`, `rank`, `score`, `text`, `title_or_type`, `clean_or_contaminated`, `source`, `corpus_hash`, `embedding_model_id`, `embedding_revision`, and `embedding_library_version`.
- `src/memcontam/baselines/retrieval_rag.py` issues a single `rag_generate` stage, includes the retrieved IDs, text, and scores in the prompt, and logs the full provenance in `retrieved_memory` and trial metadata.
- RAG remains **read-only**; `memory_before` equals `memory_after` and no `memory_write_event` is emitted.
- Prompt content and logged retrieval records are aligned; every retrieved `document_id` and `text` appears in the prompt.

### BoT mechanism

- `src/memcontam/baselines/bot_style.py` runs the five reference-aligned stages:
  1. `bot_problem_distill` — extract key information, constraints, and a distilled task description.
  2. buffer retrieve — retrieve one high-level thought template with the configured embedding provider.
  3. `bot_instantiate_solve` — instantiate the retrieved template for the current problem.
  4. `bot_thought_distill` — distill the solved trajectory into a new candidate thought template.
  5. `bot_novelty_decide` — decide whether the candidate template is novel enough to insert into the meta-buffer.
- `src/memcontam/memory/bot_buffer.py` maintains the meta-buffer; updates are accepted only after verifier success and are gated by the novelty decision.
- `src/memcontam/memory/run_state.py` persists the meta-buffer on disk keyed by the identity tuple `(run_id, task_name, baseline, arm, backbone)`, so memory survives across trials and processes while staying isolated across identities.
- The meta-buffer starts from scratch; no hand-written seed templates are injected. The optional `warm_up_path` in the config may be empty.
- `src/memcontam/cli.py` writes back accepted templates through a logged `memory_write_event` that records `status`, `new_entry_id`, parent trial lineage, and the novelty decision response.

### Temporary LLM boundary

External LLM responses in this gate are replay fixtures read from `data/replay/g0_rag_bot_faithful_v1.yaml` via `src/memcontam/clients/replay.py`. This is intentional: it keeps the gate free of provider secrets and network variance. live runs must use the identical stage structure; only the client backend changes.

---

## Verification commands

Run from the repository root after any baseline or runner change:

```bash
python -m memcontam.cli validate-config configs/g0_rag_bot_faithful_replay.yaml
python -m pytest tests/test_task_verifiers.py tests/test_cli_run.py tests/test_contamination_catalog.py tests/test_openai_compatible_client.py tests/test_aggregate.py tests/test_docs_scope.py -q
python -m memcontam.cli run configs/g0_rag_bot_faithful_replay.yaml --run-id g0_rag_bot_faithful_replay
python -m memcontam.cli aggregate runs/g0_rag_bot_faithful_replay
python scripts/inspect_g0_rag_bot_fidelity.py runs/g0_rag_bot_faithful_replay
```

Expected results:

- Config validates.
- All focused tests pass.
- Replay writes `runs/g0_rag_bot_faithful_replay/trials.jsonl`.
- Aggregate emits valid JSON.
- Inspector reports RAG/BoT evidence present.

---

## Backward compatibility

The original v0.2 multitask demo and the v0.3 Game24 gate still work without API keys:

```bash
python -m memcontam.cli validate-config configs/pilot_multitask_replay.yaml
python -m pytest tests/test_task_verifiers.py tests/test_cli_run.py tests/test_contamination_catalog.py tests/test_openai_compatible_client.py tests/test_aggregate.py -q
python -m memcontam.cli run configs/pilot_multitask_replay.yaml --run-id pilot_multitask_replay_qa
python -m memcontam.cli aggregate runs/pilot_multitask_replay_qa
```

v0.3 configs can still be validated and run, but they do not exercise the pinned encoder, versioned corpus, or persistent BoT buffer introduced in v0.4.

---

## Files changed for this slice

- `configs/g0_rag_bot_faithful_replay.yaml`
- `data/memory/catalog_v1.jsonl`
- `data/replay/g0_rag_bot_faithful_v1.yaml`
- `scripts/inspect_g0_rag_bot_fidelity.py`
- `src/memcontam/baselines/bot_runtime.py`
- `src/memcontam/baselines/bot_style.py`
- `src/memcontam/baselines/retrieval_rag.py`
- `src/memcontam/cli.py`
- `src/memcontam/clients/base.py`
- `src/memcontam/clients/recording.py`
- `src/memcontam/clients/replay.py`
- `src/memcontam/evaluation/aggregate.py`
- `src/memcontam/logging/schema.py`
- `src/memcontam/memory/bot_buffer.py`
- `src/memcontam/memory/corpus.py`
- `src/memcontam/memory/embeddings.py`
- `src/memcontam/memory/retrieval.py`
- `src/memcontam/memory/run_state.py`
- `tests/test_aggregate.py`
- `tests/test_bot_buffer.py`
- `tests/test_bot_style.py`
- `tests/test_bot_updates.py`
- `tests/test_cli_run.py`
- `tests/test_docs_scope.py`
- `tests/test_embeddings.py`
- `tests/test_fidelity_inspector.py`
- `tests/test_logging_schema.py`
- `tests/test_memory_corpus.py`
- `tests/test_method_calls.py`
- `tests/test_rag_runner.py`
- `tests/test_replay_client.py`
- `tests/test_replay_fixtures.py`
- `tests/test_retrieval_rag.py`
- `tests/test_run_state.py`
- `docs/g0-baseline-fidelity-gate-v0.4.md`
- `README.md`

---

## Known limitations

- Only `retrieval_rag` and `bot_style` pass this G0 slice.
- Optional baselines remain placeholders.
- Sentence embeddings are fixed to the pinned revision; changing the model or corpus requires a new gate tag.
