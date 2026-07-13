# memory-contamination-diagnostic

Controlled memory-contamination diagnostic harness for reasoning-memory systems.

## v0.5 Faithful Full-History, Reflexion, and Dynamic Cheatsheet Baselines

`v0.5` is the current full G0 baseline-fidelity pass for `full_history`, `reflexion_style`, and `dynamic_cheatsheet_optional` over the locked 3-task pilot set.

- `full_history` is a faithful append-only full-history baseline. Each trial makes one `full_history_generate` call and appends one sanitized transcript entry. Prior memory is rendered verbatim; no retrieval, summarization, or truncation is used in this gate.
- `reflexion_style` is a Reflexion-style verbal memory proxy / faithful adapted control flow. It calls `reflexion_generate` on every trial and `reflexion_reflect` only after verifier failure, then appends a non-empty reflection. The actor reads the latest three ordered reflections from the current identity. This is not a full reproduction of the official Reflexion agent.
- `dynamic_cheatsheet_optional` is a faithful adapted DC-Cu optional appendix comparator. Each trial calls `dynamic_cheatsheet_generate` then `dynamic_cheatsheet_curate`. A parsed non-empty `<cheatsheet>` block replaces the cheatsheet; missing or empty tags preserve prior state. Code-execution, provider-tool, and retrieval paths are removed. DC is not a new main baseline.

The matrix is `3 tasks × 3 baselines × 3 arms × 2 models = 162 trials`. The canonical replay evidence run is `g0_fh_reflexion_dc_faithful_replay` using `configs/g0_fh_reflexion_dc_faithful_replay.yaml` and inspected by `scripts/inspect_g0_fh_reflexion_dc_fidelity.py`.

External LLM responses are replay fixtures in this gate, so the verification requires no API keys or live model access. The replay output is a fidelity/QA artifact, not benchmark/manuscript evidence. Live runs must keep the same stage structure.

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

## v0.4 Faithful Adapted RAG/BoT Baselines

`v0.4` is the current partial G0 baseline-fidelity pass for `retrieval_rag` and `bot_style`.

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

## v0.3 Partial G0 Fidelity Gate (historical scaffold release)

`v0.3` tagged the first partial G0 baseline-fidelity pass for `retrieval_rag` and `bot_style`. It used a deterministic hash-projection retriever and an in-memory meta-buffer without pinned model revisions or versioned corpus hashing. The v0.3 config and inspector are preserved for historical comparison, but v0.4 is the current implementation report.

- `retrieval_rag` used deterministic sentence-embedding retrieval with logged cosine scores and provenance.
- `bot_style` implemented distill → retrieve → instantiate → update-buffer, with a `memory_write_event` recording lineage.

These two baselines were raised from prompt-label proxies to faithful adapted baselines. The remaining baselines (`no_memory`, `full_history`, `reflexion_style`, `dynamic_cheatsheet_optional`, `expel_optional`) were explicitly out of scope for this slice and were not claimed to pass G0.

This was not a benchmark result or a full paper reproduction. Historical verification commands:

```bash
python -m memcontam.cli validate-config configs/pilot_game24.yaml
python -m pytest tests/test_cli_run.py tests/test_aggregate.py tests/test_logging_schema.py tests/test_docs_scope.py -q
python -m memcontam.cli run configs/pilot_game24.yaml --run-id g0_rag_bot_gate_replay
python -m memcontam.cli aggregate runs/g0_rag_bot_gate_replay
python .sisyphus/evidence/inspect_g0_replay.py
```

## v0.2 Multitask Replay Gate

`v0.2` is a replay-only QA demo across three locked pilot tasks: Game24, Math Equation Balancer, and WordSorting. It validates the logging contract, multitask builder/verifier dispatch, optional live-smoke wiring, repeated-failure tracking, local proxy baselines, contamination lineage fields, and shallow aggregate metrics without API keys or network access in replay mode.

This release is not a benchmark result or full paper reproduction. The proxy baseline claim is:

> retrieval-only RAG lower-bound, Reflexion-style verbal memory proxy, BoT-style thought-template proxy; not full reproduction.

The current G0 fidelity work is scoped to RAG + BoT only; `no_memory`, `full_history`, `reflexion_style`, Dynamic Cheatsheet, and ExpeL are not claimed to pass G0 in this plan. The canonical replay evidence run is `g0_rag_bot_faithful_replay` (config `configs/g0_rag_bot_faithful_replay.yaml`), inspected by `scripts/inspect_g0_rag_bot_fidelity.py`. See `docs/g0-baseline-fidelity-gate-v0.2.md` for the full implementation result and verification commands.

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

## Documentation

- v0.5 G0 full pass report: `docs/g0-baseline-fidelity-gate-v0.5.md`
- v0.4 G0 partial pass report: `docs/g0-baseline-fidelity-gate-v0.4.md`
- v0.3 G0 partial pass report: `docs/g0-baseline-fidelity-gate-v0.3.md`
- v0.2 G0 gap analysis: `docs/g0-baseline-fidelity-gate-v0.2.md`
- v0.2 technical notes: `docs/replay-qa-demo-v0.2.md`
- v0.1 technical notes: `docs/replay-qa-demo-v0.1.md`
