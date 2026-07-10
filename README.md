# memory-contamination-diagnostic

Controlled memory-contamination diagnostic harness for reasoning-memory systems.

## v0.3 Partial G0 Fidelity Gate

`v0.3` tags the partial G0 baseline-fidelity pass for `retrieval_rag` and `bot_style`.

- `retrieval_rag` now uses a deterministic sentence-embedding retriever with logged cosine scores and provenance.
- `bot_style` now implements distill → retrieve → instantiate → update-buffer, with a `memory_write_event` recording lineage.

These two baselines are raised from prompt-label proxies to faithful adapted baselines. The remaining baselines (`no_memory`, `full_history`, `reflexion_style`, `dynamic_cheatsheet_optional`, `expel_optional`) are explicitly out of scope for this slice and are not claimed to pass G0.

This is not a benchmark result or a full paper reproduction. Verification commands:

```bash
python -m memcontam.cli validate-config configs/pilot_game24.yaml
python -m pytest tests/test_cli_run.py tests/test_aggregate.py tests/test_logging_schema.py tests/test_docs_scope.py -q
python -m memcontam.cli run configs/pilot_game24.yaml --run-id g0_rag_bot_gate_replay
python -m memcontam.cli aggregate runs/g0_rag_bot_gate_replay
python .sisyphus/evidence/inspect_g0_replay.py
```

See [`docs/g0-baseline-fidelity-gate-v0.3.md`](docs/g0-baseline-fidelity-gate-v0.3.md) for the full v0.3 report and [`docs/g0-baseline-fidelity-gate-v0.2.md`](docs/g0-baseline-fidelity-gate-v0.2.md) for the pre-implementation gap analysis.

## v0.2 Multitask Replay Gate

`v0.2` is a replay-only QA demo across three locked pilot tasks: Game24, Math Equation Balancer, and WordSorting. It validates the logging contract, multitask builder/verifier dispatch, optional live-smoke wiring, repeated-failure tracking, local proxy baselines, contamination lineage fields, and shallow aggregate metrics without API keys or network access in replay mode.

This release is not a benchmark result or full paper reproduction. The proxy baseline claim is:

> retrieval-only RAG lower-bound, Reflexion-style verbal memory proxy, BoT-style thought-template proxy; not full reproduction.

The current G0 fidelity work is scoped to RAG + BoT only; `no_memory`, `full_history`, `reflexion_style`, Dynamic Cheatsheet, and ExpeL are not claimed to pass G0 in this plan. The canonical replay evidence run is `g0_rag_bot_gate_replay` (config `configs/pilot_game24.yaml`), inspected by `.sisyphus/evidence/inspect_g0_replay.py`. See `docs/g0-baseline-fidelity-gate-v0.2.md` for the full implementation result and verification commands.

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

- v0.3 G0 partial pass report: `docs/g0-baseline-fidelity-gate-v0.3.md`
- v0.2 G0 gap analysis: `docs/g0-baseline-fidelity-gate-v0.2.md`
- v0.2 technical notes: `docs/replay-qa-demo-v0.2.md`
- v0.1 technical notes: `docs/replay-qa-demo-v0.1.md`
