# memory-contamination-diagnostic

Controlled memory-contamination diagnostic harness for reasoning-memory systems.

## v0.1 Replay/QA Demo

`v0.1` is a replay-only Game24 QA demo. It validates the logging contract, local proxy baselines, contamination lineage fields, and shallow aggregate metrics without API keys, network access, vLLM, or upstream method repositories.

This release is not a benchmark result or full paper reproduction. The proxy baseline claim is:

> retrieval-only RAG lower-bound, Reflexion-style verbal memory proxy, BoT-style thought-template proxy; not full reproduction.

## Quick Start

```bash
python -m memcontam.cli validate-config configs/pilot_game24.yaml
python -m pytest tests/test_logging_schema.py tests/test_cli_run.py tests/test_game24_verifier.py tests/test_aggregate.py -q
python -m memcontam.cli run configs/pilot_game24.yaml --run-id pilot_game24_replay_qa
python -m memcontam.cli aggregate runs/pilot_game24_replay_qa
```

The bundled config emits 150 replay trial rows:

```text
5 Game24 samples x 5 baselines x 3 arms x 2 replay model labels
```

## Implemented In v0.1

- Replay CLI runner with JSONL trial logging.
- Metric-ready `TrialLog` schema with reproducibility metadata.
- Controlled contamination exposure and filter-decision evidence.
- Local `retrieval_rag`, `reflexion_style`, and `bot_style` proxy baselines.
- Shallow aggregate JSON output with unsupported metrics marked as `not_computed`.
- Tests for schema, CLI replay/proxy behavior, Game24 verification, and aggregate metrics.

## Documentation

- Technical notes: `docs/replay-qa-demo-v0.1.md`
