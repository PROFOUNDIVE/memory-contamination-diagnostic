# v0.1 Replay/QA Demo Technical Notes

`v0.1` is a replay-only QA demo for the Game24 memory-contamination diagnostic harness. It validates that the runner can emit metric-ready `TrialLog` rows, proxy baseline evidence, contamination lineage fields, and shallow aggregate metrics without live model calls.

This release is not a benchmark result, not a full method reproduction, and not an admission-control proof.

## Scope

| Area | v0.1 status |
|---|---|
| Task | Game24 pilot only |
| Execution | Local replay from `configs/pilot_game24.yaml` |
| Model calls | No network, no API key, no vLLM server required |
| Baselines | `no_memory`, `full_history`, `retrieval_rag`, `reflexion_style`, `bot_style` |
| Arms | `clean`, `contaminated`, `contaminated_filter` |
| Evidence | JSONL trial logs plus shallow aggregate JSON |

## Main Commands

Run from the repository root:

```bash
python -m memcontam.cli validate-config configs/pilot_game24.yaml
python -m pytest tests/test_logging_schema.py tests/test_cli_run.py tests/test_game24_verifier.py tests/test_aggregate.py -q
python -m memcontam.cli run configs/pilot_game24.yaml --run-id pilot_game24_replay_qa
python -m memcontam.cli aggregate runs/pilot_game24_replay_qa
```

Expected replay size for the bundled config is 150 trial rows:

```text
5 Game24 samples x 5 baselines x 3 arms x 2 replay model labels = 150 rows
```

## Architecture

### Replay runner

`src/memcontam/cli.py` owns the runnable demo path:

- `validate-config` loads and checks the config shape.
- `run` loads Game24 JSONL samples, expands baseline/arm/model combinations, builds prompts, reads replay responses, verifies Game24 answers, and writes `trials.jsonl`.
- `aggregate` validates `trials.jsonl` rows and prints shallow JSON metrics.

Replay responses come from `configs/pilot_game24.yaml` through `ReplayClient`. Missing replay responses, malformed input JSONL, empty input, invalid run ids, and missing contamination catalogs fail loudly.

### Trial logging schema

`src/memcontam/logging/schema.py` defines the canonical `TrialLog` row. v0.1 includes:

- Reproducibility metadata in `metadata`, including git commit, config hash, provider/model labels, query date, seed/order, and policy versions.
- `ContaminationExposure` with exact keys for condition, exposure flag, source ids, contamination types, memory-before ids, retrieved ids, exposure mode, and reason.
- Label constraints for bad-memory uptake, repeated failure, and recovery-after-filter status.
- Strict non-negative `latency_ms` and normalized replay `token_usage`.

### Proxy baselines

The RAG/Reflexion/BoT entries are working local proxies, not upstream paper implementations.

| Baseline | Implemented behavior | Not implemented |
|---|---|---|
| `retrieval_rag` | Lexical memory retrieval, prompt injection under `Retrieved memory:`, logged `retrieved_memory` and `retrieved_scores` | Embeddings, vector DB, reranking, grounded citation pipeline |
| `bot_style` | Lexical retrieval with `k=1`, prompt injection under `Thought template:` | Buffer-of-Thought buffer training, adaptive thought evolution |
| `reflexion_style` | Recent memory/reflection strings injected under `Reflections:` | Failure-triggered reflection generation, iterative retry loop, memory learning loop |

Safe claim language:

```text
retrieval-only RAG lower-bound, Reflexion-style verbal memory proxy, BoT-style thought-template proxy; not full reproduction.
```

### Contamination evidence

`contamination_exposure` means controlled availability or retrieval opportunity. It does not prove that a model used the contaminated memory.

v0.1 computes/logs:

- Clean condition rows with explicit non-exposure.
- Contaminated rows with source lineage when contaminated memory is available.
- Filter rows with `filter_decision` and post-filter exposure state.
- Conservative uptake labels: exposure alone is not `uptake_detected`.
- `memory_write_event = null` unless explicit parent/source links exist.

### Aggregation

`src/memcontam/evaluation/aggregate.py` reads `trials.jsonl`, validates each row with `TrialLog.model_validate`, groups by `task_name`, `baseline`, `arm`, and `backbone`, then computes shallow metrics:

- `n_trials`
- `verified_success_count`, `verified_success_rate`
- `contaminated_condition_count`, `contaminated_condition_rate`
- `controlled_exposure_count`, `controlled_exposure_rate`
- `trial_level_uptake_count`, `trial_level_uptake_rate`
- `contaminated_descendant_count`, `contaminated_descendant_rate`
- `filter_drop_count`
- `token_usage_total`
- `latency_ms_min`, `latency_ms_mean`, `latency_ms_max`
- `repeated_failure_count`, `repeated_failure_rate`
- `vanilla_to_contamination_degradation_rate`

Unsupported or unevaluable metrics return the string `not_computed` instead of silently approximating.

## Artifact Map

| Purpose | Path |
|---|---|
| Config for demo run | `configs/pilot_game24.yaml` |
| CLI/replay runner | `src/memcontam/cli.py` |
| Replay client | `src/memcontam/clients/replay.py` |
| Trial schema | `src/memcontam/logging/schema.py` |
| Lexical retrieval | `src/memcontam/memory/retrieval.py` |
| Proxy baseline policies | `src/memcontam/baselines/*.py` |
| Aggregate implementation | `src/memcontam/evaluation/aggregate.py` |
| CLI replay/proxy/provenance tests | `tests/test_cli_run.py` |
| Schema tests | `tests/test_logging_schema.py` |
| Aggregate tests | `tests/test_aggregate.py` |
| Generated replay output | `runs/<run-id>/trials.jsonl` |

## Verification Surface

The release has been checked through the CLI surface:

- Config validation exits successfully.
- Pytest suite for schema, CLI, Game24 verifier, and aggregate passes.
- Replay run writes JSONL trial logs.
- Aggregate command prints parseable JSON.
- Every generated row can be validated with `TrialLog.model_validate(row)`.

## Limitations And Next Work

- Only Game24 is covered in v0.1.
- Math Equation Balancer and WordSorting are future task-file work.
- vLLM/OpenAI-compatible live execution is future readiness work.
- Dynamic Cheatsheet and ExpeL are appendix/future comparators only.
- Group-level research claims require later experimental runs beyond this replay QA demo.
