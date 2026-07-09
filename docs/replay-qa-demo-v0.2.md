# v0.2 Multitask Replay Gate Technical Notes

`v0.2` extends the replay-only QA demo to three locked pilot tasks: Game24, Math Equation Balancer (MEB), and WordSorting. It validates that the CLI runner can dispatch multiple task builders/verifiers, optionally route to a live OpenAI-compatible client for smoke testing, track repeated failures per task identity, and still emit metric-ready `TrialLog` rows without live model calls in replay mode.

This release is not a benchmark result, not a full method reproduction, and not an admission-control proof.

## Scope

| Area | v0.2 status |
|---|---|
| Tasks | Game24, Math Equation Balancer, WordSorting pilot |
| Execution | Local replay from `configs/pilot_multitask_replay.yaml` |
| Model calls | Replay by default; optional live-smoke via `live_smoke.enabled` |
| Baselines | `no_memory`, `full_history`, `retrieval_rag`, `reflexion_style`, `bot_style` |
| Arms | `clean` |
| Evidence | JSONL trial logs plus shallow aggregate JSON |

## Main Commands

Run from the repository root:

```bash
python -m memcontam.cli validate-config configs/pilot_multitask_replay.yaml
python -m pytest tests/test_task_verifiers.py tests/test_cli_run.py tests/test_contamination_catalog.py tests/test_openai_compatible_client.py tests/test_aggregate.py -q
python -m memcontam.cli run configs/pilot_multitask_replay.yaml --run-id pilot_multitask_replay_qa
python -m memcontam.cli aggregate runs/pilot_multitask_replay_qa
```

Expected replay size for the bundled config is 90 trial rows:

```text
3 tasks x 3 samples x 5 baselines x 1 arm x 2 replay model labels = 90 rows
```

## New In v0.2

### Multitask dispatch

`src/memcontam/cli.py` no longer hard-codes Game24. A `TASK_DISPATCH` table maps task names to builder and verifier callables:

- `game24`: existing builder + `verify_expression`
- `math_equation_balancer`: new builder + `verify_answer`
- `word_sorting`: new builder + `verify_words`

Unsupported task names fail loudly during config execution.

### Task builders and verifiers

`src/memcontam/tasks/math_equation_balancer.py` and `src/memcontam/tasks/word_sorting.py` now validate required row fields (`input`, `verifier_spec`, `words`) and normalize inputs before returning a `TaskInstance`.

`src/memcontam/verifiers/math_equation_balancer.py` accepts either the target equation string or the numeric `target_value`, normalizes whitespace, and returns `malformed_answer` / `wrong_answer` / `ok` reasons.

`src/memcontam/verifiers/word_sorting.py` accepts a token list, checks for non-string or empty input, compares against `sorted_words`, and returns `malformed_answer` / `wrong_order` / `ok` reasons.

### Live-smoke client path

`run_config` can switch from `ReplayClient` to `OpenAICompatibleClient` when `live_smoke.enabled` is true. The default remains replay, so no API key is required for the bundled config. The override is only intended for manual smoke checks against a small sample.

### Repeated-failure tracking

The old `_repeated_failure_label` helper has been replaced by `_RepeatedFailureTracker`, which keys incorrect results by `(task_name, sample_id, baseline, arm, backbone)`. This prevents cross-task false positives: the same `sample_id` failing in two different tasks is recorded as two independent first failures.

### Contamination catalog expansion

`data/contamination/catalog_v0.jsonl` adds entries for MEB and WordSorting, with the same baseline targeting patterns used for Game24. The catalog is still scoped to the locked three-task pilot set.

### QA cleanup

Redundant manual QA scripts that duplicated pytest coverage were removed. Smoke checks remain as thin `__main__` assertions in `scripts/qa_task_adapters.py`.

## Architecture

### Replay runner

`src/memcontam/cli.py` owns the runnable demo path:

- `validate-config` loads and checks the config shape.
- `run` dispatches task builders, expands baseline/arm/model combinations, builds prompts, reads replay responses (or live-smoke responses), verifies answers, and writes `trials.jsonl`.
- `aggregate` validates `trials.jsonl` rows and prints shallow JSON metrics.

### Trial logging schema

Unchanged from v0.1. See `docs/replay-qa-demo-v0.1.md`.

### Proxy baselines

Unchanged from v0.1. The same lexical retrieval proxies are applied across all three tasks.

### Aggregation

Unchanged from v0.1. Metrics are still grouped by `task_name`, `baseline`, `arm`, and `backbone`.

## Artifact Map

| Purpose | Path |
|---|---|
| Config for demo run | `configs/pilot_multitask_replay.yaml` |
| CLI/replay runner | `src/memcontam/cli.py` |
| Replay client | `src/memcontam/clients/replay.py` |
| OpenAI-compatible client | `src/memcontam/clients/openai_compatible.py` |
| MEB adapter | `src/memcontam/tasks/math_equation_balancer.py` |
| WordSorting adapter | `src/memcontam/tasks/word_sorting.py` |
| MEB verifier | `src/memcontam/verifiers/math_equation_balancer.py` |
| WordSorting verifier | `src/memcontam/verifiers/word_sorting.py` |
| Contamination catalog | `data/contamination/catalog_v0.jsonl` |
| Multitask replay tests | `tests/test_cli_run.py` |
| Verifier tests | `tests/test_task_verifiers.py` |
| Catalog tests | `tests/test_contamination_catalog.py` |
| Live-smoke tests | `tests/test_openai_compatible_client.py` |
| Generated replay output | `runs/<run-id>/trials.jsonl` |

## Verification Surface

The release has been checked through the CLI surface:

- Config validation exits successfully.
- Pytest suite for verifiers, CLI replay/proxy behavior, catalog constraints, live-smoke wiring, and aggregate metrics passes.
- Replay run writes JSONL trial logs.
- Aggregate command prints parseable JSON.
- Every generated row can be validated with `TrialLog.model_validate(row)`.

## Limitations And Next Work

- Arms are still `clean` only in the bundled multitask config; contaminated arms for MEB/WordSorting are future work.
- Live-smoke is a manual override, not a scheduled benchmark path.
- Group-level research claims require later experimental runs beyond this replay QA demo.
