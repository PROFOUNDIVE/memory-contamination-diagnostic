# Logging Contract v1 Operator Rules

`configs/logging_contract_replay.yaml` is an offline, strict `logging_v1` replay gate. It proves the locked 39-row matrix only: three tasks, `no_memory × clean`, and four memory baselines across clean, contaminated, and contaminated-filter arms. Optional Dynamic Cheatsheet comparators are outside this denominator.

## Artifacts And Joins

| Artifact | Role | Required joins |
|---|---|---|
| `run.json` | Completed-run manifest and immutable run metadata | `run_metadata_id`, `run_id`, `stage`, `schema_version` |
| `trials.jsonl` | Canonical trial result and final event for each trial | `trial_id`, `answer_call_id`, `failure_id` when failed |
| `calls.jsonl` | Authoritative prompt, response, telemetry, and source-span stream | `call_id`, `trial_id`, `run_metadata_id` |
| `failures.jsonl` | Durable failure origin and continuation record | `failure_id`, `trial_id` |
| `filter_events.jsonl` | Filter input decision and post-answer outcome | `filter_id`, `trial_id`, `apply` then `outcome` |
| `memory_events.jsonl` | Typed memory mutation plus before/after snapshots | `memory_id`, `trial_id`, `source_trial_id` |

All strict rows and events share one `run_id`, `run_metadata_id`, and `stage`. `event_seq` is unique and global across the five JSONL streams. A filtered trial writes `apply` before its calls and `outcome` after its calls; a memory event and the final trial row follow. A failed trial writes its recorded call, then its failure, then its failed trial row.

## Exposure, Filter, And Memory Lineage

The answer call is the exposure authority. `trials.jsonl.prompt_messages` must exactly equal the answer call messages, and `answer_call_id` must resolve in `calls.jsonl`. Source spans name the rendered memory entries and their lineage. `supported` exposure references that answer call: `final_prompt` requires exposed source IDs in its spans, while `not_in_final_prompt` records the supported negative result, including an empty span list when no memory was rendered. Clean rows are `not_applicable`; do not infer contamination exposure from retrieval or memory snapshots alone.

For `contaminated_filter`, the filter decision records ground truth, kept and removed IDs. Its outcome records final-answer source IDs and the verifier verdict, so ground truth, decision, final spans, and verification join by `trial_id`. Memory events reconcile the trial's before/after entry IDs and canonical snapshot hashes with mutation lineage. Read-only baselines emit no memory event.

## Failure, Null, And Empty Semantics

Succeeded strict trials require non-null raw response, parsed answer, and verifier result. Failed strict trials require a failure link and may use null response, answer, and verifier result; they are excluded from accuracy-like denominators. Empty JSONL streams are valid when their event type did not occur. An empty source-span list is only a supported negative final-prompt result, never proof of exposure.

## Stage And Legacy Boundary

Strict aggregation requires an explicit matching `--stage`. This gate is `stage=replay`, `provider=replay`, `embedding.offline_fallback=true`, and `live_smoke.enabled=false`; it neither needs nor reads API credentials. Legacy artifacts have no `run.json` strict manifest and require explicit `--allow-legacy`. They cannot be mixed with strict replay artifacts or relabeled as v1 evidence.

This gate is replay QA only. It must not be used as pilot, main, or benchmark evidence. No API-connected pilot was run. Main readiness requires separate later evidence and an explicit decision.
