# Logging Contract v2, Phase-11 Operator Rules

This document defines the `logging_v2` contract used by
`configs/logging_contract_phase11_replay.yaml`. It is an offline contract and
replay QA specification. It is not a pilot, main run, benchmark, or empirical
result. No API-connected pilot was run.

The historical `logging_v1` rules remain in
[`logging-contract-v1.md`](logging-contract-v1.md). Legacy artifacts and v1
artifacts are readable under their own rules, but neither is promoted to this
contract.

## 1. Artifact roles and joins

Phase-11 keeps the existing six-file topology:

| Artifact | Authority | Required identity and joins |
|---|---|---|
| `run.json` | Immutable run metadata, contract, evaluation law, target set, counts, and status | `run_id`, `run_metadata_id`, `stage`, `schema_version` |
| `trials.jsonl` | One canonical result row per trial | `trial_id`, `answer_call_id`, `pair_id`, `evaluation_law_id`, `target_set_id` |
| `calls.jsonl` | Prompt, response, telemetry, and render-time source spans | `call_id`, `trial_id`, `run_metadata_id`, `event_seq` |
| `failures.jsonl` | Failure origin and continuation record | `failure_id`, `trial_id` |
| `filter_events.jsonl` | Filter apply decision and post-answer outcome | `filter_id`, `trial_id`, `event_seq` |
| `memory_events.jsonl` | Memory mutation deltas and direct lineage edges | `memory_id`, `trial_id`, entry/version IDs, `event_seq` |

`run.json` is the join anchor. All strict rows and events share its run and
metadata identity. `trials.jsonl.prompt_messages` must equal the messages on
the resolved answer call. Event ordering is checked with `event_seq`; a
filtered trial has an apply event before calls and an outcome after calls.

## 2. Evaluation law and online or frozen semantics

One strict run has exactly one typed evaluation law. The manifest records:

```yaml
evaluation:
  evaluation_law_id: phase11_logging_contract_online_replay_v1
  regime: online                 # online | frozen
  task_law_id: locked_three_tasks_limit1_v1
  inference_law_id: logging_contract_phase11_replay_fixture_v1
  checkpoint_policy_id: null
```

The law identity, regime, task law, inference law, and checkpoint policy are
joinable metadata, not free-form notes. Compare different laws in separate
runs. `online` permits memory updates for writing baselines and has no
checkpoint reference. `frozen` is read-only in this scope. A frozen
`retrieval_rag` trial records `memory_update_mode=disabled` and a typed
checkpoint reference. `no_memory` records
`memory_update_mode=not_applicable`. A frozen memory-writing baseline is
rejected rather than silently emulated.

The Phase-11 replay gate is offline: provider `replay`, offline embedding
fallback enabled, and `live_smoke.enabled=false`. The exact verification
commands are in [the release workflow](#5-offline-verification-and-release-workflow).

## 3. Fixed target set and contamination classes

The target set is fixed before analysis and stored in `run.json`:

```yaml
target_contamination_set:
  target_set_id: controlled_injected_derived_v1
  definition_version: phase11_v1
  included_classes: [injected, derived]
  require_exact_lineage: true
```

For a memory state `M_t`, the controlled target set is the fixed membership

`B-star(M_t) = { entry in M_t | class(entry) is injected or derived }`,

subject to the exact-lineage requirement. `natural` entries are logged but
are not members of this controlled set. A different class policy requires a
different target-set ID and a separate run.

The four classes have these meanings:

* `clean`: no target error and no injected ancestor.
* `injected`: a protocol-created corruption root. Its root ID is itself.
* `derived`: a descendant with recorded exact direct ancestry to an injected
  root.
* `natural`: an ordinary system error admitted by the baseline's prespecified
  error predicate, with no injected ancestor.

The binary v1 field `clean_or_contaminated` is not enough to establish any of
these Phase-11 classes.

## 4. Lineage and the limited PROV-style boundary

Each logged item or source span records `lineage_status` as `exact`,
`approximate`, or `unavailable`, plus a `lineage_basis`. Exact derived
evidence requires direct parent IDs and injected root IDs. The authoritative
lineage is the child-local `LineageEdge` record in `memory_events.jsonl`.

Signature or similarity evidence is `approximate`. It can remain auditable,
but it cannot establish exact derivation, exact propagation, or exact target
exposure. Missing evidence is `unavailable`, not an inferred relationship.

This is a limited PROV-style record of selected entities, versions, direct
relations, and activities. It is not a full PROV-DM model and does not claim
complete provenance reconstruction. The contract records direct evidence
needed for this diagnostic only.

## 5. Exposure truth table

Exposure is evaluated against the explicit answer call, its render-time source
spans, and the run's fixed target set. Retrieval membership, memory presence,
content matching, or a similarity score cannot substitute for an answer-call
span.

| Condition | Exposure state | Exact exposed IDs |
|---|---|---|
| Clean arm | `not_applicable` | none |
| Exact target span is rendered in the answer call | `supported`, exposed | target entry and root IDs |
| Target memory exists but is absent from answer spans | `supported`, not exposed | none |
| Filtering leaves no target memory | `supported`, not exposed, distinct filtered reason | none |
| Target membership depends only on approximate lineage | `not_evaluable` | none |
| Auxiliary call has target evidence, final answer call does not | `supported`, not exposed | none |

`source_entry_ids` contains every entry rendered in the selected answer call.
`exposed_entry_ids` is the exact intersection with `B-star(M_t)`. Natural
entries can appear in source spans but never become controlled exposure under
this target set. Exposure records use of evidence in the answer prompt only;
they do not establish causal use, a causal effect, or an intervention result.

## 6. Trajectory and checkpoint pairing

`trajectory_pair_id` identifies the common trajectory context and excludes the
arm so clean, contaminated, and filter arms can be paired. It includes the
run seed and order, task, baseline, backbone, and frozen checkpoint identity
when applicable. `checkpoint_index` identifies the task or sample position on
that trajectory. `pair_id` combines trajectory identity, checkpoint index, and
sample identity.

Phase-11 degradation uses `pair_id`, never `sample_id` alone. A pair is usable
only when compatible law, target set, regime, model, seed, and checkpoint
context match, with exactly one clean and one contaminated row. Filter arms
remain separate. Incomplete or duplicate pairs are reported as not computed.

## 7. Failure, null, and empty semantics

Successful strict trials require a response, parsed answer, and verifier
result. Failed trials require a `failure_id`; response, parsed answer, and
verifier result may be null and failed trials are excluded from accuracy-like
denominators. Empty JSONL streams are valid when their event type did not
occur. An empty source-span list is a supported negative answer-prompt result,
not evidence of exposure. Read-only trials must not create memory events.

## 8. Normalized storage and non-authoritative artifacts

An immutable memory entry/version body is stored once. Canonical mutation
evidence stores IDs, before/after hashes, and event deltas: `new`, `updated`,
or `removed`. Direct parent edges are stored per child. Transitive ancestor
lists, full paths, ancestor content, and repeated full snapshots are not
canonical fields.

Full memory snapshots are optional bounded checkpoints only. Their interval,
identity, and source run are typed in the checkpoint policy. The current
39-row replay gate retains its compatibility snapshot view, but does not add a
lineage closure or an unbounded snapshot stream.

Derived summaries, indexes, traversal results, and analysis caches are
disposable non-authoritative artifacts. They must be reproducible from the
authoritative manifest, rows, calls, and direct-edge or delta records. They
must not change the evaluated memory state.

Evidence logs, provenance indexes, and analysis caches must never be injected
into an agent's retrievable memory, retrieval candidate pool, or prompt
context. Logging must observe the mechanism without changing it.

## 9. Migration boundary

| Artifact family | Identity | Allowed handling | Phase-11 status |
|---|---|---|---|
| Legacy | No strict `run.json` contract manifest | Read only with explicit `--allow-legacy` | Not Phase-11 evidence |
| Phase-10 / `logging_v1` | `logging_v1`, historical v1 fields and rules | Read and aggregate under v1 rules | Not automatically promoted |
| Phase-11 / `logging_v2` | `logging_v2`, `contract_level=phase11` | Validate and aggregate with `--contract phase11` | Eligible for this offline contract |

Migration means creating a new v2 run with complete law, target, lineage,
exposure, and pairing fields. It does not mean filling missing fields in an
old row, relabeling it, or backfilling historical `runs/*`. Keep
[`logging-contract-v1.md`](logging-contract-v1.md) unchanged as the historical
v1 operator contract.

## 10. Offline verification and release workflow

Run these commands from the repository root:

```bash
python -m memcontam.cli validate-config configs/logging_contract_phase11_replay.yaml
python -m pytest tests/test_phase11_logging_contract_gate.py -q
python -m pytest tests/test_docs_scope.py -q
python -m ruff check src tests scripts

RUN_ID="phase11-logging-contract-replay-$(date -u +%Y%m%dT%H%M%SZ)"
python -m memcontam.cli run configs/logging_contract_phase11_replay.yaml --run-id "$RUN_ID"
python -m memcontam.cli aggregate "runs/$RUN_ID" --stage replay --contract phase11
```

The replay gate must remain offline and must not be described as a pilot,
main run, benchmark, manuscript result, or release readiness decision. A
release report records the command outputs and inspected artifacts after they
exist. It must preserve the explicit statement that no API-connected pilot
was run.
