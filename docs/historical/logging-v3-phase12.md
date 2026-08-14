# Logging v3, Phase-12 Contract

## Contract boundary

Phase-12 records use `schema_version=logging_v3` and `contract_level=phase12`.
`logging_v2` remains bound to Phase-11. Legacy readers remain available, but no old
artifact is promoted by filling fields or changing a label.

The contract is anchored to audited HEAD
`830b89c8c169ffa9cdea472887fdae134dbae7cf` and Experiment Design SHA-256
`984fe2881690d93a8ccced87abf03de4bf0012158462cf07ed23505414073eb0`.

## Tagged schemas

Run metadata is tagged by `metadata_kind`:

| Kind | Scientific status | Governance applicability |
|---|---|---|
| `pre_route` | Non-scientific or admitted pre-route scientific | Route selection and seed allocation forbidden |
| `selected_route` | Scientific | Admission, route selection, and seed allocation required |
| `exploratory_code_non_scientific` | Non-scientific | Source route, source allocation, activation, and admission evidence forbidden |
| `exploratory_code_scientific` | Scientific | Admission, source route, source allocation, and exploratory activation required |

Trials are tagged by `trial_kind`. `branch_free_prefix` uses only a
`branch_free_prefix` execution key. `memory_branch` uses a `memory_arm` key with one
Methods-facing arm. `nomem_singleton` uses the singleton `*` key and has no arm.

Events are tagged by `record_type`: `tool_event`, `retrieval_event`, `context_event`,
`admission_event`, `intervention_event`, `checkpoint_event`, `eligibility_event`, or
`failure_event`. Unknown tags and extra fields are rejected. Sensitivity records use
their own `kind` discriminator and reject unrelated factor fields.

## Run identity and applicability

One run template represents one task family, baseline condition, execution key,
trajectory seed slot, and sensitivity cell. The prefix key is arm-free and requires a
prefix template key. A memory-arm key projects to the theory protocol only for Clean,
Contam, and Filter. Correct and Irrelevant project to no protocol. NoMem forbids a
prefix or arm projection.

Every run records protocol version, evidence layer, run family, model snapshot through
its template, task and condition, execution key, seed, sensitivity cell, registry
versions, policy versions, embedding and tool hashes, and applicable governance.
Applicability is fail-closed. A missing required reference or a forbidden extra
reference invalidates the record rather than producing a partial join.

## Artifact topology

`run.json` is the run anchor. A completed Phase-12 run has canonical public streams
for:

* `trials.jsonl`
* `calls.jsonl`
* `tool_events.jsonl`
* `retrieval_events.jsonl`
* `context_events.jsonl`
* `failures.jsonl`
* `memory_events.jsonl`
* `admission_events.jsonl`
* `intervention_events.jsonl`
* `checkpoint_events.jsonl`
* `eligibility_events.jsonl`

An event joins to one known run and trial, has a unique event ID, and receives a
monotone `event_seq`. A public artifact manifest records status, SHA-256, and row count.
Finalization is atomic. Failed temporary directories remain inspectable. Empty streams
are valid when an event type didn't occur.

Hidden labels live only in `audit/audit_labels.jsonl`. Public rows reject audit fields.
The Filter and baseline execution paths cannot import or read the hidden audit
authority.

## Observable separation

Retrieval events record candidate IDs and scores. Context events record final included
and removed IDs. These records don't create exposure by themselves.

For Contam and Filter, theory exposure is supported only from the explicit answer
call's exact target source spans. Presence without spans is rejected. Clean has no
applicable theory exposure. Correct and Irrelevant use the separate auxiliary inclusion
record and cannot extend the theory variable. A positive operational-use value needs a
versioned attribution rule and cannot exceed supported exposure.

Exact lineage needs stable child and parent or version IDs plus a write-time edge.
Similarity, retrieval rank, prompt visibility, or later reconstruction is approximate
or unavailable evidence, not a substitute for the edge.

## Failure and denominator policy

Phase-12 uses independent `execution_status` and `analysis_inclusion` axes.

* A valid response, whether correct or incorrect, is `completed` and `included`.
* A malformed or unparsable model response is `model_behavior`, remains `completed`
  and `included`, and receives verified score zero when no valid answer exists.
* Provider API, infrastructure, verifier, protocol, and implementation failures are
  `invalidated` and `excluded_prespecified`.
* An invalidated row needs a linked rerun under the frozen rerun policy before the
  aggregate is valid.
* Missing five-arm panels or unsupported observations remain `not_estimable`.

This policy prevents model-output failures from disappearing from accuracy while
keeping engineering loss separate from model behavior.

## No-pooling and reconstruction

Aggregation separates metadata kind, protocol version, evidence layer, run family,
task, condition, sensitivity cell, and applicable route, allocation, or activation
IDs. The full compatibility key also binds execution, prefix, registry, admission, and
policy identity. Text-only and Python-sandbox records use different protocol, tool,
run, aggregate, and claim identities. They cannot enter one superiority claim.

Seed-level arm means are computed before paired contrasts. Trial-level resampling,
complete-case dropping, and weight renormalization are forbidden. Run, aggregate,
claim-scope, and archive manifests preserve unsupported aggregates and nonclaims rather
than deleting them.

## Machine-checked tagged and outcome index

- `execution_key:branch_free_prefix`
- `execution_key:memory_arm`
- `execution_key:nomem_singleton`
- `metadata_kind:pre_route`
- `metadata_kind:selected_route`
- `metadata_kind:exploratory_code_non_scientific`
- `metadata_kind:exploratory_code_scientific`
- `trial_kind:branch_free_prefix`
- `trial_kind:memory_branch`
- `trial_kind:nomem_singleton`
- `record_type:tool_event`
- `record_type:retrieval_event`
- `record_type:context_event`
- `record_type:admission_event`
- `record_type:intervention_event`
- `record_type:checkpoint_event`
- `record_type:eligibility_event`
- `record_type:failure_event`
- `execution_status:completed`
- `execution_status:invalidated`
- `failure_class:none`
- `failure_class:model_behavior`
- `failure_class:provider_api`
- `failure_class:infrastructure`
- `failure_class:verifier`
- `failure_class:protocol`
- `failure_class:implementation`
- `analysis_inclusion:included`
- `analysis_inclusion:excluded_prespecified`
- `parse_status:parsed`
- `parse_status:invalid`
- `parse_status:not_produced`
