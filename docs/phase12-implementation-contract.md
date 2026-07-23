# Phase-12 Implementation Contract

## Status and authority

This document describes the implemented Phase-12 repository contract. It does not
activate a run, select a route, allocate seeds, approve exploratory work, or report an
empirical result.

The frozen inputs are:

| Input | Identity |
|---|---|
| Audited repository HEAD | `830b89c8c169ffa9cdea472887fdae134dbae7cf` |
| Experiment Design | `984fe2881690d93a8ccced87abf03de4bf0012158462cf07ed23505414073eb0` |
| Logging schema | `logging_v3` |
| Contract level | `phase12` |

The Experiment Design file is `Phase 12-Compatible Pilot Main and Exploratory
Experiment Design(3).md`. Existing Baseline-Fidelity-V2 documents remain the authority
for baseline method claims and F1 status. Historical `logging_v1` and `logging_v2`
artifacts remain readable under their own contracts, but they aren't Phase-12 evidence.

## Model roles and evidence layers

Pilot-A, Pilot-B, Main-A, Main-B, and any selected type extension use one fixed GPT-4o
snapshot. Main-A has `analysis_status=primary`; Main-B has
`analysis_status=robustness`. A selected extension uses
`analysis_status=confirmatory_extension`.

Main-C is a reduced replication on one separately frozen frontier-model snapshot. It
uses `analysis_status=exploratory_model_specificity`, the same text-only permission,
paired seed slots, registered baseline conditions, and its registered reduced arm set.
Main-C cannot be promoted to confirmatory evidence. A difference between Main-C and
the GPT-4o runs is model-specific heterogeneity, not a failure of the GPT-4o study.

Evidence layers are isolated. Build and calibration evidence do not become main or
extension evidence by relabeling. Main-C's model role also remains separate from the
GPT-4o confirmatory role.

## Protocol and arm axes

The theory protocol index and the Methods-facing arm are different axes.

| Experimental arm | Protocol projection | Meaning |
|---|---|---|
| `clean` | `clean` | Unmodified branch |
| `correct` | none | Matched correct auxiliary control |
| `irrelevant` | none | Matched irrelevant auxiliary control |
| `contam` | `contam` | Registered false candidate branch |
| `filter` | `filter` | Contam branch passed through the provenance-and-support filter |

Correct and Irrelevant aren't theory protocols. The Filter never receives hidden
origin, correctness, injection, contamination, or future-outcome labels. Hidden audit
labels are stored separately and may be joined only after execution.

## Baseline conditions and fidelity labels

The primary text-only implementation supports NoMem, FH-bounded, RAG-Frozen,
BoT-style proxy, and the registered cross-trial Reflexion-style proxy. Full History may
be labeled FH only after a complete-fit proof. DC-RS remains an optional appendix and
code-augmentation comparator, not a new primary baseline.

Every run uses the fidelity label frozen by its registered condition. The schema
recognizes `negative_control`, `source_aligned`, `adapted`, `style_proxy`, and
`bounded`. These labels qualify a repository condition. They don't claim a complete
upstream-method reproduction.

RAG supports `frozen` in the primary implementation. `online_ext` and `online_self`
are schema-recognized but execution-rejected for `phase12_primary_v1`.
`not_applicable` is required for non-RAG conditions. ExpeL, natural-error replay,
query-mediated attacks, direct cheatsheet or insight injection, and provider-native
code interpreters remain deferred.

Confirmatory and mandatory robustness runs use `text_only`. The separate exploratory
matrix covers NoMem, BoT, and DC-RS under `text_only` and `python_sandbox` with
`protocol_version=phase12_code_exploratory_v1`. A local OCI sandbox is the executable
Python contract. The subprocess implementation is a non-scientific test double, not a
security boundary.

## Prefix, branch, and execution-key law

Memory-bearing conditions execute one branch-free clean prefix. Joint eligibility and
the selected lower-quantile checkpoint are recomputed for the applicable task,
condition, horizon, and sensitivity cell. Clean, Correct, Irrelevant, Contam, and
Filter suffixes then share the compatible source checkpoint. Filter starts from the
exact Contam checkpoint and exposes only its active state to readers.

`PrefixExecutionKey(kind="branch_free_prefix")` applies only to the arm-free prefix.
It is not the Clean arm. `MemoryArmExecutionKey(kind="memory_arm", arm=...)` applies
to a materialized memory-bearing branch. `NoMemExecutionKey(kind="nomem_singleton",
key="*")` applies to one memory-free execution. NoMem may have five display aliases,
but it must not execute or be counted five times.

Sensitivity cells are tagged by one factor: base, timing, horizon, affinity, FH
budget, embedding, or behavior. A cell cannot carry fields from another factor.
Timing and horizon changes recompute eligibility and the selected population. The base
population isn't silently reused.

## Route feasibility and external governance

`configs/phase12/main_3w.yaml` and `configs/phase12/main_5w.yaml` are candidate planning
inputs. Neither is a selected Main alias. Under the frozen Pilot-B rates and registry
inputs, the 3w candidate reserves `14770` base calls plus `739` rerun calls, `15509`
total against capacity `17000`, and is feasible. The 5w candidate reserves `28336`
base calls plus `1417` rerun calls, `29753` total against capacity `29000`, and is
infeasible. These reports are feasibility outputs, not a route decision.

After Pilot-B, the full pre-route MFT gate, and before main unblinding, a researcher
must provide an approved `RouteSelectionManifest` and `SeedAllocationManifest`. The
implementation validates their hashes, selected candidate, reports, requested counts,
registry identity, and exact abstract-slot-to-seed mapping. It never authors or infers
the selection.

Scientific exploratory execution also needs a `SelectedPackageResourceManifest` and
`ExploratoryActivationManifest`. Both must bind the selected route and allocation, the
immutable code-matrix plan and registry, its exploratory slot-to-seed mapping, and the
budget and reproducibility reserve inequalities. The committed
`configs/phase12/exploratory_code.yaml` stays `activation_status=inactive`.

Pre-route runs carry no route selection or route-bound seed allocation. Selected-route
runs carry validated governance IDs. Non-scientific exploratory runs carry no source
route, allocation, or activation evidence. Scientific exploratory runs require all
three applicable governance references.

## Aggregation and no-pooling rules

Trajectory seeds are the independent units. Scores are reduced within seed and arm
before paired five-arm contrasts or seed bootstrap intervals are computed. Trial rows
aren't resampled as independent observations. Incomplete arm panels, unsupported
observables, and insufficient estimability remain `not_estimable`; rows aren't selected
or weights renormalized to manufacture a result.

The compatibility key isolates schema version, contract level, metadata kind,
protocol version, evidence layer, run family, task family, baseline condition, run
template, execution key, prefix identity, sensitivity cell, registry and policy
versions, scientific admission, and applicable governance IDs. In addition:

* `logging_v1`, `logging_v2`, and `logging_v3` evidence aren't pooled.
* `phase12_primary_v1` text evidence isn't pooled with
  `phase12_code_exploratory_v1` tool evidence.
* GPT-4o confirmatory rows aren't pooled with frontier Main-C rows.
* Build, calibration, main, and extension layers stay separate.
* Correct and Irrelevant auxiliary inclusion isn't pooled into theory exposure.
* Different sensitivity cells or selected populations stay separate.

## Claim boundary

This implementation provides schemas, deterministic replay, validation, planning,
logging, aggregation, certificates, and archive checks. The repository documentation
does not report a paid provider run, benchmark evidence, manuscript evidence, causal
effect, production contamination finding, or scientific outcome.

Retrieval, final-context inclusion, theory exposure, and operational use are separate
records. Retrieval alone isn't exposure. Final-context inclusion alone isn't use or
derivation. Positive theory exposure requires the applicable inherited protocol arm
and an exact answer-call source span. Operational use requires a separate versioned
attribution rule. Similarity, prompt visibility, or hash overlap cannot create a
recorded exact parent edge.

## Machine-checked contract index

- `protocol_index:clean`
- `protocol_index:contam`
- `protocol_index:filter`
- `experimental_arm:clean`
- `experimental_arm:correct`
- `experimental_arm:irrelevant`
- `experimental_arm:contam`
- `experimental_arm:filter`
- `route_candidate:3w`
- `route_candidate:5w`
- `rag_mode:frozen`
- `rag_mode:online_ext`
- `rag_mode:online_self`
- `rag_mode:not_applicable`
- `fidelity_label:negative_control`
- `fidelity_label:source_aligned`
- `fidelity_label:adapted`
- `fidelity_label:style_proxy`
- `fidelity_label:bounded`
- `tool_mode:text_only`
- `tool_mode:python_sandbox`
- `evidence_layer:build`
- `evidence_layer:calibration`
- `evidence_layer:main`
- `evidence_layer:extension`
- `run_family:readiness`
- `run_family:pilot_a`
- `run_family:pilot_b`
- `run_family:behavioral`
- `run_family:main_a`
- `run_family:main_b`
- `run_family:main_c`
- `run_family:sequential`
- `run_family:extension`
- `run_family:exploratory_code`
- `sensitivity_kind:base`
- `sensitivity_kind:timing`
- `sensitivity_kind:horizon`
- `sensitivity_kind:affinity`
- `sensitivity_kind:fh_budget`
- `sensitivity_kind:embedding`
- `sensitivity_kind:behavior`
