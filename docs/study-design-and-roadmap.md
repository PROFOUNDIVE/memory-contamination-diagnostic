# Study design, readiness, datasets, and roadmap

## Scope and claim boundary

This page states the current scientific contract directly so that it can be understood from a
public clone. It separates the confirmatory design, implemented repository capability, historical
evidence, and prospective roadmap. The roadmap cannot change the confirmatory contract until a
new prospective freeze binds the revised scope before outcomes are inspected.

The study asks how controlled relevant false memory affects verifier-based continual or
multi-step reasoning, conditional on a prespecified task, baseline, checkpoint, and support
population. The strongest supported claim is a controlled intervention effect within those
boundaries, not universal contamination detection or causal attribution of every mechanism.

## Confirmatory design

### Tasks and matched conditions

The current confirmatory tasks are Game24, Math Equation Balancer, and Word Sorting. They remain
separate task streams rather than one pooled benchmark.

Each memory-bearing baseline is branched from the same clean checkpoint:

| Condition | Human meaning | Confirmatory role |
|---|---|---|
| Clean | Continue from the unchanged clean checkpoint | Reference arm |
| Correct | Insert a matched valid native memory object | Auxiliary semantic control |
| Irrelevant | Insert a matched valid but task-inapplicable object | Auxiliary relevance/insertion control |
| Contam | Insert a matched relevant false native object with a deterministic witness | Intervention arm |

An incorrect answer is not automatically contamination. The controlled target includes false
roots deliberately inserted under the protocol and descendants attributable to those roots.
Naturally occurring errors are separate unless they are written to memory and meet the registered
classification rule.

The primary Clean-versus-Contam contrast is a total controlled-intervention effect. It can combine
insertion, occupancy, retrieval competition, relevance, and semantic falsehood. Correct-versus-
Contam is closest to a semantic-polarity comparison only when matching is adequate;
Irrelevant-versus-Contam still combines relevance and semantic-status differences.

### Memory baselines

| Baseline | Memory mechanism |
|---|---|
| FH / FH-bounded | Append-only prior interaction history under the exact registered context contract |
| RAG-Frozen | Retrieval-only access to a closed task corpus and immutable index/runtime contract |
| BoT-style | Distilled reasoning templates in a bounded native thought buffer |
| Reflexion-style | Verifier-triggered verbal reflections retained for later work |
| NoMem | No persistent memory; a Clean-only singleton negative control |

Exact FH and FH-bounded are separate strata. RAG-Frozen is also separate from online or
self-updating RAG variants. `NoMem` is not crossed with Correct, Irrelevant, or Contam.

Filter-Challenge-v1 / Filter-v5 is historical and exploratory mitigation work only. It cannot
alter confirmatory timing, readiness, support, checkpoints, seeds, estimands, route selection,
authorization, or aggregates.

## Horizons, structural readiness, and support

- `H_run` is the number of ordinary post-intervention trials actually provider-generated and
  stored. It determines execution calls, tokens, latency, storage, and budget.
- `H_primary` is an independently registered finite analysis window for the principal estimand.
  It must satisfy `H_primary <= H_run`; neither value may be inferred from the other.

Structural readiness is evaluated before outcomes. For a fixed seed, task, route, baseline, and
checkpoint, the native state must accept every matched condition and complete the full registered
suffix without implementation, capacity, serialization, or fidelity failure. Readiness does not
require prior failure, rich memory, a free slot, nonempty state, or realized retrieval.

Support follows the comparison being estimated:

1. a baseline's Level-1 effect uses that baseline's structurally ready seeds;
2. a direct baseline interaction uses pairwise common support at the same fixed checkpoint;
3. strict all-baseline common support is a sensitivity analysis, not a gate for the first two.

Retrieval and final-context inclusion are observed mechanism outcomes, not readiness conditions.

## Implemented repository state

| Surface | Repository status | What it does not establish |
|---|---|---|
| Three reduced-Main task registries | Implemented and hash-manifested | Final expanded task scope or task × baseline injection support |
| Native three-task contexts | Implemented | Scientific readiness for proposed additional tasks |
| FH-bounded, RAG-Frozen, BoT-style, Reflexion-style branching | Implemented for the reduced panel | Final DC-RS registration |
| NoMem singleton behavior | Implemented | A fifth crossed memory-bearing condition |
| Authority, execution-registry, and provenance validators | Deterministic and no-provider | Approval or existence of a final freeze packet |
| Clean-prefix calibration | Provider-backed reduced-panel feasibility/cost surface using a superseded all-baseline joint law | Current Main support, seeds, route, final budget, suffix intervention, Filter, NoMem, or scientific outcomes |
| Earlier-phase and baseline-fidelity records | Retained as historical calibration/provenance | Current authority or confirmatory mitigation evidence |

Generic execution code contains optional DC-RS support, and historical fidelity evidence describes
an adapted optional comparator. This is not final integration into the confirmatory registry and
does not freeze DC-RS checkpoints, estimands, capacity, or pairwise support.

No confirmatory Main outcome has been inspected or reported.

## Main dataset paths and coverage

The repository contains one hash-manifested reduced-Main source pool for each current task:

| Task | Frozen repository path | Source identity recorded by the manifest | Pool coverage |
|---|---|---|---|
| Game24 | `data/phase13/main/game24_main_v1.jsonl` | `buffer-of-thought-llm@d771df690ca03c82ae84c206734b762110920d85:benchmarks/gameof24.jsonl` | 95 unique rows from 98 source rows; 3 duplicate canonical signatures removed |
| Math Equation Balancer | `data/phase13/main/math_equation_balancer_main_v1.jsonl` | `dynamic-cheatsheet@5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9:data/MathEquationBalancer` | 250 of 250 source rows |
| Word Sorting | `data/phase13/main/word_sorting_main_v1.jsonl` | `buffer-of-thought-llm@d771df690ca03c82ae84c206734b762110920d85:benchmarks/word_sorting.jsonl` | 250 of 250 source rows |

`data/phase13/main/main_registry_manifest_v1.json` records source identities, content hashes,
counts, and exclusions. `data/phase13/main/exclusions_v1.json` records signatures reserved away
from Main because they occur in candidate, calibration, or pilot material.

These counts describe dataset coverage only. They do not show that every row supports every
baseline or condition, that contamination can be injected at every checkpoint, or that the full
`H_run` suffix can execute. Final task × baseline readiness and injection support must be
recomputed under the final backbone, baseline set, task set, RAG corpus, horizons, and native
capacity contracts. These three files do not cover the proposed MMLU-Pro or GPQA additions.

## Prospective pre-Main roadmap

The intended expansion stops at the following items. No further task or baseline expansion is
planned for Core Main.

1. **Integrate DC-RS as the additional adaptive-memory baseline.** Freeze its native adapter,
   fidelity evidence, checkpoint semantics, capacity, Level-1 estimand, and pairwise support.
2. **Add a prospectively fixed MMLU-Pro subset and GPQA Diamond.** Freeze exact item identities,
   scoring/verifier contracts, contamination candidates, matched controls, and exclusions.
3. **Revise and freeze the RAG-Frozen clean corpus.** Freeze contents, provenance, index,
   embedding/runtime contract, branch deltas, and hashes before outcomes.
4. **Evaluate GPT-5.6 Luna and choose the backbone prospectively.** Check task accuracy, format
   failures, token/call use, and latency without inspecting Main outcomes. The current registered
   GPT-4o package remains authoritative until a later freeze selects a replacement.
5. **Freeze the horizons.** The roadmap proposes `H_run = 50` and retention of
   `H_primary = 5` unless revised before Main. These values remain prospective until bound by the
   final authority and execution closure.
6. **Compute final structural readiness and contamination-injection support.** Produce
   task × baseline Level-1 support, pairwise common-support status at one shared prospectively
   frozen task/route checkpoint, reason codes, and `NOT_ESTIMABLE` handling.
7. **Produce the final prospective authority and execution freeze.** Bind all approved scope,
   registries, parameters, hashes, analysis rules, budgets, provenance, and separate provider
   authorization before confirmatory outcomes are inspected.

After the relevant scientific scope is frozen, implementation-ready cells may execute
sequentially under that same contract. Sequential execution is an ordering choice, not permission
to change the contract between cells.

## Minimum final-freeze contents

The final closure must bind at least:

- tasks, baseline strata, matched conditions, NoMem role, and backbone snapshot;
- task datasets, exclusions, contamination candidates, controls, and RAG corpus;
- exact checkpoints, suffix identities, `H_run`, `H_primary`, and analysis windows;
- baseline-specific readiness, pairwise support, precision targets, and seed allocation;
- estimands, estimators, intervals, multiplicity, missingness, and `NOT_ESTIMABLE` rules;
- execution templates, call/transport ceilings, token/cost budget, and execution owner;
- repository/config identities, artifact hashes, provenance manifest/seal, and separate live-call
  authorization.

Historical Filter artifacts cannot satisfy these confirmatory requirements.
