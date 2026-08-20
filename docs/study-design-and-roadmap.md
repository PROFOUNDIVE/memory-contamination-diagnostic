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

The intended Core registry contains six separate task strata: Game24, Math Equation Balancer,
Word Sorting, MMLU-Pro Engineering, MMLU-Pro Physics, and GPQA Diamond. MMLU-Pro Engineering and
Physics use distinct memory trajectories, and GPQA Diamond remains a separate task stream rather
than being pooled with either MMLU-Pro stratum.

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

The scientific target is six task strata × five selected memory-bearing baselines × four matched
arms, plus one NoMem singleton per applicable task/seed contract. Task-local readiness can still
produce a partial-crossed execution package without redefining that intended target.

### Memory baselines

| Baseline | Memory mechanism |
|---|---|
| FH-bounded | Append-only prior interaction history under the registered complete-record retention and context contract |
| RAG-Frozen | Retrieval-only access to a closed task corpus and immutable index/runtime contract |
| BoT-style | Distilled reasoning templates in a bounded native thought buffer |
| Reflexion-style | Verifier-triggered verbal reflections retained for later work |
| DC-RS adapted | Cumulative raw-interaction archive with BGE-M3 retrieval, whole-cheatsheet curation/synthesis, and persistent complete-cheatsheet replacement |
| NoMem | No persistent memory; a Clean-only singleton negative control |

`FH-bounded` is the sole selected Core FH identity; `FH-exact` remains distinct and is excluded
from confirmatory Main-A. RAG-Frozen is also separate from online or self-updating RAG variants.
`NoMem` is not crossed with Correct, Irrelevant, or Contam.

Filter-Challenge-v1 / Filter-v5 is historical and exploratory mitigation work only. It cannot
alter confirmatory timing, readiness, support, checkpoints, seeds, estimands, route selection,
authorization, or aggregates.

## Horizons, structural readiness, and support

- `H_run = 50` is the Core provider-backed execution horizon. It determines generated ordinary
  post-intervention calls, tokens, latency, storage, and budget.
- `H_primary = 50` is the independently registered primary analysis horizon, represented by
  `core_prefix_50`. Prefixes 5, 10, and 20 are prespecified sensitivities; the equal numerical
  values of `H_run` and `H_primary` do not make execution and analysis the same object.

The selected Core backbone is OpenAI Responses API `gpt-5.6-luna` with
`reasoning.effort=none`, current-turn reasoning context, `store=false`, default service tier, and
no provider-native or external code execution. This text-only contract applies across the five
selected memory-bearing baselines; it does not itself authorize provider-backed execution.

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
| Six Core task registries | Three original registries are hash-manifested; sealed MMLU-Pro Engineering, MMLU-Pro Physics, and GPQA Diamond bundles and task-order validation are implemented | Task × baseline structural/injection support |
| Native six-task ordinary runtime | All six task identities and the five selected memory-bearing baseline identities are registered in the prospective runtime | Readiness or authorization of every task × baseline cell |
| FH-bounded, RAG-Frozen, BoT-style, Reflexion-style, and DC-RS adapted branching | Four matched arms execute through baseline-native state and branch contracts | Automatic promotion of cells that remain `NOT_READY` or pending gates |
| DC-RS adapted runtime | Six-task validation, cumulative archive, BGE-M3 retrieval, retrieve-curate-generate-update order, state serialization, text-only enforcement, and complete-cheatsheet persistence are implemented | Main-A readiness, support, precision, cost, or authorization |
| FH-bounded ↔ DC-RS capacity contract | The common registered-token law is materialized and runtime-bound at 8192 generator-visible tokens | Equality of native mechanisms or capacity matching of the cumulative DC-RS archive |
| Luna provider contract | Model/client pairing, default service tier, answer/writer ceilings, and no-tool request boundaries are implemented | A dated provider snapshot, execution approval, or scientific result |
| New-MCQ RAG clean package | Task-specific 24-document clean corpora, serialized clean indices, manifests, and deterministic validators are implemented | Promotion of the three new-MCQ RAG-Frozen cells, which remain `NOT_READY` with reason `NEW_MCQ_RAG_REQUIRED_ARTIFACTS_UNFROZEN` |
| NoMem singleton behavior | Implemented | A crossed memory-bearing condition |
| Authority, execution-registry, and provenance validators | Deterministic and no-provider | Approval or existence of a final freeze packet |
| Clean-prefix calibration | Provider-backed reduced-panel feasibility/cost surface using a superseded all-baseline joint law | Current Main support, seeds, route, final budget, suffix intervention, Filter, NoMem, or scientific outcomes |
| Earlier-phase and baseline-fidelity records | Retained as historical calibration/provenance | Current authority or confirmatory mitigation evidence |

`DC-RS adapted` is now an intended confirmatory memory-bearing condition in the Experiment-owned
registry rather than an optional extension. Its implementation does not bypass task-local
Readiness-0, fixed-checkpoint, structural-support, required-pair, budget, freeze, or authorization
gates.

No confirmatory Main outcome has been inspected or reported.

## Main dataset paths and coverage

The repository contains hash-bound source pools or sealed dataset bundles for all six task strata:

| Task | Frozen repository path | Source identity recorded by the manifest | Pool coverage |
|---|---|---|---|
| Game24 | `data/phase13/main/game24_main_v1.jsonl` | `buffer-of-thought-llm@d771df690ca03c82ae84c206734b762110920d85:benchmarks/gameof24.jsonl` | 95 unique rows from 98 source rows; 3 duplicate canonical signatures removed |
| Math Equation Balancer | `data/phase13/main/math_equation_balancer_main_v1.jsonl` | `dynamic-cheatsheet@5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9:data/MathEquationBalancer` | 250 of 250 source rows |
| Word Sorting | `data/phase13/main/word_sorting_main_v1.jsonl` | `buffer-of-thought-llm@d771df690ca03c82ae84c206734b762110920d85:benchmarks/word_sorting.jsonl` | 250 of 250 source rows |
| MMLU-Pro Engineering | Sealed Core dataset bundle | `TIGER-Lab/MMLU-Pro@475d58ba0cc18a15fd5d4221f41919199e692331` with the registered selection identity | 250 rows |
| MMLU-Pro Physics | Sealed Core dataset bundle | `TIGER-Lab/MMLU-Pro@475d58ba0cc18a15fd5d4221f41919199e692331` with the registered selection identity | 250 rows |
| GPQA Diamond | Sealed Core dataset bundle | `Idavidrein/gpqa@633f5ee89ab8ad4522a9f850766b73f62147ffdd` | 198 rows |

`data/phase13/main/main_registry_manifest_v1.json` records source identities, content hashes,
counts, and exclusions. `data/phase13/main/exclusions_v1.json` records signatures reserved away
from Main because they occur in candidate, calibration, or pilot material.

These counts describe dataset coverage only. They do not show that every row supports every
baseline or condition, that contamination can be injected at every checkpoint, or that the full
`H_run` suffix can execute. Final task × baseline readiness and injection support remain separate
materialization and validation outputs.

## Prospective pre-Main roadmap

The intended expansion stops at the following items. No further task or baseline expansion is
planned for Core Main.

1. **Complete the three new-MCQ RAG-Frozen artifact packages.** Freeze and validate the remaining
   leakage-gate evidence, BGE-M3 snapshot/runtime binding, task-local candidate certification and
   relevance objects, and matched branch indices before any cell promotion.
2. **Compute final structural readiness and contamination-injection support.** Produce
   task × baseline Level-1 support, pairwise common-support status at one shared prospectively
   frozen task/route checkpoint, reason codes, and `NOT_ESTIMABLE` handling.
3. **Close the six-task/five-baseline execution package.** Bind exact checkpoints, treatment and
   control registries, GPQA display permutations, Luna snapshot/runtime metadata, execution
   templates, supported seed allocation, and route feasibility.
4. **Produce the final prospective execution freeze.** Bind all approved registries, hashes,
   analysis rules, budgets, provenance, and separate provider authorization before confirmatory
   outcomes are inspected.

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
