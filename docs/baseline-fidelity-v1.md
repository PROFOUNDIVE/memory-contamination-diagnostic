# Baseline Fidelity V1

> **Superseded for V2:** This historical report cannot support a Baseline-Fidelity-V2 fidelity claim. The sole V2 authority is [`docs/baseline-fidelity-v2.md`](baseline-fidelity-v2.md), with evidence provenance in [`docs/baseline-fidelity-v2-evidence.md`](baseline-fidelity-v2-evidence.md).

This file is the sole authoritative Baseline-Fidelity-V1 contract. It describes the
implemented `baseline_fidelity_v1` policy and its Phase-11-compatible extension seams.
Older baseline reports remain historical evidence, not alternate specifications.

## Scope and Non-Goals

Baseline Fidelity V1 covers faithful adapted versions of five fixed main baselines,
their clean execution paths, deterministic replay contracts, native memory behavior,
closed failure handling, audit artifacts, and inactive contamination/filter seams.
These are adapted baselines, not complete reproductions of the source methods.

Replay runs use `stage=replay`, `execution_class=offline_contract_replay`,
`provider=replay`, `scientific_result=false`, and `scientific_gate_id=null`. Their
artifacts are QA and contract evidence only. They are never benchmark, manuscript,
empirical, or scientific results. Pilot and main runs use live OpenAI-compatible
execution, but they are scientific only when `scientific_result=true` and a separately
accepted, non-null `scientific_gate_id` is present.

New pilot and main configurations permit only `arm=clean`. The `contaminated` and
`contaminated_filter` arms remain parseable solely for historical and deferred
compatibility. This contract does not claim production contamination, production
filtering, filtered aggregation, automatic legacy migration, local vLLM lifecycle
management, causal validity, or release readiness.

## Fixed Main Baselines

The fixed main set is exactly:

| Baseline | Native state | Semantic calls | Native update |
|---|---|---|---|
| `no_memory` | none | `no_memory_generate` | none |
| `full_history` | ordered raw task/response records | `full_history_generate` | append every completed answer response |
| `retrieval_rag` | immutable corpus | `rag_generate` | none |
| `bot_style` | reusable thought templates | `bot_problem_distill`, `bot_instantiate_solve`, `bot_thought_distill` | deterministic native novelty admission |
| `reflexion_style` | latest three corrective reflections | `reflexion_generate`, then `reflexion_reflect` only after an authenticated incorrect attempt | append valid corrective reflections |

Every baseline uses the shared task serialization, final-answer parser, verifier
boundary, provider client protocol, outcome type, logging path, and failure taxonomy.
A parseable answer is an experimental outcome whether it is correct or incorrect.
The no-memory baseline makes exactly one `no_memory_generate` call, uses no state or
memory source spans, and parses the response with the shared parser.

## Shared Execution and Failure Contracts

`BaselineExecutionOutcome.status` separates execution validity from correctness.
A valid incorrect answer is a successful trial with `status="succeeded"`,
`verifier_result=False`, no `error_type`, no failure disposition, no
`scientific_ineligibility_reason`, and no `FailureEvent`. Aggregation retains it in the
denominator. Only execution and contract failures use the closed taxonomy below.

| Failure disposition | Error type | Scientific ineligibility reason |
|---|---|---|
| `no_memory_invalid_final_answer` | `BaselineOutputError` | `invalid_final_answer` |
| `full_history_invalid_final_answer` | `BaselineOutputError` | `invalid_final_answer` |
| `rag_invalid_final_answer` | `BaselineOutputError` | `invalid_final_answer` |
| `rag_retrieval_failed` | `RetrievalContractError` | `retrieval_failed` |
| `rag_embedding_failed` | `EmbeddingContractError` | `embedding_failed` |
| `rag_manifest_invalid` | `CorpusContractError` | `manifest_invalid` |
| `rag_embedding_dimension_mismatch` | `EmbeddingContractError` | `embedding_dimension_mismatch` |
| `rag_embedding_provider_unpinned` | `EmbeddingContractError` | `embedding_provider_unpinned` |
| `bot_invalid_problem_distillation` | `BaselineOutputError` | `invalid_problem_distillation` |
| `bot_invalid_solve_result` | `BaselineOutputError` | `invalid_solve_result` |
| `bot_invalid_thought_distillation` | `BaselineOutputError` | `invalid_thought_distillation` |
| `reflexion_invalid_generation` | `BaselineOutputError` | `invalid_reflexion_generation` |
| `reflexion_invalid_reflection` | `BaselineOutputError` | `invalid_reflection` |
| `provider_call_failed` | `ProviderCallFailure` | `provider_call_failed` |
| `verifier_contract_failed` | `VerifierContractError` | `verifier_contract_failed` |

Each failed outcome carries exactly one matching row. Unknown error types,
dispositions, reasons, and combinations are invalid. `attempts_exhausted` is not an
execution failure value. Transport retries remain call-level `retry_count` values and
must not stand in for semantic attempts.

## Exact Full History

Full History stores `FullHistoryPayload(task_input, raw_response)` records and renders
all of them in chronological order. It performs one `full_history_generate` call for
the current task.

Every completed response is appended before final-answer parsing, including an empty
string. A parse failure never rolls back that append. If no response object exists,
nothing is appended. Rendered history contains the prior raw tasks and responses only.
It excludes correctness, parsed answers, verifier data, recursive prompts, and writer
metadata. A parseable incorrect response remains a successful trial.

## Retrieval-Only RAG

RAG is read-only. It queries with canonical task JSON, retrieves exactly the top three
documents, renders only their text, and makes one `rag_generate` call. Retrieval uses
normalized embeddings and normalized dot product, with `document_id` ascending as the
deterministic tie break.

The pinned embedding contract is:

```text
model: BAAI/bge-m3
revision: 5617a9f61b028005a4858fdac845db406aefb181
dimension: 1024
normalize_embeddings: true
top_k: 3
```

The immutable corpus is identified by the shared `CorpusIdentity`. Retrieval,
embedding, manifest, dimension, and provider-pin failures occur before the answer call
and use their exact taxonomy rows. RAG never writes memory, and a parseable incorrect
answer remains successful.

## BoT-Style Memory

BoT first requires strict, unfenced problem-distillation JSON with non-empty
`key_information`, `restrictions`, and `distilled_task`. It retrieves at most one
template by description at score `>= 0.7`; otherwise it uses the fixed fallback
procedure. `bot_instantiate_solve` then requires strict JSON containing a non-empty
`solution_trace` and `final_answer`. The shared parser reads that final answer.

The runtime next calls `bot_thought_distill` with the exact solution trace and final
answer. Its strict result contains a non-empty description and template, one of the
three accepted categories, and unique explicitly used memory IDs that are all visible.
Only those explicitly used IDs become supports and exact parents.

The verifier ordering is fixed:

1. Parse the structured solve result.
2. Validate thought distillation.
3. Determine and freeze novelty, admission, parents, supports, and prospective state.
4. Invoke the verifier.
5. Materialize the frozen transition without changing admission.
6. Attach the Boolean verifier result as `source_outcome`, or `None` if the verifier contract fails.

Native admission occurs when the buffer is empty or maximum description similarity is
strictly below `0.7`. Equality rejects. There is no `bot_novelty_decide` model call.
Correctness and verifier availability never gate native admission. Invalid thought
distillation invokes no verifier and writes nothing, but records
`memory_write_event.status="rejected_invalid_distillation"`. A verifier-contract
failure keeps an already admitted native transition with `source_outcome=None`, while
the trial fails under the closed verifier row. A valid incorrect answer keeps its
frozen native decision and returns `status="succeeded"`, `verifier_result=False`.

## Reflexion-Style Memory

Reflexion keeps a physical latest-three reflection buffer. An attempt becomes
authenticated only after a response is received, its final answer parses, and the
verifier returns a Boolean result. Every authenticated incorrect attempt has the sole
failure class `incorrect_answer`; only such an attempt can trigger
`reflexion_reflect`.

The reflection response is strict corrective JSON with
`failure_class="incorrect_answer"`, non-empty reflection text, and unique explicitly
used IDs drawn from visible memory. With the normal two-attempt bound, the maximum
semantic sequence is attempt 1 generation, reflection 1 after incorrectness, attempt 2
generation, and a terminal reflection after attempt 2 is also incorrect.

A valid terminal reflection is stored before return. That terminal case is a valid
incorrect trial: it uses the attempt-2 generation call as `answer_call_id`, returns
`status="succeeded"` and `verifier_result=False`, and has no error, failure disposition,
ineligibility reason, or `FailureEvent`. Malformed generation authenticates no attempt,
triggers no reflection or continuation, and writes nothing, while retaining its
completed generation call and `answer_call_id`. A verifier-contract failure also
authenticates no attempt. Malformed reflection creates no card or event and prevents a
later attempt.

Semantic attempt and reflection-event records live in validated trial metadata. They
remain separate from transport retry counts.

## Stream, Corpus, and Checkpoint Identities

Mutable baseline state is partitioned by the complete `StreamIdentity` tuple:

```text
(run_id, task_family, baseline, arm, backbone)
```

`StreamPairKey` contains the same fields except `arm`. `CorpusIdentity` is defined once
in `memcontam.baselines.contracts` and contains
`(manifest_id, corpus_version, task_family, embedding_provider_identity)`.

`NativeCheckpoint` preserves ordered card/envelope pairs and native parameters. A
cross-arm clone deep-copies cards, envelopes, and parameters while replacing only the
identity's arm. Matched checkpoints must differ in arm, agree on every non-arm field,
share one `StreamPairKey`, retain complete arm-specific `StreamIdentity` values, and
have arm-specific checkpoint hashes. The clean source remains immutable. These
checkpoint rules are compatibility seams and don't enable production contamination.
An inactive compatibility fixture may insert exactly one native contamination root;
all pre-existing IDs, payloads, envelopes, order, and parameters remain unchanged.

## Logging V2 and Exact Lineage

Baseline Fidelity V1 keeps the existing logging models. It adds no top-level schema
fields. Artifacts retain:

```text
schema_version: logging_v2
contract_level: phase11
memory_policy_version: baseline_fidelity_v1
prompt_version: baseline_fidelity_v1
retry_policy_version: baseline_fidelity_v1
```

The compatibility tuple remains `(schema_version, contract_level,
memory_policy_version, prompt_version, retry_policy_version,
contamination_catalog_version, config_hash, git_commit)`. Execution class, scientific
flags and gate, provider profile ID, dependency lock hash, baseline execution contract
version, and failure taxonomy version are resolved-config values covered by
`config_hash`.

Trial status records execution success or failure; verifier result records correctness.
Existing trial metadata records scientific ineligibility and Reflexion semantic
events. Existing failure, call, memory, source-span, and lineage streams record their
respective evidence. Exact parents come only from explicitly declared direct parent
IDs. Updater context, visibility, source evidence, similarity, and contamination
exposure remain distinct and must never be promoted into parent edges. Memory support
IDs must be a subset of declared parent IDs.

## Provider, Resolved Configuration, and Dependency Lock

Provider dispatch is closed:

```text
replay + offline_contract_replay + replay -> ReplayClient
pilot/main + live + openai_compatible -> OpenAICompatibleClient
all other combinations -> reject before run-directory creation
```

The provider profile records the provider, normalized credential-free base URL,
API-key environment-variable name, timeout, retry limit, sorted served models, and
model snapshots. Its ID is the SHA-256 hash of sorted compact JSON. Credential values
never enter logs, hashes, artifacts, errors, or debug output.

Each run atomically writes `provider_profile.json` and redacted
`resolved_config.json` before the first provider call. `run.json`, the resolved
configuration hash, provider profile ID, model snapshots, dependency lock hash, and
Git commit provide the audit join needed to reconstruct execution without secrets.

Task 0 alone owns dependency ranges, the `pip-tools==7.6.0` development pin, and
`requirements.lock` plus `requirements-dev.lock`. Those files are generated only with
`python -m piptools compile --generate-hashes`, contain exact transitive pins and
hashes, and are installed with `--require-hashes`. This lock contract is scoped only to
CPython 3.11 on Linux x86_64. It makes no cross-platform or universal Python 3.11+
claim. Lock generation forbids `pip freeze`, alternative resolvers, unpinned fallback,
ad hoc post-lock package installation, and a `vllm` dependency. No later task may
mutate the locks to make a verification command pass.

## Inactive Admission and Filtered-State Seams

The two filter seams are inactive, runner-independent, and separately owned.
`src/memcontam/memory/admission.py` and
`tests/test_filter_extension_contract.py` own the Task 11 admission evaluator. It is a
pure deterministic check of writer authorization, schema, support, and recursive
parent admission. Missing or future references, rejected parents, cycles,
unauthorized writers, and invalid supports fail closed. It cannot read hidden origin
or contamination labels, mutate state, branch the protocol, call an oracle, or build
active/quarantine state.

`src/memcontam/memory/filtered_state.py`,
`tests/test_filtered_state_contract.py`, and `tests/fixtures/filter_extension/` own the
Task 12 filtered-state seam. It consumes precomputed Task 11 decisions and creates a
deterministic, total, disjoint partition into inactive `active` and `quarantine`
checkpoints. It preserves every card ID, envelope, payload, order key, parameter, and
native order. It doesn't recompute admission, call a provider or verifier, read oracle
labels, mutate source state, register with the runner, or enable
`contaminated_filter` execution.

## Testing, Skills, and Conventional Commits

Implementation work follows red, green, refactor; a narrow test first; affected-suite
verification; audit; and one coherent green commit. Contract work uses these skills as
applicable: `executing-plans`, `test-driven-development`, `ponytail`,
`ponytail-review`, `ponytail-debt`, `ponytail-gain`, `systematic-debugging`,
`verification-before-completion`, `requesting-code-review`,
`receiving-code-review`, `writing-plans`, and `brainstorming` when drift permits more
than one valid repair.

Repository commits use `<type>: <imperative subject>` or
`<type>(<scope>): <imperative subject>`. Allowed types are `build`, `test`, `feat`,
`fix`, `refactor`, `docs`, and `chore`. When present, the scope is one of `deps`,
`contract`, `runtime`, `logging`, `clients`, `full-history`, `retrieval`, `rag`, `bot`,
`reflexion`, `filters`, `runner`, `integration`, or `docs`. Commits must stage only
owned artifacts, report the verification run, and omit generated run/cache data and
forbidden automated co-author or tool footers.

The Task 16 documentation gate is:

```bash
python -m pytest -q tests/test_baseline_policy_compatibility.py
python -m ruff check .
git diff --check
```

The gate also runs the accepted Task 16 inline `PYDOC` assertion, which requires every
required heading to occur exactly once.

## Deferred Work

Production contamination execution, production filtering, runner registration of the
inactive seams, filtered aggregation, scientific live admission, local vLLM lifecycle
management, and automatic compatibility for historical artifacts remain deferred.
Historical contaminated and filtered arms are compatibility data only.

The Task 3A and Task 15 dependency dry-run blocker is also deferred. The exact runtime
hash dry-run rejects the unpinned unsafe transitive `setuptools>=77.0.3` requested by
`torch==2.13.0`; the development dry-run rejects `pip>=22.2` requested by
`pip-tools==7.6.0`. The former Task 3A lock test was removed after its `--no-deps`
workaround produced a false green. Resolving these issues requires explicit Task 0
dependency ownership. Baseline Fidelity V1 records the operational limitation and
does not mutate or weaken the committed lock contract.
