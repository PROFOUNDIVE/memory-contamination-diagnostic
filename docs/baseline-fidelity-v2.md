# Baseline Fidelity V2

## Authority and Claim Boundary

This document is the sole authority for Baseline-Fidelity-V2 in this repository.
[`baseline-fidelity-v2-evidence.md`](baseline-fidelity-v2-evidence.md) records the evidence
provenance and current gate results. Baseline-Fidelity-V1 and the G0 reports remain
readable historical records, but they cannot support a V2 fidelity claim.

V2 covers faithful adapted baselines and explicitly named proxy baselines. Its replay
artifacts are QA and fidelity evidence, not benchmark or manuscript-quality evidence.
They don't establish scientific results, causal effects, production readiness, or a
complete reproduction of any upstream method.

`v0.8` is a repository research-artifact tag for the completed V2 source-contract
remediation. It is not a package release or overall V2 certification. `pyproject.toml`
remains at `0.1.0`.

## Exact Method Claims

Only these mechanism claims are allowed:

| Identifier | Baseline-Fidelity-V2 claim |
|---|---|
| `no_memory` | one-call no-persistent-memory baseline |
| `full_history` | context-bounded full-history with full append-only store |
| `retrieval_rag` | training-free dense retrieval with black-box input-layer augmentation |
| `bot_style` | deterministic paper-aligned BoT-style proxy |
| `reflexion_style` | failure-gated verbal-reflection adaptation with one same-sample retry |
| `dynamic_cheatsheet_rs_optional` | adapted optional DC-RS appendix comparator |

Full History keeps every canonical task and raw response pair in its append-only store,
but renders only a newest, pair-atomic suffix that fits the declared prompt budget. RAG
is read-only, requires a non-empty identified corpus, retrieves top three records, and
shows the answer model only retrieved text. BoT uses native structured stages,
thresholded description retrieval, one of three coarse fallback structures on a miss,
grounded thought distillation, and deterministic native admission. Reflexion keeps an
append-only physical reflection store while showing the actor only the latest three
reflections. DC-RS stores raw generated output separately from its parsed answer and
remains optional.

These labels don't claim exact Lewis-style RAG training or marginalization, an exact
`PROFOUNDIVE/buffer-of-thought-llm` runtime, the complete official Reflexion agent,
unbounded Full History after prompt truncation, or primary-baseline status for DC-RS.

## V1 and V2 No-Pooling Rule

V1 and V2 artifacts must fail closed when pooled. Compatibility requires matching
memory policy, prompt, retry policy, execution contract, failure taxonomy, embedding
identity, corpus identity, and fidelity gate layer. Test-double retrieval evidence
can't be pooled with pinned-semantic retrieval evidence.

There is no artifact migration from V1 to V2. V1 readers may inspect historical data,
but they can't reconstruct missing raw model output, prompt-visible IDs, exact lineage,
or source-contract evidence. F1A evidence cannot be represented as F1B or F1C evidence.

## Fidelity Gate Status

Overall V2 certification: **BLOCKED**.

F1A structural integration replay: **PASS**.
F1B source-contract replay: **PASS**.
F1C pinned real-retriever and mocked-live boundary: **BLOCKED**.

| Layer | Status | What the layer establishes |
|---|---|---|
| F1A structural integration replay | **PASS** | Runner shape, native adapter routing, stage topology, outcomes, strict joins, state isolation, and audit artifacts with an explicit test-double embedding provider. |
| F1B source-contract replay | **PASS** | Native stage outputs, byte-locked prompts, model-visible information, failure triples, source spans, state deltas, call counts, RAG top-three retrieval, and optional DC-RS behavior. |
| F1C pinned real-retriever and mocked-live boundary | **BLOCKED** | The gate exists and enforces cache-only BGE-M3, fake-provider rejection, denied sockets, and mocked OpenAI-compatible answer dispatch. This checkout lacks the pinned model cache, so it reports `missing_cached_bge_m3`. |

The F1C gate requires `BAAI/bge-m3` revision
`5617a9f61b028005a4858fdac845db406aefb181`, loaded with cache-only semantics.
It must not download weights or substitute fake embeddings. The mocked-live client tests
answer dispatch and provider metadata without a paid model call. A V2 fidelity pass can
be claimed only after the same F1C verifier returns `overall=pass` with the pinned cache.

## Prompt and Provider Versions

All V2 layers use this compatibility tuple:

```text
schema_version: logging_v2
contract_level: phase11
memory_policy_version: baseline_fidelity_v2
prompt_version: baseline_fidelity_v2
retry_policy_version: baseline_fidelity_v2
baseline_execution_contract_version: baseline_fidelity_v2
failure_taxonomy_version: baseline_fidelity_v2
```

| Layer | Answer provider and version | Embedding provider |
|---|---|---|
| F1A | replay, `baseline_fidelity_v2_structural_fixture` | explicit `test_double`, non-scientific offline replay only |
| F1B | replay, `baseline_fidelity_v2_source_contract_fixture` | explicit `test_double`, non-scientific offline replay only |
| F1C | mocked OpenAI-compatible transport, `mocked_openai_compatible_v1` | `BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`, dimension 1024, normalized, cache-only |

F1A and F1B model names are replay labels. F1C uses mocked answer transport. None is a
live scientific provider snapshot.

## Canonical Reproduction Commands

Run from the repository root. Use new run IDs if the canonical directories already
exist.

F1A:

```bash
python -m memcontam.cli validate-config configs/baseline_fidelity_v2_structural_replay.yaml
python -m pytest -q tests/test_baseline_fidelity_replay.py
python -m memcontam.cli run configs/baseline_fidelity_v2_structural_replay.yaml --run-id bfv2-structural-replay
python -m memcontam.cli aggregate runs/bfv2-structural-replay --stage replay --contract phase11
```

F1B:

```bash
python -m memcontam.cli validate-config configs/baseline_fidelity_v2_source_contract_replay.yaml
python -m pytest -q tests/test_baseline_source_contract_replay.py
python -m memcontam.cli run configs/baseline_fidelity_v2_source_contract_replay.yaml --run-id bfv2-source-contract-replay
python scripts/inspect_baseline_fidelity_v2.py runs/bfv2-source-contract-replay
python -m memcontam.cli aggregate runs/bfv2-source-contract-replay --stage replay --contract phase11
python scripts/report_baseline_resource_usage.py runs/bfv2-source-contract-replay
python scripts/build_bfv2_evidence_manifest.py --config configs/baseline_fidelity_v2_source_contract_replay.yaml --run-dir runs/bfv2-source-contract-replay --inspector-output .sisyphus/evidence/baseline-fidelity-v2/task-13/f1b-inspector.json --output .sisyphus/evidence/baseline-fidelity-v2/evidence_manifest.json
```

F1C:

```bash
python scripts/verify_bge_m3_fidelity.py
python -m pytest -q tests/test_bge_m3_fidelity.py tests/test_live_embedding_policy.py tests/test_openai_compatible_client.py
```

Documentation and compatibility gate:

```bash
python -m pytest -q tests/test_docs_scope.py tests/test_baseline_policy_compatibility.py
python -m ruff check src tests scripts
git diff --check
```

## Unresolved Non-Claims

V2 does not claim benchmark or manuscript evidence, empirical score validity, causal
attribution, scientific live admission, production contamination, production filtering,
filtered aggregation, automatic V1 migration, local vLLM provisioning, cross-platform
dependency reproducibility, a compute-matched main baseline, or certification readiness.

The clean V2 source-fidelity work does not activate historical `contaminated` or
`contaminated_filter` arms. Valid incorrect trials remain in the experimental outcome
set. A future compute-matched no-memory control, if added, must be auxiliary and must
not replace or pool with method-native baselines.
