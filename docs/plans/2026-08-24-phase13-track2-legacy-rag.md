# Phase 13 Track 2 Legacy RAG Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Materialize deterministic, runtime-consumable current-Main legacy RAG packages for Game24, MathEquationBalancer, and WordSorting when every higher-authority leakage threshold is frozen.

**Current status:** `NOT_READY -- MEB_STRUCTURAL_SIMILARITY_THRESHOLD_UNFROZEN`. Protocol-v8 §19.4 requires a registered MEB structural-similarity threshold fixed from construction and calibration data. The Addendum and repository contain no such numerical threshold or frozen registry, so production materialization fails closed and no Track 2 package or seal is published.

**Architecture:** A dedicated audit command reads held-out registries and emits only hashed task signatures plus opaque reason codes. The ordinary materializer consumes that opaque registry, deterministic task generators, literal Addendum semantic records, and the frozen Phase 12 intervention triplets, then stages and validates task-local corpora and BGE-M3 indexes before publication. Runtime loading reconstructs the existing `RagFrozenStateV3` from validated serialized artifacts.

**Tech Stack:** Python 3.11, Pydantic v2, `fractions.Fraction`, existing BGE-M3 provider and branch-index primitives, pytest, Ruff.

---

### Task 1: Lock the byte and generator contracts

**Files:**
- Create: `tests/test_phase13_legacy_rag.py`
- Create: `src/memcontam/readiness/phase13_legacy_rag_bytes.py`
- Create: `src/memcontam/readiness/phase13_legacy_rag_generators.py`

1. Write failing tests for canonical NFC JSON bytes, Game24 canonical solutions, WordSorting candidate ordering, calibration/evaluation exclusion, and repeatability.
2. Run the focused tests and confirm RED failures are due to the missing module.
3. Implement the exact Addendum byte serializer and all three generators.
4. Bind MEB canonical construction to the repaired total order `+ < - < * < /`.
5. Re-run the focused tests to GREEN.

### Task 2: Lock corpus records and the information boundary

**Files:**
- Create: `src/memcontam/readiness/phase13_legacy_rag_semantics.py`
- Create: `src/memcontam/readiness/phase13_legacy_rag_audit.py`
- Modify: `tests/test_phase13_legacy_rag.py`

1. Write failing tests for exact A/B/C rendering, six D examples from `D_build` only, and opaque audit output.
2. Implement the 54 literal Addendum records and exact renderers.
3. Implement the independent held-out reader that applies the frozen WordSorting thresholds and publishes only task-local SHA-256 signatures and governed audit-contract metadata.
4. Prove the ordinary materializer accepts only the opaque registry schema and never receives raw held-out rows.

### Task 3: Materialize and validate frozen packages

**Files:**
- Create: `src/memcontam/readiness/phase13_legacy_rag_models.py`
- Create: `src/memcontam/readiness/phase13_legacy_rag_materialize.py`
- Create: `src/memcontam/readiness/phase13_legacy_rag_validate.py`
- Create: `data/phase13/rag/legacy/**`
- Modify: `tests/test_phase13_legacy_rag.py`

1. Write failing package, tamper, branch-base, and reproducibility tests.
2. Stage all three tasks' build/calibration registries, candidate audit ledgers, selected D records, exact 24-document clean corpora, four branch corpora, BGE-M3 vectors, hashes, runtime bindings, and reports only after the MEB threshold is frozen.
3. Publish only after the staged package validates.
4. Record the current MEB 16/64 calibration/build partition while preserving its historical RHS-completion pilot as non-current evidence.
5. Repeat materialization into a second temporary root and compare all scientifically consequential non-runtime bytes and identities.

### Task 4: Bind CLI and runtime consumption

**Files:**
- Modify: `src/memcontam/readiness/phase13_cli.py`
- Create: `src/memcontam/readiness/phase13_legacy_rag_runtime.py`
- Modify: `tests/test_phase13_legacy_rag.py`

1. Add deterministic `audit-legacy-rag`, `materialize-legacy-rag`, and `validate-legacy-rag` commands.
2. Load a validated task/branch package into `RagFrozenStateV3` without rebuilding vectors.
3. Exercise clean and intervention branch top-3 retrieval through the runtime loader.
4. Confirm a cross-task binding and a tampered vector/hash fail closed.

### Task 5: Verify the complete Track 2 closure

**Files:**
- Validate all changed source, tests, plan, and `data/phase13/rag/legacy/**` artifacts.

1. Run LSP diagnostics on every changed Python file.
2. Run focused Phase 13 legacy RAG tests and relevant existing RAG/verifier tests.
3. Run Ruff on changed Python files, then the repository-configured broader Ruff gate.
4. Run the CLI help, valid audit/materialize/validate flow, one invalid input, and live local top-3 retrieval using the exact cached BGE-M3 snapshot.
5. Until the MEB threshold is frozen, report `NOT_READY -- MEB_STRUCTURAL_SIMILARITY_THRESHOLD_UNFROZEN`, publish no package seal, create a recoverable blocker checkpoint, and do not continue into Track 3.
