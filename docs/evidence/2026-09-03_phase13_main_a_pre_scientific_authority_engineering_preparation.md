# Pre-Scientific-Authority Engineering Preparation Report

## 1. Authority routing inspected

Read-only router inspected first:

`/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/References/Theoretical Artifacts/AGENTS.md`

At the start of this preparation, it routed the current stack to:

1. `Phase 13 — THEORETICAL ARTIFACT revised-v1.md`
2. `Phase 13-Compatible Baseline Memory and Filter Design revised-v5.md`
3. `Phase 13-Compatible Contamination Construction Intervention Timing and Sensitivity Protocol revised-v8.md`
4. `2026-08-24_Phase13_MainA_PostCutoff_Acceleration_Addendum.md`, narrow scope only
5. `Phase 13-Compatible Pilot Main and Exploratory Experiment Design revised-v10.md`

Relevant fixed authority includes validity rather than representative-target equality for multi-solution tasks (Theory §3.2), memory/source instrumentation (Baseline §6.1), prospective stream/checkpoint freezing (Protocol §9.1), the measured-unit evidence join and exact MR-P5 import/revalidate/seal boundary (Experiment Design), and MEB operator insertion with a complete equation and no bare target (Addendum §8.5.2). No authority file was modified.

During finalization, the shared router changed concurrently and added `2026-09-03_Phase13_MainA_Corrective_Scientific_Decision_Authority.md`. That file still self-identifies as `CANONICAL_CANDIDATE` at line 6 while its body says SD-01 through SD-05 are approved at lines 130-146. This conflicts with the governing task instruction that those decisions are not approved. The candidate was therefore not consumed as decision authority, and no authority synchronization or production correction was performed.

## 2. Exact 09-03 references located under `7주차`

- `/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/7주차/2026-09-03_Phase13_MainA_PostAudit_Scientific_Decision_Packet_revised_v4.md`
- `/home/hyunwoo/gdrive_undergrad_research/PeerJ fast-track/7주차/2026-09-03_Phase13_MainA_Prompt_Provenance_and_Recommended_Scientific_Decisions.md`

The packet is `PENDING_USER_SCIENTIFIC_DECISIONS`. The provenance document is `RECOMMENDED_DECISIONS_FOR_USER_APPROVAL` and explicitly not authority. Its SD-01 through SD-05 recommendations were not adopted. The later router/candidate-status inconsistency described above must be resolved explicitly before authority synchronization; this report does not resolve it by inference.

## 3. Repository and sealed identity

- Branch: `phase13-main-execution-entrypoint-closure`
- Current HEAD: `d1ac6c84236ec63c367775d24aa953176d321ce0`
- Current HEAD tree: `6cdcd22911ec6ce04687dfec4f7d76c322bd483b`
- Sealed audited commit: `d1ac6c84236ec63c367775d24aa953176d321ce0`
- Sealed audited tree: `6cdcd22911ec6ce04687dfec4f7d76c322bd483b`

The checkout exactly matched the sealed audit identity before this preparation work. The initial worktree had no staged, modified, or untracked files.

## 4. A-02 findings and engineering evidence

Production path:

`phase13_main_live_cli.main` → `ProductionMainRuntime.execute_ordinary` → `phase13_main_live_runtime_support.verifier` → baseline adapter → task verifier.

- Authority-compatible implementation: `src/memcontam/verifiers/math_equation_balancer.py:17` `verify_answer` parses the complete equation, preserves operands/order, restricts operators, applies precedence with exact `Fraction` arithmetic, accepts alternative valid assignments, and rejects bare targets.
- Historical implementation: `src/memcontam/verifiers/math_equation_balancer.py:74` `verify_rhs_completion_answer` accepts either the representative target string or `target_value`.
- Obsolete production dispatch: `src/memcontam/readiness/phase13_main_live_runtime_support.py:57`, especially lines 63-66, selects `verify_rhs_completion_answer` for MEB.
- Dispatch consumers: `src/memcontam/readiness/phase13_main_live_runtime.py:207` and `:239` bind that dispatcher into ordinary and prefix contexts; `src/memcontam/experiment/phase13_ordinary_runtime.py:194` routes it through every selected baseline adapter.
- Existing direct tests: `tests/test_task_verifiers.py:20` through `:144` already cover canonical/alternative validity, bare target, order, operators, precedence, wrong equations, malformed inputs, exact fractions, and historical separation.
- Added independent matrix: `tests/test_phase13_pre_authority_a02.py` covers all six required classes. Two strict xfails expose production’s alternative-valid rejection and bare-target acceptance.

Later correction seam is exactly the MEB branch of `verifier()`. Production wiring was deliberately not changed.

## 5. E-01 findings and engineering evidence

Current binding machinery is an exact allowlist, not a transitive closure:

- `src/memcontam/readiness/phase13_main_execution_bindings.py:18` declares `EXPECTED_ARTIFACT_PATHS`.
- `validate_artifact_bindings()` at `:157` verifies only listed path/hash pairs.
- `validate_semantic_joins()` at `:189` combines selected role hashes, but does not discover imported dependencies.
- `data/phase13/main/mr_p5/execution_package_v1.json:16` contains the current artifact list.
- `src/memcontam/readiness/phase13_main_execution_models.py:164` has no repository commit or tree identity field.

`scripts/diagnose_phase13_mr_p5_closure.py` computes a provider-free, AST-derived approximation of local Python imports from the production live CLI, including importable parent-package `__init__.py` files, and compares it with current MR-P5 Python bindings. Current result: 36 bound Python paths, 189 paths in the approximation, and 154 omitted paths. Material omissions include package initializers, MEB task/verifier modules, all non-BoT baseline adapters, `phase13_legacy_rag_runtime.py`, `phase13_core_datasets.py`, task builders, verifier modules, memory/checkpoint modules, and supporting runtime/provider modules. The diagnostic is intentionally read-only; `--require-closed` fails with exit 1 and creates no freeze.

This is not a verified lower or upper execution bound: `ast.walk()` includes imports inside conditional and `TYPE_CHECKING` branches, while reflection, dynamic imports, optional-import resolution, and environment-dependent execution can be missed. A synthetic test proves parent-package initialization is traversed. After authority synchronization, commit/tree binding, lock/runtime binding, consequential configuration/data binding, and verified dynamic execution coverage are candidate engineering components for a fail-closed realization; they are not asserted here as the uniquely authority-mandated representation, and unresolved prompt artifacts remain unbound.

## 6. F-01 findings and engineering evidence

Creation-to-loss flow:

1. Baseline results carry outcomes, memory snapshots, retrieval/context events, native entries, and write envelopes in `src/memcontam/experiment/phase12/runtime_registry.py:49`.
2. `build_production_trial_evidence()` in `src/memcontam/readiness/phase13_production_runtime_evidence.py:59` derives trial/order identity, parse status, verifier outcome, memory-before/after, new/removed entries, retrieval/context, target spans, lineage, retention/eviction inputs, and terminal-missingness state.
3. `production_archive_from_ordinary()` in `src/memcontam/readiness/phase13_production_runtime_join.py:28` creates `ProductionTrialRecord` rows and terminal provider evidence.
4. `ProductionMainRuntime.execute_ordinary()` at `src/memcontam/readiness/phase13_main_live_runtime.py:220` creates and validates the archive only when a memory branch exists.
5. The archive is then discarded; `dispatch_output()` returns unit-level runtime evidence and provider calls only.
6. `persist_unit_dispatch()` in `src/memcontam/readiness/phase13_main_live_dispatch.py:70` durably writes that condensed unit evidence, not the validated per-trial archive.

Additional gaps:

- `ProductionTrialRecord` at `src/memcontam/readiness/phase13_production_observability.py:60` has no parsed-answer field.
- The real production backend constructs NoMem with `checkpoint=None`; this yields `branch=None`, bypasses archive creation, and `build_production_trial_evidence()` rejects baseline `nomem`. Current durable NoMem unit evidence therefore has no 50-row per-trial archive.
- The in-memory archive model serializes, round-trips, and validates the existing registered fixture. This proves current model/fixture conformance, not completeness of the full measured-unit evidence contract.

`tests/test_phase13_pre_authority_f01.py` behaviorally demonstrates the memory-bearing archive create/validate/condense flow and the backend-to-runtime NoMem checkpoint path without choosing a new evidence schema. After authority synchronization, a correction must satisfy the Experiment-owned durable measured-unit evidence join and restart integrity, including NoMem answer/verifier rows. Whether evidence is embedded or sidecar-bound remains unselected here.

## 7. G-01 findings and engineering evidence

Failure path:

- Preflight runs at `src/memcontam/readiness/phase13_main_live_cli.py:77` before `prepare_main_run()` and before dispatch intent.
- Dispatch intent is durably claimed in `src/memcontam/readiness/phase13_main_runner.py:111`; `DispatchTechnicalFailure` is the only dispatch exception translated to terminal missingness at `:119`.
- Provider failures are first represented as failed baseline outcomes in `src/memcontam/experiment/phase12/runtime_registry.py:110`, with terminal provider evidence built in `src/memcontam/readiness/phase13_production_observability.py:201`.
- Typed live/runtime/runner exceptions preserve their own code in `phase13_main_live_cli.main()` lines 144-152.
- Any remaining `OSError`, `ValidationError`, or generic `ValueError`, regardless of phase, is translated at lines 153-154 to `MAIN_LIVE_PREFLIGHT_INVALID`.

`tests/test_phase13_pre_authority_g01.py` provider-freely drives the run-command branch through real `run_pending()`, proves the ledger reached `DISPATCH_INTENT_PERSISTED`, raises a typed `ValueError` subclass from the dispatch backend, and observes the incorrect caller-visible `MAIN_LIVE_PREFLIGHT_INVALID` identity. Later correction points are the live CLI phase boundary, `run_pending()` dispatch boundary, and terminal provider-to-`DispatchTechnicalFailure` adapter. No replacement reason code is selected here.

## 8. Tests, fixtures, and diagnostics added or executed

Added:

- `tests/test_phase13_pre_authority_a02.py`
- `scripts/diagnose_phase13_mr_p5_closure.py`
- `tests/test_phase13_mr_p5_transitive_closure_diagnostic.py`
- `tests/test_phase13_pre_authority_f01.py`
- `tests/test_phase13_pre_authority_g01.py`

Final provider-free evidence:

- Consolidated four-file preparation suite with `PYTHONPATH=src` → 14 passed, 2 strict xfailed.
- `PYTHONPATH=src python -m pytest -q tests/test_task_verifiers.py` → 18 passed.
- Focused MR-P4/MR-P5 package and production-contract checks → 10 passed, 20 deselected.
- `PYTHONPATH=src python -m pytest -q --runxfail tests/test_phase13_pre_authority_a02.py` → expected nonzero: 6 passed, 2 failed exactly at the obsolete production dispatch assertions.
- Ruff over the diagnostic and four tests → all checks passed.
- Repository Python policy checker over the diagnostic and four tests → no violations in 5 files.
- Basedpyright LSP diagnostics → no diagnostics in all five changed Python files.
- Manual CLI QA in a terminal: `--help` rendered the expected options; normal mode reported 36 bound paths, 189 AST-derived paths, and 154 omissions with exit 0; `--require-closed` reported 154 omissions with exit 1.

Environment note: four initial `.venv/bin/pytest ...` invocations failed before collection because this checkout’s `.venv` lacks the pytest executable. The environment’s editable `memcontam` installation points at a sibling checkout, so all final import-bearing verification explicitly uses `PYTHONPATH=src` to exercise this worktree. Preliminary broad commands for `tests/test_phase13_mr_p5_p6.py` and `tests/test_phase13_main_live_cli.py tests/test_phase13_main_production_backend.py` exceeded their timeouts after partial progress; focused provider-free checks are recorded below.

## 9. Expected failing regressions and current defects reproduced

- A-02 strict xfail: alternative valid equation is rejected by production dispatch.
- A-02 strict xfail: bare numeric target is accepted by production dispatch.
- E-01 fail-closed diagnostic: the current AST-derived local-import approximation contains 154 paths outside MR-P5’s Python allowlist.
- F-01 passing characterization: the registered fixture archive is valid in memory, memory-bearing runtime returns condensed dispatch after archive validation, parsed answer is absent from the archive schema, and NoMem skips archive creation.
- G-01 passing characterization: a dispatch failure after durable intent is persisted becomes `MAIN_LIVE_PREFLIGHT_INVALID`.

## 10. Production surfaces deliberately left unchanged

No file under `src/`, `data/`, `configs/`, `runs/`, or the authority mount was changed. In particular, verifier dispatch, MR-P5 models/manifests/hashes/status, observability models/persistence, failure translation/reason codes, prompt/task realization, scientific provenance, retrieval semantics, and provider execution remained unchanged.

## 11. Unresolved genuine blockers

None for completing this engineering-preparation report. Before scientific-decision registration or substantive correction, however, the concurrent authority state is genuinely inconsistent: the router now lists the corrective document, its header remains `CANONICAL_CANDIDATE`, its body claims approval, and the governing task says SD-01 through SD-05 are not approved. Explicit user resolution and coherent authority status/routing are required; this report makes no scientific choice.

## 12. Exact post-authority work deferred

After the router/status/user-instruction inconsistency is explicitly resolved and prospective decisions are coherently approved and registered: synchronize authority; implement the authorized A-02/E-01/F-01/G-01 production corrections together with other approved corrective findings; materialize authorized prompt artifacts; run independent provider-free conformance; reclose affected MR-P4; create a new MR-P5 freeze with a verified dependency realization; obtain new MR-P6 authorization; stop. Corrective Main-A execution requires a separate later command. No step in that sequence was performed here.

## 13. Git/worktree change summary

Final `git status --short` contains exactly six untracked preparation artifacts: this report, one diagnostic script, and four test files. `git diff -- src data configs runs` is empty; no tracked production, scientific-data, configuration, or run artifact changed. The generated `uv.lock` created by an earlier tool invocation was removed. No destructive Git operation, branch/worktree creation, reset, commit, or authority write occurred.

## 14. STOP declaration

STOP before scientific-decision registration, authority synchronization, substantive production corrective implementation, MR-P4/MR-P5/MR-P6 advancement, Readiness-0, or Main-A execution.

## Final validation record

Final consolidated preparation suite: 14 passed, 2 intended strict xfails. Existing verifier suite: 18 passed. Focused existing MR-P4/MR-P5 checks: 10 passed. Ruff and repository Python policy checks: clean. Python LSP diagnostics: clean. Markdown LSP: unavailable because this repository has no `.md` language server; the report was directly inspected. Manual diagnostic QA: help exit 0, normal exit 0 with `36/189/154`, fail-closed exit 1 with 154 omissions. Worktree: exactly the six preparation artifacts summarized in §§8 and 13, with no production-path diff.
