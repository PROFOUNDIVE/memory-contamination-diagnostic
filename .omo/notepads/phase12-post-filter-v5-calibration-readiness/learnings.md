## 2026-07-31 Task 1
- The external authority mount reports mode `0o664` through descriptor `fstat`; the committed identity value is decimal `436`, not the hexadecimal `81b4` shown by `stat %f`.

## 2026-07-31 Task 2
- The live seam can reuse the existing paired executor contracts without reclassifying fixture-only registries; the added zero-call branches terminate before provider construction.

## 2026-07-31 Task 2 ledger/archive follow-up
- A durable stage reservation must account for calls, both token classes, microusd, and wall seconds together; an unresolved row remains the conservative accounting source after a restart, while settlement replaces only that reservation with observed usage.
- Archive range checks must be computed from canonical UTF-8 bytes and revalidated independently of a recomputed record hash or advisory `valid` field.

## 2026-07-31 Task 3
- Freeze-A has 72 control trials and 90 native-stage method-call capacities because BoT contributes two stages per probe.
- A repeatability byte comparison is not a mutation test: copied artifacts must be altered and fed through the same Freeze-A validator used by the preview path.

## 2026-07-31 Plan digest stabilization
- The approved plan digest must canonicalize only the `- [ ]` or `- [x]` state on numbered Task 1–6 and F1–F5 checklist lines. The resulting bytes retain all substantive plan content and match the independent `e8d44600fb3a9177ae691fd8f49ac1c06305b004db7ccd50d391c9876356a230` descriptor.

## 2026-07-31 Task 4 waiting terminal
- Report 5 is branch-conditional: an authoritative `AWAITING_SCREENING_AUTHORIZATION` screening result binds reports 1–4 and explicitly records `null` for Freeze-B, SearchConfig, and BCT authorization-request digests. It is not evidence of `AWAITING_BCT_AUTHORIZATION`.

## 2026-07-31 Task 5 waiting branch
- The first reachable raw terminal remains authoritative across downstream stages: a BCT stage result may have `stage=bct`, but it preserves Task 4's `AWAITING_SCREENING_AUTHORIZATION` terminal and records no BCT pair, side, archive, ledger, or provider attempt.

## 2026-07-31 Task 5 LSP follow-up
- Pytest resolved `memcontam.experiment` as an implicit namespace package, while the running Pyright server did not reliably index a newly added descendant through that boundary. A regular package marker plus explicit `src` execution-environment paths makes fresh Pyright and restarted LSP resolution agree.

## 2026-07-31 Task 6
- The first reachable screening terminal remains authoritative: readiness can freeze the inactive code prespec and bind the nine-report handoff without provider, Pilot-B, Main, or code execution.

## 2026-07-31 Task 6 LSP follow-up
- Pyright may retain direct-child import failures after new modules are committed; an explicit workspace include set plus a regular package export boundary makes a fresh language-server index resolve all current descendants.
- The existing regular package boundary is sufficient; do not alter the working direct imports merely to satisfy a stale server cache.

## 2026-07-31 Task 6 prescribed test path
- A plan-mandated pytest node is part of the contract: relocate the one authoritative assertion to that path instead of re-exporting a test module through the non-package `tests` directory.

## 2026-07-31 Final F3 real QA
- Independent zero-call CLI QA observed the authoritative early terminal `AWAITING_SCREENING_AUTHORIZATION`; expired and digest-swapped synthetic authorization descriptors instead produced the invalid-calibration terminal with zero provider calls.
- Concurrent unresolved screening reservations stayed charged in the JSONL ledger and blocked a BCT reservation; timeout invalidation retained its reservation. Fresh readiness verification approved the untouched bundle and rejected a copied bundle with a recomputed stale prior-report hash.

## 2026-07-31 Final-wave continuation
- Descriptor-safe report reads must be applied to every report and stage path, not just plan/authorization descriptors; otherwise symlink substitution changes verifier control flow.

## 2026-07-31 Facade split
- A facade that re-exports names is insufficient for monkeypatch-based private seams because implementation functions retain their implementation-module globals; registering the implementation module under the stable facade name preserves those test seams while keeping the facade below the source-size limit.

## 2026-07-31 Full-suite timeout diagnosis
- Collection completed with `485 tests collected in 1.97s`. A direct import probe printed `True` for both `bct_live is bct_live_impl` and the corresponding `sys.modules` identity, ruling out facade recursion. The expensive final-verifier nodes are finite: code-quality `19.98s`, integration `40.30s`, scope `33.16s`, and terminal `64.28s`; a 27-node plan-compliance slice took `143.16s`.

## 2026-07-31 Final-verifier fixture memoization
- Test-local command records are immutable normalized values. Freezing synthetic git timestamps makes identical fixture inputs yield the same implementation commit; caching by that commit plus a SHA-256 over every fixture relative path and byte content avoids repeating the seven guarded setup commands while preserving distinct repositories and mutable evidence roots. The full modes module completed `42 passed in 142.90s`.

## 2026-08-01 F1 timing investigation hypotheses
- H1 fixture Git/evidence churn: `_fixture()` still creates independent Git repositories, commits, evidence bundles, and source repositories after command-record cache hits. Distinguishing observation: per-node durations remain high for fixture callers other than integration, while a controlled immutable-template toggle reduces the same nodes and preserves byte-equivalent independent repositories.
- H2 guarded subprocess execution: integration verification runs seven positive and three mutation guarded subprocesses. Distinguishing observation: integration nodes dominate `--durations=50` after fixture setup work is removed; cached command records affect setup only, not `verify_integration()` reruns.
- H3 imports/package effects: collection was `485 tests collected in 1.90s`, and the facade toggle measured `17.90s` at `1a97067` versus `19.98s` at current HEAD for the code-quality node. Distinguishing observation: collection or `-X importtime` would need to account for material runtime; those observations presently refute import/facade effects as the 600-second cause.
- H4 CPU/environment jitter: a previous exact run passed in `593.56s`, while fresh attempts exceeded 600 seconds. Distinguishing observation: repeated same-HEAD exact/file runs retain comparable node ordering but show wall-time variance only after measured deterministic hot work is insufficient to explain the gap.
- H5 replay cache side effect: the rejected F1 artifact recorded untracked `data/embedding_cache/phase12-filter-v5-bct-replay-gate/` after replay, but the exact pytest selector itself does not need that cache. Distinguishing observation: the exact selector leaves the worktree clean; replay alone recreates the named cache and cleanup is limited to that directory.

## 2026-08-01 F1 fixture-template calibration
- Root cause: repeated deterministic final-verifier fixture construction was the removable hot work. The exact selector measured `485 passed in 659.84s` before this calibration; the final fresh selector measured `487 passed in 515.97s` (`WALL_SECONDS=538.42`) after it.
- A controlled probe measured full fixture construction at `10.50s` and byte-equivalent independent `git clone --local --no-hardlinks` materialization at `0.01s`. The template key hashes every fixture-relative path and byte plus the fixed marker/plan bytes; clones receive independent repository, evidence, source, plan, summary, and untracked-source paths.
- Terminal approvals are cached only as immutable bytes after a cache key binds base/implementation/source commits, plan, summary, and every non-Git repository/source byte. Each caller writes fresh approval files under its own `tmp_path`.
- Commit: `a26a7d3 test(filter-v5): cache verifier fixture templates`; changed files: `tests/test_phase12_filter_v5_final_verifier_modes.py`, `tests/test_phase12_filter_v5_terminal_semantics.py`.

## 2026-08-01 BCT facade typing boundary
- A `TYPE_CHECKING` import plus local `__all__` exposes the two CLI exports to static analysis without replacing the runtime `sys.modules` alias, preserving implementation-module identity and monkeypatch-sensitive private seams.

## 2026-08-01 F2 remediation verification
- The replay runner binds `RunMetadata.config_hash` to canonical JSON of the fully resolved configuration, including the normalized provider profile, rather than to raw YAML bytes. The replay gate now independently resolves `--expected-config`, compares that digest to both `run.json` and `resolved_config.json`, and rejects a copied/wrong configuration.
- Current pure-LOC measurements for the archive/live split are 32, 49, 41, 174, 60, 122, 188, 23, 199, 102, 38, and 56 respectively; every split module is below the 250-line ceiling.
- Fresh selectors completed under the 600-second limit: `501 passed in 452.47s` for `tests/test_phase12_filter_v5_*.py` and `768 passed in 474.77s` for the required broad selector.

## 2026-08-01 F2-001 archive split
- The archive facade now explicitly re-exports contract values from focused models, durable JSONL storage, resource-ledger, live-record, and evidence-report modules; canonical JSON/hash, locking, fsync, and descriptor-safe reads each retain one owner.
- The active Pyright server retained stale unresolved-new-module diagnostics after the split, including imports from a reverted relative-import probe. Fresh targeted mypy, runtime facade import, and 47 focused archive/ledger/evidence/waiting tests resolve the new modules; no configuration workaround was added.

## 2026-08-01 Final F2 verdict refresh
- Fresh F2 review reran all six prior blocker surfaces: every archive/live split module remained under 250 pure LOC, facade and monkeypatch seams had clean LSP/mypy coverage, authorization and replay mutation tests passed, immutable cache isolation was included in the exact selector, and both required selectors completed (`501 passed in 450.28s`; `768 passed in 598.51s`).
