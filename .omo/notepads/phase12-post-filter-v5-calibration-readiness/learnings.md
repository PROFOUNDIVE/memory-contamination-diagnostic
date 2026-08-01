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

## 2026-08-01 Final F3 real manual QA
- Fresh interactive CLI QA with the repository `src` path explicitly bound validated zero-call config/preflight and screening cost preview, then exercised absent, copied, expired, and digest-swapped authorization paths. Every path blocked before factory construction and reported zero provider calls; the expired descriptor reaches the shared runtime expiry guard and reports `AUTHORIZATION_RUNTIME_MISMATCH` under the common invalid-calibration terminal.
- Fresh readiness fixture CLI runs preserved all six plan terminals: `AWAITING_SCREENING_AUTHORIZATION`, invalid calibration, `FILTER_V5_PILOT_B_NOT_ESTIMABLE`, `AWAITING_BCT_AUTHORIZATION`, invalid BCT evidence, and ready-for-separate-authorization despite a behavioral false negative.
- Fresh ledger/archive QA recorded duplicate-run rejection, two concurrent reservations, unresolved crash reservation retention, timeout invalidation retention, public/audit reconciliation, hidden-label rejection, byte-range tamper rejection, copied-archive symlink rejection, and descriptor no-follow rejection. Focused QA passed `42`; fresh evidence verification returned `APPROVE`.

## 2026-08-01 Final F4 scope fidelity
- The exact external `find -P | sort -z | xargs sha256sum` snapshot matched Task 1 by `cmp`; all no-follow [A3]-[A8] hashes and the sealed v1 SHA-256 stream also matched.
- The 38-commit linear series from `32efbe15` through `61381a5` changed 92 approved paths only. All nine reports retain provider calls `0`, the live/replay roots are absent, and the code prespec remains inactive and unpriced.

## 2026-08-01 Final F5 evidence reconciliation
- Exact baseline command: `env -u OPENAI_API_KEY PYTHONPATH=/home/hyunwoo/git/memory-contamination-diagnostic-filter-v5/src python scripts/verify_phase12_filter_v5_bct_evidence.py --through readiness --bundle docs/evidence/phase12-filter-v5-bct-v1 --plan .omo/plans/phase12-post-filter-v5-calibration-readiness.md --artifact-root runs/phase12-filter-v5-bct-live-v1` returned `APPROVE`, but that positive result is not sufficient for F5.
- Independent byte rehashes confirmed approved plan `e8d44600fb3a9177ae691fd8f49ac1c06305b004db7ccd50d391c9876356a230`, descriptor `7b878988972b5bc3c1a2ba24785b978cc26b973e1e44e8059ff8d3133227842e`, config `76c710a8fcb6b77b8c759dacffa4610ae0335e05f1729d6c8bb58fab27213548`, Freeze-A `0ebd95194bf81c8fbf02416669a1787965cefbf017123253cb5d6022d17b6f20`, source universe `6120bb46e40232d41e315b24e1771c23e1223440f2956ba532f75edb3bb0a9b6`, and all nine report bytes/seals; the source-universe member hashes also matched live bytes.
- Isolated mutations rejected with concrete codes: source-universe `EVIDENCE_SOURCE_UNIVERSE_INVALID`; stale seal, resealed provider counter, stage disposition, and `all_passed=true` `EVIDENCE_REPORT_CONTRACT_INVALID`; report symlink `EVIDENCE_REPORT_INVALID`; plan/descriptor symlinks `PLAN_READ_INVALID`/`PLAN_DESCRIPTOR_INVALID`; authority manifest `EVIDENCE_READINESS_INVALID`.
- Isolated raw-byte mutations of `configs/phase12/filter_v5_bct_calibration.yaml`, `data/phase12/filter_v5_bct_v1/freeze_a.json`, and the source-universe member `data/tasks/game24_pilot.jsonl` each returned `APPROVE`. Exact regression command `env -u OPENAI_API_KEY TMPDIR=$ATTEMPT_DIR/final-f5-scratch/tmp PYTHONPATH=/home/hyunwoo/git/memory-contamination-diagnostic-filter-v5/src python -m pytest tests/test_phase12_filter_v5_evidence_security.py tests/test_phase12_filter_v5_bct_archive.py tests/test_phase12_filter_v5_bct_waiting.py tests/test_phase12_filter_v5_pilot_b_readiness.py -q` returned `22 passed in 4.99s`; scratch was removed. VERDICT: REJECT.

## 2026-08-01 Final F5 repair
- Root cause: readiness reopened report bytes and stage files but never verified report-bound Freeze-A/request values, the current calibration/Methods/authority inputs, or the `source_files` hashes inside the source universe. A resealed terminal raised an unhandled `ValueError` instead of returning its error code.
- TDD RED: `env -u OPENAI_API_KEY PYTHONPATH=/home/hyunwoo/git/memory-contamination-diagnostic-filter-v5/src python -m pytest tests/test_phase12_filter_v5_evidence_security.py::test_readiness_verifier_rejects_current_frozen_input_mutation tests/test_phase12_filter_v5_evidence_security.py::test_readiness_verifier_rejects_source_universe_member_mutation tests/test_phase12_filter_v5_evidence_security.py::test_readiness_verifier_reports_resealed_terminal_tampering_with_a_code -q` returned `4 failed`: config, Freeze-A, and raw Game24 mutations were admitted; terminal rejection emitted no stdout code.
- GREEN: `bct_archive_evidence_inputs.py` descriptor-opens/re-hashes frozen authority, Methods, config, Freeze-A/request, Freeze-A manifests, source universe, and every source-universe member; the CLI translates `ValueError` to a concrete code. The exact same command returned `4 passed in 1.74s`; fresh local-scratch subprocess coverage returned `8 passed in 2.62s`, and the expanded evidence/readiness slice returned `26 passed in 6.26s`.
- Final exact readiness command returned `APPROVE`; targeted `MYPYPATH=src python -m mypy ...` reported no issues in four files, Ruff passed, changed-file LSP diagnostics were clear, `git diff --check` passed, all nine report counters remain zero, and live/replay/scratch roots remain absent. VERDICT: APPROVE.
