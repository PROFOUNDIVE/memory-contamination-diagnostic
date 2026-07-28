# Pilot-A Clean Failure Audit

## Freeze note

The completed Game24 Pilot-A at `pilot-a-game24-20260727T195633Z` is frozen as immutable calibration evidence at completed-result commit `9cb1741b2593ccdc3e844cfc9c962692654514be`. All Filter-v5 and Pilot-B work is downstream build/calibration; it cannot retroactively alter, reparse, relabel, or pool with Pilot-A.

Permitted claim only: **The registered Game24 Pilot-A infrastructure and operational-evidence-filter-v4 contract passed.** No contamination-effect, accuracy-degradation, semantic Filter-effectiveness, causal/significance, or baseline-ordering claim is supported.

## Audit population

- Clean trials: **50** (40 prefix, 10 suffix, including 2 NoMem suffix).
- Every non-null verifier outcome recomputed: **50**.
- Stored/recomputed correctness disagreements: **0**.
- Logged answer-call references that point to a trailing native stage rather than the parsed-answer source: **23**. Classification rows preserve both references; this audit does not modify production logging.

## Failure classes

| Primary category | Count |
|---|---:|
| `verified_correct` | 4 |
| `format_only_invalid` | 7 |
| `semantic_incorrect` | 30 |
| `unsupported_or_no_solution` | 9 |
| `baseline_stage_malformed` | 0 |
| `verifier_or_parser_contract_failure` | 0 |
| `not_classifiable` | 0 |

The low registered clean accuracy is primarily semantic/nonresponsive, not formatting-only: 30 semantic failures plus 9 unsupported/nonresponsive failures versus 7 independently witnessed format-only failures.

## Clean-prefix accuracy

| Baseline | Correct | Total | Accuracy |
|---|---:|---:|---:|
| `fh_bounded` | 2 | 10 | 0.200 |
| `rag_frozen` | 0 | 10 | 0.000 |
| `bot_style` | 0 | 10 | 0.000 |
| `reflexion_style` | 1 | 10 | 0.100 |
| `nomem` | 0 | 0 | n/a |

## Clean-suffix accuracy

| Baseline | Correct | Total | Accuracy |
|---|---:|---:|---:|
| `fh_bounded` | 1 | 2 | 0.500 |
| `rag_frozen` | 0 | 2 | 0.000 |
| `bot_style` | 0 | 2 | 0.000 |
| `reflexion_style` | 0 | 2 | 0.000 |
| `nomem` | 0 | 2 | 0.000 |

## Strict clean-solvability

Only one intended suffix-probe control is strictly clean-solvable under the unchanged contract:

- `seed:1:memory_branch:fh_bounded:clean:game24_pilot_004`

Format-only sensitivity adds two suffix cells, which remain ineligible under the primary rule:

- `seed:1:memory_branch:rag_frozen:clean:game24_pilot_004`
- `seed:1:memory_branch:reflexion_style:clean:game24_pilot_004`

Across all observed clean execution positions, verifier-accepted task/baseline cells are FH-bounded on `game24_pilot_001` and `game24_pilot_004`, plus Reflexion-style on `game24_pilot_001`. For paired Filter-v5 suffix probes, only the seed-1 FH-bounded `game24_pilot_004` cell is primary-eligible.

## Decision memo

1. **Primary cause:** semantic/nonresponsive failure dominates; formatting-only recovery is a minority sensitivity.
2. **Concentration:** BoT-style has no accepted clean trial and contributes most unsupported answers; RAG-Frozen has no accepted trial and three equality-format witnesses; Reflexion-style has one accepted trial and four equality-format witnesses; FH-bounded supplies all other accepted controls. No malformed native JSON stage is evidenced.
3. **Pool sufficiency:** one strict suffix control is insufficient for a defensible paired Filter-v5 probe pool.
4. **Strict cells:** the exact accepted cells are enumerated above and machine-readably in `clean_solvability_inventory.json`.
5. **Pilot-B prerequisite:** use a new disjoint calibration probe pool, frozen before Pilot-B outcomes, rather than selecting Pilot-A winners.
6. **Parser/canonicalizer prerequisite:** a narrowly versioned prospective normalization of terminal `expression = 24` to `expression` would be legitimate only before new calibration with clean controls rerun. It is not needed to preserve the strict current contract and must not retroactively change Pilot-A. The answer-call logging reference should also be corrected prospectively before relying on answer-call provenance.
7. **Legitimate corrections vs tuning:** prospective versioned parser/canonicalizer and answer-call provenance fixes, disjoint probe construction, and preregistered reruns are build/calibration corrections. Retroactively changing Pilot-A outcomes, counting manual recoveries as primary controls, selecting only observed winners, or changing seeds/tasks/eligibility after seeing outcomes is post-hoc tuning.

## Remaining uncertainties

- Pilot-A has only two suffix tasks (one selected checkpoint task per seed), so task-general clean solvability is not estimable from suffix evidence.
- The 23 answer-call reference mismatches require reconstruction from same-trial native stage order; raw calls are present, but the logged pointer itself is not semantically accurate.
- Format recovery is limited to exact equality wrappers with an independently verified left-hand expression; no prose-based or manually corrected answer is recovered.

## Evidence

- Freeze manifest: `.sisyphus/evidence/pilot-a-clean-audit/pilot_a_frozen_evidence_manifest.json`
- Trial classifications: `.sisyphus/evidence/pilot-a-clean-audit/clean_trial_classification.jsonl`
- Failure summary: `.sisyphus/evidence/pilot-a-clean-audit/clean_failure_summary.json`
- Solvability inventory: `.sisyphus/evidence/pilot-a-clean-audit/clean_solvability_inventory.json`
- Canonical run: `runs/runs/pilot-a-game24-20260727T195633Z`

Generated file SHA-256 values are reported after byte-finalization in the operator final report; all immutable source hashes are embedded in the machine-readable artifacts.
