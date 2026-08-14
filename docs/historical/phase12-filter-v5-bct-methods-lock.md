# Phase 12 Filter-v5 BCT Methods Lock

This repository lock binds approved plan SHA-256 `e8d44600fb3a9177ae691fd8f49ac1c06305b004db7ccd50d391c9876356a230`.

## Scope and Strata

Tasks are `game24`, `math_equation_balancer`, and `word_sorting`; baselines are `full_history`, `rag_frozen`, `bot_style`, and `reflexion_style`. NoMem is excluded. The twelve task-baseline strata are fixed before screening.

## Search and Construction Laws

The sixteen `SC-{suite_kappa}-{coverage}-{repeatability}-{retry}` IDs are fixed in `filter_v5_bct_calibration.yaml`. Game24 uses exact Fraction certificates, MEB uses standard-precedence certificates over all sixteen ordered operator pairs, and WordSorting uses first-difference/final-character-disagreement certificates. Pilot-A, candidate examples, future Main/reserved-extension, exact, canonical, and near-duplicate inputs are excluded.

Primary parsing is raw-only; canonicalization is sensitivity-only and unpooled. No pooling across challenge, Main, or code. `not_contradicted -> active`; it is never a semantic-safety result.

## Capacity and Stage Law

Screening reserves `18 x (1+1+2+1) = 90` calls. BCT reserves `3 x 2 x 4 x 2 x 2 x (1+1+2+1) = 480` calls. The shared maximum is 570 calls, USD 10, and 10,800 seconds. The persistent append-only ledger retains unresolved reservations. Native stages are issued or not-issued only; BoT emits `bot_problem_distill` then `bot_instantiate_solve` only when distillation parses.

## Terminal and Evidence Matrix

The terminal table is the six values in the machine lock. The nine report IDs and schema/binding rows are the `report_matrix` in the machine lock; no report is produced by this Task 1 lock.
