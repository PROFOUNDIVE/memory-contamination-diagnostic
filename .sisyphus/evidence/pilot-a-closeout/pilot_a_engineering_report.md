# Game24 Pilot-A Engineering Report

## Final status

`PILOT_A_BLOCKED_BY_IMPLEMENTATION`

## Infrastructure conclusions

- Final-head readiness passed at `5ea9a78949f54c20a63e420fc4fd593a7ea02f58` before the linked rerun.
- The preserved first attempt, `pilot-a-game24-20260727T061248Z`, completed 112 calls for two eligible seeds and validated its archive hashes, counts, references, retries, and known cost of USD 0.1243275.
- Post-run inspection invalidated that attempt scientifically: 16 Reflexion failures came from a prompt/parser schema mismatch, and scientific call records omitted latency.
- The repair exposed all four parser fields, retained per-call latency, and added explicit parent-run provenance. The required focused suite passed 100 tests; Ruff passed; changed production modules passed mypy.
- The linked rerun, `pilot-a-game24-20260727T062808Z`, named the first attempt as its parent but stopped with `JOINT_CHECKPOINT_ELIGIBILITY_EMPTY`.
- The rerun directory is empty. Provider work completed before the stop, but rows, call count, cost, and seed-level rejection evidence were not persisted. This violates the partial-archive preservation contract and prevents archive, verifier, join, and cost reconciliation for the linked attempt.
- No further provider call is authorized after the explicit empty-joint-eligibility stop condition.

## Descriptive first-attempt observations

- The invalidated first attempt materialized 26 suffix cells: 24 memory-bearing baseline/arm cells and two NoMem singleton cells.
- Recalculation agreed with all 46 non-null stored verifier outcomes.
- Contam and Filter shared the same injected root and source identity in all eight baseline/seed pairs.
- The candidate appeared in two retrieval records and six final-context records; all eight Filter decisions quarantined it, with zero Filter retrieval or final-context inclusion.
- Four BoT malformed outputs remain descriptive model behavior: two invalid solve results and two invalid thought distillations.

## Claim boundary

These records support only engineering statements about readiness, execution topology, archive integrity, observed malformed outputs, and the blocking preservation defect. They do not support causal contamination claims, Filter effectiveness, comparative robustness, generality, significance, effect sizes, threshold optimality, or manuscript-ready conclusions.
