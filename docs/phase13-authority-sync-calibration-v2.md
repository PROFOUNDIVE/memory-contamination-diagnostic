# Phase-13 Authority Sync and Calibration-v2 Readiness

## Status

The deterministic authority build is ready and emits
`DETERMINISTIC_AUTHORITY_SYNC_COMPLETE`. The conditional live-calibration branch did not run.
Its sealed terminal is `CALIBRATION_V2_EXTERNAL_BLOCK`, provider calls are `0`, and no run root,
archive, or claim exists. Every branch ends at `MAIN_A_EXECUTION_FORBIDDEN`.

This status is build and deterministic-readiness evidence only. It is not provider, Pilot-A,
Pilot-B, Main, calibration, benchmark, scientific, causal, production, or manuscript evidence.
Synthetic/test artifacts are diagnostics, not calibration evidence.

## External Block

The no-egress review found two independent contract blockers:

- `authenticated_structural_checkpoint_authority_incomplete`: the authenticated, code-pinned
  structural registry contains only `game24` seeds 10000 and 10001. The approved 36-stream panel
  therefore lacks authority for 34 required streams.
- `runtime_archive_cardinality_contract_incompatible`: the current runtime emits 160 baseline/arm
  events per source, while the Phase-13 archive validator requires exactly one completed source
  containing 10 events.

Credentials, private operator paths, request/authorization artifacts, and cache locations are
not part of the tracked report. User-issued capacities and credential/cache availability were
verified privately, but cannot overcome either authority/contract failure. Neither blocker is a
scientific observation or result. A real run requires separately authorized structural-authority
and archive-contract revision; an ad hoc live dispatch is not a valid path.

## Public Evidence

The versioned report
[`phase13-authority-sync-calibration-v2-v1.json`](evidence/phase13-authority-sync-calibration-v2-v1.json)
binds the canonical freeze, registries, historical compatibility ledger, relevant implementation
and tests, exact CLI terminal observations, and the protected dirty-root equality proof. Its
verifier independently recomputes every declared tracked hash and rejects terminal relabeling,
report tampering, stale blockers, private or untracked paths, protected-state mismatch, nonzero
provider/archive/claim state, and Main-A artifacts.

```bash
PYTHONPATH=src python -m pytest tests/test_phase13_evidence_report.py -q
PYTHONPATH=src python -m memcontam.cli phase13 validate-calibration-v2 \
  --config configs/phase13/pre_main_calibration_v2.yaml
PYTHONPATH=src python -m memcontam.cli phase13 run-calibration-v2 \
  --config configs/phase13/pre_main_calibration_v2.yaml
```
