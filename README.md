# memory-contamination-diagnostic

Controlled memory-contamination diagnostic harness for reasoning-memory systems.

## Scope and Claim Boundary

This repository covers implementation contracts, deterministic replay QA, and build-layer
evidence only. It provides no scientific, benchmark, manuscript, causal, production,
paid-provider, Pilot-A, Pilot-B, or main-run evidence.

## Current Status

Baseline-Fidelity-V2 authority and evidence are the sole current status authority, recorded in
the retained `baseline-fidelity-v2.md` and `baseline-fidelity-v2-evidence.md` documents. The
operator runbook is a frozen Phase-12 contract snapshot and not a current BFV2 status source.
Phase-12 contract authority remains the retained `phase12-implementation-contract.md` plus
`logging-v3-phase12.md`; the BGE-M3 cache note remains a supporting authority. This entrypoint
does not repeat their method tables or schemas.

## Filter-v5 Build Status

Filter-v5 implements policy `Filter-Challenge-v1`. The sealed build evidence identifies
implementation commit `3f2f9c4a7ccb8c9e9cff94f1bfc659fd0d75bb46` and evidence-recording
commit `b814b0100f66a19a7111f8f06755e550e8704a52`. All 16 deterministic MFT gates passed
with provider calls `0`. The software interface is ready.
BCT execution is blocked; all 4 BCTs are `not_executed`.

This is build evidence only. It reports no provider, Pilot, benchmark, scientific, causal,
production, or manuscript evidence.

The later documentation-maintenance descendant is not certified by that evidence. It does not
regenerate or extend the sealed evidence.

## Documentation Authorities

- [Baseline-Fidelity-V2 authority](docs/baseline-fidelity-v2.md)
- [Baseline-Fidelity-V2 evidence](docs/baseline-fidelity-v2-evidence.md)
- [BGE-M3 cache setup](docs/bge-m3-cache-setup.md)
- [Phase-12 logging-v3 contract](docs/logging-v3-phase12.md)
- [Phase-12 Filter-v5 build status](docs/phase12-filter-v5-build-status.md)
- [Phase-12 implementation contract](docs/phase12-implementation-contract.md)
- [Phase-12 operator runbook](docs/phase12-operator-runbook.md)

## Verification

```bash
conda run -n memcontam python -m pytest -q \
  tests/test_docs_scope.py::test_documentation_inventory_is_exact \
  tests/test_docs_scope.py::test_readme_is_current_authority_index \
  tests/test_docs_scope.py::test_filter_v5_status_matches_sealed_build_evidence
```
