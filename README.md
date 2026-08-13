# memory-contamination-diagnostic

Controlled memory-contamination diagnostic harness for reasoning-memory systems.

## Scope and Claim Boundary

This repository covers implementation contracts, deterministic replay QA, and build-layer
evidence only. It provides no scientific, benchmark, manuscript, causal, production,
paid-provider, Pilot-A, Pilot-B, or main-run evidence.

The no-paid-provider statement applies to the historical and canonical evidence indexed here.
Separately labeled, redacted `local_rootless_non_authoritative` rehearsal receipts may record an
optional paid local call count, but they are excluded from scientific and authoritative evidence.

## Study Scope and Supported Research Question

This repository is a bounded diagnostic harness for studying how controlled erroneous external
memory affects verifier-based continual or multi-step reasoning.

The confirmatory study is memory-mechanism-isolated and text-only. Every condition receives
equal answer-generation tool availability, uses `tool.mode = text_only`, and permits no code
execution. Tool-augmented work is separately versioned exploratory work and is not pooled with
confirmatory evidence.

The registered secondary research question is:

> How do failure recurrence, stored persistence, and applicable propagation vary across the prespecified memory mechanisms for the fixed canonical contamination candidate in each task?

Filter-v5 is a bounded mitigation baseline. It may reduce exposure to contract-invalid injected
roots and their recorded descendants, but it is not semantic-truth detection, universal
contamination removal, or a deployable security defense.

This framing does not expand the repository's build-only evidence boundary or report scientific
execution.

### Planned Confirmatory Matrix and Timeline

The matrix and timeline below describe planned study design, not executed or completed evidence.

| Axis | Planned scope |
|---|---|
| Tasks | `Game24`, `Math Equation Balancer`, and `WordSorting` |
| Memory-bearing baseline conditions | `context-bounded Full History`, `RAG-Frozen`, `BoT-style proxy`, and `Reflexion-style proxy` |
| No-memory condition | `NoMem` is a Clean-only memory-free singleton, not a five-arm memory-bearing condition |
| Planned Main-A arms | `Clean`, `Correct` (auxiliary), `Irrelevant` (auxiliary), `Contam`, and `Filter` |
| Tool and model controls | Equal tool availability with `tool.mode = text_only`, no code execution, and one primary model snapshot fixed before Pilot-A and Main |

The conservative 3-week route is the default. A 5-week extension is not selected and remains
contingent on readiness, budget, and reserved evidence.

The planned research window is an eight-week Fast Track ending no later than August 31, 2026.

## Current Status

Phase-13 authority sync and calibration-v2 deterministic readiness is complete at terminal
`DETERMINISTIC_AUTHORITY_SYNC_COMPLETE`. Live calibration remains externally blocked at
`CALIBRATION_V2_EXTERNAL_BLOCK`: provider calls are `0`, no calibration archive or claim exists,
and `MAIN_A_EXECUTION_FORBIDDEN` is unconditional. Authenticated structural checkpoint authority
is absent for 34 of 36 required streams, and the runtime emits 160 baseline/arm events per source
while the archive contract requires one completed 10-event source.

This is build/readiness evidence, not calibration or scientific evidence. It establishes no
provider, Pilot-A, Pilot-B, Main, benchmark, causal, production, or manuscript result. Synthetic
and test artifacts are not calibration evidence. Baseline-Fidelity-V2 remains the current scientific-method authority.
The Phase-12 contracts and operator runbook retain their existing scopes; the runbook remains a
frozen Phase-12 contract snapshot and not a current BFV2 status source. User-issued capacities and
credential/cache availability were verified privately, but cannot overcome the authority and
archive-contract failures. An actual run requires separately authorized authority and
archive-contract revision, not ad hoc live dispatch.

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
- [Phase-13 authority sync and calibration-v2 readiness](docs/phase13-authority-sync-calibration-v2.md)

## Verification

```bash
conda run -n memcontam python -m pytest -q \
  tests/test_docs_scope.py::test_documentation_inventory_is_exact \
  tests/test_docs_scope.py::test_readme_is_current_authority_index \
  tests/test_docs_scope.py::test_filter_v5_status_matches_sealed_build_evidence
```
