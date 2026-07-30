# Phase-12 Filter-v5 Build Status

## Scope and Authority

This page records the sealed Filter-v5 build state for policy `Filter-Challenge-v1`, schema
`filter_challenge_domain_v1`, and evidence layer `build`. The evidence is synthetic,
deterministic build evidence. It is additive and separate from
`operational-evidence-filter-v4`.

`operational-evidence-filter-v4`, with interpretation
`contract_invalid_direct_write_containment`, remains active for Pilot-A/current runtime. Its
boundary is contract-invalid direct-write containment. It is
not semantic truth detection or general contamination mitigation. This active operational policy
is separate from additive Filter-Challenge-v1 build evidence. Filter-Challenge-v1 adds build
evidence only; neither it nor `operational-evidence-filter-v4` produced Pilot results.

The sealed JSON under `.sisyphus/evidence/phase12-filter-v5-build-v1/` remains authoritative.
This page summarizes that evidence without changing its claim boundary.

## Implemented Build Surface

- Strict policy-visible and audit-only models.
- Finite `SearchConfig`, `SelectedPolicy`, inventory, and suite registries.
- Exact answer-call provenance.
- Native provisional FH-bounded, RAG-Frozen, BoT-style, and Reflexion-style adapters.
- Isolated read-only control and challenge execution with updater suppression.
- Strict eligibility, witness, tri-state, and fail-open routing.
- Separate build archives and audit records.
- Sixteen deterministic MFT gates and four guarded BCT interfaces.
- Offline Filter-v5 validation, archive, cost-preview, readiness, and evidence tooling.

Candidate exposure occurs only when the candidate entry ID is a member of the final source set
of the exact challenge answer call. Storage, retrieval, similarity, stage ordering, generic
prompt presence, and reconstructed history are insufficient.

## Verified Build Evidence

| Record | Sealed value |
|---|---|
| Base commit | `8bd8f4e85ebca919319770e319f0574af32fd458` |
| Implementation commit | `3f2f9c4a7ccb8c9e9cff94f1bfc659fd0d75bb46` |
| Evidence-recording commit | `b814b0100f66a19a7111f8f06755e550e8704a52` |
| Plan SHA-256 | `65f4c45b5db702af0f60a5296d116bc1ed64ac7440b447c676b069a8e204c12b` |
| Validation SHA-256 | `b99ebc37110c84f66dd114e4655216b1af4d23b37f60f04bea1daad384317cd2` |
| Manifest SHA-256 | `dd964902513ddcfebe10f482191310f4e57e931eb66adebfb3343def21e07571` |
| Evidence directory | `.sisyphus/evidence/phase12-filter-v5-build-v1/` |
| Deterministic MFT gates | All 16 passed |
| Provider calls | `0` |
| `scientific_result` | `false` |

All 16 deterministic MFT gates passed. Provider calls: `0`.

The 16 sealed MFT gate IDs are:

1. `MFT-FV5-01-PAIR-MATCH`
2. `MFT-FV5-02-EXPOSURE-REQUIRED`
3. `MFT-FV5-03-TRISTATE`
4. `MFT-FV5-04-FAIL-OPEN`
5. `MFT-FV5-05-ROUTE-INVARIANCE`
6. `MFT-FV5-06-SCRIPTED-CORRECT`
7. `MFT-FV5-07-SCRIPTED-IRRELEVANT`
8. `MFT-FV5-08-NO-WRITEBACK`
9. `MFT-FV5-09-CONTAM-SHADOW-SHARE`
10. `MFT-FV5-10-PARSER-BOUNDARY`
11. `MFT-FV5-11-CONTROL-CACHE`
12. `MFT-FV5-12-PROBE-KEY-INVARIANCE`
13. `MFT-FV5-13-ANSWER-CALL-PROVENANCE`
14. `MFT-FV5-14-ACTIVATION-DOMAIN`
15. `MFT-FV5-15-ELIGIBILITY-STATES`
16. `MFT-FV5-16-COVERAGE-NOT-ESTIMABLE`

`FILTER_V5_BUILD_AND_MFT_COMPLETE` is a build terminal status only. It does not report
behavioral execution or scientific results.

## Immutable Commit Boundary

Evidence at `b814b01` certifies implementation `3f2f9c4`. The later
docs-and-docs-contract-tests descendant does not alter or extend that evidence.
It does not certify the descendant HEAD.

## Current Readiness Boundary

| Readiness field | Current value |
|---|---|
| Build status | `FILTER_V5_BUILD_AND_MFT_COMPLETE` |
| Software interface | `ready` |
| BCT execution | `blocked` |
| Behavioral calls | `false` |
| Provider calls | `0` |
| `BCT-FV5-01-CERTIFIED-FALSE` | `not_executed` |
| `BCT-FV5-02-CORRECT` | `not_executed` |
| `BCT-FV5-03-IRRELEVANT` | `not_executed` |
| `BCT-FV5-04-ORDINARY-FALSE` | `not_executed` |
| Canonical patch | `pending_before_provider_backed_pilot_b` |
| Inventory | `pending_freeze` |
| Provider authorization | `absent` |

The ordered blockers are:

1. `SEARCH_CONFIG_PENDING_FREEZE`
2. `SCIENTIFIC_INVENTORY_PENDING_FREEZE`
3. `CANONICAL_PATCHES_PENDING`
4. `PROVIDER_CONFIG_DISABLED`
5. `PROVIDER_AUTHORIZATION_ABSENT`

`READY_FOR_AUTHORIZED_FILTER_V5_BEHAVIORAL_CAPABILITY_RUN` means only that the software
interface is available. It does not mean freeze, authorization, execution, Pilot-B evidence,
or Main evidence exists.

## Remaining Scientific Choices

The following choices remain unresolved. This page assigns no defaults.

- `canonicalizer`
- `ci_procedure`
- `constraint_order`
- `coverage_contract`
- `decision_rule`
- `evaluability_rate`
- `inclusion_rate`
- `inventory`
- `kappa`
- `latency_cap`
- `monetary_cost_cap`
- `operational_suite`
- `ordinary_route_rate`
- `price_registry`
- `probe_count`
- `provider_authorization`
- `replicate_count`
- `retry_count`
- `tie_break`
- `tolerance`

## Non-Claims

This page reports no scientific results, benchmark evidence, or manuscript evidence. It does
not establish causal effects, semantic truth detection, general contamination mitigation, or
production readiness. It records no paid-provider execution, Pilot-A evidence, Pilot-B
evidence, or Main evidence. It does not claim provider authorization, canonical-patch
completion, or complete upstream reproduction.

## Verification

The checks below are read-only unit tests. They do not regenerate evidence.

```bash
conda run -n memcontam python -m pytest -q \
  tests/test_docs_scope.py::test_documentation_inventory_is_exact \
  tests/test_docs_scope.py::test_readme_is_current_authority_index \
  tests/test_docs_scope.py::test_filter_v5_status_matches_sealed_build_evidence
```
