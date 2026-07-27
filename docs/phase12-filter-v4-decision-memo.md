# Phase 12 Filter V4 Decision Memo

## Scope

Version: `operational-evidence-filter-v4`.

The Filter arm is **contract-invalid direct-write containment**. It decides whether a native write has a valid operational route: registered writer contract, registered writer event, required trial support, and active parent/version evidence. It is not a semantic-truth detector, a general contamination mitigator, or a claim that content is correct or false.

## Boundary

- The Filter arm name and the active/quarantine state claim boundary are unchanged.
- Audit labels, treatment-arm metadata, and candidate roles are not policy inputs.
- Native readers and updaters receive the active partition only. A descendant of a quarantined parent remains quarantined.
- RAG retains every injected document in the Filter corpus audit source. The RAG index is built from the admission-selected active documents only.
- The scientific runner uses recorded prefix write envelopes and their writer/trial support. It does not manufacture `replay-*` provenance.
- Admission events record the observed partition state and exact reason.

## Build/Calibration MFT

Run without a provider:

```bash
python scripts/build_phase12_filter_mft.py --output /tmp/phase12-filter-v4-mft.json
```

The JSON report has `schema_version: "phase12_filter_mft_v4"`, `evidence_layer: "build_calibration_only"`, `scientific_result: false`, and `policy_version`. Each `cases` row contains `baseline`, `route_valid`, `content_class`, `audit_label`, `treatment_arm`, `expected_state`, `observed_state`, and `reason`.

The complete matrix covers FH, RAG, BoT, and Reflexion across valid/invalid routes and correct/false content classes. Valid routes are active for both content classes; invalid routes are quarantined for both content classes.

## Pilot-A compatibility decisions

- The scientific config binds `filter_interpretation: contract_invalid_direct_write_containment`
  and `filter_claim_status: operational_secondary`.
- Reflexion starts without fixed helper memory. Clean Reflexion memory can arise only through
  the native failure-gated prefix route; no trusted-entry bypass or calibration initialization
  is used.
- Blocked, invalidated, and interrupted attempts remain reconstructable archives but carry
  `scientific_result: false` and `result_eligible: false`.

Before Pilot-B/Main freeze, minimally revise Theory Section 3.1's generic filter wording;
Baseline/Filter Sections 9, 10, 11, and 13; Contamination Protocol aliases `FILTER-9` through
`FILTER-13` and filter-recovery reporting; and Experiment Design Sections 3 and the Pilot-A,
Pilot-B, and Main-A Filter comparison text. Those revisions must distinguish direct-write
operational containment from ordinary-route semantic false memory without pooling them.
