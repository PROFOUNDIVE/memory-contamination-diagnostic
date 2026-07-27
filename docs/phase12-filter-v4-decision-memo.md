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
