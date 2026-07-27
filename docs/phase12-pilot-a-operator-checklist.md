# Phase-12 Pilot-A Operator Checklist

This checklist stops before scientific Pilot-A. All plumbing and handoff output is
non-scientific (`scientific_result=false`).

## Offline Checks

```bash
python -m memcontam.cli phase12 cost-preview \
  --config configs/phase12/pilot_a_game24_minimal.yaml

python -m memcontam.cli phase12 pilot-a \
  --config configs/phase12/pilot_a_game24_scientific.yaml \
  --admission-only
```

The preview rejects a projected maximum above USD 5. Admission-only verifies the Filter-v4
readiness manifest, evidence hashes, final implementation commit, scientific config hash,
MFT, F1C, archive, and invariant results. It starts no run.

## Human-Authorized Plumbing

Only after reviewing the preview and provider configuration, use:

```bash
python -m memcontam.cli phase12 run-plumbing \
  --config configs/phase12/pilot_a_game24_minimal.yaml \
  --run-id phase12-pilot-a-plumbing \
  --arm Clean \
  --instances 1 \
  --allow-live-calls
```

The command rejects every non-Clean arm, any instance count other than one, missing
`--allow-live-calls`, and scientific-result plumbing. It writes a non-scientific live
archive, including Clean-only trial/call/failure/retrieval/context streams. Validate it
immediately:

```bash
python -m memcontam.cli phase12 validate-archive \
  --run-dir "${MEMCONTAM_ARTIFACT_ROOT}/runs/phase12-pilot-a-plumbing" \
  --mode clean-plumbing \
  --output .sisyphus/evidence/pilot-a-unblock/t7-plumbing.json
```

The archive report must remain below USD 5, preserve any model parse failures, and keep
`scientific_result=false`.

## Filter Claim Boundary

Pilot-A's `Filter` arm is `operational-evidence-filter-v4`: contract-invalid direct-write
containment. Its quarantine result is not semantic identification of false content, and a
Filter–Contam accuracy difference is not general mitigation evidence. Pilot-A checks branch,
admission, active-only visibility, logging, and archive mechanics. Evaluate ordinary-route
semantic false memory in Pilot-B before Main freeze.

## Handoff Gate

Inspect `.sisyphus/evidence/pilot-a-closeout/pilot_a_readiness_manifest_phase12_filter_v4.json`.
The linked scientific command uses `configs/phase12/pilot_a_game24_scientific.yaml`, parent
`pilot-a-game24-20260727T062808Z`, and root attempt
`pilot-a-game24-20260727T061248Z`. A missing or stale Filter-v4 artifact remains blocked rather
than being synthesized.
