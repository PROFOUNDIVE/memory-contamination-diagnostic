# Phase-12 Pilot-A Operator Checklist

This checklist stops before scientific Pilot-A. All plumbing and handoff output is
non-scientific (`scientific_result=false`).

## Offline Checks

```bash
python -m memcontam.cli phase12 cost-preview \
  --config configs/phase12/pilot_a_game24_minimal.yaml

python -m memcontam.cli phase12 pilot-a \
  --config configs/phase12/pilot_a_game24_minimal.yaml \
  --admission-only
```

The preview rejects a projected maximum above USD 5. Admission-only reads the passing
T5 F1C/micro-retrieval, T6 invariant/archive, and T7 plumbing reports. It starts no run.

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

Inspect `.sisyphus/evidence/pilot-a-unblock/t7-handoff.json`. It records the current
implementation commit, config/F1C/invariant hashes, estimated Pilot-A maximum cost,
and the exact human launch command. Its status is ready only when every T5/T6/T7 report
passes; a missing live plumbing archive remains blocked rather than being synthesized.
