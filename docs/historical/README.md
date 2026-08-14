# Historical and provenance documentation

Every document linked here is retained for reproducibility, calibration evidence, or
implementation provenance from an earlier repository package. Words such as “current,”
“authority,” “ready,” or “active” inside a snapshot refer only to that historical package. They do
not override the [current study design and status](../study-design-and-roadmap.md) or the
[current operator guide](../operator-guide.md).

Historical commands and paths are preserved as recorded. For exact reproduction, use the commit
identities named by the relevant document; do not reinterpret a historical runbook as current
operator guidance.

Filter / FilterChallenge / Filter-v5 material is exploratory mitigation evidence. It is not a
confirmatory Main arm and cannot gate readiness, checkpoints, support, seeds, route selection,
estimands, authorization, or aggregates.

## Retained snapshots

- [Baseline Fidelity V2 contract](baseline-fidelity-v2.md): historical baseline fidelity claims
  and reproduction commands.
- [Baseline Fidelity V2 evidence](baseline-fidelity-v2-evidence.md): immutable gate, resource,
  hash, and closeout provenance.
- [BGE-M3 cache setup](bge-m3-cache-setup.md): the exact historical BFV2 cache procedure.
- [Earlier-phase logging contract](logging-v3-phase12.md): historical logging, exposure,
  denominator, and no-pooling rules.
- [Earlier-phase implementation contract](phase12-implementation-contract.md): the complete
  historical implementation and study contract.
- [Earlier-phase operator runbook](phase12-operator-runbook.md): historical replay and admission
  workflow.
- [Filter-v5 methods lock](phase12-filter-v5-bct-methods-lock.md): immutable exploratory
  challenge/calibration methods.
- [Filter-v5 build status](phase12-filter-v5-build-status.md): sealed deterministic build evidence
  and blocked BCT status.

Machine-readable Filter-v5 evidence remains under [`../evidence/`](../evidence/), unchanged for
its hashes, terminal states, and authority bindings.
