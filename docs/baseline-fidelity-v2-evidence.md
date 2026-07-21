# Baseline Fidelity V2 Evidence

This file records provenance for the V2 gate. It is evidence metadata, not a second
contract. [`baseline-fidelity-v2.md`](baseline-fidelity-v2.md) remains the sole V2
authority.

## Evidence Provenance

| Field | Recorded value |
|---|---|
| Repository commit before Task 13 documentation | `09c6160452f95302dc5574d1edb1ebeaa620cd61` |
| Plan | `.sisyphus/plans/BASELINE-FIDELITY-V2_source-contract_remediation.md` |
| Plan SHA-256 | `5a5afe7f0d5fa171ff9d0b279fdd5875ee6885e718043cdfff3e59c449428e0f` |
| Evidence namespace | `.sisyphus/evidence/baseline-fidelity-v2/` |
| F1A run | `runs/bfv2-structural-replay/` |
| F1B run | `runs/bfv2-source-contract-replay/` |
| F1B inspector result | 18 trials, 32 calls, 5 failures, 10 memory events, `overall=pass` |
| F1C verifier result in this checkout | `overall=blocked`, `blocker=missing_cached_bge_m3`, exit 1 |

F1A and F1B use committed replay inputs and explicit non-scientific test-double
embeddings. F1C uses pinned BGE-M3 cache-only semantics and mocked-live answer dispatch.
Sockets are denied during its verification. No API call, model download, generated
`runs/` directory, or embedding cache belongs in a commit.

The generated F1B artifact manifest is
`.sisyphus/evidence/baseline-fidelity-v2/evidence_manifest.json`. It hashes the config,
native replay fixture, prompt fixtures, inspector report, and strict run artifacts. It
doesn't turn the blocked F1C layer into a pass.

## Resource Usage

The committed reporter was run against `runs/bfv2-source-contract-replay`. Replay
fixtures don't report provider token or latency values, so those totals are zero rather
than estimates.

| Baseline | semantic calls | transport retries | prompt tokens | completion tokens | latency ms | retrievals | memory writes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_memory` | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| `full_history` | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| `retrieval_rag` | 3 | 0 | 0 | 0 | 0 | 9 | 0 |
| `bot_style` | 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| `reflexion_style` | 9 | 0 | 0 | 0 | 0 | 0 | 2 |
| `dynamic_cheatsheet_rs_optional` | 6 | 0 | 0 | 0 | 0 | 3 | 2 |

These are method-native unequal call counts. They aren't compute-matched performance
comparisons. No F1C resource row is reported because the missing cache stops the gate
before a verified run exists.

## Artifact Hash Manifest

SHA-256 values below seal the committed inputs and verification programs used by this
evidence report.

| Artifact | SHA-256 |
|---|---|
| `configs/baseline_fidelity_v2_structural_replay.yaml` | `b0372a67dff01bcec72bd1eb284d5c72303e2b33e12bb4e6e9ecd5e1149d2886` |
| `configs/baseline_fidelity_v2_source_contract_replay.yaml` | `2fb0e9bf37b30e36662d485bc82689e25773794c20e8d6fd231be603de81e3bb` |
| `configs/baseline_fidelity_v2_bge_smoke.yaml` | `13941120b0b0005d2acb8407311b3a62834e7aa4730ea6084c38ac559404941c` |
| `data/replay/baseline_fidelity_v2_source_contract.yaml` | `1e44286a31791a73ffeadf0cec7db0c2fd75a14f14421b96fcae3633e44d97f7` |
| `data/memory/baseline_fidelity_v2_contract_corpus.jsonl` | `b608b1694bc4250a4d0e79400357ef8d01c76fe498b20cf45e76004cd01d30a7` |
| `data/memory/baseline_fidelity_v2_contract_corpus.manifest.json` | `839e84ff0b0394abd36f0b1ac89f6b56219348b3f7b4572e0daecc58630bcd3c` |
| `scripts/inspect_baseline_fidelity_v2.py` | `8f16ad86d665595c99dae0ec4de2464ed27d885050e84ed61c1312123a665a4b` |
| `scripts/verify_bge_m3_fidelity.py` | `48574cdce33b244c0eff1754ab00430d8ae42236b2734395f8a11b94628f0cdf` |
| `scripts/report_baseline_resource_usage.py` | `30cd8f5da1c32d7ae3bdf353d39552c1745b0a515dfab02768f825d7aebfc0ec` |

The corpus manifest's internal content hash is
`sha256:904943a6e4e8efc67a205d79c9a4f9cfe534b4205b683de6319e4c1a9ffbb49c`.
Prompt fixture hashes and generated run-artifact hashes are recorded in the generated
JSON manifest rather than duplicated here.

## Seal Status

F1A structural replay and F1B source-contract replay are reproducible QA gates in this
checkout. F1B's independent inspector passed and its generated artifact hashes match the
files produced by the canonical run.

The evidence package isn't a complete V2 certification seal while F1C is blocked. The
only accepted next step is to provision the exact pinned BGE-M3 revision in the local
cache and rerun the existing verifier. Changing the model, revision, provider mode, or
network policy would create different evidence, not complete this gate.
