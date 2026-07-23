# Phase-12 Operator Runbook

## Release boundary

Run these commands from the repository root. They inspect or exercise the implemented
contract. They don't select a main route, allocate seeds, activate exploratory work,
make paid provider calls, or establish a scientific finding.

The frozen repository authority is
`830b89c8c169ffa9cdea472887fdae134dbae7cf`. The Experiment Design SHA-256 is
`984fe2881690d93a8ccced87abf03de4bf0012158462cf07ed23505414073eb0`.
If either identity differs from the intended package, stop and review the drift. Don't
rewrite IDs or hashes locally.

## Environment and BGE-M3 cache

Install the repository and development tools in the active Python 3.11 environment:

```bash
python -m pip install -e '.[dev]'
```

Scientific Phase-12 admission requires the exact normalized dense embedding contract
`BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181`, dimension 1024, loaded from
the local cache only. Fake embeddings and network download during verification are
forbidden.

Use the detailed instructions in `docs/bge-m3-cache-setup.md`. The essential cache
commands are:

```bash
export HF_HOME="$HOME/.cache/huggingface"
hf download BAAI/bge-m3 \
  --revision 5617a9f61b028005a4858fdac845db406aefb181
hf cache verify BAAI/bge-m3 \
  --revision 5617a9f61b028005a4858fdac845db406aefb181 \
  --fail-on-missing-files
```

Resolve the same revision offline, then run the verifier and focused regressions:

```bash
HF_HUB_OFFLINE=1 python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download(
    repo_id="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    local_files_only=True,
))
PY

python scripts/verify_bge_m3_fidelity.py
python -m pytest -q \
  tests/test_bge_m3_fidelity.py \
  tests/test_live_embedding_policy.py \
  tests/test_openai_compatible_client.py
```

This checkout's committed BFV2 certificate remains `blocked` with
`missing_cached_bge_m3`. Cache acquisition is an external readiness step, not a reason
to edit that evidence. Until the exact cache is present and the existing verifier
passes, F1C and all real scientific Pilot-A or Main-A requests remain blocked.

## Validate canonical configs

The committed configuration set contains readiness, two pilots, two candidate main
routes, and one inactive exploratory matrix:

```bash
python -m memcontam.cli phase12 validate --config configs/phase12/readiness.yaml
python -m memcontam.cli phase12 validate --config configs/phase12/pilot_a.yaml
python -m memcontam.cli phase12 validate --config configs/phase12/pilot_b.yaml
python -m memcontam.cli phase12 validate --config configs/phase12/main_3w.yaml
python -m memcontam.cli phase12 validate --config configs/phase12/main_5w.yaml
python -m memcontam.cli phase12 validate --config configs/phase12/exploratory_code.yaml
```

The two main files must stay `selection_status=candidate`. The exploratory file must
stay `activation_status=inactive`. There is no selected-main alias, route-selection
manifest, seed-allocation manifest, selected-package resource manifest, or activation
manifest in this committed config set.

The CLI `plan` command consumes the full Phase-12 study contract, not a compact
canonical candidate file. For offline contract QA, the canonical fixture is the input:

```bash
python -m memcontam.cli phase12 plan \
  --config tests/fixtures/phase12/FX-CONFIG-001.json
```

The output lists both candidate routes and reports `scientific_result=false`. It does
not choose a route.

## Non-scientific replay workflow

Use a new, single-component run ID. Replay fixtures require no provider key:

```bash
RUN_ID="phase12-contract-replay-$(date -u +%Y%m%dT%H%M%SZ)"

python -m memcontam.cli phase12 run-prefix \
  --replay FX-BRANCH-001 \
  --fixture-root tests/fixtures/phase12 \
  --run-root runs \
  --run-id "${RUN_ID}-prefix"

python -m memcontam.cli phase12 run-branch \
  --replay FX-BRANCH-001 \
  --fixture-root tests/fixtures/phase12 \
  --run-root runs \
  --run-id "${RUN_ID}-branch"

python -m memcontam.cli phase12 aggregate \
  --run-dir "runs/${RUN_ID}-branch"

python -m memcontam.cli phase12 validate-archive \
  --run-dir "runs/${RUN_ID}-branch"
```

`run-prefix` writes the arm-free clean prefix. `run-branch` exercises prefix, matched
branches, suffix, and archive writing in the current replay shell. `aggregate` and
`validate-archive` accept either `--run-dir` or their registered replay fixture option.
Never combine both source options.

The direct replay-only aggregate and archive checks use their own fixtures:

```bash
python -m memcontam.cli phase12 aggregate \
  --replay FX-AGG-001 \
  --fixture-root tests/fixtures/phase12
python -m memcontam.cli phase12 validate-archive \
  --replay FX-ARCHIVE-001 \
  --fixture-root tests/fixtures/phase12
```

## Certificates and admission gates

BFV2 and P12I are independent gates:

* BFV2 covers F1A structural replay, F1B source-contract replay, and F1C's pinned real
  retriever with mocked OpenAI-compatible answer transport.
* P12I is issued from a non-scientific contract replay. It may truthfully bind a
  blocked BFV2 certificate while keeping scientific admission false.

P12I requires passing evidence for all eight subgates: prefix checkpoint, five-arm
branch, NoMem alias, Filter information boundary, logging-v3 joins, model-behavior
denominator, eligibility recomputation, and run archive reconstruction. The evidence
paths and hashes are reopened and checked before certificate issuance.

Real scientific admission requires a passing BFV2 certificate, a passing P12I
certificate, a valid reconstructed archive, and governance applicable to the requested
run family. Main and route-bound extension requests additionally need validated route
selection, seed allocation, and the exact registered slot-to-seed assignment.
Scientific Python-sandbox requests also need validated external resource and
exploratory activation artifacts.

Admission-only checks use `run-prefix` or `run-branch` with `--scientific`,
`--admission-only`, the requested `--run-family`, `--candidate`, and `--mode`, plus an
externally prepared `--admission-bundle`. The repository doesn't provide a bundle that
turns the blocked F1C state into a pass. A rejected check must not create a run
directory.

## Failure response

Stop on any config, schema, compatibility, governance, certificate, archive, or hash
error. There is no local fallback.

| Failure | Operator action |
|---|---|
| `missing_cached_bge_m3` or `F1C_NOT_PASS` | Keep scientific admission blocked. Provision and verify the exact cache externally. |
| `ROUTE_SELECTION_REQUIRED` or `SEED_ALLOCATION_REQUIRED` | Obtain approved external manifests. Don't copy a candidate label into run metadata. |
| `EXPLORATORY_ACTIVATION_REQUIRED` | Keep the code matrix inactive until resource and activation governance is approved. |
| `EXPLORATORY_BUDGET_INSUFFICIENT` or `REPRODUCIBILITY_RESERVE_INSUFFICIENT` | Reject activation. Don't reduce protected work or alter the frozen plan locally. |
| `SEED_ASSIGNMENT_MISMATCH` | Reject the run. Use the exact externally frozen abstract-slot mapping. |
| Archive or artifact hash mismatch | Preserve the failed material for inspection. Don't repair evidence in place. |
| Model-produced malformed output | Keep it as included `model_behavior` with score zero when no valid answer exists. |
| Provider or engineering failure | Invalidate under the frozen policy and link any permitted rerun. |

## Documentation and release checks

Run the focused documentation contract and release-boundary checks:

```bash
python -m pytest -q tests/test_phase12_docs_scope.py
python -m pytest -q \
  tests/test_phase12_canonical_configs.py \
  tests/test_phase12_cli.py \
  tests/test_phase12_scientific_admission.py \
  tests/test_phase12_claim_scope.py
python -m ruff check src tests scripts
```

Passing these checks establishes repository contract consistency only. It doesn't
report a live pilot, selected Main execution, benchmark result, manuscript result,
causal use, or production contamination diagnosis.

## Machine-checked CLI index

- `command:phase12 validate`
- `command:phase12 plan`
- `command:phase12 run-prefix`
- `command:phase12 run-branch`
- `command:phase12 aggregate`
- `command:phase12 validate-archive`
