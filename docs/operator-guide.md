# Installation and CLI operator guide

## Install from a fresh clone

The package supports Python `>=3.11,<3.14`. Clone the repository, create an isolated environment,
and install the editable package before using the CLI:

```bash
git clone https://github.com/PROFOUNDIVE/memory-contamination-diagnostic.git
cd memory-contamination-diagnostic
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

On Windows PowerShell, activate the virtual environment with
`.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`.

Conda is optional. The name `memcontam` is conventional, not a prerequisite:

```bash
conda create -n memcontam python=3.11
conda activate memcontam
python -m pip install -e '.[dev]'
```

Installation provides both the `memcontam` console script and the equivalent
`python -m memcontam.cli` module entry point. Examples below use the module form.

## Safe deterministic installation check

The following command uses only tracked files, makes no provider calls, and incurs no cost:

```bash
python -m memcontam.cli validate-config configs/pilot_multitask_replay.yaml
```

Expected output begins with `valid config:`. This is a historical replay-config validation, not a
scientific-readiness decision. A second fixture-backed deterministic check is:

```bash
python -m memcontam.cli phase12 validate \
  --config tests/fixtures/phase12/FX-CONFIG-001.json
```

## Internal CLI namespace

The literal `phase13` CLI namespace is an internal implementation/versioning label retained to
keep modules, schemas, configs, data paths, and automation stable. It is not the public scientific
name of the study.

Inspect the actual command surface with:

```bash
python -m memcontam.cli --help
python -m memcontam.cli phase13 --help
```

The root CLI also exports `phase12`, `validate-config`, `run`, and `aggregate`. They remain useful
for historical implementation and replay workflows but are not substitutes for a prospectively
frozen confirmatory execution package.

## Current internal validator and calibration commands

| Command | Provider calls or cost? | Purpose |
|---|---:|---|
| `phase13 validate-authority-freeze` | No | Validate a freeze schema, closure hash, and declared requirement sets |
| `phase13 validate-execution-registry` | No | Validate the execution registry referenced by a freeze |
| `phase13 validate-provenance` | No | Validate a sealed provenance bundle and each listed artifact hash |
| `phase13 prepare-clean-prefix` | No | Validate reduced-panel calibration inputs and write an authorization request |
| `phase13 run-clean-prefix` | Yes | Execute the separately authorized reduced-panel GPT-4o clean-prefix calibration |

The three generic validators require caller-supplied closure artifacts. The public repository does
not ship a final authority/execution/provenance bundle, so their examples use placeholders rather
than implying a runnable scientific package.

The shipped clean-prefix config also references a hash-bound readiness input that is not included
in a public clone. `prepare-clean-prefix` therefore fails closed until an approved complete input
bundle is supplied. It is not part of the fresh-clone installation check.

### Validate an authority freeze

```bash
python -m memcontam.cli phase13 validate-authority-freeze \
  --freeze <authority-freeze.json> \
  --requirements <authority-requirements.json>
```

Success prints `{"freeze_id": "...", "status": "valid"}`. The command validates schema,
relative-reference safety, uniqueness, closure hash, and declared authority/registry/parameter
sets. It does not open or hash the authority files named by the freeze and does not judge whether
the scientific choices are substantively approved.

### Validate an execution registry

```bash
python -m memcontam.cli phase13 validate-execution-registry \
  --root <artifact-root> \
  --freeze <authority-freeze.json> \
  --requirements <authority-requirements.json>
```

The command validates the freeze, opens and hashes its single referenced execution registry, and
checks registry identity, declared task/baseline/condition dimensions, template uniqueness,
`H_run`, RAG corpus reference, and capacity ordering. It does not open or hash the RAG corpus and
does not execute a template. Success prints `{"registry_id": "...", "status": "valid"}`.

### Validate a provenance bundle

```bash
python -m memcontam.cli phase13 validate-provenance \
  --root <bundle-root> \
  --manifest <provenance-manifest.json> \
  --seal <provenance-seal.json>
```

Artifact paths in the manifest resolve under `--root`. The command checks strict schemas, unique
roles and paths, relative-path safety, manifest/seal identity, and every listed artifact SHA-256.
Success prints the bundle ID, artifact count, and manifest hash.

### Prepare the reduced-panel clean-prefix calibration

```bash
python -m memcontam.cli phase13 prepare-clean-prefix \
  --config configs/phase13/clean_prefix_calibration_v1.yaml \
  --run-id <one-component-run-id> \
  --output <authorization-request.json>
```

This no-provider command validates the exact calibration kind, four-baseline panel,
clean-prefix-only scope, schedules, input hashes, and call/token/cost ceilings. It writes the
request to `--output`, which must not already exist. A valid request means only that the package is
ready for separate calibration authorization. The implemented joint-eligibility law is superseded
for current Main support planning.

Run IDs must be one path component: no absolute paths, nested components, or `..`.

### Run the authorized clean-prefix calibration

```bash
OPENAI_API_KEY=<credential> \
python -m memcontam.cli phase13 run-clean-prefix \
  --config configs/phase13/clean_prefix_calibration_v1.yaml \
  --run-id <same-run-id> \
  --request <authorization-request.json> \
  --authorization <authorization.json> \
  --expected-authorization-sha256 <64-hex-sha256> \
  --allow-live-calls
```

This command can make paid OpenAI calls. It requires a matching separate authorization,
`--allow-live-calls`, the configured credential, the pinned BGE-M3 model in a local cache, matching
config/request/implementation/budget identities, and a new run directory. The current verifier
does not enforce authorization expiration; freshness remains an external approval requirement.

The command executes clean prefixes for the three current tasks across FH-bounded, RAG-Frozen,
BoT-style, and Reflexion-style for four registered calibration seeds. It does not execute suffix
trials, NoMem, Correct/Irrelevant/Contam interventions, Filter, or Main.

Outputs follow the config-selected root:

```text
runs/phase13-clean-prefix-calibration-v1/<run-id>/
```

The run archive includes the resolved config, request, authorization, trials, calls, checkpoints,
eligibility rows, seed status, rates, accounting, artifact manifest, and seal. Runtime failures
retain partial accounting and mark the archive invalidated. A completed result is reduced-panel
feasibility/cost evidence only; it cannot determine current Main support, seeds, route, or budget.

## Operational boundaries

- A validator reporting `valid` establishes internal schema/hash closure only, not scientific
  approval, a backbone decision, a final freeze, or live-call authorization.
- Do not substitute the clean-prefix calibration config for a Main execution registry.
- Do not place secrets in configs, requests, manifests, logs, or committed documentation.
- Replay and generic validation remain secret-free; only explicit live boundaries need provider
  credentials.
- Follow the approved config and returned run directory rather than assuming a legacy artifact
  layout.
- Do not inspect confirmatory outcomes before the final prospective closure is frozen.
