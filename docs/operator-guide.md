# Installation and CLI operator guide

## Install from a fresh clone

The package supports Python `>=3.11,<3.14`. Create an isolated environment and install the
development extras before using the CLI:

```bash
git clone https://github.com/PROFOUNDIVE/memory-contamination-diagnostic.git
cd memory-contamination-diagnostic
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The installation provides both `memcontam` and `python -m memcontam.cli`; this guide uses the
module form.

## Inspect the installed surface

Use `--help` rather than relying on examples from older records:

```bash
python -m memcontam.cli --help
python -m memcontam.cli validate-config --help
python -m memcontam.cli run --help
python -m memcontam.cli aggregate --help
```

The root CLI exposes these public workflow entry points:

| Command | Purpose | Provider calls by default? |
|---|---|---:|
| `validate-config` | Parse and validate a run configuration | No |
| `run` | Execute the configured workflow | No for replay configurations |
| `aggregate` | Read a completed run directory and produce an aggregate report | No |

Versioned namespaces such as `phase12` and `phase13` are retained for compatibility and internal
workflow separation. Their accepted arguments are discoverable through their own `--help` output;
this public guide intentionally does not enumerate restricted inputs or local research artifacts.

## Deterministic configuration check

The shipped replay configuration provides a no-provider installation check:

```bash
python -m memcontam.cli validate-config configs/pilot_multitask_replay.yaml
```

Success begins with `valid config:`. Validation checks the declared stage, provider, tasks,
baselines, arms, logging contract, and replay fixture structure. It does not execute a run or
make a network request.

## Replay workflow

Replay configurations supply fixture responses instead of contacting a model provider. After
validation, create a run with a single-component identifier:

```bash
python -m memcontam.cli run <config.yaml> --run-id <run-id>
```

Run identifiers must not be absolute paths or contain parent-directory components. The configured
output location determines where the run archive is written; do not assume a fixed output layout.

`--allow-live-calls` is an explicit execution boundary. Supply it only for an approved
provider-backed configuration with the required credential available in the environment. Never
place credentials in configuration files, manifests, or logs.

## Run archive

A strict run records its configuration and machine-readable execution streams in the configured
run directory. The archive includes:

- `run.json`, `resolved_config.json`, and `provider_profile.json` for run identity and resolved
  settings;
- `trials.jsonl`, `calls.jsonl`, and `failures.jsonl` for execution records; and
- `memory_events.jsonl` and `filter_events.jsonl` when the selected workflow emits those events.

Resolved configuration and provider-profile records are designed to describe the execution
contract without storing provider credentials.

## Aggregation

Aggregate a completed run directory with:

```bash
python -m memcontam.cli aggregate <run-directory>
```

Use `aggregate --help` to inspect compatibility options before processing an older run archive.
Aggregation validates the selected archive contract before reporting results. It does not rerun
trials or contact a provider.

## Operational boundaries

- A syntactically valid configuration is not scientific approval or live-call authorization.
- Use replay fixtures for local development, testing, and review whenever possible.
- Keep live execution approval separate from source control and configuration review.
- Treat historical documents as provenance, not as current operator instructions.
