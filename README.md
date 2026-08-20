# memory-contamination-diagnostic

`memory-contamination-diagnostic` is a research harness for studying how persistent reasoning
memory can affect later verified task performance. It provides task adapters, memory-baseline
implementations, deterministic configuration checks, and offline replay support.

## What a fresh clone supports

- Validate a shipped replay configuration without provider access.
- Run fixture-backed replay workflows without credentials or network calls.
- Inspect the CLI and its input requirements locally.

A repository checkout is not an authorization to run paid providers, produce scientific claims, or
interpret replay output as a study result.

## Quick start

Python `>=3.11,<3.14` is supported.

```bash
git clone https://github.com/PROFOUNDIVE/memory-contamination-diagnostic.git
cd memory-contamination-diagnostic
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Inspect the command surface:

```bash
python -m memcontam.cli --help
```

Run the deterministic installation check:

```bash
python -m memcontam.cli validate-config configs/pilot_multitask_replay.yaml
```

The command validates configuration only. It makes no provider calls and does not create a run.

## Safety boundary

- Keep credentials out of configuration files, logs, and commits.
- Treat `--allow-live-calls` as an explicit paid-execution boundary.
- Use fixture-backed replay when developing or reviewing behavior locally.
- Preserve historical records as provenance; they are not current operating instructions.

## Documentation

- [Installation and CLI operator guide](docs/operator-guide.md)
- [Study scope and publication boundary](docs/study-design-and-roadmap.md)
- [Historical and provenance documentation](docs/historical/README.md)
