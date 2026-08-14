# memory-contamination-diagnostic

This repository is a research harness for a narrow question: **when a reasoning system carries
external memory across problems, how does a controlled false memory change later verified
performance and failure propagation?**

The repository contains implementation, deterministic validation, hash-bound source pools,
calibration tooling, and historical evidence. It does not contain inspected confirmatory Main
outcomes, and nothing in a checkout authorizes provider-backed scientific execution.

## Current confirmatory design

The current scientific contract covers three separate task streams: Game24, Math Equation
Balancer, and Word Sorting. Each memory-bearing baseline continues from the same fixed clean
checkpoint under four matched conditions:

- **Clean:** continue from the unchanged clean checkpoint.
- **Correct:** insert a matched, valid memory object.
- **Irrelevant:** insert a matched valid object that does not apply to the target task family.
- **Contam:** insert a matched, relevant, objectively false memory object.

The four confirmatory memory-bearing baseline strata are:

- **FH / FH-bounded:** append-only interaction history under the exact registered context bound.
- **RAG-Frozen:** retrieval from a prospectively closed corpus and immutable index/runtime
  contract, without online memory writes.
- **BoT-style:** storage and reuse of distilled reasoning templates in a bounded native buffer.
- **Reflexion-style:** verifier-triggered verbal reflections retained for later work.

Exact FH and FH-bounded are distinct strata and are not pooled. `NoMem` is a Clean-only,
memory-free negative-control singleton, not a fifth baseline crossed with the four conditions.

The primary controlled contrast is Clean versus Contam. It is a total intervention contrast and
can include insertion, capacity, retrieval competition, relevance, and semantic-falsehood effects.
Correct and Irrelevant help interpret that total contrast without isolating every component by
themselves.

Filter / FilterChallenge / Filter-v5 is historical and explicitly exploratory mitigation work.
It is not a confirmatory Main arm, does not gate readiness or support, cannot determine
checkpoints, seeds, route selection, or authorization, and does not enter the primary estimand.

## Horizons, readiness, and support

`H_run` is the number of post-intervention trials actually generated and stored. `H_primary` is
the separately prespecified finite analysis window for the principal estimand; it may be shorter
than `H_run`, but neither value determines the other.

A seed's fixed task × route × baseline checkpoint is structurally ready when its native memory
state can accept every matched intervention/control and complete the registered suffix without
implementation, capacity, serialization, or fidelity failure. Contamination-injection support is
the set of seeds structurally ready at that fixed checkpoint. Baseline effects use
baseline-specific support; direct baseline comparisons use pairwise common support. Strict
all-baseline support is sensitivity-only.

## Current state and proposed work

Implemented surfaces include the three current task registries, native task contexts, four
memory-bearing baseline branches, the NoMem singleton, deterministic contract validators, and a
separately authorized reduced-panel clean-prefix calibration path. The calibration path implements
a superseded all-baseline joint-eligibility law; it cannot determine current Main support, seeds,
route, or final budget.

The remaining roadmap is prospective rather than frozen: integrate DC-RS, fix a MMLU-Pro subset,
add GPQA Diamond, revise and freeze the RAG-Frozen clean corpus, evaluate GPT-5.6 Luna and choose
the backbone prospectively, set `H_run = 50`, retain `H_primary = 5` unless revised before Main,
compute final task × baseline structural/injection support, and produce the final prospective
authority and execution freeze. No further task or baseline expansion is planned for Core Main.

The current confirmatory contract remains the three-task, four-memory-baseline, registered GPT-4o
package until a later public freeze authorizes changes. After the relevant scope is frozen,
implementation-ready cells may execute sequentially under that same contract and separate
authorization. No Main evidence is reported.

## Fresh-clone setup

Python `>=3.11,<3.14` is supported. From a shell with Git and a supported Python installed:

```bash
git clone https://github.com/PROFOUNDIVE/memory-contamination-diagnostic.git
cd memory-contamination-diagnostic
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Conda is optional. `memcontam` is only a conventional environment name:

```bash
conda create -n memcontam python=3.11
conda activate memcontam
python -m pip install -e '.[dev]'
```

After installation, run a deterministic, no-provider validation using only tracked files:

```bash
python -m memcontam.cli validate-config configs/pilot_multitask_replay.yaml
```

Expected output begins with `valid config:`. This validates a historical replay configuration; it
does not establish scientific readiness or authorize live calls.

## Documentation

- [Study design, implementation status, datasets, and roadmap](docs/study-design-and-roadmap.md)
- [Installation and CLI operator guide](docs/operator-guide.md)
- [Historical and provenance documentation](docs/historical/README.md)
