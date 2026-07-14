# G0 DC-RS and Reflexion Same-Sample Retry Fidelity Follow-up

**Tag:** post-`v0.5` follow-up  
**Repo:** `memory-contamination-diagnostic`  
**Scope:** optional appendix comparator for `dynamic_cheatsheet_rs_optional` and tightened same-sample retry control flow for `reflexion_style` over the locked 3-task pilot set  
**Evidence run:** `runs/g0_dc_rs_reflexion_fidelity_followup_replay/trials.jsonl`

> For the historical full-history pass, see [`docs/g0-baseline-fidelity-gate-v0.5.md`](g0-baseline-fidelity-gate-v0.5.md). For the prior RAG/BoT report, see [`docs/g0-baseline-fidelity-gate-v0.4.md`](g0-baseline-fidelity-gate-v0.4.md).

---

## TL;DR

This follow-up is published after the Task 8 replay evidence exists. It records the source-bounded fidelity claims for two adapted baselines:

- Faithful adapted DC-RS optional appendix comparator: top-3 cosine retrieval over prior same-identity input/output pairs, label-free pre-answer cheatsheet synthesis, then memory-conditioned generation, with native method-call costs logged.
- Faithful adapted Reflexion control flow: failed trajectory plus sanitized evaluator feedback produces linguistic reflection, latest-three reflection memory conditions a same-sample retry, stopping on success or attempt limit; no weight updates.

The canonical replay evidence run emits `108 trial rows` and `174 native method calls` (DC-RS 108, Reflexion 66). The replay output is fidelity/QA evidence only, not benchmark or manuscript-quality evidence. This document does not claim a complete reproduction, benchmark-score improvement, or primary DC baseline status.

---

## Release status

`v0.5` remains the historical full G0 baseline-fidelity pass. This follow-up tightens Reflexion to the source-required same-sample retry semantics and adds DC-RS as an optional appendix comparator. The `pyproject.toml` version remains `0.1.0`; no Python package is published. The intended Git tag for the historical full pass is still `v0.5`.

---

## Exact scope and matrix

The follow-up uses the locked 3-task pilot set with three arms and two replay model labels, restricted to DC-RS and Reflexion:

- Tasks: `game24`, `math_equation_balancer`, `word_sorting` (3 samples each)
- Baselines: `dynamic_cheatsheet_rs_optional`, `reflexion_style`
- Arms: `clean`, `contaminated`, `contaminated_filter`
- Models: `gpt4o`, `frontier_reasoning`

`3 tasks × 2 baselines × 3 arms × 2 models = 108 trials`.

Counts observed in the canonical run:

- 54 per baseline
- 18 per baseline/arm
- 174 total method calls
- Stage counts: `dc_rs_synthesize=54`, `dc_rs_generate=54`, `reflexion_generate=60`, `reflexion_reflect=6`
- 6 Reflexion same-sample retry identities (game24_pilot_001)

---

## Official sources

### Dynamic Cheatsheet (DC-RS source)

- Paper: `Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory`, https://aclanthology.org/2026.eacl-long.333/
- Repository: https://github.com/suzgunmirac/dynamic-cheatsheet
- License: MIT, copyright 2025 Mirac Suzgun

### Reflexion

- Paper: `Reflexion: Language Agents with Verbal Reinforcement Learning`, https://arxiv.org/abs/2303.11366
- Repository: https://github.com/noahshinn/reflexion
- License: MIT, copyright 2023 Noah Shinn

Both projects are cited as research sources. The prompts, state shapes, and control flow here are adapted for this diagnostic harness, not copied verbatim.

---

## Adaptation table

| Must Preserve | Safely Adapted | Omitted |
|---|---|---|
| Top-k cosine retrieval over memory entries (DC-RS) | Uses pinned `sentence-transformers/all-MiniLM-L6-v2` with `offline_fallback: true` for network-free QA | Code execution, provider tool calls, web search, and live retrieval variants |
| Pre-answer cheatsheet synthesis from retrieved memory (DC-RS) | Synthesizes from same-identity `dc_rs_io_pair` records only; no gold labels or verifier spec in prompt | Post-hoc curation, multi-pass refinement, and cross-identity sharing |
| Memory-conditioned generation after synthesis (DC-RS) | Generation sees only the synthesized cheatsheet and current task input | Direct exposure of retrieved pair outputs to the generator prompt |
| Failure-gated generation/reflection loop (Reflexion) | Reflection triggered only after verifier failure with sanitized reason | Success-only reflection, multi-step planning, and environment-specific scaffolding |
| Latest-three reflection memory conditions retry (Reflexion) | Ordered latest-three reflections rendered from the current identity | Weight updates, value functions, and external learning signals |
| Same-sample retry until success or attempt limit (Reflexion) | `max_attempts` capped at 2; retry stops after generate/reflect/generate | Cross-sample retries, continued reflection after retry, and second retry cycles |

---

## Per-baseline stages, state, and write semantics

### dynamic_cheatsheet_rs_optional — faithful adapted DC-RS optional appendix comparator

- Two stages per trial in fixed order: `dc_rs_synthesize` then `dc_rs_generate`.
- Persistent state is keyed by `(run_id, task_name, baseline, arm, backbone)`; the cheatsheet and the `dc_rs_io_pair` corpus grow under that key. No `sample_id` appears in the persistent state key.
- `dc_rs_synthesize` retrieves top-3 cosine-scored `dc_rs_io_pair` records from prior same-identity entries, synthesizes a label-free `<cheatsheet>` block, and writes a `MemoryEntry` with `memory_type="dynamic_cheatsheet"` when a non-empty block is parsed.
- `dc_rs_generate` reads the current cheatsheet and current task input, emits an answer, and the trial appends a deterministic `dc_rs_io_pair` entry after verification.
- Emits `memory_write_event={"type": "dynamic_cheatsheet_rs_update", "status": "accepted", ...}` on accepted synthesis.
- Native method-call costs are logged per call.
- DC-RS remains an optional appendix comparator, not a new main baseline.

### reflexion_style — faithful adapted Reflexion control flow

- One actor stage per trial: `reflexion_generate`.
- On verifier success: no second call, no write event, state unchanged.
- On verifier failure only: calls `reflexion_reflect`, strips the response, and appends one non-empty `MemoryEntry` with `memory_type="verbal_reflection"`. The reflection is derived from the failed trajectory plus sanitized evaluator feedback; no verifier metadata or gold answer is leaked.
- With `max_attempts: 2`, a same-sample `reflexion_generate` retry runs using the latest-three ordered reflections from the current identity. It stops on success or after the attempt limit. No weight updates occur.
- `game24_pilot_001` is pinned to fail the first `reflexion_generate` and succeed on the retry, producing exactly 6 retry identities (3 arms × 2 models).
- Emits `memory_write_event={"type": "reflexion_append", "status": "accepted", ...}` on accepted reflection.
- This is a faithful adapted control flow, not a complete reproduction of the official Reflexion agent.

---

## Native call-count table

| Baseline | Stages per trial | Calls per trial | Total calls |
|---|---|---|---|
| `dynamic_cheatsheet_rs_optional` | `dc_rs_synthesize`, `dc_rs_generate` | 2 | 108 |
| `reflexion_style` | `reflexion_generate` (+ `reflexion_reflect` + retry `reflexion_generate` on failure) | 1 or 3 | 66 |
| **Total** | — | — | **174** |

Reflexion's extra 12 calls are the six failure-only reflections plus six same-sample retry generations at `game24_pilot_001`.

---

## Contamination, filter, and isolation contract

- State is keyed by `(run_id, task_name, baseline, arm, backbone)`. No entry, source ID, or trial lineage crosses this key.
- `clean` seeds contain only the paired clean record.
- `contaminated` seeds contain the paired clean record plus the paired corrupted record.
- `contaminated_filter` starts from the contaminated seed set and applies `drop_known_contaminated`, keeping the paired clean record and logging the drop count.
- DC-RS retrieves only same-identity `dc_rs_io_pair` records; retrieved records never include current/future or foreign pairs.
- Contamination exposure is determined from `memory_before` structured records, not from retrieval fields.
- Generated descendants are marked contaminated only when their prompt-visible ancestry includes a contaminated entry. Filtered/clean descendants remain clean.
- Model-visible rendered `content` never contains catalog source labels, contamination labels, verifier specs, expected/gold values, or verifier reasons.

---

## Verification commands and results

Run from the repository root:

```bash
python -m memcontam.cli validate-config configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml
python -m pytest tests/test_dc_rs_faithful.py tests/test_reflexion_faithful.py tests/test_cli_run.py tests/test_replay_client.py tests/test_replay_fixtures.py tests/test_contamination_catalog.py tests/test_logging_schema.py tests/test_aggregate.py tests/test_docs_scope.py -q
python -m pytest tests/test_task_verifiers.py tests/test_cli_run.py tests/test_contamination_catalog.py tests/test_openai_compatible_client.py tests/test_aggregate.py tests/test_docs_scope.py -q
python -m ruff check src tests scripts
python -m memcontam.cli run configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml --run-id g0_dc_rs_reflexion_fidelity_followup_replay
python -m memcontam.cli aggregate runs/g0_dc_rs_reflexion_fidelity_followup_replay
python scripts/inspect_g0_dc_rs_reflexion_fidelity.py runs/g0_dc_rs_reflexion_fidelity_followup_replay
```

Expected results:

- Config validates.
- All focused tests pass.
- Ruff reports no errors.
- Replay writes `runs/g0_dc_rs_reflexion_fidelity_followup_replay/trials.jsonl` with 108 rows.
- Aggregate emits valid JSON with method-call totals matching the table above.
- Inspector reports overall pass.

The canonical Task 8 evidence was captured with:

- Config: `configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml`
- Fixture: `data/replay/g0_dc_rs_reflexion_fidelity_followup_v1.yaml`
- Corpus: `data/memory/catalog_v2.jsonl`, version `memory_catalog_v2`
- Run ID: `g0_dc_rs_reflexion_fidelity_followup_replay`
- Inspector: `scripts/inspect_g0_dc_rs_reflexion_fidelity.py`

Inspector result summary:

```json
{
  "shape": "pass",
  "dc_rs": "pass",
  "reflexion": "pass",
  "accounting": "pass",
  "summary": {
    "trials": 108,
    "method_calls": 174,
    "dc_rs_calls": 108,
    "reflexion_calls": 66
  },
  "reasons": [],
  "overall": "pass"
}
```

Independent row count: 108. Independent method-call sum: 174.

---

## Limitations and non-claims

- This follow-up covers `dynamic_cheatsheet_rs_optional` and `reflexion_style` only. `no_memory`, `full_history`, `retrieval_rag`, `bot_style`, the historical `dynamic_cheatsheet_optional` DC-Cu baseline, and `expel_optional` are not claimed to pass this slice.
- This is an adapted baseline gate, not a complete reproduction of either official source.
- The replay output is a fidelity/QA artifact, not benchmark or manuscript-quality evidence.
- No benchmark-score improvement, primary DC baseline status, or manuscript-quality claim is made.
- No tool execution, live web search, or external code interpreter is used in the replay gate.
- DC-RS is an optional appendix comparator, not a new main baseline.

---

## Artifact file list

- `configs/g0_dc_rs_reflexion_fidelity_followup_replay.yaml`
- `data/memory/catalog_v2.jsonl`
- `data/replay/g0_dc_rs_reflexion_fidelity_followup_v1.yaml`
- `scripts/inspect_g0_dc_rs_reflexion_fidelity.py`
- `src/memcontam/baselines/dynamic_cheatsheet_optional.py`
- `src/memcontam/baselines/reflexion_style.py`
- `src/memcontam/cli.py`
- `tests/test_dc_rs_faithful.py`
- `tests/test_reflexion_faithful.py`
- `tests/test_cli_run.py`
- `tests/test_replay_client.py`
- `tests/test_replay_fixtures.py`
- `tests/test_contamination_catalog.py`
- `tests/test_logging_schema.py`
- `tests/test_aggregate.py`
- `tests/test_docs_scope.py`
- `docs/g0-dc-rs-reflexion-fidelity-followup.md`
- `README.md`
