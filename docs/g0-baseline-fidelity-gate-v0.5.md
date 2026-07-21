# G0 Baseline Fidelity Gate — v0.5 Faithful Full-History, Reflexion, and Dynamic Cheatsheet Baselines

> **Superseded for V2:** This historical report cannot support a Baseline-Fidelity-V2 fidelity claim. The sole V2 authority is [`docs/baseline-fidelity-v2.md`](baseline-fidelity-v2.md), with evidence provenance in [`docs/baseline-fidelity-v2-evidence.md`](baseline-fidelity-v2-evidence.md).

**Tag:** `v0.5`  
**Repo:** `memory-contamination-diagnostic`  
**Scope:** full G0 pass for `full_history`, `reflexion_style`, and `dynamic_cheatsheet_optional` over the locked 3-task pilot set  
**Evidence run:** `runs/g0_fh_reflexion_dc_faithful_replay/trials.jsonl`

> For the prior RAG/BoT report, see [`docs/g0-baseline-fidelity-gate-v0.4.md`](g0-baseline-fidelity-gate-v0.4.md).

---

## Release status

`v0.5` historically documented the repository research-artifact work that raised three native-memory baselines from legacy stubs to faithful adapted runtimes. The `pyproject.toml` version remained `0.1.0`; no Python package was published. This report recorded an intended `v0.5` Git tag, but it does not establish that a tag or release exists.

The replay evidence in this report is a fidelity/QA artifact, not benchmark/manuscript evidence. Model labels are replay labels. Live runs keep the same stage structure and require their own provider snapshots.

---

## Exact scope and matrix

The gate uses the locked 3-task pilot set with three arms and two replay model labels:

- Tasks: `game24`, `math_equation_balancer`, `word_sorting` (3 samples each)
- Baselines: `full_history`, `reflexion_style`, `dynamic_cheatsheet_optional`
- Arms: `clean`, `contaminated`, `contaminated_filter`
- Models: `gpt4o`, `frontier_reasoning`

`3 tasks × 3 baselines × 3 arms × 2 models = 162 trials`.

Counts observed in the canonical run:

- 54 per baseline
- 18 per baseline/arm
- 222 total method calls
- Stage counts: full_history_generate=54, reflexion_generate=54, reflexion_reflect=6, dynamic_cheatsheet_generate=54, dynamic_cheatsheet_curate=54
- 6 Reflexion reflected trials (game24_pilot_001)
- 6 DC preserved_missing_tag rows (game24_pilot_001)

---

## Official sources and revisions

### Reflexion

- Paper: `Reflexion: Language Agents with Verbal Reinforcement Learning`, https://arxiv.org/abs/2303.11366
- Repository: https://github.com/noahshinn/reflexion
- Pinned revision: `218cf0ef1df84b05ce379dd4a8e47f17766733a0`
- License: MIT, copyright 2023 Noah Shinn

### Dynamic Cheatsheet

- Paper: `Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory`, https://aclanthology.org/2026.eacl-long.333/
- Repository: https://github.com/suzgunmirac/dynamic-cheatsheet
- Pinned revision: `5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9`
- License: MIT, copyright 2025 Mirac Suzgun

Both projects are cited as research sources. The prompts and control flow here are adapted for this diagnostic harness, not copied verbatim.

---

## Per-baseline stages, state, and write semantics

### full_history — faithful append-only full-history

- One stage per trial: `full_history_generate`.
- State is an ordered list of prior transcripts. The prompt renders every prior `MemoryEntry` as sanitized text; no retrieval, summarization, truncation, or reordering occurs in this gate.
- The new response is parsed and verified, then one `MemoryEntry` with `memory_type="full_history_transcript"` is appended. Its content contains the task input, serialized prompt, raw response, parsed answer, and boolean correctness only.
- Emits `memory_write_event={"type": "full_history_append", "status": "accepted", ...}` for every trial.
- `retrieved_memory` and `retrieved_scores` are empty.

### reflexion_style — Reflexion-style verbal memory proxy / faithful adapted control flow

- One actor stage per trial: `reflexion_generate`.
- On verifier success: no second call, no write event, state unchanged.
- On verifier failure only: calls `reflexion_reflect`, strips the response, and appends one non-empty `MemoryEntry` with `memory_type="verbal_reflection"`. Empty reflections are rejected with `status="rejected_empty"` and no state change.
- The actor and reflector prompts read the latest three ordered rendered reflections from the current identity; the full identity-local list is retained for logging/lineage.
- Emits `memory_write_event={"type": "reflexion_append", "status": "accepted", ...}` on accepted reflection.
- `retrieved_memory` and `retrieved_scores` are empty.
- This is a faithful adapted control flow, not an exact reproduction of the official Reflexion agent across every task environment.

### dynamic_cheatsheet_optional — faithful adapted DC-Cu optional appendix comparator

- Two stages per trial in fixed order: `dynamic_cheatsheet_generate` then `dynamic_cheatsheet_curate`.
- State is one mutable text cheatsheet plus lineage metadata. Seeded corpus records render as catalog-order bullet lines.
- Generator prompt contains the task input and current cheatsheet. Curator prompt contains the previous cheatsheet, current input, raw output, parsed answer, and boolean correctness only, and asks for one `<cheatsheet>...</cheatsheet>` block.
- A non-empty parsed block replaces the cheatsheet with a new `MemoryEntry` of `memory_type="dynamic_cheatsheet"` and emits `status="accepted"`. Missing, incomplete, or empty tags preserve the exact prior state and emit `status="preserved_missing_tag"` or `status="preserved_empty"`.
- Emits `memory_write_event={"type": "dynamic_cheatsheet_update", "status": "accepted"|"preserved_missing_tag"|"preserved_empty", ...}`.
- `retrieved_memory` and `retrieved_scores` are empty.
- DC remains an optional appendix comparator, not a new main baseline. This is a faithful adapted DC-Cu runtime, not an exact reproduction of the official Dynamic Cheatsheet implementation.

---

## Native call-count table

| Baseline | Stages per trial | Calls per trial | Total calls |
|---|---|---|---|
| `full_history` | `full_history_generate` | 1 | 54 |
| `reflexion_style` | `reflexion_generate` (+ `reflexion_reflect` on failure) | 1 or 2 | 60 |
| `dynamic_cheatsheet_optional` | `dynamic_cheatsheet_generate`, `dynamic_cheatsheet_curate` | 2 | 108 |
| **Total** | — | — | **222** |

Reflexion's extra 6 calls are the failure-only reflections at `game24_pilot_001`.

---

## Contamination, filter, and isolation contract

- State is keyed by `(run_id, task_name, baseline, arm, backbone)`. No entry, source ID, or trial lineage crosses this key.
- `clean` seeds contain only the paired clean record.
- `contaminated` seeds contain the paired clean record plus the paired corrupted record.
- `contaminated_filter` starts from the contaminated seed set and applies `drop_known_contaminated`, keeping the paired clean record and logging the drop count.
- Contamination exposure is determined from `memory_before` structured records, not from retrieval fields.
- Generated descendants are marked contaminated only when their prompt-visible ancestry includes a contaminated entry. Filtered/clean descendants remain clean.
- Model-visible rendered `content` never contains catalog source labels, contamination labels, verifier specs, expected/gold values, or verifier reasons. Structured `memory_before`/`memory_after` retain `clean_or_contaminated`, source IDs, and lineage metadata for exposure/audit logic.

---

## Adaptations and deviations

- `full_history` is verbatim append-only transcript memory. The harness task/verifier interfaces replace any benchmark-specific scaffolding.
- `reflexion_style` keeps the official failure-gated generation/reflection loop and latest-three read window, but uses the harness task format, verifier, and replay fixtures.
- `dynamic_cheatsheet_optional` keeps the official DC-Cu generate/curate loop and conservative `<cheatsheet>` replacement, but removes code-execution instructions, provider-specific tool requests, retrieval variants, and web-search paths. It uses the harness task format and verifier.

---

## Deterministic edge fixtures

The canonical replay fixture pins two edge cases to `game24_pilot_001`:

- Its `reflexion_generate` answer is verifier-invalid across the 3 arms × 2 models, producing exactly 6 `reflexion_reflect` calls.
- Its `dynamic_cheatsheet_curate` response omits the required `<cheatsheet>` tags across the same 6 trials, producing exactly 6 `preserved_missing_tag` rows.

These six occurrences each prove failure-only reflection and conservative cheatsheet preservation.

---

## Verification commands and results

Run from the repository root:

```bash
python -m memcontam.cli validate-config configs/g0_fh_reflexion_dc_faithful_replay.yaml
python -m pytest tests/test_full_history_faithful.py tests/test_reflexion_faithful.py tests/test_dynamic_cheatsheet_faithful.py tests/test_cli_run.py tests/test_contamination_catalog.py tests/test_logging_schema.py tests/test_replay_client.py tests/test_replay_fixtures.py tests/test_aggregate.py tests/test_docs_scope.py -q
python -m pytest tests/test_task_verifiers.py tests/test_cli_run.py tests/test_contamination_catalog.py tests/test_openai_compatible_client.py tests/test_aggregate.py tests/test_docs_scope.py -q
python -m ruff check src tests scripts
python -m memcontam.cli run configs/g0_fh_reflexion_dc_faithful_replay.yaml --run-id g0_fh_reflexion_dc_faithful_replay
python -m memcontam.cli aggregate runs/g0_fh_reflexion_dc_faithful_replay
python scripts/inspect_g0_fh_reflexion_dc_fidelity.py runs/g0_fh_reflexion_dc_faithful_replay
```

Expected results:

- Config validates.
- All focused tests pass.
- Ruff reports no errors.
- Replay writes `runs/g0_fh_reflexion_dc_faithful_replay/trials.jsonl` with 162 rows.
- Aggregate emits valid JSON with method-call totals matching the table above.
- Inspector reports overall pass.

The canonical Task 9 evidence was captured with:

- Config SHA256: `42c5e736d833356216c9a5d3709d7ccc027de25e82456b8307ea6678f354a9a5`
- Corpus: `data/memory/catalog_v2.jsonl`, version `memory_catalog_v2`
- Fixture: `data/replay/g0_fh_reflexion_dc_faithful_v1.yaml`, version `v1`

---

## Limitations and non-claims

- This gate covers `full_history`, `reflexion_style`, and `dynamic_cheatsheet_optional` only. `no_memory`, `retrieval_rag`, `bot_style`, and `expel_optional` are not claimed to pass this slice.
- This is an adapted baseline gate, not a full paper reproduction.
- The replay output is a fidelity/QA artifact, not benchmark/manuscript evidence.
- No formal admission control proof, independence from backbone choice, or all-baseline pass is claimed.
- DC is an optional appendix comparator, not a new main baseline.

---

## Artifact file list

- `configs/g0_fh_reflexion_dc_faithful_replay.yaml`
- `data/memory/catalog_v2.jsonl`
- `data/replay/g0_fh_reflexion_dc_faithful_v1.yaml`
- `scripts/inspect_g0_fh_reflexion_dc_fidelity.py`
- `src/memcontam/baselines/full_history.py`
- `src/memcontam/baselines/reflexion_style.py`
- `src/memcontam/baselines/dynamic_cheatsheet_optional.py`
- `src/memcontam/cli.py`
- `tests/test_full_history_faithful.py`
- `tests/test_reflexion_faithful.py`
- `tests/test_dynamic_cheatsheet_faithful.py`
- `tests/test_cli_run.py`
- `tests/test_contamination_catalog.py`
- `tests/test_logging_schema.py`
- `tests/test_aggregate.py`
- `tests/test_replay_client.py`
- `tests/test_replay_fixtures.py`
- `tests/test_docs_scope.py`
- `tests/test_g0_fh_reflexion_dc_inspector.py`
- `docs/g0-baseline-fidelity-gate-v0.5.md`
- `README.md`

---

## Backward compatibility

The v0.4 faithful RAG/BoT gate and the v0.2 multitask replay demo still run without API keys. See [`docs/g0-baseline-fidelity-gate-v0.4.md`](g0-baseline-fidelity-gate-v0.4.md) for the RAG/BoT report and [`docs/g0-baseline-fidelity-gate-v0.2.md`](g0-baseline-fidelity-gate-v0.2.md) for the pre-implementation gap analysis.
