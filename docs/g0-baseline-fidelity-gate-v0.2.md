# G0 Baseline Fidelity Gate — v0.2 Analysis

> For the v0.3 implementation result, see [`g0-baseline-fidelity-gate-v0.3.md`](g0-baseline-fidelity-gate-v0.3.md).

**Repo:** `memory-contamination-diagnostic`  
**Baseline date:** 2026-07-09  
**Scope:** v0.2 multitask replay harness (`docs/replay-qa-demo-v0.2.md`)  
**Gate purpose:** Decide whether the current baseline implementations are faithful adapted baselines or still stub/proxy prompt labels. Gate must pass before E1/E2 main experiments are run.

---

## 1. What G0 checks

G0 asks: *Do the baselines preserve the core read/write/update mechanisms of the papers/repos they claim to represent?*  
If a baseline is only a prompt header (e.g., `Reflections: {last 3 memory entries}`), it fails G0 even if the runner logs trial rows cleanly.

This is a **non-experiment gate**. Passing G0 does not produce a benchmark result; it unlocks the next gates (G1–G3) and the actual experiments (E1–E6) by ensuring the independent variables (baseline families) are methodologically sound.

---

## Implementation scope guardrails

**This implementation plan targets a partial G0 pass: RAG + BoT only.**

The current sprint raises `retrieval_rag` and `bot_style` from prompt-label proxies to faithful adapted baselines. The following baselines are explicitly **out of implementation scope for this plan** and remain in their current proxy or placeholder state:

- `no_memory` — unchanged one-line prompt; no fidelity work.
- `full_history` — unchanged static content dump; no transcript append or context tracking.
- `reflexion_style` — unchanged last-3-entry header; no failure-triggered reflection generation.
- `Dynamic Cheatsheet` — placeholder only; no generator/curator/cheatsheet loop.
- `ExpeL` — placeholder only; no experience pool or insight extraction.

This plan does **not** claim exact or full reproduction of any paper or official repo. It adapts the minimal mechanisms needed for a RAG + BoT partial G0 pass inside the v0.2 replay harness.

### Verification commands for this scope

Run from the repository root after any baseline or runner change:

```bash
python -m memcontam.cli validate-config configs/pilot_multitask_replay.yaml
python -m pytest tests/test_task_verifiers.py tests/test_cli_run.py tests/test_contamination_catalog.py tests/test_openai_compatible_client.py tests/test_aggregate.py tests/test_logging_schema.py -q
python -m memcontam.cli run configs/pilot_multitask_replay.yaml --run-id <safe-single-path-component>
python -m memcontam.cli aggregate runs/<safe-single-path-component>
```

The replay config emits 90 trial rows and requires no API keys.

---

## 2. v0.2 baseline inventory

`src/memcontam/cli.py:73-79` registers five main baselines and two optional placeholders:

| Baseline file | Current role | Lines |
|---|---|---|
| `src/memcontam/baselines/no_memory.py` | Ignores memory; one-line prompt | 7-9 |
| `src/memcontam/baselines/full_history.py` | Concatenates all memory entries as `History:` | 7-10 |
| `src/memcontam/baselines/retrieval_rag.py` | Lexical top-3 retrieval as `Retrieved memory:` | 8-12 |
| `src/memcontam/baselines/reflexion_style.py` | Injects last 3 entries as `Reflections:` | 7-10 |
| `src/memcontam/baselines/bot_style.py` | Lexical top-1 retrieval as `Thought template:` | 8-12 |
| `src/memcontam/baselines/dynamic_cheatsheet_optional.py` | Placeholder docstring only | 1 |
| `src/memcontam/baselines/expel_optional.py` | Placeholder docstring only | 1 |

The README already states the v0.2 claim:

> retrieval-only RAG lower-bound, Reflexion-style verbal memory proxy, BoT-style thought-template proxy; not full reproduction.  
> — `README.md:9-11`

That claim is consistent with the code, but G0 requires raising the three memory baselines from *proxy* to *faithful adapted baseline* before they can support main quantitative claims.

---

## 3. Runner-level facts that constrain every baseline

The CLI runner in `src/memcontam/cli.py:274-383` currently:

1. Builds a fresh `MemoryState` per trial from the contamination catalog (`_memory_entries_for_arm`, 124-148).
2. Passes it to the baseline policy.
3. **Never mutates memory afterward.**
4. Always writes `memory_write_event=None` (`cli.py:374-376`).
5. Writes `memory_after` as a static dump of the initial `memory.entries` (`cli.py:376`).

Consequences:
- No baseline can accumulate state across trials.
- `source_trial_id` on `MemoryEntry` (`memory/stores.py:6-16`) is never populated.
- `memory_write_event` is schema-only; aggregate.py looks for it but never sees a real one in production runs.
- Repeated-failure tracking works per trial identity, but it is not tied to memory content.

These are **cross-cutting gaps** that must be fixed for any memory baseline to pass G0.

---

## 4. Per-baseline fidelity gap

### 4.1 no_memory

**Current code:** `build_prompt` ignores memory entirely.

```python
class NoMemoryPolicy:
    def build_prompt(self, task: TaskInstance, memory: MemoryState) -> list[dict[str, str]]:
        return [{"role": "user", "content": f"Solve this {task.task_name} instance: {task.input}"}]
```

**Missing for a faithful no-memory baseline:**
- Task-specific answer format / parser contract so the verifier can parse the final answer deterministically.
- A system prompt that clarifies whether the model should answer directly or use chain-of-thought.
- No-memory is otherwise the closest to its intended behavior; the main gap is **prompt/formalization**, not a missing mechanism.

### 4.2 full_history

**Current code:** joins all `memory.entries.content` under `History:`.

```python
history = "\n".join(entry.content for entry in memory.entries)
return [{"role": "user", "content": f"History:\n{history}\n\nSolve: {task.input}"}]
```

**Official / repo reference:** Dynamic Cheatsheet paper §2.3 defines *Full-History Appending (FH)* as concatenating all prior input-output pairs verbatim. That is closer to the current code than the DC variants, but even FH should append real trial transcripts.

**Missing mechanisms:**
- Append-only interaction history (current memory is a static catalog slice, not prior trials).
- Record of prompt, raw response, parsed answer, and verifier result per trial.
- Order/seed preservation across trials.
- Context-length tracking and truncation policy.
- Clear role formatting (user/assistant) instead of raw content dump.

### 4.3 retrieval_rag

**Current code:** lexical Jaccard-like retrieval from `memory.entries`.

```python
retrieved = lexical_retrieve(str(task.input), memory.entries)
context = "\n".join(entry.content for entry, _score in retrieved)
return [{"role": "user", "content": f"Retrieved memory:\n{context}\n\nSolve: {task.input}"}]
```

`src/memcontam/memory/retrieval.py:8-19` implements the lexical scorer.

**Official / repo reference:**
- RAG (Lewis et al., NeurIPS 2020) — dense retriever + generator.
- Canonical implementation: ParlAI RAG (`facebookresearch/ParlAI`) with DPR/FAISS backend.
- DPR repo (`facebookresearch/DPR`) — bi-encoder query/document encoders, FAISS index, retrieval output with provenance.

**Missing mechanisms:**
- Corpus / index build (not a static list).
- Query encoder / retriever model.
- Dense or hybrid retrieval, not lexical overlap.
- Top-k document selection with logged scores.
- Passage provenance: `id`, `title`, `text`, `score`, `has_answer`.
- Retrieved-document-conditioned generation (the current baseline just prepends text).

### 4.4 reflexion_style

**Current code:** injects the last 3 memory entries under `Reflections:`.

```python
reflections = "\n".join(entry.content for entry in memory.entries[-3:])
return [{"role": "user", "content": f"Reflections:\n{reflections}\n\nSolve: {task.input}"}]
```

**Official / repo reference:**
- Paper: *Reflexion: Language Agents with Verbal Reinforcement Learning* (Shinn et al., NeurIPS 2023).
- Repo: `github.com/noahshinn/reflexion`.
- Algorithm 1 in the paper: Actor generates trajectory → Evaluator scores → Self-Reflection model produces verbal feedback → memory append → next trial conditions on memory.

**Missing mechanisms:**
- Verifier/evaluator-driven failure signal (reflection only after failed attempts).
- Separate self-reflection generation step (a dedicated model call from failed trial + feedback).
- Persistent episodic memory across trials (current memory is static catalog, not accumulated reflections).
- Strategy variants: `NONE`, `LAST_ATTEMPT`, `REFLEXION`, `LAST_ATTEMPT_AND_REFLEXION`.
- Structured reflection formatting (headers/bullets, first-person lesson).
- Cross-trial reuse of past reflections on the next attempt.

### 4.5 bot_style

**Current code:** lexical top-1 retrieval under `Thought template:`.

```python
templates = lexical_retrieve(str(task.input), memory.entries, k=1)
template_text = templates[0][0].content if templates else ""
return [{"role": "user", "content": f"Thought template:\n{template_text}\n\nSolve: {task.input}"}]
```

**Official / repo reference:**
- Paper: *Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models* (Yang et al., NeurIPS 2024 Spotlight).
- Repo: `github.com/YangLing0818/buffer-of-thought-llm`.
- Local repo: `/home/hyunwoo/git/buffer-of-thought-llm`.

**Missing mechanisms:**
- Problem distillation first (extract key info, constraints, meta-problem).
- Persistent meta-buffer / template bank of high-level thought-templates.
- Semantic retrieval (embedding/LightRAG), not lexical top-1.
- Template instantiation: adapt retrieved template into task-specific reasoning/code.
- Buffer update after solving: distill solved trajectory into a new template.
- Novelty / similarity gate before inserting into buffer.
- Code/text dual-mode reasoning and optional self-correction loop.

### 4.6 dynamic_cheatsheet_optional

**Current code:** one-line docstring placeholder.

**Official / repo reference:**
- Paper: *Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory* (Suzgun et al., 2025).
- Repo: `github.com/suzgunmirac/dynamic-cheatsheet`.
- Local repo: `/home/hyunwoo/git/dynamic-cheatsheet`.

**Missing mechanisms (entire baseline):**
- Persistent cheatsheet state across queries.
- Generator + curator split.
- Cumulative update loop after each answer.
- Retrieval-synthesis variant (top-k similar past examples).
- Hybrid cumulative + retrieval variant.
- Full-history baseline distinction.
- Code-execution loop (`EXECUTE CODE!` + Python runner).
- Benchmark/resume/eval harness.

### 4.7 expel_optional

**Current code:** one-line docstring placeholder.

**Official / repo reference:**
- Paper: *ExpeL: LLM Agents Are Experiential Learners* (Zhao et al., AAAI 2024).
- Repo: `github.com/LeapLabTHU/ExpeL`.

**Missing mechanisms (entire baseline):**
- Success/failure experience pool from training tasks.
- Trial-and-error retry/reflection loop for experience gathering.
- Insight extraction from success/failure pairs and success lists.
- ADD / EDIT / UPVOTE / DOWNVOTE insight operations.
- Eval-time retrieval of similar past experiences (FAISS + sentence embedder).
- Prompt injection of insights + few-shot examples at evaluation.

---

## 5. Summary table

| Baseline | Current state | Faithful adapted baseline requires | G0 status |
|---|---|---|---|
| no_memory | Ignores memory | Task-specific answer format + parser contract | Minor gap |
| full_history | Static content dump | Append-only trial transcript, order/seed, context tracking | Major gap |
| retrieval_rag | Lexical top-3 retrieval | Dense retriever, index, provenance, score logging | Major gap |
| reflexion_style | Last-3 memory header | Failure-triggered reflection, memory append, reuse loop | Major gap |
| bot_style | Lexical top-1 template | Distill → retrieve → instantiate → update buffer | Major gap |
| dynamic_cheatsheet_optional | Placeholder | Full generator/curator/cheatsheet loop | Not implemented |
| expel_optional | Placeholder | Experience pool + insight extraction + eval retrieval | Not implemented |

---

## 6. Cross-cutting runner gaps

| Gap | Location | Why it blocks G0 |
|---|---|---|
| No memory mutation | `src/memcontam/cli.py:320-376` | Memory baselines cannot accumulate state, so they are prompt-label ablations. |
| `memory_write_event` always `None` | `src/memcontam/cli.py:374-376` | Schema supports provenance but no real write events are emitted. |
| `memory_after` == `memory_before` | `src/memcontam/cli.py:376` | No downstream observer can detect memory change. |
| `source_trial_id` unused | `src/memcontam/memory/stores.py:11` | Lineage from trial → memory entry is broken. |
| Lexical retrieval shared by RAG and BoT | `src/memcontam/memory/retrieval.py:8-19` | Both baselines use the same wrong retrieval mechanism. |
| Optional baselines not wired | `src/memcontam/baselines/*optional.py` | `BASELINE_POLICIES` excludes them; they are documentation-only. |

---

## 7. Reference mapping

| Method | Paper | Official repo | Local repo / notes |
|---|---|---|---|
| Reflexion | Shinn et al., NeurIPS 2023 | `github.com/noahshinn/reflexion` | `References/MDs/Reflexion Language Agents with Verbal Reinforcement Learning.md` |
| BoT | Yang et al., NeurIPS 2024 Spotlight | `github.com/YangLing0818/buffer-of-thought-llm` | `/home/hyunwoo/git/buffer-of-thought-llm` |
| Dynamic Cheatsheet | Suzgun et al., 2025 | `github.com/suzgunmirac/dynamic-cheatsheet` | `/home/hyunwoo/git/dynamic-cheatsheet` |
| ExpeL | Zhao et al., AAAI 2024 | `github.com/LeapLabTHU/ExpeL` | `References/Paper cards/ExpeL - LLM Agents Are Experiential Learners.md` |
| RAG / DPR | Lewis et al., NeurIPS 2020; Karpukhin et al., EMNLP 2020 | `facebookresearch/ParlAI`, `facebookresearch/DPR` | — |

---

## 8. Suggested G0 pass criteria

For the **main five baselines**, G0 passes when:

1. **no_memory**: deterministic answer format and parser; memory remains empty.
2. **full_history**: each trial appends its own (prompt, response, verifier summary) to memory; `memory_after` differs from `memory_before` with a logged `memory_write_event`.
3. **retrieval_rag**: explicit retriever/index, logged top-k results with ids/scores/text, and generation conditioned on retrieved docs.
4. **reflexion_style**: failed trials trigger a reflection-generation step; reflections are appended and reused on later attempts; `memory_write_event` records the parent trial and source entry ids.
5. **bot_style**: problem distillation, semantic template retrieval, template instantiation, and post-solution buffer update with novelty gating.

For the **optional baselines**, G0 is satisfied by documenting them as `not_implemented` and keeping them out of `BASELINE_POLICIES` until they are implemented.

---

## 9. Downstream implications

- **E1 (Game24 contamination uptake pilot)** should not run until G0 passes for at least `reflexion_style`, `bot_style`, and `retrieval_rag`; otherwise contamination signal is confounded with prompt-label noise.
- **E2 (Main 3-task diagnostic)** requires all five main baselines to pass G0.
- **G1 (Related Work / baseline eligibility)** can proceed in parallel, but its evidence table should mark current implementations as `proxy` until G0 passes.
- **G2 (Task/data/contamination catalog freeze)** can proceed; catalog does not depend on baseline fidelity.

---

## 10. Open decisions

1. Should `full_history` be implemented as verbatim transcript append (FH baseline in DC paper) or as a summarized interaction log?
2. Should `retrieval_rag` use a sentence embedding retriever or a small task-specific BM25 index?
3. Should `reflexion_style` generate reflections via a separate LLM call, or can a deterministic heuristic produce reflection text from verifier feedback?
4. Should `bot_style` initialize the meta-buffer with hand-written task templates (as in the official repo’s `math.txt` and `test_templates.py`) or build it from scratch during the pilot?
5. Should optional baselines be implemented at all for the main paper, or remain appendix-only?

---

## 11. Open decision answers

### D1. Full-history implementation

**Decision:** `full_history`는 **verbatim transcript append**로 구현한다.

- 요약 log가 아니라, prior trial의 prompt / response / parsed answer / verifier result를 그대로 누적한다.
- FH baseline의 의미를 유지하기 위해 summarization mechanism은 넣지 않는다.

---

### D2. Retrieval RAG implementation

**Decision:** `retrieval_rag`는 **sentence embedding retriever**를 사용한다.

- BM25는 main baseline이 아니라 **pilot / ablation only**로 둔다.
- Main RAG baseline은 lexical proxy가 아니라 semantic retrieval 기반이어야 한다.
- top-k retrieved entries의 id, score, text, provenance를 로그에 남긴다.

---

### D3. Reflexion-style implementation

**Decision:** `reflexion_style`는 **separate LLM reflection call**로 구현한다.

- deterministic heuristic reflection은 **proxy only**로 둔다.
- 실패 trial 이후 evaluator/verifier feedback을 바탕으로 별도 reflection generation step을 수행한다.
- 생성된 reflection은 persistent memory에 append하고 이후 trial에서 재사용한다.

---

### D4. BoT-style meta-buffer initialization

**Decision:** `bot_style`의 meta-buffer는 **from scratch**로 시작한다.

- hand-written task templates로 초기화하지 않는다.
- BoT 논문의 template 예시는 생성된 template의 예시로 보고, seed template로 사용하지 않는다.
- pilot 중 problem distillation → template creation/update → retrieval/instantiation 흐름으로 buffer를 구축한다.

---

### D5. Optional baselines

**Decision:** `dynamic_cheatsheet_optional`과 `expel_optional`은 main paper에서 구현하지 않고 **appendix-only / not implemented**로 둔다.

- Main baseline은 다음 5개로 고정한다:
  1. `no_memory`
  2. `full_history`
  3. `retrieval_rag`
  4. `reflexion_style`
  5. `bot_style`
- DC / ExpeL은 Related Work 또는 Limitations에서 다루되, placeholder result는 포함하지 않는다.

---

## 12. Updated G0 target

| Baseline | G0 target |
|---|---|
| `no_memory` | Deterministic answer format + parser contract |
| `full_history` | Verbatim transcript append |
| `retrieval_rag` | Sentence-embedding retrieval; BM25 only for pilot/ablation |
| `reflexion_style` | Separate LLM reflection call; heuristic only as proxy |
| `bot_style` | From-scratch meta-buffer construction |
| `dynamic_cheatsheet_optional` | Appendix-only / not implemented |
| `expel_optional` | Appendix-only / not implemented |

E1 should wait until `retrieval_rag`, `reflexion_style`, and `bot_style` meet this G0 target. E2 should wait until all five main baselines meet the target.

---

## 13. Implementation result and verification evidence

This implementation plan produced a **partial G0 pass for `retrieval_rag` and `bot_style` only**. The canonical replay evidence run uses:

- Config: `configs/pilot_game24.yaml`
- Run ID: `g0_rag_bot_gate_replay`
- Inspection script: `.sisyphus/evidence/inspect_g0_replay.py`

The inspection script validates that the replay artifact at `runs/g0_rag_bot_gate_replay/trials.jsonl` contains `TrialLog` rows with RAG retrieval provenance (`retrieved_memory` and `retrieved_scores`) and BoT writeback lineage (`memory_write_event` plus a changed `memory_after`).

The following baselines remain **out of scope** for this G0 implementation slice and are not claimed to pass G0:

- `no_memory`
- `full_history`
- `reflexion_style`
- `Dynamic Cheatsheet`
- `ExpeL`

This is an adapted baseline gate, not an exact or full reproduction of any paper or official repository.

### Exact verification commands

Run from the repository root after any baseline or runner change:

```bash
python -m memcontam.cli validate-config configs/pilot_game24.yaml
python -m pytest tests/test_cli_run.py tests/test_aggregate.py tests/test_logging_schema.py tests/test_docs_scope.py -q
python -m memcontam.cli run configs/pilot_game24.yaml --run-id g0_rag_bot_gate_replay
python -m memcontam.cli aggregate runs/g0_rag_bot_gate_replay
python .sisyphus/evidence/inspect_g0_replay.py
```

The replay run emits Game24 pilot trial rows and requires no API keys or live model access. These rows are QA evidence for the G0 gate, not benchmark results.