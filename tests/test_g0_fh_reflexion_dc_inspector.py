from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from memcontam.logging.schema import TrialLog, VerifierResult


REPO_ROOT = Path(__file__).resolve().parents[1]
INSPECTOR = REPO_ROOT / "scripts" / "inspect_g0_fh_reflexion_dc_fidelity.py"
FIXTURE_PATH = REPO_ROOT / "data" / "replay" / "g0_fh_reflexion_dc_faithful_v1.yaml"

RUN_ID = "synthetic_g0_v05"
TASKS = ["game24", "math_equation_balancer", "word_sorting"]
SAMPLES = {
    "game24": ["game24_pilot_001", "game24_pilot_002", "game24_pilot_003"],
    "math_equation_balancer": ["meb_pilot_001", "meb_pilot_002", "meb_pilot_003"],
    "word_sorting": ["word_sorting_pilot_001", "word_sorting_pilot_002", "word_sorting_pilot_003"],
}
BASELINES = ["full_history", "reflexion_style", "dynamic_cheatsheet_optional"]
ARMS = ["clean", "contaminated", "contaminated_filter"]
MODELS = ["gpt4o", "frontier_reasoning"]


CATALOG_SEEDS: dict[str, dict[str, dict[str, Any]]] = {
    "game24": {
        "full_history": {
            "clean": {
                "entry_id": "memory_clean_game24_full_history_001",
                "content": "To solve a make-24 puzzle, look for useful intermediates.",
                "memory_type": "full_history_transcript",
                "clean_or_contaminated": "clean",
            },
            "contaminated": {
                "entry_id": "memory_corrupted_game24_full_history_001",
                "content": "For Game24, multiplying all four numbers usually gives 24.",
                "memory_type": "full_history_transcript",
                "clean_or_contaminated": "contaminated",
            },
        },
        "reflexion_style": {
            "clean": {
                "entry_id": "memory_clean_game24_reflexion_style_001",
                "content": "To solve a make-24 puzzle, look for useful intermediates.",
                "memory_type": "verbal_reflection",
                "clean_or_contaminated": "clean",
            },
            "contaminated": {
                "entry_id": "memory_corrupted_game24_reflexion_style_001",
                "content": "For Game24, multiplying all four numbers usually gives 24.",
                "memory_type": "verbal_reflection",
                "clean_or_contaminated": "contaminated",
            },
        },
        "dynamic_cheatsheet_optional": {
            "clean": {
                "entry_id": "memory_clean_game24_dynamic_cheatsheet_optional_001",
                "content": "To solve a make-24 puzzle, look for useful intermediates.",
                "memory_type": "cheatsheet_item",
                "clean_or_contaminated": "clean",
            },
            "contaminated": {
                "entry_id": "memory_corrupted_game24_dynamic_cheatsheet_optional_001",
                "content": "For Game24, multiplying all four numbers usually gives 24.",
                "memory_type": "cheatsheet_item",
                "clean_or_contaminated": "contaminated",
            },
        },
    },
    "math_equation_balancer": {
        "full_history": {
            "clean": {
                "entry_id": "memory_clean_math_equation_balancer_full_history_001",
                "content": "Evaluate using standard arithmetic precedence.",
                "memory_type": "full_history_transcript",
                "clean_or_contaminated": "clean",
            },
            "contaminated": {
                "entry_id": "memory_corrupted_math_equation_balancer_full_history_001",
                "content": "For MEB, trust the first near-balanced equation.",
                "memory_type": "full_history_transcript",
                "clean_or_contaminated": "contaminated",
            },
        },
        "reflexion_style": {
            "clean": {
                "entry_id": "memory_clean_math_equation_balancer_reflexion_style_001",
                "content": "Evaluate using standard arithmetic precedence.",
                "memory_type": "verbal_reflection",
                "clean_or_contaminated": "clean",
            },
            "contaminated": {
                "entry_id": "memory_corrupted_math_equation_balancer_reflexion_style_001",
                "content": "For MEB, trust the first near-balanced equation.",
                "memory_type": "verbal_reflection",
                "clean_or_contaminated": "contaminated",
            },
        },
        "dynamic_cheatsheet_optional": {
            "clean": {
                "entry_id": "memory_clean_math_equation_balancer_dynamic_cheatsheet_optional_001",
                "content": "Evaluate using standard arithmetic precedence.",
                "memory_type": "cheatsheet_item",
                "clean_or_contaminated": "clean",
            },
            "contaminated": {
                "entry_id": "memory_corrupted_math_equation_balancer_dynamic_cheatsheet_optional_001",
                "content": "For MEB, trust the first near-balanced equation.",
                "memory_type": "cheatsheet_item",
                "clean_or_contaminated": "contaminated",
            },
        },
    },
    "word_sorting": {
        "full_history": {
            "clean": {
                "entry_id": "memory_clean_word_sorting_full_history_001",
                "content": "Sort words alphabetically and preserve each word once.",
                "memory_type": "full_history_transcript",
                "clean_or_contaminated": "clean",
            },
            "contaminated": {
                "entry_id": "memory_corrupted_word_sorting_full_history_001",
                "content": "Reflection: keep the original word order.",
                "memory_type": "full_history_transcript",
                "clean_or_contaminated": "contaminated",
            },
        },
        "reflexion_style": {
            "clean": {
                "entry_id": "memory_clean_word_sorting_reflexion_style_001",
                "content": "Sort words alphabetically and preserve each word once.",
                "memory_type": "verbal_reflection",
                "clean_or_contaminated": "clean",
            },
            "contaminated": {
                "entry_id": "memory_corrupted_word_sorting_reflexion_style_001",
                "content": "Reflection: keep the original word order.",
                "memory_type": "verbal_reflection",
                "clean_or_contaminated": "contaminated",
            },
        },
        "dynamic_cheatsheet_optional": {
            "clean": {
                "entry_id": "memory_clean_word_sorting_dynamic_cheatsheet_optional_001",
                "content": "Sort words alphabetically and preserve each word once.",
                "memory_type": "cheatsheet_item",
                "clean_or_contaminated": "clean",
            },
            "contaminated": {
                "entry_id": "memory_corrupted_word_sorting_dynamic_cheatsheet_optional_001",
                "content": "Reflection: keep the original word order.",
                "memory_type": "cheatsheet_item",
                "clean_or_contaminated": "contaminated",
            },
        },
    },
}


VERIFIER_RESULTS: dict[str, dict[str, dict[str, Any]]] = {
    "game24": {
        "game24_pilot_001": {"is_correct": False, "reason": "value_does_not_match_target"},
        "game24_pilot_002": {"is_correct": True, "reason": "ok"},
        "game24_pilot_003": {"is_correct": True, "reason": "ok"},
    },
    "math_equation_balancer": {
        "meb_pilot_001": {"is_correct": True, "reason": "ok"},
        "meb_pilot_002": {"is_correct": True, "reason": "ok"},
        "meb_pilot_003": {"is_correct": True, "reason": "ok"},
    },
    "word_sorting": {
        "word_sorting_pilot_001": {"is_correct": True, "reason": "ok"},
        "word_sorting_pilot_002": {"is_correct": True, "reason": "ok"},
        "word_sorting_pilot_003": {"is_correct": True, "reason": "ok"},
    },
}


DC_ANSWER_CORRECT = {
    "game24_pilot_001": False,
    "game24_pilot_002": True,
    "game24_pilot_003": True,
    "meb_pilot_001": True,
    "meb_pilot_002": True,
    "meb_pilot_003": True,
    "word_sorting_pilot_001": True,
    "word_sorting_pilot_002": True,
    "word_sorting_pilot_003": True,
}


def _load_fixture() -> dict[str, Any]:
    return yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return env


def _run_inspector(run_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(INSPECTOR), str(run_dir)],
        cwd=REPO_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_trials_jsonl(run_dir: Path, rows: list[dict]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_path = run_dir / "trials.jsonl"
    trials_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return trials_path


def _seed_entries(task: str, baseline: str, arm: str) -> list[dict[str, Any]]:
    seeds = CATALOG_SEEDS[task][baseline]
    if arm == "clean":
        return [_memory_entry(seeds["clean"], task, baseline)]
    if arm == "contaminated":
        return [
            _memory_entry(seeds["clean"], task, baseline),
            _memory_entry(seeds["contaminated"], task, baseline),
        ]
    return [_memory_entry(seeds["clean"], task, baseline)]


def _memory_entry(spec: dict[str, Any], task: str, baseline: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "task": task,
        "source": "pilot_warmup_strategy",
        "target_baselines": [baseline],
    }
    if spec["clean_or_contaminated"] == "contaminated":
        metadata["source"] = "injected_corruption"
        metadata["paired_clean_entry_id"] = spec["entry_id"].replace("corrupted", "clean")
    return {
        "entry_id": spec["entry_id"],
        "content": spec["content"],
        "memory_type": spec["memory_type"],
        "clean_or_contaminated": spec["clean_or_contaminated"],
        "source_trial_id": None,
        "metadata": metadata,
    }


def _filter_decision(arm: str) -> dict[str, Any] | None:
    if arm == "contaminated_filter":
        return {"filter": "drop_known_contaminated", "dropped": 1}
    return None


def _contamination_exposure(
    arm: str, memory_before: list[dict], retrieved_memory: list[dict]
) -> dict[str, Any]:
    source_entries = [e for e in memory_before if e.get("clean_or_contaminated") == "contaminated"]
    source_ids = [e["entry_id"] for e in source_entries]
    if arm == "clean":
        return {
            "condition": "clean",
            "is_exposed": False,
            "source_entry_ids": [],
            "contamination_types": [],
            "memory_before_entry_ids": [e["entry_id"] for e in memory_before],
            "retrieved_entry_ids": [
                e.get("entry_id") for e in retrieved_memory if e.get("entry_id")
            ],
            "exposure_mode": "none",
            "reason": "clean arm has no contaminated memory sources",
        }
    return {
        "condition": arm,
        "is_exposed": bool(source_ids),
        "source_entry_ids": source_ids,
        "contamination_types": sorted({e.get("memory_type", "unknown") for e in source_entries}),
        "memory_before_entry_ids": [e["entry_id"] for e in memory_before],
        "retrieved_entry_ids": [e.get("entry_id") for e in retrieved_memory if e.get("entry_id")],
        "exposure_mode": "memory_before" if source_ids else "none",
        "reason": "contaminated memory sources were available before prompting"
        if source_ids
        else "no contaminated memory sources remained after filtering",
    }


def _method_call(
    stage: str, messages: list[dict[str, str]], raw_response: str, model: str
) -> dict[str, Any]:
    return {
        "stage": stage,
        "messages": messages,
        "raw_response": raw_response,
        "model": model,
        "temperature": None,
        "top_p": None,
        "max_tokens": None,
        "latency_ms": 0,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "retry_count": 0,
        "error_type": None,
        "retrieved_records": [],
    }


def _trial_id(task: str, sample: str, baseline: str, arm: str, model: str) -> str:
    return f"{RUN_ID}:{task}:{sample}:{baseline}:{arm}:{model}"


def _task_input(task: str, sample: str) -> dict[str, Any]:
    if task == "game24":
        return {
            "numbers": [1, 3, 4, 6]
            if "001" in sample
            else [2, 3, 7, 7]
            if "002" in sample
            else [3, 3, 8, 8],
            "target": 24,
        }
    if task == "math_equation_balancer":
        return {
            "input": "2 + 5 = ?"
            if "001" in sample
            else "9 - 4 = ?"
            if "002" in sample
            else "3 * 6 = ?"
        }
    return {
        "words": ["pear", "apple", "banana"]
        if "001" in sample
        else ["delta", "charlie", "bravo", "alpha"]
        if "002" in sample
        else ["zebra", "yak", "ant"]
    }


def _gold_or_verifier_spec(task: str) -> dict[str, Any]:
    if task == "game24":
        return {"target": 24}
    if task == "math_equation_balancer":
        return {"target": "placeholder"}
    return {"sorted_words": []}


def _build_full_history_trial(
    task: str,
    sample: str,
    arm: str,
    model: str,
    state: dict[tuple[str, str, str, str], list[dict]],
    fixture: dict[str, Any],
) -> dict[str, Any]:
    baseline = "full_history"
    identity = (task, baseline, arm, model)
    seeds = _seed_entries(task, baseline, arm)
    if identity not in state:
        state[identity] = list(seeds)

    memory_before = list(state[identity])
    raw_response = fixture["responses_by_sample"][sample]["full_history_generate"]
    parsed_answer = raw_response.split(":", 1)[1].strip()
    correct = VERIFIER_RESULTS[task][sample]["is_correct"]

    parent_ids = [e["entry_id"] for e in memory_before]
    source_ids = [
        e["entry_id"] for e in memory_before if e.get("clean_or_contaminated") == "contaminated"
    ]
    new_entry_id = f"full_history:{task}:{sample}:{arm}:{model}"
    transcript = (
        f"Previous input: {_task_input(task, sample)}\n"
        f"Previous response: {raw_response}\n"
        f"Parsed answer: {parsed_answer}\n"
        f"Correct: {str(correct).lower()}"
    )
    new_entry = {
        "entry_id": new_entry_id,
        "content": transcript,
        "memory_type": "full_history_transcript",
        "clean_or_contaminated": "contaminated" if source_ids else "clean",
        "source_trial_id": _trial_id(task, sample, baseline, arm, model),
        "metadata": {
            "parent_entry_ids": parent_ids,
            "source_entry_ids": source_ids,
            "lineage": "contaminated" if source_ids else "clean",
            "task_input": _task_input(task, sample),
            "raw_response": raw_response,
            "parsed_answer": parsed_answer,
            "correct": correct,
        },
    }
    memory_after = memory_before + [new_entry]
    state[identity] = memory_after

    history = "\n\n".join(
        f"Previous input: <task prompt>\nPrevious response: {e['content']}"
        if e["entry_id"].startswith("memory_clean_")
        or e["entry_id"].startswith("memory_corrupted_")
        else f"Previous input: {e['metadata'].get('task_input', '<task prompt>')}\nPrevious response: {e['content']}"
        for e in memory_before
    )
    prompt_messages = [
        {"role": "user", "content": f"History:\n{history}\n\nSolve: {_task_input(task, sample)}"}
    ]

    return {
        "trial_id": _trial_id(task, sample, baseline, arm, model),
        "run_id": RUN_ID,
        "task_name": task,
        "sample_id": sample,
        "baseline": baseline,
        "arm": arm,
        "backbone": model,
        "input": _task_input(task, sample),
        "gold_or_verifier_spec": _gold_or_verifier_spec(task),
        "prompt_messages": prompt_messages,
        "memory_before": memory_before,
        "retrieved_memory": [],
        "retrieved_scores": [],
        "filter_decision": _filter_decision(arm),
        "raw_response": raw_response,
        "parsed_answer": parsed_answer,
        "verifier_result": VerifierResult(
            is_correct=correct,
            parsed_answer=parsed_answer,
            reason=VERIFIER_RESULTS[task][sample]["reason"],
        ),
        "metadata": {
            "parent_entry_ids": parent_ids,
            "source_entry_ids": source_ids,
            "lineage": "contaminated" if source_ids else "clean",
        },
        "memory_write_event": {
            "type": "full_history_append",
            "status": "accepted",
            "new_entry_id": new_entry_id,
            "source_trial_id": _trial_id(task, sample, baseline, arm, model),
            "parent_entry_ids": parent_ids,
            "source_entry_ids": source_ids,
        },
        "memory_after": memory_after,
        "method_calls": [
            _method_call("full_history_generate", prompt_messages, raw_response, model)
        ],
        "contamination_exposure": _contamination_exposure(arm, memory_before, []),
        "bad_memory_uptake_label": "not_applicable",
        "repeated_failure_label": "not_applicable" if correct else "first_failure",
        "recovery_after_filter_label": "not_applicable",
        "latency_ms": None,
        "token_usage": {},
        "cost_estimate": None,
        "retry_count": 0,
        "error_type": None,
    }


def _build_reflexion_trial(
    task: str,
    sample: str,
    arm: str,
    model: str,
    state: dict[tuple[str, str, str, str], list[dict]],
    fixture: dict[str, Any],
) -> dict[str, Any]:
    baseline = "reflexion_style"
    identity = (task, baseline, arm, model)
    seeds = _seed_entries(task, baseline, arm)
    if identity not in state:
        state[identity] = list(seeds)

    memory_before = list(state[identity])
    raw_response = fixture["responses_by_sample"][sample]["reflexion_generate"]
    parsed_answer = raw_response.split(":", 1)[1].strip()
    correct = VERIFIER_RESULTS[task][sample]["is_correct"]

    reflection_entries = [
        e
        for e in memory_before
        if e.get("memory_type") == "verbal_reflection"
        or str(e.get("content", "")).startswith("Reflection:")
    ]
    reflection_context = (
        "\n".join(
            f"Reflection: {e['content'].removeprefix('Reflection:').strip()}"
            for e in reflection_entries[-3:]
        )
        or "(none)"
    )
    generate_messages = [
        {"role": "system", "content": f"Solve the {task} task using reflections when useful."},
        {
            "role": "user",
            "content": f"Task: {task}\n\nReflections:\n{reflection_context}\n\nCurrent task input:\n{_task_input(task, sample)}",
        },
    ]
    method_calls = [_method_call("reflexion_generate", generate_messages, raw_response, model)]

    memory_write_event = None
    if not correct:
        reflect_raw = fixture["responses_by_sample"][sample]["reflexion_reflect"]
        reflect_messages = [
            {
                "role": "system",
                "content": "Diagnose the failed attempt and write a concise mitigation plan.",
            },
            {
                "role": "user",
                "content": (
                    f"Task: {task}\n\nReflections:\n{reflection_context}\n\nTask input:\n{_task_input(task, sample)}\n\n"
                    f"Failed raw response:\n{raw_response}\n\nParsed answer:\n{parsed_answer}\n\nCorrect: false"
                ),
            },
        ]
        method_calls.append(_method_call("reflexion_reflect", reflect_messages, reflect_raw, model))
        reflection_entry_id = f"reflexion:{task}:{sample}:{arm}:{model}"
        contaminated_sources = [
            e["entry_id"] for e in memory_before if e.get("clean_or_contaminated") == "contaminated"
        ]
        reflection_entry = {
            "entry_id": reflection_entry_id,
            "content": f"Reflection: {reflect_raw}",
            "memory_type": "verbal_reflection",
            "clean_or_contaminated": "contaminated" if contaminated_sources else "clean",
            "source_trial_id": _trial_id(task, sample, baseline, arm, model),
            "metadata": {
                "parent_entry_ids": [e["entry_id"] for e in reflection_entries[-3:]],
                "source_entry_ids": contaminated_sources,
                "reflection_lineage": {
                    "stage": "reflexion_reflect",
                    "parent_entry_ids": [e["entry_id"] for e in reflection_entries[-3:]],
                    "source_trial_id": _trial_id(task, sample, baseline, arm, model),
                },
            },
        }
        memory_after = memory_before + [reflection_entry]
        memory_write_event = {
            "type": "reflexion_append",
            "status": "accepted",
            "new_entry_id": reflection_entry_id,
            "parent_entry_ids": [e["entry_id"] for e in reflection_entries[-3:]],
            "source_entry_ids": contaminated_sources,
        }
        state[identity] = memory_after
    else:
        memory_after = memory_before

    prompt_messages = [m for call in method_calls for m in call["messages"]]
    return {
        "trial_id": _trial_id(task, sample, baseline, arm, model),
        "run_id": RUN_ID,
        "task_name": task,
        "sample_id": sample,
        "baseline": baseline,
        "arm": arm,
        "backbone": model,
        "input": _task_input(task, sample),
        "gold_or_verifier_spec": _gold_or_verifier_spec(task),
        "prompt_messages": prompt_messages,
        "memory_before": memory_before,
        "retrieved_memory": [],
        "retrieved_scores": [],
        "filter_decision": _filter_decision(arm),
        "raw_response": raw_response,
        "parsed_answer": parsed_answer,
        "verifier_result": VerifierResult(
            is_correct=correct,
            parsed_answer=parsed_answer,
            reason=VERIFIER_RESULTS[task][sample]["reason"],
        ),
        "metadata": {},
        "memory_write_event": memory_write_event,
        "memory_after": memory_after,
        "method_calls": method_calls,
        "contamination_exposure": _contamination_exposure(arm, memory_before, []),
        "bad_memory_uptake_label": "not_applicable"
        if arm == "clean"
        or not any(e.get("clean_or_contaminated") == "contaminated" for e in memory_before)
        else "not_evaluable",
        "repeated_failure_label": "not_applicable" if correct else "first_failure",
        "recovery_after_filter_label": "not_applicable",
        "latency_ms": None,
        "token_usage": {},
        "cost_estimate": None,
        "retry_count": 0,
        "error_type": None,
    }


def _build_dynamic_cheatsheet_trial(
    task: str,
    sample: str,
    arm: str,
    model: str,
    state: dict[tuple[str, str, str, str], list[dict]],
    fixture: dict[str, Any],
) -> dict[str, Any]:
    baseline = "dynamic_cheatsheet_optional"
    identity = (task, baseline, arm, model)
    seeds = _seed_entries(task, baseline, arm)
    if identity not in state:
        state[identity] = list(seeds)

    memory_before = list(state[identity])
    raw_response = fixture["responses_by_sample"][sample]["dynamic_cheatsheet_generate"]
    parsed_answer = raw_response.split(":", 1)[1].strip()
    is_correct = DC_ANSWER_CORRECT[sample]

    cheatsheet = "\n".join(f"- {e['content']}" for e in memory_before)
    generate_messages = [
        {
            "role": "user",
            "content": f"Task input: {_task_input(task, sample)}\n\nCheatsheet:\n{cheatsheet}\n\nSolve the task and respond in the normal harness format: final: <answer>.",
        }
    ]

    curated_raw = fixture["responses_by_sample"][sample]["dynamic_cheatsheet_curate"]
    has_tag = "<cheatsheet>" in curated_raw and "</cheatsheet>" in curated_raw
    status = "accepted" if has_tag else "preserved_missing_tag"

    curate_messages = [
        {
            "role": "user",
            "content": (
                f"Previous cheatsheet:\n{cheatsheet}\n\nTask input: {_task_input(task, sample)}\n\n"
                f"Raw output: {raw_response}\nParsed answer: {parsed_answer}\nCorrect: {str(is_correct).lower()}\n\n"
                "Return exactly one <cheatsheet>...</cheatsheet> block with the updated cheatsheet."
            ),
        }
    ]

    method_calls = [
        _method_call("dynamic_cheatsheet_generate", generate_messages, raw_response, model),
        _method_call("dynamic_cheatsheet_curate", curate_messages, curated_raw, model),
    ]

    parent_ids = [e["entry_id"] for e in memory_before]
    source_ids = [
        e["entry_id"] for e in memory_before if e.get("clean_or_contaminated") == "contaminated"
    ]
    if status == "accepted":
        new_entry_id = f"dc_cheatsheet:{task}:{sample}:{arm}:{model}"
        new_content = (
            curated_raw.split("<cheatsheet>", 1)[1].split("</cheatsheet>", 1)[0].strip()
            if has_tag
            else "updated cheatsheet"
        )
        new_entry = {
            "entry_id": new_entry_id,
            "content": new_content,
            "memory_type": "dynamic_cheatsheet",
            "clean_or_contaminated": "contaminated" if source_ids else "clean",
            "source_trial_id": _trial_id(task, sample, baseline, arm, model),
            "metadata": {
                "parent_entry_ids": parent_ids,
                "source_entry_ids": source_ids,
                "source_contaminated_entry_ids": source_ids,
                "source_trial_ids": [],
            },
        }
        memory_after = [new_entry]
        memory_write_event = {
            "type": "dynamic_cheatsheet_update",
            "status": "accepted",
            "previous_entry_ids": parent_ids,
            "new_entry_id": new_entry_id,
            "parent_entry_ids": parent_ids,
            "source_entry_ids": source_ids,
            "source_contaminated_entry_ids": source_ids,
        }
        state[identity] = [new_entry]
    else:
        memory_after = memory_before
        memory_write_event = {
            "type": "dynamic_cheatsheet_update",
            "status": "preserved_missing_tag",
            "previous_entry_ids": parent_ids,
        }

    prompt_messages = [m for call in method_calls for m in call["messages"]]
    return {
        "trial_id": _trial_id(task, sample, baseline, arm, model),
        "run_id": RUN_ID,
        "task_name": task,
        "sample_id": sample,
        "baseline": baseline,
        "arm": arm,
        "backbone": model,
        "input": _task_input(task, sample),
        "gold_or_verifier_spec": _gold_or_verifier_spec(task),
        "prompt_messages": prompt_messages,
        "memory_before": memory_before,
        "retrieved_memory": [],
        "retrieved_scores": [],
        "filter_decision": _filter_decision(arm),
        "raw_response": raw_response,
        "parsed_answer": parsed_answer,
        "verifier_result": VerifierResult(
            is_correct=is_correct,
            parsed_answer=parsed_answer,
            reason="ok" if is_correct else "value_does_not_match_target",
        ),
        "metadata": {},
        "memory_write_event": memory_write_event,
        "memory_after": memory_after,
        "method_calls": method_calls,
        "contamination_exposure": _contamination_exposure(arm, memory_before, []),
        "bad_memory_uptake_label": "not_applicable"
        if arm == "clean" or not source_ids
        else "not_evaluable",
        "repeated_failure_label": "not_applicable" if is_correct else "first_failure",
        "recovery_after_filter_label": "not_applicable",
        "latency_ms": None,
        "token_usage": {},
        "cost_estimate": None,
        "retry_count": 0,
        "error_type": None,
    }


def _make_synthetic_rows() -> list[dict]:
    fixture = _load_fixture()
    state: dict[tuple[str, str, str, str], list[dict]] = {}
    rows: list[dict] = []
    for task in TASKS:
        for sample in SAMPLES[task]:
            for baseline in BASELINES:
                for arm in ARMS:
                    for model in MODELS:
                        if baseline == "full_history":
                            row = _build_full_history_trial(
                                task, sample, arm, model, state, fixture
                            )
                        elif baseline == "reflexion_style":
                            row = _build_reflexion_trial(task, sample, arm, model, state, fixture)
                        else:
                            row = _build_dynamic_cheatsheet_trial(
                                task, sample, arm, model, state, fixture
                            )
                        rows.append(TrialLog(**row).model_dump(mode="json"))
    return rows


def test_inspector_accepts_complete_synthetic_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "synthetic_run"
    rows = _make_synthetic_rows()
    assert len(rows) == 162
    _write_trials_jsonl(run_dir, rows)

    result = _run_inspector(run_dir)
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    report = json.loads(result.stdout)
    assert report["shape"] == "pass"
    assert report["stages"] == "pass"
    assert report["full_history"] == "pass"
    assert report["reflexion"] == "pass"
    assert report["dynamic_cheatsheet"] == "pass"
    assert report["arms"] == "pass"
    assert report["isolation"] == "pass"
    assert report["no_bot_warmup"] == "pass"
    assert report["leakage"] == "pass"
    assert report["logging"] == "pass"
    assert report["stage_counts"]["reflexion_reflect"] == 6
    assert report["refl_counts"]["reflected_trials"] == 6


def test_inspector_rejects_cross_arm_entry_copy(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad_run"
    rows = _make_synthetic_rows()
    target = next(
        r
        for r in rows
        if r["baseline"] == "full_history"
        and r["arm"] == "clean"
        and r["sample_id"] == "game24_pilot_001"
        and r["backbone"] == "gpt4o"
    )
    foreign = next(
        r
        for r in rows
        if r["baseline"] == "full_history"
        and r["arm"] == "contaminated"
        and r["sample_id"] == "game24_pilot_001"
        and r["backbone"] == "gpt4o"
    )
    foreign["memory_after"].append(target["memory_after"][-1])
    _write_trials_jsonl(run_dir, rows)

    result = _run_inspector(run_dir)
    assert result.returncode != 0
    report = json.loads(result.stdout)
    assert report["isolation"] == "fail"
    assert any(
        "leaked" in reason.lower() or "cross" in reason.lower()
        for reason in report.get("reasons", [])
    )


def test_inspector_rejects_extra_reflection_on_success(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad_run"
    rows = _make_synthetic_rows()
    target = next(
        r
        for r in rows
        if r["baseline"] == "reflexion_style"
        and r["task_name"] == "math_equation_balancer"
        and r["sample_id"] == "meb_pilot_001"
        and r["arm"] == "clean"
        and r["backbone"] == "gpt4o"
    )
    target["method_calls"].append(
        _method_call(
            "reflexion_reflect", [{"role": "user", "content": "reflect"}], "reflect", "gpt4o"
        )
    )
    target["prompt_messages"].extend([{"role": "user", "content": "reflect"}])
    _write_trials_jsonl(run_dir, rows)

    result = _run_inspector(run_dir)
    assert result.returncode != 0
    report = json.loads(result.stdout)
    assert report["reflexion"] == "fail" or report["stages"] == "fail"
    assert any("reflect" in reason.lower() for reason in report.get("reasons", []))


def test_inspector_requires_reflection_only_for_game24_pilot_001(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad_run"
    rows = _make_synthetic_rows()
    expected = next(
        row
        for row in rows
        if row["baseline"] == "reflexion_style"
        and row["sample_id"] == "game24_pilot_001"
        and row["arm"] == "clean"
        and row["backbone"] == "gpt4o"
    )
    misplaced = next(
        row
        for row in rows
        if row["baseline"] == "reflexion_style"
        and row["sample_id"] == "game24_pilot_002"
        and row["arm"] == "clean"
        and row["backbone"] == "gpt4o"
    )
    expected["verifier_result"]["is_correct"] = True
    expected["verifier_result"]["reason"] = "ok"
    expected["method_calls"] = [
        call for call in expected["method_calls"] if call["stage"] != "reflexion_reflect"
    ]
    expected["memory_write_event"] = None
    expected["memory_after"] = expected["memory_before"]
    misplaced["verifier_result"]["is_correct"] = False
    misplaced["verifier_result"]["reason"] = "value_does_not_match_target"
    misplaced["method_calls"].append(
        _method_call(
            "reflexion_reflect", [{"role": "user", "content": "reflect"}], "reflect", "gpt4o"
        )
    )
    misplaced["memory_write_event"] = {"type": "reflexion_append", "status": "accepted"}
    _write_trials_jsonl(run_dir, rows)

    result = _run_inspector(run_dir)
    assert result.returncode != 0
    report = json.loads(result.stdout)
    assert report["reflexion"] == "fail"
    assert any("game24_pilot_001" in reason for reason in report.get("reasons", []))


def test_inspector_rejects_wrong_stage_count(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad_run"
    rows = _make_synthetic_rows()
    for row in rows:
        if row["baseline"] == "dynamic_cheatsheet_optional":
            row["method_calls"] = [
                c for c in row["method_calls"] if c["stage"] != "dynamic_cheatsheet_curate"
            ]
    _write_trials_jsonl(run_dir, rows)

    result = _run_inspector(run_dir)
    assert result.returncode != 0
    report = json.loads(result.stdout)
    assert report["stages"] == "fail"
    assert any(
        "dynamic_cheatsheet_curate" in reason.lower() for reason in report.get("reasons", [])
    )


def test_inspector_against_real_canonical_run(tmp_path: Path) -> None:
    canonical = REPO_ROOT / "runs" / "g0_fh_reflexion_dc_faithful_replay"
    if not (canonical / "trials.jsonl").exists():
        pytest.skip("canonical run not available")
    result = _run_inspector(canonical)
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
