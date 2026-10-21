"""Reconstruct authenticated historical stages, or reopen the identical report read-only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from memcontam.evaluation.phase13_historical_exposure import (
    STAGES, HistoricalIntegrityError, checked, join_original, observation,
    reconstruct_stages, summarize, validate_output_path, validate_population,
)


def reconstruct(historical: Path, analysis: Path, output: Path):
    inputs = {}

    def read(path: Path, expected: str | None = None):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise HistoricalIntegrityError("INPUT_HASH_MISMATCH:" + str(path))
        if str(path) in inputs and inputs[str(path)] != digest:
            raise HistoricalIntegrityError("INPUT_CHANGED_DURING_READ:" + str(path))
        inputs[str(path)] = digest
        return raw

    def document(path: Path, expected: str | None = None):
        return json.loads(read(path, expected))

    receipt = document(output.parent / "task4/DoneClaim.json")
    published = {item["path"]: item["sha256"] for item in receipt["outputs"]}
    documents = {}
    for name in ("historical-inputs.json", "reuse-manifest.json", "missing-artifacts.json", "task4/authenticated-records.json"):
        path = output.parent / name
        documents[name] = document(path, published[str(path)])
    inventory, reuse = documents["historical-inputs.json"], documents["reuse-manifest.json"]
    validate_population(inventory)
    authority = reuse["authority"]["manifest"]
    if authority["sha256"] != "5cf0c3772739aa5124866752a99bcf5dfcbb77bd3e364de438b2ebc54eee34d4":
        raise HistoricalIntegrityError("AUTHORITY_MANIFEST_PIN")
    read(Path(authority["path"]), authority["sha256"])
    for binding in reuse["authenticated_existing_outputs"]:
        if not Path(binding["path"]).is_relative_to(analysis):
            raise HistoricalIntegrityError("EXISTING_ANALYSIS_ROOT_JOIN")
        read(Path(binding["path"]), binding["sha256"])
    for path, digest in document(analysis / "native_input_hashes.json")["files"].items():
        read(Path(path), digest)
    for binding in inventory["sources"]:
        read(Path(binding["path"]), binding["sha256"])
    parent = document(historical / "registration.json", reuse["registration"]["sha256"])
    variant = document(historical / "continuations/c003/registration.json", reuse["c003_registration"]["sha256"])
    if parent["runroot"] != str(historical) or parent["source_head"] != inventory["source_head"] or variant["source_head"] != inventory["source_head"] or variant["source_registration_sha256"] != reuse["registration"]["sha256"]:
        raise HistoricalIntegrityError("REGISTRATION_SOURCE_JOIN")
    source = Path(__file__).resolve().parents[1]
    read(Path(__file__).resolve())
    read(source / "src/memcontam/evaluation/phase13_historical_exposure.py")
    for name, digest in parent["source_hashes"].items():
        read(source / name, digest)
    for segment in ("c001", "c002", "c003"):
        contract_path = historical / "continuations" / segment / ("registration.json" if segment == "c003" else "amendment.json")
        contract = document(contract_path)
        if contract["source_head"] != parent["source_head"] or contract["eligible_cells"] != parent["cells"][1:]:
            raise HistoricalIntegrityError("CONTINUATION_SCOPE_JOIN")
        if segment == "c001" and contract["registration_sha256"] != reuse["registration"]["sha256"]:
            raise HistoricalIntegrityError("AMENDMENT_PARENT_JOIN")
        for name, digest in contract["adapter_hashes"].items():
            read(contract_path.parent / name, digest)
        for name, digest in contract["prior_hashes"].items():
            read(historical / name, digest)
    for cell in inventory["cells"]:
        registration = parent if cell["cohort"] == "original-g2" else variant
        registered = next(item for item in registration.get("cells", registration.get("eligible_cells")) if (item["task"], item["baseline"]) == (cell["task"], cell["baseline"]))
        if cell["registered_suffix_order"] != registered["suffix_ids"]:
            raise HistoricalIntegrityError("REGISTERED_SUFFIX_JOIN")
    reused = {item["trial_id"]: item for item in documents["task4/authenticated-records.json"]}
    if len(reused) != 200:
        raise HistoricalIntegrityError("REUSED_POPULATION_CARDINALITY")
    archives, rows, calls_seen = {}, [], set()
    for slot in inventory["rows"]:
        row = dict(slot)
        row["adapter_hashes"] = variant["adapter_hashes"] if slot["segment"] == "c003" else {}
        prefix = "src/memcontam/baselines/bot" if slot["baseline"] == "bot_style" else "src/memcontam/baselines/dynamic_cheatsheet"
        row["historical_source_hashes"] = {name: digest for name, digest in parent["source_hashes"].items() if name.startswith(prefix) or name.endswith(("runtime_registry.py", "phase13_production_runtime_join.py"))}
        row["stages"] = {stage: observation(None, "UNATTEMPTED_OR_MISSING_ORIGINAL", " | ".join(slot["trial_paths"])) for stage in STAGES}
        row["native_object_ids"] = []
        row["answer_request_binding"] = None
        if slot["archive"] is not None:
            binding = slot["archive"]
            if binding["path"] not in archives:
                archives[binding["path"]] = document(Path(binding["path"]), binding["sha256"])
            archive = archives[binding["path"]]
            expected_ids = {item["trial_id"] for item in inventory["rows"] if item["archive"] is not None and item["archive"]["path"] == binding["path"]}
            archived_ids = [record["evidence"]["trial_id"] for record in archive["records"]]
            if len(archived_ids) != len(expected_ids) or set(archived_ids) != expected_ids:
                raise HistoricalIntegrityError("ARCHIVE_REGISTERED_SUFFIX_CARDINALITY")
            trial = document(Path(slot["trial"]["path"]), slot["trial"]["sha256"])
            evidence, calls, injected = join_original(slot, archive, trial)
            if archive["registration_packet_sha256"] != "5d8b8a3120d3e8c457549c8cce6c4e5fa42fa29f8a62e2b6f5403850dc6ee1a8":
                raise HistoricalIntegrityError("OBSERVABILITY_PACKET_JOIN")
            if any(call.call_id in calls_seen for call in calls):
                raise HistoricalIntegrityError("DUPLICATE_POPULATION_CALL_ID")
            calls_seen.update(call.call_id for call in calls)
            row["stages"] = reconstruct_stages(evidence, calls, injected)
            for value in row["stages"].values():
                if value["value"] is None and value["eligible"]:
                    value["missing_artifact"] = binding["path"] + f"#/records/{slot['archive_record_index']}/evidence/lineage"
                    if "CALL" in value["reason"] or "CURATOR" in value["reason"] or "RETRIEVAL" in value["reason"]:
                        value["missing_artifact"] = slot["trial"]["path"] + ("#/retrieval" if "RETRIEVAL" in value["reason"] else "#/outcome/method_calls")
            row["native_object_ids"] = list(injected)
            row["recorded_lineage"] = [node.model_dump(mode="json") for node in evidence.lineage]
            row["registered_target_set"] = evidence.target_set.model_dump(mode="json")
            row["original_call_bindings"] = []
            for call in calls:
                if call.provider_request_contract is None:
                    raise HistoricalIntegrityError("MISSING_REQUEST_CONTRACT")
                row["original_call_bindings"].append({"call_id": call.call_id, "input_sha256": call.provider_request_contract["input_sha256"], "response_id": call.provider_response_id,
                                                    "response_text_sha256": hashlib.sha256(call.raw_response.encode()).hexdigest() if call.raw_response is not None else None})
            answer = next((call for call in calls if call.call_id == slot["answer_call_id"]), None)
            if answer:
                if answer.provider_request_contract is None:
                    raise HistoricalIntegrityError("MISSING_ANSWER_REQUEST_CONTRACT")
                row["answer_request_binding"] = {"call_id": answer.call_id, "input_sha256": answer.provider_request_contract["input_sha256"], "source_spans": [span.model_dump(mode="json") for span in answer.source_spans], "artifact": slot["trial"]["path"], "json_pointer": "/outcome/method_calls"}
            if slot["trial_id"] in reused and slot["cohort"] == "c003-input16384":
                for stage in ("stored", "retrieved", "actual_answer_context"):
                    if row["stages"][stage]["value"] != reused[slot["trial_id"]][stage]["value"]:
                        raise HistoricalIntegrityError("SUPPORTED_REUSE_CONTRADICTION")
        for stage in ("curator", "derivation", "strategy_visible"):
            if slot["baseline"] != "dc_rs":
                row["stages"][stage] = observation(None, "BASELINE_HAS_NO_CURATOR", eligible=False)
        if slot["arm"] != "contam":
            row["stages"]["registered_z"] = observation(None, "ARM_HAS_NO_REGISTERED_CONTAM_TARGET", eligible=False)
        row["stages"]["operational_use"] = observation(None, "NOT_REGISTERED", "separately registered operational attribution", eligible=False)
        rows.append(row)
    for cell in inventory["cells"]:
        selected = [row for row in rows if row["cell_key"] == cell["cell_key"]]
        if sum(row["observation_status"] == "completed" for row in selected) != cell["completed_archive_rows"]:
            raise HistoricalIntegrityError("COMPLETED_DENOMINATOR_JOIN")
    for path, digest in inputs.items():
        checked({"path": path, "sha256": digest})
    return rows, summarize(rows), inputs, documents["missing-artifacts.json"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--existing-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-output", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.absolute()
    validate_output_path(output, (args.historical_root, args.existing_analysis), args.verify_output)
    rows, summary, inputs, missing = reconstruct(args.historical_root, args.existing_analysis, output)
    files = {"rows.jsonl": "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows).encode()}
    csv_text = io.StringIO(newline="")
    writer = csv.DictWriter(csv_text, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
    files["summary.csv"] = csv_text.getvalue().encode()
    missing["task9_pending_stage_authentication_not_artifact_absence"] = False
    missing["stage_gaps"] = [{"population_key": row["population_key"], "stage": stage, **value} for row in rows for stage, value in row["stages"].items() if value["status"] == "unknown"]
    files["missing-artifacts.json"] = (json.dumps(missing, sort_keys=True, indent=2) + "\n").encode()
    files["report.md"] = ("# Historical Exposure Reconstruction\n\nHISTORICAL_RECONSTRUCTION_COMPLETE_WITH_DECLARED_GAPS\n\n"
        "Source generation and current authority currency: unknown. Historical-only; no adoption, operational-use or causal claim. No rankings, embeddings, thresholds or provider execution.\n\n"
        "Original-g2: 800 unattempted slots. c003-input16384: 400 completed, 5 technical-missing, 395 unattempted. These are separate cohorts, not pooled estimates. 64 segment views are locators, not populations.\n\n"
        "200 reused c003 BoT Correct/Contam records retain supported storage and zero retrieval/context. Other rows require their own original joins. DC strategy visibility does not establish target parentage. Actual context uses answer-call request hashes and rendered source spans, never response text or threshold predictions.\n\n"
        "summary.csv groups task/baseline/arm/cohort/segment/stage. expected = known + unknown; known = true + false. completed counts native completed outcomes independently of evidence coverage. Rates are true/known and known/expected; zero eligibility is null (empty CSV field). Non-applicable stages are unknown/null with eligible=false, not negative observations.\n\n"
        "c001/c002 bind the original root registration via sealed amendments; absent literal registration.json files remain declared. Task4 inputs are consumed from the report parent, alongside task4/DoneClaim.json. Verify-output redoes input joins and compares bytes without writes. Provider-transport proof remains limited to prior audits; native request records are not a new provider database join.\n").encode()
    manifest = {"status": "HISTORICAL_RECONSTRUCTION_COMPLETE_WITH_DECLARED_GAPS", "expected": len(rows), "inputs": inputs,
                "outputs": {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
                "current_freshness": "unknown", "source_generation": "unknown", "enforced_immutability": False,
                "scientific_adoption": False, "paid_launch": False, "provider_calls": 0}
    files["manifest.json"] = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    if args.verify_output:
        if {path.name for path in output.iterdir()} != set(files):
            raise HistoricalIntegrityError("REPORT_FILE_SET_MISMATCH")
        for name, raw in files.items():
            if (output / name).is_symlink() or (output / name).read_bytes() != raw:
                raise HistoricalIntegrityError("REPORT_REOPEN_MISMATCH:" + name)
    else:
        output.mkdir()
        for name, raw in files.items():
            with (output / name).open("xb") as stream:
                stream.write(raw)
    print(json.dumps({"status": manifest["status"], "verify_only": args.verify_output, "rows": len(rows), "summary_rows": len(summary), "input_bindings": len(inputs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
