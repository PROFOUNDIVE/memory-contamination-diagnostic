from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Final

import pytest

E: Final = Path(os.environ.get("RECOVERY_EVIDENCE", "/home/hyunwoo/phase13_artifacts/phase13-parallel-recovery-20261021-001"))
D2: Final = Path(__file__).resolve().parents[1]
D1: Final = Path(os.environ.get("RECOVERY_D1", "/home/hyunwoo/git/memory-contamination-diagnostic-phase13-recovery-d1-20261021-001"))
CAND: Final = Path(os.environ.get("RECOVERY_CANDIDATE", "/home/hyunwoo/phase13_controlplane/v11_5_4_11_h1_capture_20261021_candidate001"))
REPORT: Final = D1 / "scripts/reconstruct_phase13_historical_exposure.py"
PACKET: Final = D2 / "scripts/prepare_phase13_snapshot_diagnostic.py"
COMPAT: Final = CAND / "recovery/h1_capture_compat.py"
LATEST: Final = E / "d1/report-task15-v2"
HIST: Final = Path("/home/hyunwoo/phase13_artifacts/PeerJ-fast-track/11\uC8FC\uCC28/diagnostic-p/diagnostic-p-v3-defect6-native50-20261007-g2")
ANALYSIS: Final = Path("/home/hyunwoo/phase13_artifacts/diagnostic-p-investigation-20261015")
JOB: Final = HIST.parent.parent / "h1_readiness_review_source019_20261019_002"
REVIEW: Final = JOB / "reviewer-merge-boundary-001/native-review/REVIEW_RESULT.json"


def fingerprint(root: Path) -> tuple[tuple[str, str, int, int], ...]:
    return tuple((str(path), hashlib.sha256(path.read_bytes()).hexdigest(),
                  path.stat().st_mtime_ns, path.stat().st_mode)
                 for path in sorted(root.rglob("*")) if path.is_file())


def child(tmp: Path, code: str, root: Path = D2) -> subprocess.CompletedProcess[str]:
    script = tmp / f"child-{uuid.uuid4().hex}.py"
    script.write_text(code)
    argv = [sys.executable, "-B", "-m", "tests.phase13_recovery_offline", str(script)]
    result = subprocess.run(argv, cwd=D2, env={**os.environ, "PYTHONPATH": str(root / "src")},
                            capture_output=True, text=True, timeout=120, check=False)
    script.with_suffix(".receipt.json").write_text(json.dumps({"argv": argv, "cwd": str(D2),
        "PYTHONPATH": str(root / "src"), "exit_code": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr}, indent=2))
    return result


def cli(tmp: Path, argv: list[str], root: Path = D2) -> subprocess.CompletedProcess[str]:
    return child(tmp, f"import runpy, sys\nsys.argv = {argv!r}\nrunpy.run_path(sys.argv[0], run_name='__main__')\n", root)


def report_args(output: Path, verify: bool = True) -> list[str]:
    return [str(REPORT), "--historical-root", str(HIST), "--existing-analysis", str(ANALYSIS),
            "--output-dir", str(output), *(["--verify-output"] if verify else [])]


@pytest.mark.parametrize("entry", [REPORT, PACKET, COMPAT])
def test_help_when_only_offline_entrypoints_are_requested(tmp_path: Path, entry: Path) -> None:
    result = cli(tmp_path, [str(entry), "--help"], D1 if entry == REPORT else D2)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("lane", ["D1", "D2"])
def test_valid_verification_when_m1_and_q_wait_externally(tmp_path: Path, lane: str) -> None:
    assert json.loads((E / "m1/task13-resume-002/capture-status.json").read_text())["status"] == "CAPTURE_REQUIRES_EXTERNAL_ACTION"
    assert json.loads((E / "q/pending.json").read_text())["gates"]["human_panel"]["status"] == "UNVERIFIED"
    output, root, argv = (LATEST, D1, report_args(LATEST)) if lane == "D1" else (
        E / "d2/packet", D2, [str(PACKET), "--verify-packet", str(E / "d2/packet")])
    before = fingerprint(output)
    result = cli(tmp_path, argv, root)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["status"] in {"HISTORICAL_RECONSTRUCTION_COMPLETE_WITH_DECLARED_GAPS", "DIAGNOSTIC_READY_FOR_SEPARATE_AUTHORIZATION"}
    assert fingerprint(output) == before


def test_native_export_corruption_when_only_m1_consumes_it(tmp_path: Path) -> None:
    database, export = tmp_path / "native.sqlite", tmp_path / "export.json"
    shutil.copyfile(E / "task-2/normalized-native.sqlite", database)
    shutil.copyfile(REVIEW.with_name("native-export.json"), export)
    code = (f"import importlib, json, sys\nfrom pathlib import Path\nsys.path.insert(0, {str(COMPAT.parent)!r})\n"
        "compat = importlib.import_module('h1_capture_compat')\n"
        f"compat.pins.modules()[2].DATABASE = Path({str(database)!r})\n"
        "gate = compat.admission.Gate(compat.pins.pin(compat.pins.CANDIDATE))\n"
        f"review = json.loads(Path({str(REVIEW)!r}).read_text())\n"
        f"proof = compat.verify_native(gate, review, Path({str(export)!r}))\n"
        "assert proof['sql_identity']['part_count'] == 299\nprint('NATIVE_GRAPH_VERIFIED')\n")
    assert child(tmp_path, code).returncode == 0
    before = fingerprint(LATEST), fingerprint(E / "d2/packet")
    export.write_text("{}")
    result = child(tmp_path, code)
    assert result.returncode == 1 and "pin hash/readonly mismatch" in result.stderr, result.stderr
    assert cli(tmp_path, report_args(LATEST), D1).returncode == 0
    assert cli(tmp_path, [str(PACKET), "--verify-packet", str(E / "d2/packet")]).returncode == 0
    assert (fingerprint(LATEST), fingerprint(E / "d2/packet")) == before


@pytest.fixture
def rebound(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Explicit scratch amendment of locators only; never historical authentication."""
    parent = tmp_path / "d1"
    (parent / "task4").mkdir(parents=True)
    proposal = json.loads((E / "d2/proposal.json").read_text())
    authority = tmp_path / "MANIFEST.json"
    shutil.copyfile(proposal["authority_manifest"]["path"], authority)
    proposal["authority_manifest"]["path"] = str(authority)
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(proposal))
    receipt = json.loads((E / "d1/task4/DoneClaim.json").read_text())
    receipt["outputs"] = []
    for name in ("historical-inputs.json", "reuse-manifest.json", "missing-artifacts.json", "task4/authenticated-records.json"):
        path = parent / name
        shutil.copyfile(E / "d1" / name, path)
        if name == "reuse-manifest.json":
            reuse = json.loads(path.read_text())
            reuse["authority"]["manifest"]["path"] = str(authority)
            path.write_text(json.dumps(reuse))
        receipt["outputs"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (parent / "task4/DoneClaim.json").write_text(json.dumps(receipt))
    report, packet = parent / "report", tmp_path / "packet"
    result = cli(tmp_path, report_args(report, False), D1)
    assert result.returncode == 0, result.stderr
    result = cli(tmp_path, [str(PACKET), "--proposal", str(proposal_path), "--output-dir", str(packet), "--provider-free"])
    assert result.returncode == 0, result.stderr
    assert cli(tmp_path, report_args(report), D1).returncode == 0
    assert cli(tmp_path, [str(PACKET), "--verify-packet", str(packet)]).returncode == 0
    return authority, report, packet


def test_all_actual_consumers_refuse_when_shared_manifest_digest_changes(
    tmp_path: Path, rebound: tuple[Path, Path, Path],
) -> None:
    authority, report, packet = rebound
    before = fingerprint(LATEST), fingerprint(E / "d2/packet"), fingerprint(report), fingerprint(packet)
    authority.write_bytes(authority.read_bytes() + b" ")
    for argv, root, reason in ((report_args(report), D1, "INPUT_HASH_MISMATCH"),
        ([str(PACKET), "--verify-packet", str(packet)], D2, "PACKET_BINDING_MISMATCH")):
        dispatch = tmp_path / f"{root.name}-simulated-dispatch"
        code = (f"import runpy, sys\nfrom pathlib import Path\nsys.argv = {argv!r}\n"
                "try:\n    runpy.run_path(sys.argv[0], run_name='__main__')\n"
                "except SystemExit as error:\n    if error.code: raise\n"
                f"Path({str(dispatch)!r}).write_text('SIMULATED_ONLY')\n")
        result = child(tmp_path, code, root)
        assert result.returncode == 1 and reason in result.stderr, result.stderr
        assert not dispatch.exists()
    assert cli(tmp_path, report_args(LATEST), D1).returncode == 0
    assert cli(tmp_path, [str(PACKET), "--verify-packet", str(E / "d2/packet")]).returncode == 0
    assert (fingerprint(LATEST), fingerprint(E / "d2/packet"), fingerprint(report), fingerprint(packet)) == before


@pytest.mark.parametrize("lane", ["D1", "D2", "M1"])
def test_output_collision_refuses_before_any_write(tmp_path: Path, lane: str) -> None:
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "sentinel").write_text("preserved")
    options = {"D1": (report_args(target, False), D1), "D2": ([str(PACKET), "--proposal", str(E / "d2/proposal.json"), "--output-dir", str(target), "--provider-free"], D2),
        "M1": ([str(COMPAT), "--review-result", str(REVIEW), "--candidate", str(JOB / "CANDIDATE_H1_READINESS.json"), "--candidate-sha256", "e301a463caa38f7f717161884efae42d76c0ce5ebb8db6c67bb27343541501cc", "--export", str(REVIEW.with_name("native-export.json")), "--output", str(target), "--verify-only"], D2)}
    argv, root = options[lane]
    before = fingerprint(target)
    result = cli(tmp_path, argv, root)
    assert result.returncode == 1, result.stderr
    assert fingerprint(target) == before


@pytest.mark.parametrize("lane", ["D1", "D2"])
def test_overlapping_input_namespace_refuses_when_output_is_nested(tmp_path: Path, lane: str) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    proposal = json.loads((E / "d2/proposal.json").read_text())
    shutil.copyfile(proposal["handoff"]["path"], inputs / "handoff.md")
    proposal["handoff"]["path"] = str(inputs / "handoff.md")
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(proposal))
    historical = report_args(inputs / "forbidden", False)
    historical[historical.index("--existing-analysis") + 1] = str(inputs)
    argv, root, code = (historical, D1, "OUTPUT_INPUT_COLLISION") if lane == "D1" else (
        [str(PACKET), "--proposal", str(proposal_path), "--output-dir", str(inputs / "forbidden"), "--provider-free"], D2, "PACKET_OUTPUT_NESTED_IN_INPUT")
    result = cli(tmp_path, argv, root)
    assert result.returncode == 1 and code in result.stderr, result.stderr


def test_completed_request_refuses_redispatch_when_restarted(tmp_path: Path) -> None:
    code = ("from pathlib import Path\nfrom tests.test_phase13_snapshot_budget import "
            "test_completion_survives_new_process_without_replay_or_double_charge, synthetic_evidence_policy\n"
            "import pytest\nwith pytest.MonkeyPatch.context() as patch:\n"
            "    synthetic_evidence_policy.__wrapped__(patch)\n"
            f"    test_completion_survives_new_process_without_replay_or_double_charge(Path({str(tmp_path)!r}))\n"
            "print('DURABLE_RESTART_DUPLICATE_RECEIPT_REQUEST_REFUSED')\n")
    result = child(tmp_path, code)
    assert result.returncode == 0, result.stderr
    assert "DURABLE_RESTART_DUPLICATE_RECEIPT_REQUEST_REFUSED" in result.stdout


def test_real_compatibility_cli_preserves_source_closure_refusal(tmp_path: Path) -> None:
    argv = [str(COMPAT), "--review-result", str(REVIEW), "--candidate", str(JOB / "CANDIDATE_H1_READINESS.json"),
        "--candidate-sha256", "e301a463caa38f7f717161884efae42d76c0ce5ebb8db6c67bb27343541501cc",
        "--export", str(REVIEW.with_name("native-export.json")), "--output", str(tmp_path / "proof.json"), "--verify-only"]
    code = ("import importlib, runpy, subprocess, sys, pytest\n"
        "from tests.phase13_recovery_offline import OfflineTransportDenied\n"
        "guard = subprocess.Popen\nraw = importlib.reload(subprocess).Popen\n"
        "reads = [('rev-parse','HEAD'), ('status','--porcelain'), ('ls-files','--stage','-v','-z'),\n"
        " ('rev-parse','--path-format=absolute','--git-path','index'), ('ls-files','-z'),\n"
        " ('ls-tree','-r','--name-only','-z','HEAD'), ('diff','--binary','--full-index','--no-ext-diff','--no-textconv','HEAD','--')]\n"
        "def readonly_git(args, *positional, **kwargs):\n"
        "    if (not positional and isinstance(args,list) and args[:4] == ['git','--no-optional-locks','-C',\n"
        "        '/home/hyunwoo/git/memory-contamination-diagnostic-phase13-rag-observability-repair']\n"
        "        and tuple(args[4:]) in reads and not kwargs.get('shell') and kwargs.get('executable') is None):\n"
        "        return raw(args, **kwargs)\n"
        "    return guard(args, *positional, **kwargs)\n"
        "subprocess.Popen = readonly_git\n"
        "with pytest.raises(OfflineTransportDenied): subprocess.run(['opencode','run'],check=True)\n"
        "with pytest.raises(OfflineTransportDenied): subprocess.run(['git','status'],check=True)\n"
        f"sys.argv = {argv!r}\nrunpy.run_path(sys.argv[0], run_name='__main__')\n")
    result = child(tmp_path, code)
    assert result.returncode == 1 and "current source closure complete file/mode/index inventory changed" in result.stderr, result.stderr
    assert not (tmp_path / "proof.json").exists()


def test_stale_lane_output_digest_refuses_promotion(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    shutil.copytree(E / "d2/packet", packet)
    path = packet / "readiness.json"
    path.write_bytes(path.read_bytes() + b" ")
    result = cli(tmp_path, [str(PACKET), "--verify-packet", str(packet)])
    assert result.returncode == 1 and "PACKET_CHECKSUM_MISMATCH" in result.stderr


def test_external_denial_remains_active_when_python_child_is_allowed(tmp_path: Path) -> None:
    code = ("import os, socket, subprocess, pytest\n"
        "from tests.phase13_recovery_offline import OfflineTransportDenied\n"
        "from memcontam.clients.openai_responses import OpenAIResponsesClient\n"
        "assert not any('API_KEY' in key or 'CREDENTIAL' in key for key in os.environ)\n"
        "with pytest.raises(OfflineTransportDenied): socket.create_connection(('example.invalid',443))\n"
        "with pytest.raises(OfflineTransportDenied): subprocess.run(['opencode','run'],check=True)\n"
        "for method in ('chat','count_compiled_v3','send_compiled_v3'):\n"
        "    with pytest.raises(OfflineTransportDenied): getattr(OpenAIResponsesClient.__new__(OpenAIResponsesClient),method)()\n"
        "print('CREDENTIAL_NETWORK_PROVIDER_COUNT_LAUNCH_DENIED')\n")
    result = child(tmp_path, code)
    assert result.returncode == 0, result.stderr
