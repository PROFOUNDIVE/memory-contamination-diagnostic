from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC = ROOT / "scripts/diagnose_phase13_mr_p5_closure.py"


def _run(*arguments: str, repository_root: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, str(DIAGNOSTIC), "--repository-root", str(repository_root), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )


def test_mr_p5_diagnostic_reproduces_known_transitive_dependency_gaps() -> None:
    result = _run()

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert {
        "src/memcontam/tasks/math_equation_balancer.py",
        "src/memcontam/verifiers/math_equation_balancer.py",
        "src/memcontam/readiness/phase13_legacy_rag_runtime.py",
        "src/memcontam/readiness/phase13_core_datasets.py",
        "src/memcontam/baselines/full_history_phase12.py",
        "src/memcontam/baselines/reflexion_phase12.py",
        "src/memcontam/baselines/retrieval_rag_phase12.py",
    } <= set(report["omitted_local_imports"])


def test_mr_p5_diagnostic_can_fail_closed_without_creating_a_freeze() -> None:
    result = _run("--require-closed")

    assert result.returncode == 1
    assert json.loads(result.stdout)["omitted_local_import_count"] > 0


def test_mr_p5_diagnostic_traverses_parent_package_initializers(tmp_path: Path) -> None:
    files = {
        "src/memcontam/__init__.py": "",
        "src/memcontam/readiness/__init__.py": "",
        "src/memcontam/readiness/phase13_main_live_cli.py": "import memcontam.feature.leaf\n",
        "src/memcontam/feature/__init__.py": "import memcontam.support\n",
        "src/memcontam/feature/leaf.py": "",
        "src/memcontam/support.py": "",
        "data/phase13/main/mr_p5/execution_package_v1.json": '{"artifacts": []}',
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    result = _run(repository_root=tmp_path)

    assert result.returncode == 0, result.stderr
    assert set(json.loads(result.stdout)["omitted_local_imports"]) == set(files) - {
        "data/phase13/main/mr_p5/execution_package_v1.json"
    }


def test_mr_p5_diagnostic_accepts_a_separately_versioned_package(tmp_path: Path) -> None:
    files = {
        "src/memcontam/__init__.py": "",
        "src/memcontam/readiness/__init__.py": "",
        "src/memcontam/readiness/phase13_main_live_cli.py": "",
        "data/phase13/main/mr_p5/execution_package_v2.json": json.dumps(
            {
                "artifacts": [
                    {
                        "path": "src/memcontam/__init__.py",
                    },
                    {
                        "path": "src/memcontam/readiness/__init__.py",
                    },
                    {
                        "path": "src/memcontam/readiness/phase13_main_live_cli.py",
                    },
                ]
            }
        ),
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    result = _run(
        "--package",
        "data/phase13/main/mr_p5/execution_package_v2.json",
        "--require-closed",
        repository_root=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["omitted_local_import_count"] == 0
