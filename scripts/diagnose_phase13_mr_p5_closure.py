# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# ─── How to run ───
# uv run scripts/diagnose_phase13_mr_p5_closure.py --repository-root .
# Add --require-closed to return nonzero while local imports remain outside MR-P5.

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ClosureReport:
    entrypoint: str
    bound_python_path_count: int
    local_import_closure_count: int
    omitted_local_import_count: int
    omitted_local_imports: tuple[str, ...]


def _package_initializers(source_root: Path, parts: tuple[str, ...]) -> tuple[Path, ...]:
    initializers = (
        source_root.joinpath(*parts[:depth], "__init__.py")
        for depth in range(1, len(parts) + 1)
    )
    return tuple(path for path in initializers if path.is_file())


def _module_paths(source_root: Path, module: str) -> tuple[Path, ...]:
    parts = tuple(part for part in module.split(".") if part)
    base = source_root.joinpath(*parts)
    module_file = base.with_suffix(".py")
    if module_file.is_file():
        return (*_package_initializers(source_root, parts[:-1]), module_file)
    package_file = base / "__init__.py"
    if package_file.is_file():
        return _package_initializers(source_root, parts)
    return ()


def _module_name(source_root: Path, path: Path) -> tuple[str, ...]:
    parts = path.relative_to(source_root).with_suffix("").parts
    return parts[:-1] if parts[-1] == "__init__" else parts


def _local_imports(source_root: Path, path: Path) -> tuple[Path, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    current = _module_name(source_root, path)
    package = current if path.name == "__init__.py" else current[:-1]
    imported: set[Path] = set()
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            keep = len(package) - (node.level - 1)
            relative = (*package[:keep], *((node.module or "").split(".") if node.module else ()))
            module = ".".join(relative) if node.level else node.module or ""
            candidates.extend((module, *(f"{module}.{alias.name}" for alias in node.names if alias.name != "*")))
        for candidate in candidates:
            if not candidate.startswith("memcontam"):
                continue
            imported.update(_module_paths(source_root, candidate))
    return tuple(sorted(imported))


def _closure(source_root: Path, entrypoint: Path) -> tuple[Path, ...]:
    entrypoint_parts = entrypoint.relative_to(source_root).parts[:-1]
    pending = [*_package_initializers(source_root, entrypoint_parts), entrypoint]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        pending.extend(imported for imported in _local_imports(source_root, path) if imported not in visited)
    return tuple(sorted(visited))


def _report(repository_root: Path, package_relative: Path) -> ClosureReport:
    source_root = repository_root / "src"
    package_path = repository_root / package_relative
    package = json.loads(package_path.read_text(encoding="utf-8"))
    bound = {
        repository_root / row["path"]
        for row in package["artifacts"]
        if row["path"].endswith(".py")
    }
    entrypoint = repository_root / "src/memcontam/readiness/phase13_main_live_cli.py"
    closure = set(_closure(source_root, entrypoint))
    omitted = tuple(sorted(str(path.relative_to(repository_root)) for path in closure - bound))
    return ClosureReport(
        entrypoint=str(entrypoint.relative_to(repository_root)),
        bound_python_path_count=len(bound),
        local_import_closure_count=len(closure),
        omitted_local_import_count=len(omitted),
        omitted_local_imports=omitted,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare an AST-derived approximation of Main-A local imports with MR-P5 bindings."
    )
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument(
        "--package",
        type=Path,
        default=Path("data/phase13/main/mr_p5/execution_package_v1.json"),
    )
    parser.add_argument("--require-closed", action="store_true")
    args = parser.parse_args()
    report = _report(args.repository_root.resolve(), args.package)
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return int(args.require_closed and bool(report.omitted_local_imports))


if __name__ == "__main__":
    raise SystemExit(main())
