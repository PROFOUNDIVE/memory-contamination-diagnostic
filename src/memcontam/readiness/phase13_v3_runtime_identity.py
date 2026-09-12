from __future__ import annotations

import os
import platform
import re
import site
import sys
from importlib import metadata, util
from pathlib import Path
from typing import Final

import memcontam

from .phase13_v3_authority_models import FrozenModel
from .phase13_v3_resource_files import read_files

ROOT: Final = Path(__file__).resolve().parents[3]
PREFIX: Final = "/home/hyunwoo/miniconda3/envs/memcontam"
EXECUTABLE: Final = PREFIX + "/bin/python"
LOCK_HASHES: Final = (
    "c644c8f5f9517461f2545e49e3f5f4907c83ad6340d639b54e7bbac15f126465",
    "d4929b7d063b9834e7649c21206e59a360b5db5a73950c323c771185ae8a1446",
)


class RuntimeIdentityError(ValueError):
    code = "MAIN_RUNTIME_IDENTITY_DRIFT"

    def __init__(self) -> None:
        super().__init__(self.code)


class RuntimeIdentityV3(FrozenModel):
    executable: str
    prefix: str
    base_prefix: str
    implementation: str
    version: str
    requirements_sha256: str
    requirements_dev_sha256: str
    versions: tuple[tuple[str, str], ...]
    pythonpath: str
    module_origin: str
    package_paths: tuple[str, ...]


def freeze_runtime_identity() -> RuntimeIdentityV3:
    source = str(ROOT / "src")
    origin = str(ROOT / "src/memcontam/__init__.py")
    spec = util.find_spec("memcontam")
    if (
        Path.cwd() != ROOT or sys.executable != EXECUTABLE
        or sys.prefix != PREFIX or sys.base_prefix != PREFIX
        or platform.python_implementation() != "CPython" or platform.python_version() != "3.11.15"
        or os.environ.get("PYTHONPATH") != source
        or os.environ.get("PYTHONNOUSERSITE") != "1" or site.ENABLE_USER_SITE
        or "PYTHONHOME" in os.environ or "VIRTUAL_ENV" in os.environ
        or spec is None or spec.origin != origin or memcontam.__file__ != origin
        or tuple(memcontam.__path__) != (str(ROOT / "src/memcontam"),)
        or source not in sys.path
    ):
        raise RuntimeIdentityError()
    source_position = sys.path.index(source)
    for position, entry in enumerate(sys.path):
        path = Path(entry or os.getcwd())
        if ".venv" in path.parts:
            raise RuntimeIdentityError()
        if (str(path).startswith(str(ROOT.parent / "memory-contamination-diagnostic"))
                and not path.is_relative_to(ROOT) and position < source_position):
            raise RuntimeIdentityError()
    for name, module in tuple(sys.modules.items()):
        if name == "memcontam" or name.startswith("memcontam."):
            filename = getattr(module, "__file__", None)
            if filename is not None and not Path(filename).resolve().is_relative_to(ROOT / "src"):
                raise RuntimeIdentityError()
    resources = read_files(ROOT, ("requirements.lock", "requirements-dev.lock"))
    by_name = {resource.binding.path: resource for resource in resources}
    hashes = tuple(by_name[name].binding.sha256 for name in ("requirements.lock", "requirements-dev.lock"))
    if hashes != LOCK_HASHES:
        raise RuntimeIdentityError()
    versions: dict[str, str] = {}
    for resource in resources:
        for name, version in re.findall(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", resource.raw.decode(), re.MULTILINE):
            normalized = re.sub(r"[-_.]+", "-", name).lower()
            if normalized in versions and versions[normalized] != version:
                raise RuntimeIdentityError()
            versions[normalized] = version
    try:
        if len(versions) != 102 or any(metadata.version(name) != version for name, version in versions.items()):
            raise RuntimeIdentityError()
    except metadata.PackageNotFoundError as error:
        raise RuntimeIdentityError() from error
    return RuntimeIdentityV3(
        executable=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix,
        implementation=platform.python_implementation(), version=platform.python_version(),
        requirements_sha256=hashes[0], requirements_dev_sha256=hashes[1],
        versions=tuple(sorted(versions.items())), pythonpath=source,
        module_origin=origin, package_paths=tuple(memcontam.__path__),
    )


def validate_runtime_identity(expected: RuntimeIdentityV3) -> None:
    if freeze_runtime_identity() != expected:
        raise RuntimeIdentityError()
