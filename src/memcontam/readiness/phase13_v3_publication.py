from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path
from typing import Final
from uuid import uuid4

from .phase13_authority_files import AuthorityFileError, authority_directory
from .phase13_v3_resource_files import normalized_paths


P4_PATHS: Final = (
    "authority_v3/current_authority_v3.json",
    "cost_envelope_v3/activated_policy_v3.json",
    "cost_envelope_v3/base_inputs_v3.json",
    "cost_envelope_v3/cost_witness_v3.json",
    "mr_p4/corrected_v3/provider_free_conformance_v3.json",
    "mr_p4/corrected_v3/manifest_v3.json",
)
P5_PATHS: Final = (
    "cost_envelope_v3/complete_inputs_v3.json", "cost_envelope_v3/cost_proof_v3.json",
    "main_live_contract_v3.json", "mr_p5/execution_package_v3.json",
)
P6_PATHS: Final = ("mr_p6/authorized_execution_v3.json", "mr_p6/authorized_execution_v3.sha256")
OUTPUT_PATHS: Final = (*P4_PATHS, *P5_PATHS, *P6_PATHS)


class ArtifactError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def publish_artifacts(root: Path, artifacts: tuple[tuple[str, bytes], ...]) -> None:
    names = normalized_paths(tuple(name for name, _ in artifacts))
    if any(name not in OUTPUT_PATHS for name in names):
        raise ArtifactError("MAIN_HISTORICAL_OUTPUT_FORBIDDEN")
    with ExitStack() as stack:
        try:
            directory = stack.enter_context(authority_directory(root))
        except AuthorityFileError as error:
            raise ArtifactError("MAIN_PATH_UNSAFE") from error
        directories = {"": directory}
        observations: list[tuple[int, str, int]] = []
        temporaries: list[tuple[int, str]] = []
        published: list[tuple[int, str, int]] = []
        try:
            for name, raw in artifacts:
                relative = ""
                parent = directory
                for component in name.split("/")[:-1]:
                    relative = f"{relative}/{component}".lstrip("/")
                    if relative not in directories:
                        try:
                            os.mkdir(component, 0o755, dir_fd=parent)
                        except FileExistsError:
                            pass
                        try:
                            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                        except OSError as error:
                            raise ArtifactError("MAIN_PATH_UNSAFE") from error
                        stack.callback(os.close, child)
                        observations.append((parent, component, child))
                        directories[relative] = child
                    parent = directories[relative]
                target = name.split("/")[-1]
                temporary = f".{target}.{uuid4().hex}.tmp"
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                     0o644, dir_fd=parent)
                stack.callback(os.close, descriptor)
                temporaries.append((parent, temporary))
                remaining = memoryview(raw)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("incomplete artifact write")
                    remaining = remaining[written:]
                os.fsync(descriptor)
                try:
                    os.link(temporary, target, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
                except FileExistsError as error:
                    raise ArtifactError("MAIN_ARTIFACT_EXISTS") from error
                published.append((parent, target, descriptor))
                os.fsync(descriptor)
                os.fsync(parent)
            for parent, component, descriptor in observations:
                named = os.stat(component, dir_fd=parent, follow_symlinks=False)
                opened = os.fstat(descriptor)
                if (named.st_dev, named.st_ino, named.st_mode) != (opened.st_dev, opened.st_ino, opened.st_mode):
                    raise ArtifactError("MAIN_PATH_UNSAFE")
        except (OSError, ArtifactError) as error:
            for parent, target, descriptor in reversed(published):
                named = os.stat(target, dir_fd=parent, follow_symlinks=False)
                opened = os.fstat(descriptor)
                if (named.st_dev, named.st_ino) == (opened.st_dev, opened.st_ino):
                    os.unlink(target, dir_fd=parent)
                    os.fsync(parent)
            if isinstance(error, ArtifactError):
                raise
            raise ArtifactError("MAIN_ARTIFACT_PUBLICATION_FAILED") from error
        finally:
            for parent, temporary in temporaries:
                os.unlink(temporary, dir_fd=parent)
