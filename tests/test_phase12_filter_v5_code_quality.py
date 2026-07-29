from __future__ import annotations

from pathlib import Path

import pytest
from tests.test_phase12_filter_v5_final_verifier_modes import _fixture, _request

from memcontam.experiment.phase12.filter_challenge.final_verifier import (
    FinalVerifierError,
    verify_final_report,
)
from memcontam.experiment.phase12.filter_challenge.final_verifier_quality import (
    _structural_findings,
    quality_commands,
)


@pytest.mark.parametrize(
    ("relative_path", "source"),
    (
        (
            "src/memcontam/experiment/phase12/cli.py",
            "from memcontam.clients.factory import build_llm_client\nbuild_llm_client(None)\n",
        ),
        (
            "scripts/verify_provider.py",
            "import memcontam.clients.openai_compatible as provider\n"
            "def outer():\n"
            "    client = provider.OpenAICompatibleClient\n"
            "    return client(None)\n",
        ),
        (
            "src/memcontam/ordinary.py",
            "import memcontam.clients.factory as provider\n"
            "class Nested:\n"
            "    factory = getattr(provider, 'build_llm_client')\n"
            "    def construct(self):\n"
            "        local = self.factory\n"
            "        return local(None)\n",
        ),
        (
            "scripts/chained_provider.py",
            "import memcontam.clients.openai_responses as provider\n"
            "def outer():\n"
            "    first = second = provider.OpenAIResponsesClient\n"
            "    return second(None, allow_live_calls=True)\n",
        ),
        (
            "scripts/unaliased_factory.py",
            "import memcontam.clients.factory\n"
            "memcontam.clients.factory.build_llm_client(None)\n",
        ),
        (
            "scripts/unaliased_compatible.py",
            "import memcontam.clients.openai_compatible\n"
            "memcontam.clients.openai_compatible.OpenAICompatibleClient(None)\n",
        ),
        (
            "scripts/unaliased_responses.py",
            "import memcontam.clients.openai_responses\n"
            "memcontam.clients.openai_responses.OpenAIResponsesClient(None)\n",
        ),
    ),
)
def test_code_quality_rejects_provider_calls_outside_filter_challenge_and_nested_scopes(
    tmp_path: Path, relative_path: str, source: str
) -> None:
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")

    assert _structural_findings(path) == [f"provider:{path.as_posix()}"]


def test_code_quality_rejects_changed_script_provider_call(tmp_path: Path) -> None:
    fixture = _fixture(
        tmp_path,
        provider_source=(
            "from memcontam.clients.factory import build_llm_client\n"
            "build_llm_client(None)\n"
        ),
    )

    with pytest.raises(FinalVerifierError, match="CODE_QUALITY_REJECTED"):
        verify_final_report(_request(fixture, "code-quality", tmp_path / "f2.json"))


def test_code_quality_allows_provider_implementation_definition_without_construction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "src" / "memcontam" / "clients" / "openai_compatible.py"
    path.parent.mkdir(parents=True)
    path.write_text("class OpenAICompatibleClient:\n    pass\n", encoding="utf-8")

    assert _structural_findings(path) == []


def test_quality_commands_exclude_package_qualified_tests_from_mypy() -> None:
    paths = (
        "src/memcontam/experiment/phase12/filter_challenge/final_verifier_quality.py",
        "tests/test_phase12_filter_v5_final_verifier_modes.py",
        "tests/test_phase12_filter_v5_code_quality.py",
    )

    commands = quality_commands(Path(__file__).resolve().parents[1], paths, "HEAD", "HEAD")

    assert all(command["exit_code"] == 0 for command in commands)


def test_code_quality_payload_binds_changed_commit_metadata(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    report = verify_final_report(_request(fixture, "code-quality", tmp_path / "f2.json"))

    assert report["base_commit"] == fixture.base_commit
    assert report["implementation_commit"] == fixture.evidence.implementation_commit
    assert report["changed_paths"] == ["src/filter_v5_marker.py"]
