import subprocess


def test_no_ignored_files_are_tracked() -> None:
    result = subprocess.run(
        ["git", "ls-files", "-ci", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "", (
        "tracked files match .gitignore; remove them with `git rm --cached`:\n"
        f"{result.stdout}"
    )
