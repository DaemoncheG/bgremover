import subprocess
import sys


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "main.py", *args],
        text=True,
        capture_output=True,
    )


def test_list_models():
    result = run_cli("--list-models")
    assert result.returncode == 0
    assert "u2net" in result.stdout.splitlines()


def test_missing_input_outputs_help():
    result = run_cli()
    assert result.returncode != 0
    assert "usage:" in result.stdout.lower()
