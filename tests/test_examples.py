"""Run every lesson against the mock server (no GPU, no network).

These tests prove the lessons' client code, parsing and assertions work against
the vLLM API *shape*. They cannot prove vLLM itself behaves this way; for that
run `make live-run` against a real server (see the README).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(EXAMPLES))

from _common import load_env, parse_metrics  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def mock_url() -> Iterator[str]:
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "tools" / "mock_server.py"), str(port)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        proc.stdout.readline()  # "Mock vLLM server on ..."
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def run_example(
    name: str, base_url: str, **env: str
) -> subprocess.CompletedProcess[str]:
    full_env = {**os.environ, "VLLM_BASE_URL": base_url, "VLLM_MODEL": "", **env}
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
        env=full_env,
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("01_check_endpoint.py", "max_model_len=4096"),
        ("02_chat_completion.py", "finish_reason: stop"),
        ("03_streaming.py", "TTFT"),
        ("04_sampling.py", "n=3 candidates"),
        ("05_structured_output.py", "matches the schema"),
        ("06_tool_calling.py", "Tool call: get_weather({'city': 'Berlin'})"),
        ("07_batching.py", "speedup:"),
        ("08_prefix_caching.py", "tokens cached"),
        ("09_metrics.py", "mean time to first token"),
        ("10_openai_sdk.py", "stream:"),
    ],
)
def test_lessons_against_mock(name: str, expected: str, mock_url: str) -> None:
    result = run_example(name, mock_url)
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected in result.stdout


def test_tool_round_trip_reaches_final_answer(mock_url: str) -> None:
    out = run_example("06_tool_calling.py", mock_url).stdout
    assert "Final answer:" in out and "18" in out


@pytest.mark.parametrize(
    "path", sorted(EXAMPLES.glob("[0-9]*.py")), ids=lambda p: p.name
)
def test_lessons_skip_cleanly_without_a_server(path: Path) -> None:
    """With nothing listening, every lesson prints SKIP and exits 0."""
    result = run_example(path.name, f"http://127.0.0.1:{free_port()}")
    assert result.returncode == 0, result.stderr
    assert "SKIP" in result.stdout


def test_parse_metrics_sums_labels_and_strips_total() -> None:
    text = (
        "# HELP x\n"
        'vllm:request_success_total{finished_reason="stop",model_name="m"} 3\n'
        'vllm:request_success_total{finished_reason="length",model_name="m"} 2\n'
        'vllm:num_requests_running{model_name="m"} 1.5\n'
        "vllm:time_to_first_token_seconds_sum 4\n"
    )
    parsed = parse_metrics(text)
    assert parsed["vllm:request_success"] == 5
    assert parsed["vllm:num_requests_running"] == 1.5
    assert parsed["vllm:time_to_first_token_seconds_sum"] == 4


def test_load_env_never_overrides_and_ignores_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nA_SET=from_file\nexport B_NEW='quoted'\nC_EMPTY=\nnot a line\n"
    )
    monkeypatch.setenv("A_SET", "from_shell")
    monkeypatch.delenv("B_NEW", raising=False)
    monkeypatch.delenv("C_EMPTY", raising=False)
    load_env(env_file)
    assert os.environ["A_SET"] == "from_shell"
    assert os.environ["B_NEW"] == "quoted"
    assert "C_EMPTY" not in os.environ
    monkeypatch.delenv("B_NEW")
