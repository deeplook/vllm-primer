"""Shared helpers for the lessons: .env, HTTP to a vLLM server, metrics."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SKIP_HINT = (
    "Start a server first: `make mock` (fake, no GPU), `make live-up` then "
    "`make live-forward` (real, AWS GPU), or point VLLM_BASE_URL at your own."
)


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ; real variables win."""
    env_file = path or ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip().strip("\"'")
        if value:  # an empty value in .env means "unset"
            os.environ.setdefault(key.strip(), value)


load_env()

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("VLLM_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _open(path: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    req = urllib.request.Request(
        BASE_URL + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers=_headers(),
        method="GET" if payload is None else "POST",
    )
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"{err.code} from {path}: {err.read().decode()[:300]}")


def get_json(path: str, timeout: float = 10) -> Any:
    with _open(path, None, timeout) as resp:
        return json.load(resp)


def get_text(path: str, timeout: float = 10) -> str:
    with _open(path, None, timeout) as resp:
        return str(resp.read().decode())


def post_json(path: str, payload: dict[str, Any], timeout: float = 120) -> Any:
    with _open(path, payload, timeout) as resp:
        return json.load(resp)


def stream_json(path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """POST with stream=True and yield each server-sent-event JSON chunk."""
    with _open(path, {**payload, "stream": True}, 120) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                return
            yield json.loads(data)


def endpoint_up() -> bool:
    try:
        _open("/health", None, 3).close()
        return True
    except Exception:
        return False


def require_endpoint() -> str | None:
    """Return the model name to use, or print a SKIP line and return None."""
    if not endpoint_up():
        print(f"SKIP: no vLLM server at {BASE_URL}. {SKIP_HINT}")
        return None
    return model_name()


def model_name() -> str:
    """$VLLM_MODEL, else the first model the server reports."""
    if configured := os.environ.get("VLLM_MODEL"):
        return configured
    return str(get_json("/v1/models")["data"][0]["id"])


def chat(model: str, content: str, **params: Any) -> dict[str, Any]:
    """One non-streaming chat completion with a single user message."""
    body: dict[str, Any] = post_json(
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": content}], **params},
    )
    return body


# --- Prometheus metrics ----------------------------------------------------

_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)$")


def parse_metrics(text: str) -> dict[str, float]:
    """Sum every sample per metric name; counters lose their _total suffix."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not (m := _SAMPLE.match(line.strip())):
            continue
        name = m.group(1).removesuffix("_total")
        try:
            totals[name] = totals.get(name, 0.0) + float(m.group(3))
        except ValueError:
            continue
    return totals


def metrics() -> dict[str, float]:
    return parse_metrics(get_text("/metrics"))


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)
