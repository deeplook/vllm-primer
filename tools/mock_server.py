#!/usr/bin/env python3
"""Tiny vLLM-shaped mock server for CI and offline development.

Speaks the subset of vLLM's OpenAI-compatible API that the lessons use:
/health, /version, /v1/models, /v1/chat/completions (plain, streaming,
n, seed, logprobs, json_schema output, tool calls) and /metrics. Replies are
canned, so it proves the lessons' client code works, not that vLLM does.

Usage: mock_server.py [PORT]   (default 8000)
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

PORT = 8000
MODEL = "mock-model"
BLOCK = 16  # prefix-cache granularity, like vLLM's KV block size
LATENCY = 0.05  # seconds per request; makes concurrency visible in lesson 07
TOKEN_DELAY = 0.005  # seconds per streamed token

LOCK = threading.Lock()
STATS: dict[str, float] = {
    "requests": 0,
    "running": 0,
    "prompt_tokens": 0,
    "generation_tokens": 0,
    "prefix_queries": 0,
    "prefix_hits": 0,
    "ttft_sum": 0.0,
    "ttft_count": 0,
}
SEEN_PROMPTS: list[list[str]] = []


def tokens(text: str) -> list[str]:
    return text.split()


def example_value(schema: dict[str, Any], name: str = "") -> Any:
    """Build a value that satisfies a (simple) JSON schema."""
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        return {
            key: example_value(sub, key)
            for key, sub in schema.get("properties", {}).items()
        }
    if kind == "array":
        return [example_value(schema.get("items", {}))]
    if kind == "integer":
        return 1
    if kind == "number":
        return 18.5
    if kind == "boolean":
        return True
    return "Berlin" if name == "city" else "example"


def prompt_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages)


def cache_lookup(prompt: list[str]) -> int:
    """Longest block-aligned prefix shared with an earlier prompt."""
    best = 0
    for seen in SEEN_PROMPTS:
        n = 0
        while n < min(len(seen), len(prompt)) and seen[n] == prompt[n]:
            n += 1
        best = max(best, n - n % BLOCK)
    SEEN_PROMPTS.append(prompt)
    del SEEN_PROMPTS[:-50]
    return best


def reply_text(body: dict[str, Any], index: int) -> str:
    messages = body.get("messages", [])
    last = str(messages[-1].get("content") or "") if messages else ""
    seed = body.get("seed")
    if body.get("temperature", 1.0) == 0:
        key = f"{last}|greedy"
    elif seed is not None:
        key = f"{last}|{seed}|{index}"
    else:
        key = f"{last}|{random.random()}"
    word = hashlib.sha256(key.encode()).hexdigest()[:6]
    return f"Mock reply {word} to: {last[:40]}"


def tool_call_for(body: dict[str, Any]) -> dict[str, Any] | None:
    tools = body.get("tools") or []
    if not tools or any(m.get("role") == "tool" for m in body["messages"]):
        return None
    fn = tools[0]["function"]
    args = example_value(fn.get("parameters", {"type": "object"}))
    return {
        "id": "call_mock_0",
        "type": "function",
        "function": {"name": fn["name"], "arguments": json.dumps(args)},
    }


def completion(body: dict[str, Any]) -> dict[str, Any]:
    prompt = tokens(prompt_text(body["messages"]))
    n = int(body.get("n", 1))
    choices: list[dict[str, Any]] = []
    for i in range(n):
        message: dict[str, Any] = {"role": "assistant", "content": None}
        finish = "stop"
        fmt = body.get("response_format") or {}
        if call := tool_call_for(body):
            message["tool_calls"] = [call]
            finish = "tool_calls"
        elif fmt.get("type") == "json_schema":
            schema = fmt["json_schema"]["schema"]
            message["content"] = json.dumps(example_value(schema))
        elif any(m.get("role") == "tool" for m in body["messages"]):
            message["content"] = "The tool result says: " + str(
                body["messages"][-1].get("content")
            )
        else:
            message["content"] = reply_text(body, i)
        choice: dict[str, Any] = {
            "index": i,
            "message": message,
            "finish_reason": finish,
            "logprobs": None,
        }
        if body.get("logprobs") and message["content"]:
            top = int(body.get("top_logprobs") or 0)
            choice["logprobs"] = {
                "content": [
                    {
                        "token": tok,
                        "logprob": -0.1,
                        "top_logprobs": [
                            {"token": tok, "logprob": -0.1} for _ in range(top)
                        ],
                    }
                    for tok in tokens(message["content"])
                ]
            }
        choices.append(choice)
    generated = sum(len(tokens(c["message"]["content"] or "")) or 4 for c in choices)
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", MODEL),
        "choices": choices,
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": generated,
            "total_tokens": len(prompt) + generated,
        },
    }


def render_metrics() -> str:
    s = STATS
    label = f'model_name="{MODEL}"'
    lines = [
        f"vllm:num_requests_running{{{label}}} {s['running']}",
        f"vllm:kv_cache_usage_perc{{{label}}} 0.01",
        f"vllm:prompt_tokens_total{{{label}}} {s['prompt_tokens']}",
        f"vllm:generation_tokens_total{{{label}}} {s['generation_tokens']}",
        f'vllm:request_success_total{{finished_reason="stop",{label}}} {s["requests"]}',
        f"vllm:prefix_cache_queries_total{{{label}}} {s['prefix_queries']}",
        f"vllm:prefix_cache_hits_total{{{label}}} {s['prefix_hits']}",
        f"vllm:time_to_first_token_seconds_sum{{{label}}} {s['ttft_sum']}",
        f"vllm:time_to_first_token_seconds_count{{{label}}} {s['ttft_count']}",
    ]
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        pass  # quiet

    def _send(self, code: int, payload: Any, ctype: str = "application/json") -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/health":
            self._send(200, b"", "text/plain")
        elif path == "/version":
            self._send(200, {"version": "mock"})
        elif path == "/v1/models":
            model = {
                "id": MODEL,
                "object": "model",
                "owned_by": "mock",
                "max_model_len": 4096,
            }
            self._send(200, {"object": "list", "data": [model]})
        elif path == "/metrics":
            with LOCK:
                self._send(200, render_metrics().encode(), "text/plain")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.split("?")[0] != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        start = time.perf_counter()
        prompt = tokens(prompt_text(body["messages"]))
        with LOCK:
            STATS["running"] += 1
            STATS["prefix_queries"] += len(prompt)
            STATS["prefix_hits"] += cache_lookup(prompt)
        first = 0.0
        result: dict[str, Any] = {"usage": {"completion_tokens": 0}}
        try:
            time.sleep(LATENCY)
            result = completion(body)
            first = time.perf_counter() - start
            if body.get("stream"):
                self._stream(body, result)
            else:
                self._send(200, result)
        finally:
            with LOCK:
                STATS["running"] -= 1
                STATS["requests"] += 1
                STATS["prompt_tokens"] += len(prompt)
                STATS["generation_tokens"] += result["usage"]["completion_tokens"]
                STATS["ttft_sum"] += first
                STATS["ttft_count"] += 1

    def _stream(self, body: dict[str, Any], result: dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def emit(chunk: dict[str, Any]) -> None:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        base = {"id": "chatcmpl-mock", "object": "chat.completion.chunk"}
        base["model"] = body.get("model", MODEL)
        message = result["choices"][0]["message"]
        text = message.get("content") or ""
        emit({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
        for word in tokens(text):
            time.sleep(TOKEN_DELAY)
            delta = {"content": word + " "}
            emit({**base, "choices": [{"index": 0, "delta": delta}]})
        done = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        emit({**base, "choices": done})
        if (body.get("stream_options") or {}).get("include_usage"):
            emit({**base, "choices": [], "usage": result["usage"]})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Mock vLLM server on http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
