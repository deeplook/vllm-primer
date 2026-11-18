#!/usr/bin/env python3
"""Lesson 03: Stream tokens as server-sent events and time them.

With stream=True the server sends `data: {...}` lines, ending with
`data: [DONE]`. Measuring the gap before the first chunk gives time to first
token (TTFT); the rest is decode speed. stream_options.include_usage adds a
final chunk with token counts.
"""

from __future__ import annotations

import sys
import time

from _common import require_endpoint, stream_json


def main() -> None:
    model = require_endpoint()
    if model is None:
        sys.exit(0)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Count from one to five."}],
        "max_tokens": 48,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first: float | None = None
    pieces: list[str] = []
    usage = None
    for chunk in stream_json("/v1/chat/completions", payload):
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk["choices"]:
            if text := choice["delta"].get("content"):
                if first is None:
                    first = time.perf_counter() - start
                pieces.append(text)
                print(text, end="", flush=True)
    total = time.perf_counter() - start
    print()
    assert first is not None and pieces, "no content streamed"
    assert usage is not None, "include_usage should add a final usage chunk"
    print(f"chunks: {len(pieces)}, TTFT: {first * 1000:.0f} ms, total: {total:.2f} s")
    print(f"completion tokens reported by the server: {usage['completion_tokens']}")


if __name__ == "__main__":
    main()
