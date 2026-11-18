#!/usr/bin/env python3
"""Lesson 07: Continuous batching, why concurrency is nearly free.

vLLM schedules every in-flight request into the same forward pass, so sending
eight requests at once takes far less than eight times as long as one. This
lesson times the same requests sequentially and concurrently and compares
tokens per second.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

from _common import chat, require_endpoint

N = 8
PROMPT = "Write two sentences about the sea."


def run(model: str) -> int:
    body = chat(model, PROMPT, temperature=0.7, max_tokens=48)
    return int(body["usage"]["completion_tokens"])


def main() -> None:
    model = require_endpoint()
    if model is None:
        sys.exit(0)
    run(model)  # warm-up (CUDA graphs, caches)

    t0 = time.perf_counter()
    seq_tokens = sum(run(model) for _ in range(N))
    seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N) as pool:
        par_tokens = sum(pool.map(lambda _: run(model), range(N)))
    par = time.perf_counter() - t0

    print(f"sequential: {seq:.2f} s, {seq_tokens / seq:6.1f} tokens/s")
    print(f"concurrent: {par:.2f} s, {par_tokens / par:6.1f} tokens/s")
    print(f"speedup:    {seq / par:.1f}x for {N} requests")
    assert par < seq, "concurrent requests should finish sooner than sequential"


if __name__ == "__main__":
    main()
