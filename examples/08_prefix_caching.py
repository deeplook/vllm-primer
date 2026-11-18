#!/usr/bin/env python3
"""Lesson 08: Automatic prefix caching.

vLLM keeps the KV cache of earlier prompts in blocks and reuses any block-
aligned prefix a new prompt shares with them, so a long, stable system
prompt is only computed once. This lesson sends the same long prefix twice
and reads the hit counter from /metrics. (Prefix caching is on by default in
current vLLM; if the counter does not move, check for --no-enable-prefix-caching.)
"""

from __future__ import annotations

import sys
import time
import uuid

from _common import chat, metrics, require_endpoint

# A fresh nonce up front keeps earlier runs from pre-warming this prefix.
PREFIX = (
    f"Session {uuid.uuid4().hex}. You are a careful assistant. Reference document: "
    + "The history of computing begins with mechanical calculators. " * 60
)


def timed(model: str, question: str) -> float:
    t0 = time.perf_counter()
    chat(model, PREFIX + question, temperature=0, max_tokens=16)
    return time.perf_counter() - t0


def main() -> None:
    model = require_endpoint()
    if model is None:
        sys.exit(0)

    before = metrics()
    t1 = timed(model, "Question one: what is 2+2?")
    mid = metrics()
    t2 = timed(model, "Question two: what is 3+3?")
    after = metrics()

    hits1 = mid["vllm:prefix_cache_hits"] - before["vllm:prefix_cache_hits"]
    hits2 = after["vllm:prefix_cache_hits"] - mid["vllm:prefix_cache_hits"]
    queried = after["vllm:prefix_cache_queries"] - mid["vllm:prefix_cache_queries"]
    print(f"call 1: {t1 * 1000:5.0f} ms, {hits1:.0f} cached prompt tokens")
    print(f"call 2: {t2 * 1000:5.0f} ms, {hits2:.0f} of {queried:.0f} tokens cached")
    assert hits2 > hits1, "second call should reuse the shared prefix"


if __name__ == "__main__":
    main()
