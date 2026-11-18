#!/usr/bin/env python3
"""Lesson 09: Read the Prometheus metrics vLLM exposes.

GET /metrics returns Prometheus text. Counters only go up, gauges move both
ways, and histograms come as _sum/_count pairs, so sum/count is a mean. This
lesson sends one request and shows the counters it moved, plus the average
time to first token since the server started.
"""

from __future__ import annotations

import sys

from _common import chat, metrics, require_endpoint

COUNTERS = [
    "vllm:request_success",
    "vllm:prompt_tokens",
    "vllm:generation_tokens",
]
GAUGES = ["vllm:num_requests_running", "vllm:kv_cache_usage_perc"]


def main() -> None:
    model = require_endpoint()
    if model is None:
        sys.exit(0)

    before = metrics()
    chat(model, "Say hello.", max_tokens=16)
    after = metrics()

    for name in COUNTERS:
        print(
            f"{name:<28} +{after[name] - before[name]:.0f}  (total {after[name]:.0f})"
        )
    for name in GAUGES:
        print(f"{name:<28} = {after[name]:.3f}")
    count = after["vllm:time_to_first_token_seconds_count"]
    mean = after["vllm:time_to_first_token_seconds_sum"] / count
    print(f"mean time to first token over {count:.0f} requests: {mean * 1000:.0f} ms")

    assert after["vllm:request_success"] == before["vllm:request_success"] + 1
    assert after["vllm:generation_tokens"] > before["vllm:generation_tokens"]


if __name__ == "__main__":
    main()
