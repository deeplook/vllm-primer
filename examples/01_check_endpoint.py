#!/usr/bin/env python3
"""Lesson 01: Check that a vLLM server is up and see what it serves.

vLLM exposes /health (liveness), /version, and the OpenAI-style /v1/models,
which also reports max_model_len, the longest sequence the server accepts.
Skips cleanly when no server is reachable.
"""

from __future__ import annotations

import sys

from _common import BASE_URL, get_json, require_endpoint


def main() -> None:
    if require_endpoint() is None:
        sys.exit(0)
    print(f"server:  {BASE_URL}")
    print(f"version: {get_json('/version').get('version')}")
    for model in get_json("/v1/models")["data"]:
        print(f"model:   {model['id']} (max_model_len={model.get('max_model_len')})")


if __name__ == "__main__":
    main()
