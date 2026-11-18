#!/usr/bin/env python3
"""Lesson 10: Use the official OpenAI SDK against vLLM.

Because vLLM speaks the OpenAI API, switching an app from a hosted model to
your own server is one line: base_url. The key is only checked when the
server was started with --api-key.
"""

from __future__ import annotations

import os
import sys

from _common import BASE_URL, require_endpoint


def main() -> None:
    try:
        from openai import OpenAI
    except ImportError:
        print("SKIP: openai not installed. Run: uv sync --all-groups")
        sys.exit(0)
    model = require_endpoint()
    if model is None:
        sys.exit(0)

    client = OpenAI(
        base_url=f"{BASE_URL}/v1", api_key=os.environ.get("VLLM_API_KEY") or "EMPTY"
    )
    print("models:", [m.id for m in client.models.list().data])

    reply = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hi in one word."}],
        max_tokens=16,
    )
    content = reply.choices[0].message.content
    print("reply: ", content)
    assert content and content.strip()

    print("stream: ", end="")
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Count to three."}],
        max_tokens=32,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
