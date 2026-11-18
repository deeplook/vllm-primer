#!/usr/bin/env python3
"""Lesson 02: One chat completion, and what the response tells you.

Shows the OpenAI-compatible request/response shape: the message, the
finish_reason ("stop" = the model ended, "length" = max_tokens cut it off)
and the token usage the server counted.
"""

from __future__ import annotations

import sys

from _common import chat, require_endpoint


def main() -> None:
    model = require_endpoint()
    if model is None:
        sys.exit(0)

    body = chat(model, "Name the capital of France in one word.", max_tokens=16)
    choice = body["choices"][0]
    print("reply:        ", choice["message"]["content"])
    print("finish_reason:", choice["finish_reason"])
    usage = body["usage"]
    print(
        f"usage:         {usage['prompt_tokens']} prompt + "
        f"{usage['completion_tokens']} completion = {usage['total_tokens']} tokens"
    )
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert choice["message"]["content"].strip(), "empty reply"


if __name__ == "__main__":
    main()
