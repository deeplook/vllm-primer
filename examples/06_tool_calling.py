#!/usr/bin/env python3
"""Lesson 06: Function calling with the tools API.

The server must be started with --enable-auto-tool-choice and a
--tool-call-parser that matches the model family (tools/aws_gpu.py passes
`hermes`, which fits Qwen2.5). The model replies with tool_calls instead of
text; you run the function and send the result back as a `tool` message.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from _common import post_json, require_endpoint

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def main() -> None:
    model = require_endpoint()
    if model is None:
        sys.exit(0)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather in Berlin?"}
    ]
    body = post_json(
        "/v1/chat/completions",
        {"model": model, "messages": messages, "tools": TOOLS, "max_tokens": 128},
    )
    msg = body["choices"][0]["message"]
    calls = msg.get("tool_calls")
    if not calls:
        print("No tool call returned (is the server running with tool support?)")
        print("reply:", msg.get("content"))
        return

    call = calls[0]
    args = json.loads(call["function"]["arguments"])
    print(f"Tool call: {call['function']['name']}({args})")
    assert call["function"]["name"] == "get_weather" and "city" in args

    result = {"city": args["city"], "temp_c": 18, "condition": "cloudy"}
    messages += [
        msg,
        {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)},
    ]
    body = post_json(
        "/v1/chat/completions",
        {"model": model, "messages": messages, "max_tokens": 128},
    )
    print("Final answer:", body["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
