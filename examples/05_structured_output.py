#!/usr/bin/env python3
"""Lesson 05: Constrain output to a JSON Schema.

response_format={"type": "json_schema", ...} makes vLLM decode only tokens
that keep the output valid against the schema, so the reply always parses
and has the right fields, even from a small model.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from _common import chat, require_endpoint

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "temp_c": {"type": "number"},
        "condition": {"type": "string", "enum": ["sunny", "cloudy", "rainy"]},
    },
    "required": ["city", "temp_c", "condition"],
    "additionalProperties": False,
}

TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
}


def validate(value: dict[str, Any], schema: dict[str, Any]) -> None:
    """Check the few schema features used above (no extra dependency)."""
    assert set(schema["required"]) <= set(value), f"missing keys in {value}"
    assert set(value) <= set(schema["properties"]), f"extra keys in {value}"
    for key, spec in schema["properties"].items():
        assert isinstance(value[key], TYPES[spec["type"]]), f"{key} has wrong type"
        if "enum" in spec:
            assert value[key] in spec["enum"], f"{key}={value[key]!r} not in enum"


def main() -> None:
    model = require_endpoint()
    if model is None:
        sys.exit(0)

    body = chat(
        model,
        "Make up the current weather in Berlin.",
        max_tokens=120,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "weather", "schema": SCHEMA},
        },
    )
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop", "output truncated by max_tokens"
    content = choice["message"]["content"]
    print("raw:   ", content)
    parsed = json.loads(content)
    validate(parsed, SCHEMA)
    print("valid:  matches the schema")


if __name__ == "__main__":
    main()
