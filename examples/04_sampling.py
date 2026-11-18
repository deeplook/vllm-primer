#!/usr/bin/env python3
"""Lesson 04: Temperature, seed, n, and logprobs.

temperature=0 is greedy and repeatable. With temperature>0 the output varies,
unless you pass a seed, which makes it reproducible. `n` returns several
candidates from one request (they share the prompt's prefill), and `logprobs`
exposes the model's confidence in each token.
"""

from __future__ import annotations

import sys

from _common import chat, require_endpoint

PROMPT = "Write a five word slogan for a bicycle shop."


def text(body: dict[str, object]) -> str:
    choices = body["choices"]
    assert isinstance(choices, list)
    return str(choices[0]["message"]["content"]).strip()


def main() -> None:
    model = require_endpoint()
    if model is None:
        sys.exit(0)

    greedy = [text(chat(model, PROMPT, temperature=0, max_tokens=24)) for _ in range(2)]
    print("greedy #1:", greedy[0])
    assert greedy[0] == greedy[1], "temperature=0 should repeat exactly"

    seeded = [
        text(chat(model, PROMPT, temperature=1.0, seed=42, max_tokens=24))
        for _ in range(2)
    ]
    print("seed=42:  ", seeded[0])
    assert seeded[0] == seeded[1], "the same seed should repeat exactly"

    body = chat(model, PROMPT, temperature=1.0, n=3, max_tokens=24)
    print("n=3 candidates:")
    for choice in body["choices"]:
        print("  -", choice["message"]["content"].strip())
    assert len(body["choices"]) == 3

    body = chat(
        model, PROMPT, temperature=0, max_tokens=8, logprobs=True, top_logprobs=2
    )
    first = body["choices"][0]["logprobs"]["content"][0]
    print(f"first token {first['token']!r} logprob={first['logprob']:.3f}")
    assert first["logprob"] <= 0


if __name__ == "__main__":
    main()
