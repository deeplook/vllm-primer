# Changelog

## [Unreleased]

### Added
- Lessons 01-10: endpoint check, chat completion, streaming, sampling, structured
  output, tool calling, continuous batching, prefix caching, metrics and the
  OpenAI SDK, all against a server at `VLLM_BASE_URL`.
- `tools/mock_server.py`: a vLLM-shaped mock (chat, SSE streaming, `n`, `seed`,
  `logprobs`, JSON-schema output, tool calls, Prometheus metrics) for offline
  use and CI.
- `tools/aws_gpu.py`: `plan`, `up`, `status`, `wait`, `logs`, `forward`, `down`
  and `purge` for a single `g4dn.xlarge` (T4) running the pinned vLLM image.
  No inbound ports (SSM tunnel), IMDSv2, encrypted volume, auto-terminate, and
  every resource tagged `Project=vllm-primer`. AWS credentials come only from
  `~/.aws`; deleting needs explicit ids plus a confirmation variable.
- `.env` / `.env.example` configuration; ambient `HF_TOKEN` is never forwarded.
- Makefile with `check-all` (lint, strict mypy, tests), `live-*` and
  `clean-remote*` targets, CI on the locked environment, optional pre-commit hook.

### Verified
- 2026-11-18: all ten lessons ran against real vLLM v0.33.0 serving
  Qwen2.5-0.5B-Instruct on a `g4dn.xlarge` (T4) in `eu-central-1`, through the
  SSM tunnel, with `tools/aws_gpu.py` launching and terminating the instance.
  Observed: 5.3x batching speedup for 8 concurrent requests, 656 of 682 prompt
  tokens served from the prefix cache on the second call, ~60 ms time to first
  token. The whole run cost well under $0.50.

### Known gaps
- vLLM v0.33 prints a deprecation warning for `--model`; it still works, but
  `aws_gpu.py` should pass the model positionally before the image pin is raised.
- Spot has not been tried yet (the account's Spot quota was 0 at the time).
