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

### Known gaps
- The lessons have been exercised against the mock only. A run on a real GPU
  instance (`make live-up` ... `make live-run`) has not been done yet.
