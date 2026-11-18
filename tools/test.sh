#!/usr/bin/env bash
# Smoke test: start the mock vLLM server and hit its endpoints with curl.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PORT="${MOCK_PORT:-18000}"

echo "==> Starting mock server on :$PORT"
python3 tools/mock_server.py "$PORT" >/dev/null &
MOCK_PID=$!
trap 'kill $MOCK_PID' EXIT
for _ in $(seq 20); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  sleep 0.25
done

echo "==> /v1/models"
curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q 'mock-model'

echo "==> /v1/chat/completions"
curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock-model","messages":[{"role":"user","content":"hi"}]}' \
  | grep -q 'Mock reply'

echo "==> /metrics"
curl -sf "http://127.0.0.1:$PORT/metrics" | grep -q 'vllm:request_success_total'

echo "==> All smoke tests passed"
