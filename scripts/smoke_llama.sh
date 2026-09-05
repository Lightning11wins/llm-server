#!/usr/bin/env bash
# Smoke-test a GGUF model directly with llama-server (no llm-server involved).
# Usage: scripts/smoke_llama.sh <path/to/model.gguf> [extra llama-server args...]
# Starts llama-server on port 8099, waits for health, runs one completion,
# prints timings and VRAM use, then stops the server.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="$1"; shift
PORT=8099
LOG="$(mktemp)"

bin/llama.cpp/llama-server -m "$MODEL" --host 127.0.0.1 --port $PORT --no-webui "$@" > "$LOG" 2>&1 &
PID=$!
# On exit: stop the server and print only warnings/errors from its log (full log kept if it failed).
cleanup() {
	kill $PID 2>/dev/null; wait $PID 2>/dev/null
	echo "--- llama-server warnings/errors:"
	grep -E '^[0-9.]+ [WE] ' "$LOG" | grep -v -E 'CORS|security risk|more info:|^[0-9.]+ W srv  llama_server: -+$' || echo "(none)"
	if [ "${FAILED:-0}" = 1 ]; then echo "--- full log kept at $LOG"; else rm -f "$LOG"; fi
}
trap cleanup EXIT

echo "Waiting for llama-server (pid $PID)..."
for i in $(seq 1 600); do
	if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
	if ! kill -0 $PID 2>/dev/null; then echo "llama-server exited early"; FAILED=1; exit 1; fi
	sleep 1
done
echo "Ready after ${i}s."

echo "--- nvidia-smi:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true

completion() {  # $1 = label, $2 = JSON request body
	echo "--- $1:"
	curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d "$2" | python3 -c '
import json, sys
r = json.load(sys.stdin)
print(repr(r["content"][:200]))
t = r["timings"]
print("prompt: %d tok @ %.1f tok/s   gen: %d tok @ %.1f tok/s" % (
	t["prompt_n"], t["prompt_per_second"], t["predicted_n"], t["predicted_per_second"]))
'
}

completion "short prompt" '{"prompt":"The three primary colors are","n_predict":64,"temperature":0.7}'

# ~1000-token prompt to measure prompt processing (what a tool-heavy agent mostly does).
LONG=$(python3 -c 'import json; print(json.dumps(" ".join(["The quick brown fox jumps over the lazy dog near the river bank at dawn."] * 90) + " Summary:"))')
completion "long prompt" "{\"prompt\":$LONG,\"n_predict\":32,\"temperature\":0.7,\"cache_prompt\":false}"
