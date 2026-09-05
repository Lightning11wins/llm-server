#!/usr/bin/env bash
# End-to-end test: start llm-server, exercise every model in models/ through the
# llm CLI with a short TTL, confirm eviction, then stop the server.
# Usage: scripts/e2e.sh [model ...]   (defaults to all directories in models/)
set -uo pipefail
cd "$(dirname "$0")/.."

PORT=8098
TTL=2  # eviction monitor polls every 5s, so waits below are TTL + 6
LOG="$(mktemp)"
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
	MODELS=($(ls -d models/*/ | xargs -n1 basename))
fi

PORT=$PORT venv/bin/python -c "import server, uvicorn; uvicorn.run(server.app, host='127.0.0.1', port=$PORT)" > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null; echo; echo "--- server log:"; cat "$LOG"; rm -f "$LOG"' EXIT

for i in $(seq 1 30); do
	curl -sf "http://127.0.0.1:$PORT/list" > /dev/null 2>&1 && break
	sleep 0.5
done

fail=0
for m in "${MODELS[@]}"; do
	echo "=== $m"
	echo "--- load:"
	./llm --port $PORT load --model "$m" --ttl $TTL || fail=1
	echo "--- run:"
	./llm --port $PORT run --model "$m" --ttl $TTL --prompt "The capital of France is" --max-tokens 24 --temperature 0.7 || fail=1
	echo "--- nvidia-smi:"
	nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true
	echo "--- loaded:"
	./llm --port $PORT list --loaded true
	# Wait for eviction before the next model so VRAM is not shared between models.
	echo "--- waiting $((TTL + 6))s for TTL eviction"
	sleep $((TTL + 6))
	./llm --port $PORT list --loaded true | grep -q '"name"' && { echo "STILL LOADED after TTL"; fail=1; }
	echo "--- nvidia-smi after eviction:"
	nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true
done

echo "--- leftover llama-server processes:"
pgrep -af llama-server | grep -v pgrep || echo "(none)"

echo
[ $fail -eq 0 ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
