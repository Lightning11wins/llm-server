#!/usr/bin/env bash
# End-to-end test: start llm-server, exercise every model in models/ through the
# llm CLI with a short TTL, confirm eviction, then stop the server.
# Usage: scripts/e2e.sh [model ...]   (defaults to all directories in models/)
set -uo pipefail
cd "$(dirname "$0")/.."

# Keep temp files inside the project: torch needs a writable temp dir when
# server.py is imported, and the server inherits the fd for $LOG below, which
# AppArmor revalidates against the profile at exec.
export TMPDIR="$PWD/tmp"
mkdir -p "$TMPDIR" || exit 1
# The profile grants no write access to __pycache__; see run.sh.
export PYTHONDONTWRITEBYTECODE=1

PORT=8098
TTL=2  # eviction monitor polls every 5s, so waits below are TTL + 6
LOG="$(mktemp)"
MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
	MODELS=($(ls -d models/*/ | xargs -n1 basename))
fi

# Run the server under the AppArmor profile when it is loaded, so the tests
# exercise the same confinement as run.sh. Checked by entering the profile,
# not by looking in /etc/apparmor.d: an installed but unloaded profile would
# make aa-exec fail and the server never start.
CONFINE=""
if aa-exec -p llm-server -- true 2> /dev/null; then
	CONFINE="aa-exec -p llm-server --"
else
	echo "note: the llm-server AppArmor profile is not loaded; running unconfined" >&2
fi

$CONFINE venv/bin/python -c "import server, uvicorn; uvicorn.run(server.app, host='127.0.0.1', port=$PORT)" > "$LOG" 2>&1 &
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

	# Unload mid-generation: the running request must be cancelled and the model released.
	echo "--- unload during generation:"
	./llm --port $PORT load --model "$m" --ttl 60 > /dev/null || fail=1
	rout="$(mktemp)"
	./llm --port $PORT run --model "$m" --prompt "Write a long story about a robot." --max-tokens 4000 > "$rout" 2>&1 &
	rpid=$!
	sleep 3
	if kill -0 $rpid 2>/dev/null; then
		./llm --port $PORT unload --model "$m" || fail=1
		if wait $rpid; then
			echo "(generation finished just before the unload)"  # hit EOS in the gap; not a failure
		elif ! grep -q "was unloaded; request cancelled" "$rout"; then
			echo "RUN FAILED for another reason:"; tail -n 3 "$rout"; fail=1
		fi
	else
		# Nothing left to cancel: the model hit EOS first. Still check the unload itself.
		echo "(generation finished before the unload)"
		./llm --port $PORT unload --model "$m" || fail=1
	fi
	./llm --port $PORT list --loaded true | grep -q '"name"' && { echo "STILL LOADED after unload"; fail=1; }
	./llm --port $PORT unload --model "$m" || fail=1  # idempotent
	rm -f "$rout"
	echo "--- nvidia-smi after unload:"
	nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true
done

echo "--- leftover llama-server processes:"
pgrep -af llama-server | grep -v pgrep || echo "(none)"

echo
[ $fail -eq 0 ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
exit $fail
