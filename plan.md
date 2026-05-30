# LLM Server — Design Plan

A local LLM server that serves HuggingFace models over HTTP for use by other apps and scripts on the same machine. Priority: simplest, most concise implementation possible — the less code the better; large amounts of code are hard to read, understand, and maintain.

## File Structure

```
llm-server/
├── models/          # HuggingFace model subdirectories
├── server.py        # Main server
├── server.log       # Runtime log (generated)
├── llm.sh           # CLI testing script
├── README.md        # Agent-readable API reference
└── plan.md          # This file (not in final project)
```

## Stack

- **Runtime:** Python, FastAPI + uvicorn
- **Inference:** HuggingFace `transformers` + `torch` (CUDA via `device_map="auto"`)
- **Streaming:** `TextIteratorStreamer` (background thread) → FastAPI `StreamingResponse`

## server.py

### Constants (top of file)

```python
PORT = 8080
DEFAULT_TTL = 300           # seconds
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_REPETITION_PENALTY = 1.0
```

### Model Storage

Models are HuggingFace-format subdirectories inside `models/`. The model name used in API requests is the directory name (e.g. `llama-3` → `models/llama-3/`).

### Model Lifecycle

- Models are loaded with `AutoModelForCausalLM` + `AutoTokenizer`, `device_map="auto"` (auto GPU allocation).
- Each loaded model has a TTL countdown. After the TTL expires with no activity, the model is unloaded from memory.
- TTL rule: after each request completes, TTL is set to `max(time_remaining, request_ttl)`. Example: model loaded with ttl=30; 20s later (10s remaining) a run request arrives with ttl=15; after the request, TTL is set to 15 because 15 > 10.
- A model is never unloaded while inference is actively running.
- The `load` endpoint blocks until the model is fully in memory before responding. If the model is already loaded, refresh its TTL and return success immediately.

### Concurrency

- Each model has its own `asyncio.Lock`. Requests for the same model are queued; only one runs at a time.
- Requests for **different** models run in parallel (each holds its own lock independently).

### API

All request bodies are JSON. All responses are JSON except `/run`, which is SSE.

#### `GET /list`

Query params:
- `loaded` (optional): `true` | `false` | `any` (default: `any`)

Response:
```json
[{"name": "llama-3", "loaded": true}, ...]
```

#### `POST /load`

Body:
```json
{"model": "llama-3", "ttl": 300}
```

Blocks until fully loaded. Response:
```json
{"status": "loaded", "model": "llama-3"}
```

#### `POST /run`

Body:
```json
{
  "model": "llama-3",
  "prompt": "Hello, world",
  "ttl": 300,
  "autoload": false,
  "max_tokens": 512,
  "temperature": 1.0,
  "top_p": 1.0,
  "repetition_penalty": 1.0
}
```

Required: `model`, `prompt`. If `autoload` is true, `ttl` is also required. Returns `text/event-stream` (SSE):
```
data: {"token": "Hello"}
data: {"token": " there"}
...
data: [DONE]
```

If an error occurs after streaming has begun, send `data: {"error": "..."}` then close the stream.

#### Error responses (non-streaming)

| Condition | Status |
|-----------|--------|
| Model name not found in `models/` | 404 |
| Model exists but is not loaded (`autoload=false`) | 400 |
| Server/inference error | 500 |

```json
{"error": "description of what went wrong"}
```

### Logging (server.log)

Plain text, one event per line, always with a datetime stamp. Logged events:
- Server start / stop
- Request received (method + endpoint)
- Request completed (duration)
- Model loaded / unloaded
- Any other important events

## Dependencies

### requirements.txt

List all dependencies except `torch`. `torch` is excluded because the correct wheel is CUDA-version-specific and must be installed separately.

### .gitignore

Include `venv/` and `__pycache__/`.

### Install instructions (in README)

Two-step setup:
```bash
# 1. Create and activate virtualenv
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install torch — pick the line matching your CUDA version (check with nvidia-smi)
pip install torch                                                       # CPU only
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4
```

`device_map="auto"` in `server.py` handles GPU vs CPU automatically at runtime — no code changes needed between environments.

## llm.sh

Bash script. Sends requests to the server over HTTP; does not invoke `server.py` directly.

Interface: subcommand + named flags.

```bash
./llm.sh list [--loaded true|false|any]
./llm.sh load --model <name> [--ttl <seconds>]
./llm.sh run  --model <name> --prompt <text> [--autoload --ttl <s>] [--max-tokens <n>] [--temperature <f>]
```

For `run`, tokens are printed to stdout as they arrive (using `curl --no-buffer` to consume the SSE stream live).

## README.md

Short, concise, intended for coding agents integrating with this server. Covers: how to start the server, full API reference with example requests and responses, and `llm.sh` usage. Trim any filler — make every word count.
