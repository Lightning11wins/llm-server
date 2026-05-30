# LLM Server

Serves local HuggingFace models over HTTP. Place model directories in `models/`.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Install torch — pick the line matching your CUDA version (check with nvidia-smi):
pip install torch                                                       # CPU only
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4
```

## Start

```bash
python3 server.py
```

Default port: `8080`. Change `PORT` at the top of `server.py`. Logs are written to `logs/<timestamp>.log`.

## API

All request bodies are JSON. Errors return `{"error": "..."}` with HTTP 404/400/500.

### GET /list

```
GET /list?loaded=any|true|false
```

```json
[{"name": "llama-3", "loaded": true}, ...]
```

### POST /load

```json
{"model": "llama-3", "ttl": 300}
```

Blocks until loaded. If already loaded, refreshes TTL. `ttl` (seconds) defaults to 300.

Returns `{"status": "loaded", "model": "llama-3"}`.

### POST /run

```json
{
  "model": "llama-3",
  "prompt": "Hello",
  "ttl": 300,
  "autoload": false,
  "max_tokens": 512,
  "temperature": 1.0,
  "top_p": 1.0,
  "repetition_penalty": 1.0
}
```

Required: `model`, `prompt`. If `autoload=true`, `ttl` is also required. All other fields optional.

Streams `text/event-stream`:
```
data: {"token": "Hello"}
data: {"token": " there"}
data: [DONE]
```

On mid-stream error: `data: {"error": "..."}` then stream closes.

## llm.sh

```bash
./llm.sh list [--loaded true|false|any]
./llm.sh load --model <name> [--ttl <s>]
./llm.sh run  --model <name> --prompt <text> [--autoload --ttl <s>] \
              [--max-tokens <n>] [--temperature <f>] [--top-p <f>] [--repetition-penalty <f>]
```

## TODO

- Chat template support (`tokenizer.apply_chat_template()`) for instruction-tuned models

## Developer Notes

- **`torch` is not in `requirements.txt`** — it must be installed separately with a CUDA-version-specific index URL. See Setup above.
- **Models must be HuggingFace-format directories** — each model directory needs `config.json`, tokenizer files, and weight files. Single `.safetensors` or `.bin` files alone won't work.
- **Instruction-tuned models produce poor output without chat templates** — the server sends raw prompts directly to the tokenizer. Until chat template support is added (see TODO), base/completion models work best.
- **`do_sample=True` is always set** — generation is always stochastic. Deterministic/greedy output is not available via the API.
- **`device_map="auto"` silently spills to CPU** — if a model exceeds available VRAM, layers overflow to system RAM with no error or warning, making inference much slower.
- **Model loading is serialized** — only one model loads at a time. Concurrent load requests for different models queue behind each other.
- **Per-model inference queue** — concurrent requests to the same model are queued; requests to different loaded models run in parallel.
- **TTL expiry has up to 5s lag** — the TTL monitor polls every 5 seconds, so models may stay loaded slightly past their TTL.
