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
