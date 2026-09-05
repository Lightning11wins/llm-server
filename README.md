# LLM Server

Serves local language models over HTTP. Each model lives in its own directory under `models/` and is run by one of two backends:

| Backend        | Model format                              | Best for                                    |
|----------------|-------------------------------------------|---------------------------------------------|
| `llama-cpp`    | single GGUF file, run via `llama-server`  | quantized models, GPU + CPU offload (MoE)   |
| `transformers` | HuggingFace directory (safetensors etc.)  | small unquantized models such as GPT-2      |

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Install torch — pick the line matching your CUDA version (check with nvidia-smi):
pip install torch                                                       # CPU only
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4
```

### llama.cpp (for the `llama-cpp` backend)

The server expects `bin/llama.cpp/llama-server`. `bin/` is gitignored.

**CUDA build (recommended on NVIDIA).** There is no prebuilt Linux CUDA release, so build it. Needs the CUDA toolkit (from NVIDIA's apt repo; the distro's 12.0 package does not support gcc 13), `cmake` and `libcurl4-openssl-dev`:

```bash
git clone --depth 1 --branch b10819 https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp-src
cd /tmp/llama.cpp-src
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON -DCMAKE_BUILD_RPATH=/usr/local/cuda/lib64
cmake --build build --config Release -j 12 --target llama-server llama-cli llama-bench
mkdir -p <repo>/bin/llama.cpp && cp build/bin/* <repo>/bin/llama.cpp/
```

`CMAKE_CUDA_ARCHITECTURES=89` is the RTX 4060 (Ada); change it for other GPUs. The two RPATH flags make the binaries find their own `.so` files and the CUDA runtime from any location.

**Vulkan prebuilt (no toolchain needed, slower prompt processing).**

```bash
mkdir -p bin && cd bin
curl -L -o llama.tar.gz https://github.com/ggml-org/llama.cpp/releases/download/b10819/llama-b10819-bin-ubuntu-vulkan-x64.tar.gz
tar xzf llama.tar.gz && rm llama.tar.gz && mv llama-b* llama.cpp
```

Either way, `bin/llama.cpp/llama-cli --list-devices` should list your GPU.

## Models

Every model directory **must** contain a `model.json` naming its backend. Directories without one are listed by `/list` but rejected by `/load` and `/run`.

### llama-cpp

```json
{
  "backend": "llama-cpp",
  "model": "Qwen3.5-9B-Q4_K_M.gguf",
  "args": ["-ngl", "99", "-c", "8192", "-fa", "on"],
  "startup_timeout": 300
}
```

- `model` (required): GGUF filename inside the model directory.
- `args` (optional): extra `llama-server` flags. The server always sets `-m`, `--host 127.0.0.1`, `--port <free port>`, `--no-webui` and `--jinja`, and rejects configs that try to override them. Run `bin/llama.cpp/llama-server --help` for the full list. Useful ones: `-ngl N` (layers on GPU), `-c N` (context size), `-fa on` (flash attention), `--n-cpu-moe N` (keep the first N layers' MoE experts in CPU RAM), `-ctk q8_0 -ctv q8_0` (quantized KV cache), `--fit on` (auto-tune to fit device memory), `--reasoning-budget N` (cap thinking tokens).
- `startup_timeout` (optional, seconds): how long to wait for the model to load. Default 300.
- `template_defaults` (optional, any backend): chat-template arguments applied to every chat request unless the request's `template_args` overrides them, e.g. `{"enable_thinking": false}` to make a Qwen model answer without thinking by default.

Chat requests use the chat template embedded in the GGUF. `llama-server` renders it, separates the model's thinking into `reasoning` events and parses tool calls into `tool_calls` events.

Each subprocess logs to `logs/llama-<model>-<timestamp>.log`.

Downloading a GGUF with the HuggingFace CLI (installed with `transformers`):

```bash
mkdir -p models/qwen3.5-9b
HF_HUB_DISABLE_XET=1 hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir models/qwen3.5-9b
rm -rf models/qwen3.5-9b/.cache
```

Test a GGUF directly, without the server, to tune flags. It prints VRAM use plus short- and long-prompt speeds:

```bash
scripts/smoke_llama.sh models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf -ngl 99 -c 8192 -fa on
```

#### Tuning notes (RTX 4060 8 GB, 30 GB RAM, CUDA build b10819)

| Model | Flags | VRAM | Prompt | Generation |
|---|---|---|---|---|
| Qwen3.5-9B Q4_K_M | `-ngl 99 -c 8192 -fa on` | 5.9 GB | ~94 tok/s* | 44 tok/s |
| Qwen3.6-35B-A3B UD-Q4_K_XL | `-ngl 99 --n-cpu-moe 34 -c 32768 -fa on -ctk q8_0 -ctv q8_0 --load-mode none` | 6.7 GB | 533 tok/s | 44 tok/s |

\* measured on a 5-token prompt, so not meaningful; the 35B figure is from a 1442-token prompt.

For the MoE model: `--n-cpu-moe N` keeps the experts of the first N of 40 layers in system RAM (~0.5 GB per layer at this quant), and `--load-mode none` disables mmap, which the loader recommends when experts are on CPU. All 40 layers on CPU used 3.6 GB VRAM at 24 tok/s; 34 gives 44 tok/s; 33 gave no further gain and left under 1 GB free. Leave `-ngl 99` set explicitly, otherwise the `--fit` heuristic may pick a different split.

### transformers

```json
{"backend": "transformers"}
```

The directory must be a standard HuggingFace model directory: `config.json`, tokenizer files, and weight files. It is loaded with `device_map="auto"`.

Chat requests are rendered with the tokenizer's chat template. A model whose tokenizer has none (GPT-2) accepts raw prompts only and answers chat requests with HTTP 400. This backend does not split reasoning or tool calls out of the output: whatever the model prints arrives as `token` events.

## Start

To start, simply run `./run.sh` in the project directory. This sets up the AppArmor profile and Python environment automatically, asking before anything that needs `sudo` or downloads packages.

Binds `127.0.0.1:8080`. Change `HOST` and `PORT` at the top of `server.py`; `HOST = "0.0.0.0"` is there commented out for serving other machines, which also needs a one-line profile change. Logs are written to `logs/<timestamp>.log`.

## AppArmor

`run.sh` confines the server with the AppArmor profile in `apparmor/llm-server`, and the `llama-server` subprocess with a tighter one: read-only on `models/`, TCP only, and the only thing it can write is its own log. The server itself can write `logs/` and `tmp/`, nothing else in the project. `run.sh` copies the profile to `/etc/apparmor.d` and reloads it whenever the two differ, asking first, since that part needs `sudo`. The profile takes the checkout path from `/etc/apparmor.d/tunables/llm-server`, a one-line file `run.sh` also writes, so the profile itself has nothing machine-specific in it.

The profile also pins both processes to loopback addresses, but only on a kernel that mediates socket addresses (`/sys/kernel/security/apparmor/features/network_v9/af_inet` exists). On other kernels the rules load as plain TCP permission and `run.sh` prints a warning.

`./run.sh --unconfined` skips all of it. To remove the profile: `sudo apparmor_parser -R /etc/apparmor.d/llm-server && sudo rm /etc/apparmor.d/llm-server /etc/apparmor.d/tunables/llm-server`.

## API

All request bodies are JSON. Errors return `{"error": "..."}` with HTTP 404/400/500.

### GET /list

```
GET /list?loaded=any|true|false
```

```json
[{"name": "qwen3.5-9b", "loaded": true, "context_length": 8192}, ...]
```

`context_length` is the total number of tokens (prompt plus completion) the loaded model accepts; `null` when the model is not loaded. For `llama-cpp` it is the per-slot context (`-c` divided by the number of parallel slots).

### POST /load

```json
{"model": "qwen3.5-9b", "ttl": 300}
```

Blocks until loaded. If already loaded, refreshes TTL. `ttl` (seconds) defaults to 300.

Returns `{"status": "loaded", "model": "qwen3.5-9b", "context_length": 8192}`. Returns HTTP 409 if the model is unloaded while this load is in flight.

### POST /unload

```json
{"model": "qwen3.5-9b"}
```

Unloads the model immediately, freeing its VRAM without waiting for anything in flight. Requests generating with it are cancelled and end with `data: {"error": "Model '...' was unloaded; request cancelled after N tokens"}`.

Returns `{"status": "...", "model": "qwen3.5-9b"}` with one of:

| Status | Meaning |
|---|---|
| `unloaded` | The model was loaded and has been released. |
| `cancelled` | A load was in flight; it is released the moment it finishes and its `/load` returns 409. |
| `unloading` | A TTL eviction or an earlier unload is already releasing it. |
| `not_loaded` | Nothing to do. |

Unloading is idempotent, so any of these is a success. HTTP 404 is returned only for a name that is neither loaded nor present in `models/`.

### POST /run

Generates from either a raw `prompt` or a chat `messages` list; exactly one of the two is required.

```json
{
  "model": "qwen3.5-9b",
  "prompt": "Hello",
  "stop": ["\n\n"],
  "ttl": 300,
  "autoload": false,
  "max_tokens": 512,
  "temperature": 1.0,
  "top_p": 1.0,
  "repetition_penalty": 1.0
}
```

```json
{
  "model": "qwen3.5-9b",
  "messages": [
    {"role": "system", "content": "You are a terse sysadmin assistant."},
    {"role": "user", "content": "How much disk is free?"}
  ],
  "tools": [
    {"type": "function", "function": {
      "name": "run_shell",
      "description": "Run a shell command and return its output",
      "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
    }}
  ],
  "template_args": {"enable_thinking": false},
  "stop": [],
  "max_tokens": 512
}
```

- `prompt`: raw text, fed to the model as is. No template, no special tokens.
- `messages`: OpenAI-style chat messages, rendered server-side through the model's chat template. Roles `system`, `user`, `assistant` and `tool` mean what the template says they mean; the server only checks that each message has a string `role`. Messages that the template rejects return HTTP 400 with the template's error.
- `tools` (chat only): OpenAI function schemas. When the model decides to call one, the stream ends with a `tool_calls` event and `stop_reason` `tool`.
- `template_args` (chat only): extra variables for the chat template, merged over the model's `template_defaults`. `{"enable_thinking": false}` turns thinking off on Qwen models. Every model family spells these differently, so nothing here is interpreted by the server.
- `stop`: up to 16 strings; generation ends when the output ends with one. The stop string itself is omitted from the output on `llama-cpp` and included on `transformers`.
- `ttl`, `autoload`, `max_tokens`, `temperature`, `top_p`, `repetition_penalty`: as before. `max_tokens` covers thinking and answer together.

Streams `text/event-stream`. Each event has exactly one key:

```
data: {"reasoning": "The user wants"}        model's thinking, chat mode only, when the template separates it
data: {"token": "Hello"}                     answer text
data: {"tool_calls": [{"id": "...", "type": "function", "function": {"name": "run_shell", "arguments": "{\"command\":\"df -h\"}"}}]}
data: {"stop_reason": "eos"}                 eos | length | stop | tool
data: {"usage": {"prompt_tokens": 281, "completion_tokens": 26, "context_length": 8192}}
data: [DONE]
```

`token` and `reasoning` events stream as generated; `tool_calls`, `stop_reason` and `usage` arrive once at the end, in that order. `arguments` is a JSON string, as in the OpenAI API. `stop_reason` `eos` means the model ended its turn, `length` that `max_tokens` was hit, `stop` that a `stop` string matched, `tool` that the model requested tool calls. In chat mode on `llama-cpp` a matched stop string is reported as `eos`, since `llama-server` does not distinguish the two there.

On mid-stream error: `data: {"error": "..."}` then stream closes.

#### Writing a harness

The server is stateless: send the whole conversation every turn. `llama-server` caches the KV state of the common prefix, so a growing history costs prompt processing only for the new tail.

1. Send `messages` (and `tools`) and collect the stream. Concatenate `token` fragments into `content` and `reasoning` fragments into `reasoning_content`.
2. Append the assistant turn to your history **including `reasoning_content` and `tool_calls`**: `{"role": "assistant", "content": "...", "reasoning_content": "...", "tool_calls": [...]}`. The template decides what to keep; Qwen's drops reasoning from all but the latest turn. If you drop it yourself, the prompt no longer matches what the model produced and the KV cache is invalidated.
3. If `stop_reason` was `tool`, run each call and append `{"role": "tool", "tool_call_id": "<id>", "content": "<result>"}` per call, then go to 1.
4. Watch `usage.prompt_tokens` against `usage.context_length` and truncate your history before it no longer fits. The server does not truncate for you; a prompt that overflows the context fails.

### POST /template

```json
{"model": "qwen3.5-9b", "messages": [...], "tools": [...], "template_args": {}, "autoload": false, "ttl": 300}
```

Returns `{"prompt": "<|im_start|>system\n..."}`: the exact text a `/run` with the same fields would feed the model. For checking what a template does with your messages. Same validation and errors as `/run` in chat mode.

## llm CLI

```bash
./llm [--host <h>] [--port <p>] list [--loaded true|false|any]
./llm load   --model <name> [--ttl <s>]
./llm unload --model <name>
./llm run    --model <name> --prompt <text> [<generation flags>]
./llm chat   --model <name> [--system <text>] [--messages <file.json>] [--user <text>] \
             [--tools <file.json>] [--no-thinking] [--template-arg key=json ...] [<generation flags>]
./llm template --model <name> [--system ...] [--messages ...] [--user ...] [--tools ...] [--no-thinking] [--autoload --ttl <s>]

# generation flags:
  [--autoload --ttl <s>] [--max-tokens <n>] [--temperature <f>] [--top-p <f>] [--repetition-penalty <f>]
  [--stop <text> ...] [--json]
```

`chat` builds the message list as `--system`, then the contents of `--messages` (a JSON list, e.g. a saved history), then `--user`. Answer text goes to stdout; thinking, `stop_reason` and `usage` go to stderr; tool calls are printed to stdout as JSON. `--json` prints every event as one JSON line instead. `--no-thinking` is shorthand for `--template-arg enable_thinking=false`.

```bash
./llm chat --model qwen3.5-9b --autoload --ttl 600 --no-thinking --user "What is 2+2?"
./llm chat --model qwen3.5-9b --tools tools.json --user "What is the weather in Paris?" --json
./llm template --model qwen3.5-9b --system "Be brief." --user "Hi"
```

## Testing

```bash
./run-tests.sh                # load, run, evict and unload every model in models/
./run-tests.sh qwen3.5-9b     # just the named models
```

Per model this covers raw and stop-string completion, template rendering, chat with and without thinking, a tool call, TTL eviction and unload mid-generation. Models without a chat template must refuse chat mode.

When the AppArmor profile is loaded, this offers to reload it in complain mode for the run (needs `sudo`), so a rule that is too tight is reported at the end rather than breaking a test in the middle, and puts it back in enforce mode afterwards. Any reported denial fails the run. Complain mode applies to every process under the profile, so the script refuses to start while a `./run.sh` server is running.

## Developer Notes

- **`torch` is not in `requirements.txt`** — it must be installed separately with a CUDA-version-specific index URL. See Setup above.
- **`model.json` is required** — a directory without a valid one returns HTTP 400 on load/run with the reason.
- **Model names** may contain letters, digits, `.`, `_` and `-`, and may not start with a dot.
- **Sampling defaults differ per backend** — `transformers` samples with only the parameters the API sets; `llama-server` additionally applies its own defaults (`top_k 40`, `min_p 0.05`) unless overridden in `args`.
- **Chat mode is two different code paths** — `llama-cpp` proxies `/v1/chat/completions` (and `/apply-template` for `/template`), so template rendering, reasoning extraction and tool-call parsing are all `llama-server`'s. `transformers` renders with `apply_chat_template` and streams the raw output. The server owns nothing model-specific: it validates shapes, merges `template_defaults`, and relays events.
- **Chat requests are rendered twice on `llama-cpp`** — once through `/apply-template` before streaming starts, so bad messages fail with HTTP 400 instead of an error event inside a 200 stream, and again inside `/v1/chat/completions`. Rendering is milliseconds; the KV cache is untouched by the first pass.
- **Thinking counts against `max_tokens`** — a Qwen model left to think may spend most of a small budget on reasoning and end with `stop_reason` `length` and no answer. Turn thinking off with `template_args`, raise `max_tokens`, or cap it with `--reasoning-budget N` in the model's `args`.
- **Reasoning is only separated when the template does it** — `llama-server` extracts `<think>` blocks in chat mode. In raw `prompt` mode, and on the `transformers` backend, the tags arrive inline as `token` events.
- **`do_sample=True` is always set** on the transformers backend — generation is always stochastic.
- **`device_map="auto"` silently spills to CPU** on the transformers backend — if a model exceeds available VRAM, layers overflow to system RAM with no error, making inference much slower. Use the `llama-cpp` backend for anything that does not fit in VRAM.
- **Unloading a `llama-cpp` model kills its subprocess**, which frees VRAM fully. Unloading a `transformers` model frees the tensors but the CUDA context stays resident in this process (a few hundred MB).
- **Subprocesses die with the server** — `llama-server` children are started with `PR_SET_PDEATHSIG`, so a crashed or killed server does not leave GPU memory held by orphans. (The signal is tied to the executor thread that spawned the child; those threads live for the life of the process.)
- **A crashed `llama-server` is detected** — `/run` returns HTTP 500 for it, `/load` reloads it, and the TTL monitor evicts it on its next pass.
- **`llm run` exits non-zero** when the stream ends with an error event.
- **Model loading is serialized** — only one model loads at a time. Concurrent load requests for different models queue behind each other.
- **`/unload` does not queue** — unlike TTL eviction, it neither waits for in-flight generation nor for an unrelated load to finish, since its purpose is to free VRAM now. Everything else on the model is collateral: cancelled with an error, never left half-served. Loads do wait for it: a `/load` that arrives while VRAM is still being released starts once the release is done.
- **`/unload` cancels one load** — the one in flight, which includes a `/load` waiting for the previous release of the same model to finish. A further `/load` for the same model queued behind that one is not affected and brings the model back once it gets its turn.
- **`/unload` on a `transformers` model stops generation at the next token** — the running `generate` thread is asked to stop and joined before the tensors are dropped, so VRAM is free when `/unload` returns. A `llama-cpp` model is killed outright.
- **There is no VRAM accounting** — nothing checks free memory before a load. A `llama-cpp` model that does not fit fails its load with the CUDA error in `logs/llama-<model>-<ts>.log`; a `transformers` model silently spills to CPU. Use `/unload` to make room.
- **Per-model inference queue** — concurrent requests to the same model are queued; requests to different loaded models run in parallel.
- **TTL expiry has up to 5s lag** — the TTL monitor polls every 5 seconds, so models may stay loaded slightly past their TTL.
