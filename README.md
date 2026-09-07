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
- `args` (optional): extra `llama-server` flags. The server always sets `-m`, `--host 127.0.0.1`, `--port <free port>` and `--no-webui`, and rejects configs that try to override them. Run `bin/llama.cpp/llama-server --help` for the full list. Useful ones: `-ngl N` (layers on GPU), `-c N` (context size), `-fa on` (flash attention), `--n-cpu-moe N` (keep the first N layers' MoE experts in CPU RAM), `-ctk q8_0 -ctv q8_0` (quantized KV cache), `--fit on` (auto-tune to fit device memory).
- `startup_timeout` (optional, seconds): how long to wait for the model to load. Default 300.

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

## Start

To start, simply run `./run.sh` in the project directory. This sets up the AppArmor profile and Python environment automatically, asking before anything that needs `sudo` or downloads packages.

Binds `127.0.0.1:8080`. Change `HOST` and `PORT` at the top of `server.py`; `HOST = "0.0.0.0"` is there commented out for serving other machines, which also needs a one-line profile change. Logs are written to `logs/<timestamp>.log`.

## AppArmor

`run.sh` confines the server with the AppArmor profile in `apparmor/llm-server`, and the `llama-server` subprocess with a tighter one: read-only on `models/`, TCP only, and the only thing it can write is its own log. The server itself can write `logs/` and `tmp/`, nothing else in the project. `run.sh` copies the profile to `/etc/apparmor.d` and reloads it whenever the two differ, asking first, since that part needs `sudo`.

The profile also pins both processes to loopback addresses, but only on a kernel that mediates socket addresses (`/sys/kernel/security/apparmor/features/network_v9/af_inet` exists). On other kernels the rules load as plain TCP permission and `run.sh` prints a warning.

`./run.sh --unconfined` skips all of it. To remove the profile: `sudo apparmor_parser -R /etc/apparmor.d/llm-server && sudo rm /etc/apparmor.d/llm-server`.

## API

All request bodies are JSON. Errors return `{"error": "..."}` with HTTP 404/400/500.

### GET /list

```
GET /list?loaded=any|true|false
```

```json
[{"name": "qwen3.5-9b", "loaded": true}, ...]
```

### POST /load

```json
{"model": "qwen3.5-9b", "ttl": 300}
```

Blocks until loaded. If already loaded, refreshes TTL. `ttl` (seconds) defaults to 300.

Returns `{"status": "loaded", "model": "qwen3.5-9b"}`. Returns HTTP 409 if the model is unloaded while this load is in flight.

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

```json
{
  "model": "qwen3.5-9b",
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

## llm CLI

```bash
./llm [--host <h>] [--port <p>] list [--loaded true|false|any]
./llm load   --model <name> [--ttl <s>]
./llm unload --model <name>
./llm run    --model <name> --prompt <text> [--autoload --ttl <s>] \
             [--max-tokens <n>] [--temperature <f>] [--top-p <f>] [--repetition-penalty <f>]
```

## Testing

```bash
./run_tests.sh                # load, run, evict and unload every model in models/
./run_tests.sh qwen3.5-9b     # just the named models
```

Runs against the AppArmor profile in complain mode when it is installed, so a rule that is too tight is reported at the end rather than breaking a test in the middle.

## TODO

The prompt is passed to the model as raw text. Instruction-tuned models (Qwen etc.) need the following to be useful; none of it exists yet:

- Chat-formatted input (a `messages` list) with the model's chat template applied server-side. `llama-server` exposes `/apply-template` and `/v1/chat/completions` for this.
- Thinking mode on/off (a chat-template argument on Qwen models).
- Stop sequences and end-of-turn handling. Raw completion runs to `max_tokens` or the model's EOS.
- Tool-call schemas in requests and parsed tool calls in responses.
- System prompt support (also a chat-template concern).
- Context window / prompt-length reporting so a client knows when to truncate.

## Developer Notes

- **`torch` is not in `requirements.txt`** — it must be installed separately with a CUDA-version-specific index URL. See Setup above.
- **`model.json` is required** — a directory without a valid one returns HTTP 400 on load/run with the reason.
- **Model names** may contain letters, digits, `.`, `_` and `-`, and may not start with a dot.
- **Sampling defaults differ per backend** — `transformers` samples with only the parameters the API sets; `llama-server` additionally applies its own defaults (`top_k 40`, `min_p 0.05`) unless overridden in `args`.
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
