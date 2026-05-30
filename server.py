import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import logging as hf_logging

# ── Configuration ──────────────────────────────────────────────────────────────
PORT                    = 8080
DEFAULT_TTL             = 300
DEFAULT_MAX_TOKENS      = 512
DEFAULT_TEMPERATURE     = 1.0
DEFAULT_TOP_P           = 1.0
DEFAULT_REPETITION_PENALTY = 1.0

MODELS_DIR = Path("models")

# ── Logging ────────────────────────────────────────────────────────────────────
_handler = logging.FileHandler("server.log")
_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(_handler)

# Write uvicorn access/error logs to our file as well
for _n in ("uvicorn", "uvicorn.access", "uvicorn.error"):
    logging.getLogger(_n).addHandler(_handler)

# Silence noisy library output (progress bars, pad_token warnings, etc.)
hf_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# ── Model registry ─────────────────────────────────────────────────────────────
# {name: {"model": ..., "tokenizer": ..., "lock": asyncio.Lock, "ttl_end": float}}
models: dict = {}
load_lock = asyncio.Lock()  # serialises concurrent load operations for the same model


async def ttl_monitor():
    while True:
        await asyncio.sleep(5)
        now = time.time()
        for name in [n for n, e in list(models.items()) if now >= e["ttl_end"] and not e["lock"].locked()]:
            del models[name]
            log.info(f"Model unloaded: {name}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Server started")
    asyncio.create_task(ttl_monitor())
    yield
    log.info("Server stopped")


app = FastAPI(lifespan=lifespan)

# ── Schemas ────────────────────────────────────────────────────────────────────
class LoadReq(BaseModel):
    model: str
    ttl: float = DEFAULT_TTL


class RunReq(BaseModel):
    model: str
    prompt: str
    ttl: Optional[float] = None
    autoload: bool = False
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY

# ── Helpers ────────────────────────────────────────────────────────────────────
def require_exists(name: str) -> Path:
    p = MODELS_DIR / name
    if not p.exists():
        raise HTTPException(404, f"Model '{name}' not found in models/")
    return p


async def ensure_loaded(name: str, ttl: float):
    if name in models:
        e = models[name]
        e["ttl_end"] = time.time() + max(e["ttl_end"] - time.time(), ttl)
        return

    async with load_lock:
        if name in models:  # second check after acquiring lock
            e = models[name]
            e["ttl_end"] = time.time() + max(e["ttl_end"] - time.time(), ttl)
            return

        path = str(MODELS_DIR / name)
        log.info(f"Model loading: {name}")
        loop = asyncio.get_running_loop()
        tok   = await loop.run_in_executor(None, lambda: AutoTokenizer.from_pretrained(path))
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        model = await loop.run_in_executor(None, lambda: AutoModelForCausalLM.from_pretrained(path, device_map="auto"))
        models[name] = {"model": model, "tokenizer": tok, "lock": asyncio.Lock(), "ttl_end": time.time() + ttl}
        log.info(f"Model loaded: {name}")

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/list")
async def list_models(loaded: str = "any"):
    log.info(f"Request received: GET /list  loaded={loaded}")
    names = sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir()) if MODELS_DIR.exists() else []
    result = [{"name": n, "loaded": n in models} for n in names]
    if loaded == "true":  result = [r for r in result if     r["loaded"]]
    if loaded == "false": result = [r for r in result if not r["loaded"]]
    log.info(f"Request completed: GET /list  returned={len(result)}")
    return result


@app.post("/load")
async def load(req: LoadReq):
    log.info(f"Request received: POST /load  model={req.model}")
    require_exists(req.model)
    await ensure_loaded(req.model, req.ttl)
    log.info(f"Request completed: POST /load  model={req.model}")
    return {"status": "loaded", "model": req.model}


@app.post("/run")
async def run(req: RunReq):
    log.info(f"Request received: POST /run  model={req.model}")
    require_exists(req.model)

    if req.autoload:
        if req.ttl is None:
            raise HTTPException(400, "ttl is required when autoload=true")
        await ensure_loaded(req.model, req.ttl)
    elif req.model not in models:
        raise HTTPException(400, f"Model '{req.model}' is not loaded")

    ttl   = req.ttl if req.ttl is not None else DEFAULT_TTL
    entry = models[req.model]

    async def stream():
        async with entry["lock"]:
            try:
                m, tok = entry["model"], entry["tokenizer"]
                inputs   = tok(req.prompt, return_tensors="pt").to(m.device)
                streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
                thread   = threading.Thread(
                    target=m.generate,
                    kwargs=dict(**inputs, streamer=streamer, max_new_tokens=req.max_tokens,
                                temperature=req.temperature, top_p=req.top_p,
                                repetition_penalty=req.repetition_penalty, do_sample=True),
                )
                thread.start()
                token_count = 0
                for token in streamer:
                    token_count += 1
                    yield f"data: {json.dumps({'token': token})}\n\n"
                thread.join()
                entry["ttl_end"] = time.time() + max(entry["ttl_end"] - time.time(), ttl)
                log.info(f"Request completed: POST /run  model={req.model}  tokens={token_count}")
                yield "data: [DONE]\n\n"
            except Exception as e:
                log.error(f"Inference error ({req.model}): {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
