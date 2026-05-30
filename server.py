import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, TypedDict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
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

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# ── Logging ────────────────────────────────────────────────────────────────────
_log_dir = BASE_DIR / "logs"
_log_dir.mkdir(exist_ok=True)
_ts = time.strftime('%Y-%m-%d_%H-%M-%S')
_log_file = _log_dir / f"{_ts}.log"
if _log_file.exists():
    _sfx = 2
    while (_log_dir / f"{_ts}-{_sfx}.log").exists():
        _sfx += 1
    _log_file = _log_dir / f"{_ts}-{_sfx}.log"

_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_fh  = logging.FileHandler(_log_file)
_fh.setFormatter(_fmt)
_ch  = logging.StreamHandler()
_ch.setFormatter(_fmt)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(_fh)
log.addHandler(_ch)
log.propagate = False  # prevent double-printing via root logger

# Silence noisy library output (progress bars, pad_token warnings, etc.)
hf_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# ── Model registry ─────────────────────────────────────────────────────────────
class ModelEntry(TypedDict):
    model: Any
    tokenizer: Any
    lock: asyncio.Lock
    ttl_end: float

models: dict[str, ModelEntry] = {}
load_lock = asyncio.Lock()  # serializes all model loading


async def ttl_monitor():
    while True:
        await asyncio.sleep(5)
        now = time.time()
        for name in [n for n, e in list(models.items()) if now >= e["ttl_end"] and not e["lock"].locked()]:
            del models[name]
            log.info(f"Model unloaded: {name}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    for _n in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logging.getLogger(_n).addHandler(_fh)
    log.info("Server started")
    asyncio.create_task(ttl_monitor())
    yield
    log.info("Server stopped")


app = FastAPI(lifespan=lifespan)


@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exc_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": str(exc)})


@app.exception_handler(Exception)
async def generic_exc_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ── Schemas ────────────────────────────────────────────────────────────────────
class LoadReq(BaseModel):
    model: str
    ttl: float = DEFAULT_TTL


class RunReq(BaseModel):
    model: str
    prompt: str
    ttl: float | None = None
    autoload: bool = False
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY

# ── Helpers ────────────────────────────────────────────────────────────────────
def validate_model(name: str) -> None:
    if "/" in name or ".." in Path(name).parts:
        raise HTTPException(400, f"Invalid model name: '{name}'")
    if not (MODELS_DIR / name).exists():
        raise HTTPException(404, f"Model '{name}' not found in models/")


def _refresh_ttl(entry: ModelEntry, ttl: float) -> None:
    entry["ttl_end"] = time.time() + max(entry["ttl_end"] - time.time(), ttl)


async def ensure_loaded(name: str, ttl: float):
    if name in models:
        e = models[name]
        _refresh_ttl(e, ttl)
        return

    async with load_lock:
        if name in models:  # second check after acquiring lock
            e = models[name]
            _refresh_ttl(e, ttl)
            return

        path = str(MODELS_DIR / name)
        log.info(f"Model loading: {name}")
        t0   = time.time()
        loop = asyncio.get_running_loop()
        tok   = await loop.run_in_executor(None, lambda: AutoTokenizer.from_pretrained(path))
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        model = await loop.run_in_executor(None, lambda: AutoModelForCausalLM.from_pretrained(path, device_map="auto"))
        models[name] = {"model": model, "tokenizer": tok, "lock": asyncio.Lock(), "ttl_end": time.time() + ttl}
        log.info(f"Model loaded: {name}  ({time.time() - t0:.1f}s)")

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
    try:
        validate_model(req.model)
        await ensure_loaded(req.model, req.ttl)
    except Exception as e:
        log.error(f"Request failed: POST /load  model={req.model}  {type(e).__name__}: {e}")
        if not isinstance(e, HTTPException):
            raise HTTPException(500, "Internal server error")
        raise
    log.info(f"Request completed: POST /load  model={req.model}")
    return {"status": "loaded", "model": req.model}


@app.post("/run")
async def run(req: RunReq):
    log.info(f"Request received: POST /run  model={req.model}")
    try:
        validate_model(req.model)
        if req.autoload:
            if req.ttl is None:
                raise HTTPException(400, "ttl is required when autoload=true")
            await ensure_loaded(req.model, req.ttl)
        elif req.model not in models:
            raise HTTPException(400, f"Model '{req.model}' is not loaded")
    except Exception as e:
        log.error(f"Request failed: POST /run  model={req.model}  {type(e).__name__}: {e}")
        if not isinstance(e, HTTPException):
            raise HTTPException(500, "Internal server error")
        raise

    ttl   = req.ttl if req.ttl is not None else DEFAULT_TTL
    entry = models[req.model]

    async def stream():
        async with entry["lock"]:
            token_count = 0
            try:
                m, tok   = entry["model"], entry["tokenizer"]
                loop     = asyncio.get_running_loop()
                inputs   = await loop.run_in_executor(None, lambda: tok(req.prompt, return_tensors="pt").to(m.device))
                streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=60)
                thread   = threading.Thread(
                    target=m.generate,
                    kwargs=dict(**inputs, streamer=streamer, max_new_tokens=req.max_tokens,
                                temperature=req.temperature, top_p=req.top_p,
                                repetition_penalty=req.repetition_penalty, do_sample=True),
                    daemon=True,
                )
                thread.start()
                while True:
                    token = await loop.run_in_executor(None, lambda: next(streamer, None))
                    if token is None:
                        break
                    token_count += 1
                    yield f"data: {json.dumps({'token': token})}\n\n"
                log.info(f"Request completed: POST /run  model={req.model}  tokens={token_count}")
                yield "data: [DONE]\n\n"
            except Exception as e:
                log.error(f"Inference error ({req.model}): {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                _refresh_ttl(entry, ttl)

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
