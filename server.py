import asyncio
import json
import logging
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypedDict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, TextIteratorStreamer
from transformers import logging as hf_logging


# ── Configuration ──────────────────────────────────────────────────────────────
HOST                    = "0.0.0.0"
PORT                    = 8080
DEFAULT_TTL             = 300
DEFAULT_MAX_TOKENS      = 512
DEFAULT_TEMPERATURE     = 1.0
DEFAULT_TOP_P           = 1.0
DEFAULT_REPEAT_PENALTY  = 1.0
TTL_MONITOR_INTERVAL    = 5
STREAMER_TIMEOUT        = 60

MODEL_NAME_REGEX        = r"[a-zA-Z0-9_-]+"

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR   = BASE_DIR / "logs"


# ── Logging ────────────────────────────────────────────────────────────────────
_log_dir = LOGS_DIR
_log_dir.mkdir(exist_ok=True)
_ts = time.strftime('%Y-%m-%d_%H-%M-%S')
_log_file = _log_dir / f"{_ts}.log"

# Duplicate log file handling.
if _log_file.exists():
	_sfx = 2
	while (_log_dir / f"{_ts}-{_sfx}.log").exists():
		_sfx += 1
	_log_file = _log_dir / f"{_ts}-{_sfx}.log"

_formatter = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_file_handler = logging.FileHandler(_log_file)
_file_handler.setFormatter(_formatter)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_formatter)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(_file_handler)
log.addHandler(_stream_handler)
log.propagate = False  # prevent double-printing via root logger

# Silence noisy library output (progress bars, pad_token warnings, etc.)
hf_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


# ── Model registry ─────────────────────────────────────────────────────────────
class LoadedModel(TypedDict):
	model: PreTrainedModel
	tokenizer: PreTrainedTokenizerBase
	lock: asyncio.Lock
	ttl_end: float

loaded_models: dict[str, LoadedModel] = {}
load_lock = asyncio.Lock()  # serialized model loading


# Evict when TTL expires.
async def ttl_monitor() -> None:
	while True:
		await asyncio.sleep(TTL_MONITOR_INTERVAL)
		
		# Collect models to evict.
		now = time.time()
		expired = [
			n for n, e in list(loaded_models.items())
			if now >= e["ttl_end"] and not e["lock"].locked()
		]
		
		# Evict.
		for name in expired:
			del loaded_models[name]
			log.info(f"Model unloaded: {name}")


# Setup model manager.
@asynccontextmanager
async def lifespan(app: FastAPI):
	# Rewire uvicorn to our file logger.
	logging.getLogger("uvicorn").addHandler(_file_handler)
	logging.getLogger("uvicorn.access").addHandler(_file_handler)
	logging.getLogger("uvicorn.error").addHandler(_file_handler)
	asyncio.create_task(ttl_monitor())
	
	log.info("Server started")
	yield
	log.info("Server stopped")


# Start server.
app = FastAPI(lifespan=lifespan)


@app.exception_handler(HTTPException)
async def http_exc_handler(request: Request, exc: HTTPException) -> JSONResponse:
	return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exc_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
	return JSONResponse(status_code=422, content={"error": str(exc)})


@app.exception_handler(Exception)
async def generic_exc_handler(request: Request, exc: Exception) -> JSONResponse:
	log.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
	return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ── Schemas ────────────────────────────────────────────────────────────────────
class ModelInfo(TypedDict):
	name: str
	loaded: bool

class LoadReq(BaseModel):
	model: str
	ttl: float = Field(DEFAULT_TTL, gt=0)

class RunReq(BaseModel):
	model: str
	prompt: str = Field(..., min_length=1)
	ttl: float = Field(DEFAULT_TTL, gt=0)
	autoload: bool = False
	max_tokens: int = Field(DEFAULT_MAX_TOKENS, ge=1)
	temperature: float = Field(DEFAULT_TEMPERATURE, gt=0)
	top_p: float = Field(DEFAULT_TOP_P, gt=0, le=1.0)
	repetition_penalty: float = Field(DEFAULT_REPEAT_PENALTY, gt=0)


# ── Helpers ────────────────────────────────────────────────────────────────────
# Ensure that name is an available model.
def validate_model(name: str) -> None:
	if not re.fullmatch(MODEL_NAME_REGEX, name):
		raise HTTPException(400, f"Invalid model name: '{name}'")
	if not (MODELS_DIR / name).exists():
		raise HTTPException(404, f"Model '{name}' not found in models/")


# Extend a model's if `ttl` (in seconds) is longer.
def _refresh_ttl(registered_model: LoadedModel, ttl: float) -> None:
	now = time.time()
	registered_model["ttl_end"] = now + max(registered_model["ttl_end"] - now, ttl)


# Ensure that a model is loaded, updating the model TTL as needed.
# Note: Loads are serialized via load_lock.
async def ensure_loaded(name: str, ttl: float) -> None:
	# Early exit if already loaded.
	if name in loaded_models:
		_refresh_ttl(loaded_models[name], ttl)
		return
	
	async with load_lock:
		# Re-check for loaded model.
		if name in loaded_models:
			_refresh_ttl(loaded_models[name], ttl)
			return
		
		# Load tokenizer and model off the event loop.
		path = str(MODELS_DIR / name)
		log.info(f"Model loading: {name}")
		t0 = time.time()
		loop = asyncio.get_running_loop()
		tok = await loop.run_in_executor(None, lambda: AutoTokenizer.from_pretrained(path))
		if tok.pad_token_id is None:
			tok.pad_token_id = tok.eos_token_id
		model = await loop.run_in_executor(None, lambda: AutoModelForCausalLM.from_pretrained(path, device_map="auto"))
		
		# Register and log.
		t1 = time.time()
		loaded_models[name] = {
			"model": model,
			"tokenizer": tok,
			"lock": asyncio.Lock(),
			"ttl_end": t1 + ttl,
		}
		log.info(f"Model loaded: {name}  ({t1 - t0:.1f}s)")

# ── Endpoints ──────────────────────────────────────────────────────────────────
# Get a list of all available models. Supports optional filtering by loaded=true|false|any.
@app.get("/list")
async def list_models(loaded: str = "any") -> list[ModelInfo]:
	log.info(f"Request received: GET /list  loaded={loaded}")
	if loaded not in ("any", "true", "false"):
		raise HTTPException(400, f"Invalid loaded value: '{loaded}'. Must be 'any', 'true', or 'false'.")

	# Enumerate disk.
	names = sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir()) if MODELS_DIR.exists() else []
	result: list[ModelInfo] = [{"name": n, "loaded": n in loaded_models} for n in names]
	
	# Filter by loaded state.
	if loaded == "true":  result = [r for r in result if r["loaded"]]
	if loaded == "false": result = [r for r in result if not r["loaded"]]
	
	log.info(f"Request completed: GET /list  returned={len(result)}")
	return result


# Load a model into memory or refresh its ttl if it is already loaded.
@app.post("/load")
async def load(req: LoadReq) -> dict[str, str]:
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


# Stream inference tokens as SSE. autoload=true loads model if absent.
@app.post("/run")
async def run(req: RunReq) -> StreamingResponse:
	log.info(f"Request received: POST /run  model={req.model}")

	# Validate and ensure model is ready.
	try:
		validate_model(req.model)
		if req.autoload:
			await ensure_loaded(req.model, req.ttl)
		elif req.model not in loaded_models:
			raise HTTPException(400, f"Model '{req.model}' is not loaded")
	except Exception as e:
		log.error(f"Request failed: POST /run  model={req.model}  {type(e).__name__}: {e}")
		if not isinstance(e, HTTPException):
			raise HTTPException(500, "Internal server error")
		raise

	# Set up the inference stream.
	loaded_model = loaded_models[req.model]
	ttl = req.ttl
	async def stream():
		async with loaded_model["lock"]:
			token_count = 0
			try:
				# Tokenize prompt.
				model = loaded_model["model"]
				tok = loaded_model["tokenizer"]
				loop = asyncio.get_running_loop()
				inputs = await loop.run_in_executor(None, lambda: tok(req.prompt, return_tensors="pt").to(model.device))
				
				# Start generation thread.
				streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=STREAMER_TIMEOUT)  # type: ignore[arg-type]
				thread = threading.Thread(
					target=model.generate,  # type: ignore[arg-type]
					kwargs=dict(**inputs, streamer=streamer, max_new_tokens=req.max_tokens,
								temperature=req.temperature, top_p=req.top_p,
								repetition_penalty=req.repetition_penalty, do_sample=True),
					daemon=True,
				)
				thread.start()
				
				# Yield tokens.
				while True:
					token = await loop.run_in_executor(None, lambda: next(streamer, None))
					if token is None:
						break
					token_count += 1
					yield f"data: {json.dumps({'token': token})}\n\n"
				
				# Signal completion.
				log.info(f"Request completed: POST /run  model={req.model}  tokens={token_count}")
				yield "data: [DONE]\n\n"
			except Exception as e:
				log.error(f"Inference error ({req.model}): {e}")
				yield f"data: {json.dumps({'error': str(e)})}\n\n"
			finally:
				_refresh_ttl(loaded_model, ttl)

	return StreamingResponse(stream(), media_type="text/event-stream")


# Start server.
if __name__ == "__main__":
	uvicorn.run(app, host=HOST, port=PORT)
