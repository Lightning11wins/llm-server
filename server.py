import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, TypedDict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import logging as hf_logging

from backends import Backend, GenParams, ModelConfigError, create_backend, read_model_config


# ── Configuration ──────────────────────────────────────────────────────────────
HOST                    = "0.0.0.0"
PORT                    = 8080
DEFAULT_TTL             = 300
DEFAULT_MAX_TOKENS      = 512
DEFAULT_TEMPERATURE     = 1.0
DEFAULT_TOP_P           = 1.0
DEFAULT_REPEAT_PENALTY  = 1.0
TTL_MONITOR_INTERVAL    = 5

MODEL_NAME_REGEX        = r"[a-zA-Z0-9_-][a-zA-Z0-9._-]*"  # no leading dot, no path separators

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
	backend: Backend
	lock: asyncio.Lock
	ttl_end: float
	pinned: int  # requests that have committed to using this model but not yet locked it

loaded_models: dict[str, LoadedModel] = {}
load_lock = asyncio.Lock()  # serialized model loading


# A model may be evicted when nothing is using it and it has either expired or its backend has died.
def _evictable(entry: LoadedModel) -> bool:
	idle = not entry["lock"].locked() and entry["pinned"] == 0
	return idle and (time.time() >= entry["ttl_end"] or not entry["backend"].is_alive())


# Evict models whose TTL has expired or whose backend has died.
async def ttl_monitor() -> None:
	while True:
		await asyncio.sleep(TTL_MONITOR_INTERVAL)
		for name in list(loaded_models):
			await unload_model(name, only_if_evictable=True)


# Remove a model from the registry and release its resources.
# Serialized with loading via load_lock, so a new model never starts loading while an old one is
# still releasing GPU memory. With only_if_evictable=True the model is left alone unless it is idle
# and expired/dead; the check is repeated under the lock because a request may have started meanwhile.
async def unload_model(name: str, *, only_if_evictable: bool = False) -> None:
	entry = loaded_models.get(name)
	if entry is None or (only_if_evictable and not _evictable(entry)):
		return
	async with load_lock:
		entry = loaded_models.get(name)
		if entry is None or (only_if_evictable and not _evictable(entry)):
			return
		del loaded_models[name]
		loop = asyncio.get_running_loop()
		try:
			await loop.run_in_executor(None, entry["backend"].unload)
		except Exception as e:
			log.error(f"Error unloading {name}: {type(e).__name__}: {e}")
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
	for name in list(loaded_models):
		await unload_model(name)
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


# A StreamingResponse that always runs `finalizer` once the response is finished or abandoned,
# even when the body iterator was never started (client gone before the first chunk).
class FinalizedStreamingResponse(StreamingResponse):
	def __init__(self, content, finalizer: Callable[[], None], **kwargs) -> None:
		super().__init__(content, **kwargs)
		self._finalizer = finalizer

	async def __call__(self, scope, receive, send) -> None:
		try:
			await super().__call__(scope, receive, send)
		finally:
			self._finalizer()


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
# Ensure that name is an available model with a valid model.json.
def validate_model(name: str) -> None:
	if not re.fullmatch(MODEL_NAME_REGEX, name):
		raise HTTPException(400, f"Invalid model name: '{name}'")
	if not (MODELS_DIR / name).is_dir():
		raise HTTPException(404, f"Model '{name}' not found in models/")
	try:
		read_model_config(MODELS_DIR / name)
	except ModelConfigError as e:
		raise HTTPException(400, f"Model '{name}' has an invalid config: {e}")


# Extend a model's if `ttl` (in seconds) is longer.
def _refresh_ttl(registered_model: LoadedModel, ttl: float) -> None:
	now = time.time()
	registered_model["ttl_end"] = now + max(registered_model["ttl_end"] - now, ttl)


# Ensure that a model is loaded, updating the model TTL as needed.
# Note: Loads are serialized via load_lock.
async def ensure_loaded(name: str, ttl: float) -> None:
	# Early exit if already loaded. A dead backend (crashed subprocess) is unloaded and reloaded.
	entry = loaded_models.get(name)
	if entry is not None:
		if entry["backend"].is_alive():
			_refresh_ttl(entry, ttl)
			return
		log.error(f"Model backend died, reloading: {name}")
		await unload_model(name)

	async with load_lock:
		# Re-check for loaded model.
		if name in loaded_models:
			_refresh_ttl(loaded_models[name], ttl)
			return

		# Build the backend from model.json and load it off the event loop.
		try:
			backend = create_backend(name, MODELS_DIR / name)
		except ModelConfigError as e:
			raise HTTPException(400, f"Model '{name}' has an invalid config: {e}")
		log.info(f"Model loading: {name}  backend={backend.config['backend']}")
		t0 = time.time()
		loop = asyncio.get_running_loop()
		await loop.run_in_executor(None, backend.load)

		# Register and log.
		t1 = time.time()
		loaded_models[name] = {
			"backend": backend,
			"lock": asyncio.Lock(),
			"ttl_end": t1 + ttl,
			"pinned": 0,
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
		loaded_model = loaded_models.get(req.model)
		if loaded_model is None:
			raise HTTPException(400, f"Model '{req.model}' is not loaded")
		if not loaded_model["backend"].is_alive():
			raise HTTPException(500, f"Model '{req.model}' backend has died; it will be unloaded shortly, load it again")
		loaded_model["pinned"] += 1
	except Exception as e:
		log.error(f"Request failed: POST /run  model={req.model}  {type(e).__name__}: {e}")
		if not isinstance(e, HTTPException):
			raise HTTPException(500, "Internal server error")
		raise

	# Set up the inference stream.
	ttl = req.ttl
	async def stream():
		token_count = 0
		try:
			async with loaded_model["lock"]:
				loop = asyncio.get_running_loop()
				params = GenParams(prompt=req.prompt, max_tokens=req.max_tokens, temperature=req.temperature,
								   top_p=req.top_p, repetition_penalty=req.repetition_penalty)
				gen = loaded_model["backend"].generate(params)

				# Pull tokens from the blocking generator off the event loop.
				while True:
					token = await loop.run_in_executor(None, next, gen, None)
					if token is None:
						break
					token_count += 1
					yield f"data: {json.dumps({'token': token})}\n\n"

				log.info(f"Request completed: POST /run  model={req.model}  tokens={token_count}")
				yield "data: [DONE]\n\n"
		except Exception as e:
			log.error(f"Inference error ({req.model}): {e}")
			yield f"data: {json.dumps({'error': str(e)})}\n\n"

	# Unpin from the response's finalizer rather than the generator's `finally`: if the client
	# disconnects before streaming starts, the generator is never entered and its `finally` never runs.
	def release() -> None:
		loaded_model["pinned"] -= 1
		_refresh_ttl(loaded_model, ttl)

	return FinalizedStreamingResponse(stream(), release, media_type="text/event-stream")


# Start server.
if __name__ == "__main__":
	uvicorn.run(app, host=HOST, port=PORT)
