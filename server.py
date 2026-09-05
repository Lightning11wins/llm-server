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
	unloaded: bool  # set when the model leaves the registry; in-flight requests abort on it

loaded_models: dict[str, LoadedModel] = {}
loading_models: dict[str, bool] = {}  # name being loaded -> whether an unload was requested meanwhile
unloading_models: dict[str, asyncio.Future] = {}  # name being released -> future that completes when it is
load_lock = asyncio.Lock()  # serialized model loading


# Raised inside a running request when its model is unloaded out from under it.
class ModelUnloadedError(Exception):
	pass


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


# Take a model out of the registry so no further request can reach it, and hand its entry to the
# caller, who then owns releasing it. Returns None if the model is absent or (with only_if_evictable)
# still in use. There is no await here, so exactly one caller can ever take a given entry.
def _take(name: str, *, only_if_evictable: bool) -> LoadedModel | None:
	entry = loaded_models.get(name)
	if entry is None or (only_if_evictable and not _evictable(entry)):
		return None
	del loaded_models[name]
	entry["unloaded"] = True
	return entry


# Release a taken entry's resources off the event loop. Does not raise. The release is recorded in
# unloading_models so ensure_loaded can wait for it rather than loading over memory still being freed.
async def _release_backend(name: str, backend: Backend) -> None:
	loop = asyncio.get_running_loop()
	done = loop.create_future()
	unloading_models[name] = done  # never already present: a reload waits for the release to finish
	try:
		await loop.run_in_executor(None, backend.unload)
	except Exception as e:
		log.error(f"Error unloading {name}: {type(e).__name__}: {e}")
	finally:
		del unloading_models[name]
		done.set_result(None)
	log.info(f"Model unloaded: {name}")


# Wait for every in-flight backend release, so a load never overlaps with GPU memory being freed.
async def _await_releases() -> None:
	while unloading_models:
		await asyncio.wait(list(unloading_models.values()))


# Remove an idle model from the registry and release its resources.
# Serialized with loading via load_lock, so a new model never starts loading while an old one is
# still releasing GPU memory. With only_if_evictable=True the model is left alone unless it is idle
# and expired/dead; the check is repeated under the lock because a request may have started meanwhile.
async def unload_model(name: str, *, only_if_evictable: bool = False) -> None:
	entry = loaded_models.get(name)
	if entry is None or (only_if_evictable and not _evictable(entry)):
		return
	async with load_lock:
		entry = _take(name, only_if_evictable=only_if_evictable)
		if entry is None:
			return
		await _release_backend(name, entry["backend"])


# Unload a model immediately, whatever it is doing. Unlike unload_model this neither waits for
# in-flight generation nor queues behind a load of some other model, because the point of it is to
# free VRAM now. Requests holding the entry see unloaded=True and abort with a clear error.
# Returns what was found: "unloaded", "cancelled" (mid-load), "unloading" (already going) or "not_loaded".
async def force_unload_model(name: str) -> str:
	entry = _take(name, only_if_evictable=False)
	if entry is not None:
		log.info(f"Model unloading: {name}  (in-flight requests: {entry['pinned']})")
		await _release_backend(name, entry["backend"])
		return "unloaded"
	if name in loading_models:
		loading_models[name] = True  # ensure_loaded releases it as soon as the load returns
		log.info(f"Model load cancelled: {name}")
		return "cancelled"
	if name in unloading_models:
		return "unloading"
	return "not_loaded"


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
	for name in list(loading_models):
		loading_models[name] = True  # released by ensure_loaded once the load returns
	for name in list(loaded_models):
		await unload_model(name)
	await _await_releases()
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

class UnloadReq(BaseModel):
	model: str

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
# Ensure that name is a syntactically valid model name.
def validate_model_name(name: str) -> None:
	if not re.fullmatch(MODEL_NAME_REGEX, name):
		raise HTTPException(400, f"Invalid model name: '{name}'")


# Ensure that name is an available model with a valid model.json.
def validate_model(name: str) -> None:
	validate_model_name(name)
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

		# A forced unload releases without holding load_lock, so wait for any release still in
		# progress before claiming GPU memory. Nothing else can reach this point meanwhile.
		await _await_releases()

		# Build the backend from model.json and load it off the event loop. The load cannot be
		# interrupted, so an unload arriving meanwhile is recorded in loading_models and applied
		# the moment the model is up, rather than leaving it loaded.
		loading_models[name] = False
		try:
			try:
				backend = create_backend(name, MODELS_DIR / name)
			except ModelConfigError as e:
				raise HTTPException(400, f"Model '{name}' has an invalid config: {e}")
			log.info(f"Model loading: {name}  backend={backend.config['backend']}")
			t0 = time.time()
			loop = asyncio.get_running_loop()
			await loop.run_in_executor(None, backend.load)
		finally:
			cancelled = loading_models.pop(name, False)

		if cancelled:
			await _release_backend(name, backend)
			raise HTTPException(409, f"Model '{name}' was unloaded while it was loading")

		# Register and log.
		t1 = time.time()
		loaded_models[name] = {
			"backend": backend,
			"lock": asyncio.Lock(),
			"ttl_end": t1 + ttl,
			"pinned": 0,
			"unloaded": False,
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


# Unload a model immediately, aborting any requests currently using it.
@app.post("/unload")
async def unload(req: UnloadReq) -> dict[str, str]:
	log.info(f"Request received: POST /unload  model={req.model}")
	try:
		validate_model_name(req.model)
		# A model that is loaded (or loading) is always unloadable, even if its directory or
		# model.json has since been removed or broken on disk.
		known = req.model in loaded_models or req.model in loading_models or req.model in unloading_models
		if not known and not (MODELS_DIR / req.model).is_dir():
			raise HTTPException(404, f"Model '{req.model}' not found in models/")
		status = await force_unload_model(req.model)
	except Exception as e:
		log.error(f"Request failed: POST /unload  model={req.model}  {type(e).__name__}: {e}")
		if not isinstance(e, HTTPException):
			raise HTTPException(500, "Internal server error")
		raise
	log.info(f"Request completed: POST /unload  model={req.model}  status={status}")
	return {"status": status, "model": req.model}


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
			# The model may have been unloaded while this request waited its turn on the lock.
			async with loaded_model["lock"]:
				if loaded_model["unloaded"]:
					raise ModelUnloadedError
				loop = asyncio.get_running_loop()
				params = GenParams(prompt=req.prompt, max_tokens=req.max_tokens, temperature=req.temperature,
								   top_p=req.top_p, repetition_penalty=req.repetition_penalty)
				gen = loaded_model["backend"].generate(params)

				# Pull tokens from the blocking generator off the event loop. An unload mid-stream
				# either breaks the backend (raising here) or is caught by the check below.
				try:
					while True:
						token = await loop.run_in_executor(None, next, gen, None)
						if token is None:
							break  # finished; an unload landing now does not spoil a complete answer
						if loaded_model["unloaded"]:
							raise ModelUnloadedError
						token_count += 1
						yield f"data: {json.dumps({'token': token})}\n\n"
				finally:
					# On client disconnect the worker thread may still be inside next(gen); closing
					# a running generator raises, so leave it to be finalized when the thread returns.
					if not gen.gi_running:
						gen.close()

				log.info(f"Request completed: POST /run  model={req.model}  tokens={token_count}")
				yield "data: [DONE]\n\n"
		except Exception as e:
			# An unload races the backend failing, so it is what the client is told either way.
			if loaded_model["unloaded"]:
				log.info(f"Request cancelled: POST /run  model={req.model}  tokens={token_count}  (model unloaded)")
				error = f"Model '{req.model}' was unloaded; request cancelled after {token_count} tokens"
			else:
				log.error(f"Inference error ({req.model}): {e}")
				error = str(e)
			yield f"data: {json.dumps({'error': error})}\n\n"

	# Unpin from the response's finalizer rather than the generator's `finally`: if the client
	# disconnects before streaming starts, the generator is never entered and its `finally` never runs.
	def release() -> None:
		loaded_model["pinned"] -= 1
		if not loaded_model["unloaded"]:
			_refresh_ttl(loaded_model, ttl)

	return FinalizedStreamingResponse(stream(), release, media_type="text/event-stream")


# Start server.
if __name__ == "__main__":
	uvicorn.run(app, host=HOST, port=PORT)
