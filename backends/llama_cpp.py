"""
llama.cpp backend. Runs a GGUF model in a managed `llama-server` subprocess bound to
127.0.0.1 on a free port, and proxies generation requests to it. Unloading kills
the subprocess, which reliably frees VRAM.

model.json:
    {
        "backend": "llama-cpp",
        "model": "Qwen3.5-9B-Q4_K_M.gguf",        # GGUF file inside the model dir (required)
        "args": ["-ngl", "99", "-c", "32768"],   # extra llama-server flags (optional)
        "startup_timeout": 300                   # seconds to wait for the model to load (optional)
    }

Subprocess output is written to logs/llama-<model>-<timestamp>.log.
"""
import ctypes
import json
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import IO, Any, Iterator

import requests

from . import Backend, GenParams, ModelConfigError

BASE_DIR         = Path(__file__).parent.parent
LLAMA_SERVER_BIN = BASE_DIR / "bin" / "llama.cpp" / "llama-server"
LOGS_DIR         = BASE_DIR / "logs"

DEFAULT_STARTUP_TIMEOUT = 300   # seconds
HEALTH_POLL_INTERVAL    = 0.5   # seconds
SHUTDOWN_GRACE          = 10    # seconds between SIGTERM and SIGKILL
READ_TIMEOUT            = 60    # seconds to wait for the next streamed token


def _free_port() -> int:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind(("127.0.0.1", 0))
		return s.getsockname()[1]


def _die_with_parent() -> None:
	"""preexec_fn: ask Linux to SIGKILL the child if this server process dies."""
	try:
		PR_SET_PDEATHSIG = 1
		ctypes.CDLL("libc.so.6").prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
	except Exception:
		pass  # best effort; unload() still handles the normal path


class LlamaCppBackend(Backend):
	proc: subprocess.Popen[bytes] | None = None
	log_file: IO[bytes] | None = None
	log_path: Path | None = None
	port: int = 0

	def __init__(self, name: str, model_dir: Path, config: dict[str, Any]) -> None:
		super().__init__(name, model_dir, config)

		model_file = config.get("model")
		if not isinstance(model_file, str) or not model_file:
			raise ModelConfigError("llama-cpp backend requires a string 'model' key naming the GGUF file")
		self.model_path = model_dir / model_file
		if not self.model_path.is_file():
			raise ModelConfigError(f"GGUF file not found: {model_file}")

		args = config.get("args", [])
		if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
			raise ModelConfigError("'args' must be a list of strings")
		self.args: list[str] = args

		timeout = config.get("startup_timeout", DEFAULT_STARTUP_TIMEOUT)
		if not isinstance(timeout, (int, float)) or timeout <= 0:
			raise ModelConfigError("'startup_timeout' must be a positive number")
		self.startup_timeout = float(timeout)

	@property
	def base_url(self) -> str:
		return f"http://127.0.0.1:{self.port}"

	def load(self) -> None:
		if not LLAMA_SERVER_BIN.is_file():
			raise RuntimeError(f"llama-server binary not found at {LLAMA_SERVER_BIN}")

		self.port = _free_port()
		LOGS_DIR.mkdir(exist_ok=True)
		self.log_path = LOGS_DIR / f"llama-{self.name}-{time.strftime('%Y-%m-%d_%H-%M-%S')}.log"
		self.log_file = open(self.log_path, "wb")

		cmd = [
			str(LLAMA_SERVER_BIN),
			"-m", str(self.model_path),
			"--host", "127.0.0.1",
			"--port", str(self.port),
			"--no-webui",
			*self.args,
		]
		self.proc = subprocess.Popen(
			cmd, stdout=self.log_file, stderr=subprocess.STDOUT,
			stdin=subprocess.DEVNULL, preexec_fn=_die_with_parent,
		)

		# Wait for the server to report healthy (model fully loaded).
		deadline = time.monotonic() + self.startup_timeout
		while True:
			if self.proc.poll() is not None:
				self._cleanup()
				raise RuntimeError(f"llama-server exited with code {self.proc.returncode} during load. {self._log_tail()}")
			try:
				if requests.get(f"{self.base_url}/health", timeout=2).ok:
					return
			except requests.RequestException:
				pass
			if time.monotonic() > deadline:
				self.unload()
				raise RuntimeError(f"llama-server did not become healthy within {self.startup_timeout:.0f}s. {self._log_tail()}")
			time.sleep(HEALTH_POLL_INTERVAL)

	def unload(self) -> None:
		proc = self.proc
		if proc is not None and proc.poll() is None:
			proc.terminate()
			try:
				proc.wait(SHUTDOWN_GRACE)
			except subprocess.TimeoutExpired:
				proc.kill()
				proc.wait()
		self._cleanup()

	def _cleanup(self) -> None:
		if self.log_file is not None:
			self.log_file.close()
			self.log_file = None
		self.proc = None

	def _log_tail(self, lines: int = 15) -> str:
		if self.log_path is None or not self.log_path.exists():
			return ""
		tail = self.log_path.read_text(errors="replace").splitlines()[-lines:]
		return f"See {self.log_path.name}:\n" + "\n".join(tail)

	def generate(self, params: GenParams) -> Iterator[str]:
		if self.proc is None or self.proc.poll() is not None:
			raise RuntimeError("llama-server subprocess is not running")

		body = {
			"prompt": params.prompt,
			"n_predict": params.max_tokens,
			"temperature": params.temperature,
			"top_p": params.top_p,
			"repeat_penalty": params.repetition_penalty,
			"stream": True,
			"cache_prompt": True,
		}
		with requests.post(f"{self.base_url}/completion", json=body, stream=True, timeout=(5, READ_TIMEOUT)) as resp:
			if not resp.ok:
				raise RuntimeError(f"llama-server returned HTTP {resp.status_code}: {resp.text[:500]}")
			for raw in resp.iter_lines():
				if not raw.startswith(b"data: "):
					continue  # blank separators, SSE comments/pings
				event = json.loads(raw[6:])
				if "error" in event:
					raise RuntimeError(f"llama-server error: {event['error']}")
				content = event.get("content", "")
				if content:
					yield content
				if event.get("stop"):
					break
