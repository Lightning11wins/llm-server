"""
llama.cpp backend. Runs a GGUF model in a managed `llama-server` subprocess bound to
127.0.0.1 on a free port, and proxies generation requests to it. Unloading kills
the subprocess, which reliably frees VRAM.

model.json:
    {
        "backend": "llama-cpp",
        "model": "Qwen3.5-9B-Q4_K_M.gguf",        # GGUF file inside the model dir (required)
        "args": ["-ngl", "99", "-c", "32768"],   # extra llama-server flags (optional)
        "startup_timeout": 300,                  # seconds to wait for the model to load (optional)
        "template_defaults": {"enable_thinking": false}   # chat template arguments (optional)
    }

Raw prompts go to llama-server's `/completion`; chat requests (messages) go to its
`/v1/chat/completions`, which applies the GGUF's embedded jinja chat template and
parses reasoning and tool calls out of the output.

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

from . import Backend, GenEvent, GenParams, Message, ModelConfigError, TemplateError

BASE_DIR         = Path(__file__).parent.parent
LLAMA_SERVER_BIN = BASE_DIR / "bin" / "llama.cpp" / "llama-server"
LOGS_DIR         = BASE_DIR / "logs"

DEFAULT_STARTUP_TIMEOUT = 300   # seconds
HEALTH_POLL_INTERVAL    = 0.5   # seconds
SHUTDOWN_GRACE          = 10    # seconds between SIGTERM and SIGKILL
SSE_PING_INTERVAL       = 15    # seconds; llama-server emits SSE comments at this rate while silent (e.g. prompt processing)
READ_TIMEOUT            = 60    # seconds without any bytes (token or ping) before the stream is considered dead

# Flags model.json may not set: the server chooses the model file and binding itself, an API key
# would make its own proxied requests fail, and chat mode depends on the jinja template engine.
RESERVED_ARGS = {"-m", "--model", "-mu", "--model-url", "-hf", "--hf-repo", "--host", "--port",
				 "--no-webui", "--webui", "--api-key", "--jinja", "--no-jinja"}

# llama-server's stop_type (raw completion) and finish_reason (chat) mapped onto our stop_reason.
COMPLETION_STOP_REASONS = {"eos": "eos", "limit": "length", "word": "stop"}
CHAT_STOP_REASONS       = {"stop": "eos", "length": "length", "tool_calls": "tool"}

_prctl = ctypes.CDLL(None, use_errno=True).prctl  # resolved at import so the forked child makes a single C call
PR_SET_PDEATHSIG = 1


def _free_port() -> int:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind(("127.0.0.1", 0))
		return s.getsockname()[1]


def _die_with_parent() -> None:
	"""preexec_fn: ask Linux to SIGKILL the child when the parent goes away.

	PR_SET_PDEATHSIG fires when the *thread* that forked exits. load() runs on an asyncio default
	executor worker; those threads live until interpreter shutdown, so in practice this means
	"when the server process exits", including crashes and SIGKILL. unload() is the normal path.
	"""
	_prctl(PR_SET_PDEATHSIG, signal.SIGKILL)


class LlamaCppBackend(Backend):
	proc: subprocess.Popen[bytes] | None = None
	log_file: IO[bytes] | None = None
	log_path: Path | None = None
	port: int = 0
	n_ctx: int | None = None
	chat_template: bool = False

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
		reserved = RESERVED_ARGS.intersection(a.split("=", 1)[0] for a in args)
		if reserved:
			raise ModelConfigError(f"'args' may not set {', '.join(sorted(reserved))}; the server sets these itself")
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
			"--jinja",
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
					self._read_props()
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

	def is_alive(self) -> bool:
		return self.proc is not None and self.proc.poll() is None

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

	# Read what /props says about the loaded model: the per-slot context size (n_ctx divided over the
	# parallel slots) and whether the GGUF carries a chat template. Without one llama-server would
	# silently fall back to ChatML, so such models are restricted to raw prompts like on transformers.
	def _read_props(self) -> None:
		try:
			props = requests.get(f"{self.base_url}/props", timeout=5).json()
			n_ctx = props["default_generation_settings"]["n_ctx"]
		except (requests.RequestException, ValueError, KeyError, TypeError):
			return
		self.n_ctx = n_ctx if isinstance(n_ctx, int) else None
		self.chat_template = bool(props.get("chat_template"))

	def has_chat_template(self) -> bool:
		return self.chat_template

	def context_length(self) -> int | None:
		return self.n_ctx

	def render_template(self, messages: list[Message], tools: list[dict[str, Any]] | None, template_args: dict[str, Any]) -> str:
		if not self.is_alive():
			raise RuntimeError("llama-server subprocess is not running")
		body: dict[str, Any] = {"messages": messages, "chat_template_kwargs": template_args}
		if tools:
			body["tools"] = tools
		resp = requests.post(f"{self.base_url}/apply-template", json=body, timeout=(5, 30))
		if resp.status_code == 400:
			raise TemplateError(_error_message(resp))
		if not resp.ok:
			raise RuntimeError(f"llama-server returned HTTP {resp.status_code}: {resp.text[:500]}")
		return resp.json()["prompt"]

	def generate(self, params: GenParams) -> Iterator[GenEvent]:
		if not self.is_alive():
			raise RuntimeError("llama-server subprocess is not running")
		if params.messages is not None:
			yield from self._chat(params)
		else:
			yield from self._completion(params)

	# Stream `data:` payloads from a llama-server SSE endpoint, raising on transport or server errors.
	def _sse(self, path: str, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
		body["stream"] = True
		body["cache_prompt"] = True
		body["sse_ping_interval"] = SSE_PING_INTERVAL  # keeps READ_TIMEOUT from firing during long prompt processing
		with requests.post(f"{self.base_url}{path}", json=body, stream=True, timeout=(5, READ_TIMEOUT)) as resp:
			if not resp.ok:
				raise RuntimeError(f"llama-server returned HTTP {resp.status_code}: {_error_message(resp)}")
			for raw in resp.iter_lines():
				if not raw.startswith(b"data: "):
					continue  # blank separators, SSE comments/pings
				if raw == b"data: [DONE]":
					return
				event = json.loads(raw[6:])
				if "error" in event:
					err = event["error"]
					raise RuntimeError(f"llama-server error: {err.get('message', err) if isinstance(err, dict) else err}")
				yield event

	def _usage(self, prompt_tokens: int | None, completion_tokens: int | None) -> GenEvent:
		return {"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
						  "context_length": self.n_ctx}}

	def _completion(self, params: GenParams) -> Iterator[GenEvent]:
		body: dict[str, Any] = {
			"prompt": params.prompt,
			"n_predict": params.max_tokens,
			"temperature": params.temperature,
			"top_p": params.top_p,
			"repeat_penalty": params.repetition_penalty,
			"stop": params.stop,
		}
		for event in self._sse("/completion", body):
			content = event.get("content", "")
			if content:
				yield {"token": content}
			if event.get("stop"):
				yield {"stop_reason": COMPLETION_STOP_REASONS.get(event.get("stop_type"), "eos")}
				timings = event.get("timings", {})
				prompt_n = timings.get("prompt_n")
				if prompt_n is not None:
					prompt_n += timings.get("cache_n", 0)  # prompt_n counts only the tokens not served from cache
				yield self._usage(prompt_n, timings.get("predicted_n"))
				return
		raise RuntimeError("llama-server stream ended without a final event")

	def _chat(self, params: GenParams) -> Iterator[GenEvent]:
		body: dict[str, Any] = {
			"messages": params.messages,
			"chat_template_kwargs": params.template_args,
			"max_tokens": params.max_tokens,
			"temperature": params.temperature,
			"top_p": params.top_p,
			"repeat_penalty": params.repetition_penalty,
			"stop": params.stop,
			"stream_options": {"include_usage": True},
		}
		if params.tools:
			body["tools"] = params.tools

		finish_reason: str | None = None
		usage: dict[str, Any] = {}
		tool_calls: dict[int, dict[str, Any]] = {}  # by index; arguments arrive as fragments
		for event in self._sse("/v1/chat/completions", body):
			if "usage" in event:
				usage = event["usage"] or {}
			for choice in event.get("choices", []):
				delta = choice.get("delta", {})
				if delta.get("reasoning_content"):
					yield {"reasoning": delta["reasoning_content"]}
				if delta.get("content"):
					yield {"token": delta["content"]}
				for call in delta.get("tool_calls", []):
					acc = tool_calls.setdefault(call.get("index", len(tool_calls)),
												{"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
					if call.get("id"):
						acc["id"] = call["id"]
					fn = call.get("function", {})
					acc["function"]["name"] += fn.get("name") or ""
					acc["function"]["arguments"] += fn.get("arguments") or ""
				if choice.get("finish_reason"):
					finish_reason = choice["finish_reason"]

		if finish_reason is None:
			raise RuntimeError("llama-server stream ended without a finish_reason")
		# Tool calls cut off by max_tokens have truncated arguments; they are dropped rather than
		# handed to a harness as if they were complete. The stop_reason "length" tells the story.
		if finish_reason == "tool_calls" and tool_calls:
			yield {"tool_calls": [tool_calls[i] for i in sorted(tool_calls)]}
		yield {"stop_reason": CHAT_STOP_REASONS.get(finish_reason, "eos")}
		yield self._usage(usage.get("prompt_tokens"), usage.get("completion_tokens"))


# Pull the human-readable message out of a llama-server error response.
def _error_message(resp: requests.Response) -> str:
	try:
		err = resp.json()["error"]
		return err["message"] if isinstance(err, dict) else str(err)
	except (ValueError, KeyError, TypeError):
		return resp.text[:500]
