"""
Model backends.

Every model directory under models/ must contain a `model.json` describing how to
run it. The only required key is `backend`; the rest is backend-specific:

    {"backend": "transformers"}
    {"backend": "llama-cpp", "model": "foo.gguf", "args": ["-ngl", "99", "-c", "32768"]}

Any backend may also carry `template_defaults`, a dict of chat-template arguments
applied to every chat request unless the request overrides them:

    {"backend": "llama-cpp", "model": "foo.gguf", "template_defaults": {"enable_thinking": false}}

A backend is a plain object with blocking `load()`, `unload()` and `generate()`
methods. The server calls them from a thread pool so the event loop stays free.

`generate()` yields events, each a dict with exactly one key:

    {"token": str}         a fragment of the answer
    {"reasoning": str}     a fragment of the model's thinking (chat mode, if the template separates it)
    {"tool_calls": [...]}  OpenAI-shaped tool calls, emitted once at the end (chat mode)
    {"stop_reason": str}   one of "eos", "length", "stop" or "tool"; emitted once at the end
    {"usage": {...}}       prompt_tokens, completion_tokens and context_length; emitted last
"""
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

MODEL_CONFIG_FILE = "model.json"

GenEvent = dict[str, Any]
Message = dict[str, Any]

STOP_REASONS = ("eos", "length", "stop", "tool")


@dataclass
class GenParams:
	max_tokens: int
	temperature: float
	top_p: float
	repetition_penalty: float
	stop: list[str] = field(default_factory=list)
	prompt: str | None = None                   # raw completion; exactly one of prompt/messages is set
	messages: list[Message] | None = None       # chat completion, rendered through the model's template
	tools: list[dict[str, Any]] | None = None   # OpenAI function schemas, chat mode only
	template_args: dict[str, Any] = field(default_factory=dict)  # extra variables for the chat template


class ModelConfigError(Exception):
	"""Raised when a model directory has a missing or invalid model.json."""


class TemplateError(Exception):
	"""Raised when messages/tools cannot be rendered through the model's chat template (a client error)."""


class Backend(ABC):
	def __init__(self, name: str, model_dir: Path, config: dict[str, Any]) -> None:
		self.name = name
		self.model_dir = model_dir
		self.config = config
		self.template_defaults: dict[str, Any] = config.get("template_defaults", {})

	@abstractmethod
	def load(self) -> None:
		"""Load the model into memory. Blocking. Raises on failure."""

	@abstractmethod
	def unload(self) -> None:
		"""Release all resources held by the model. Blocking. Must not raise."""

	@abstractmethod
	def generate(self, params: GenParams) -> Iterator[GenEvent]:
		"""Yield generation events (see module docstring). Blocking iterator. Raises on failure."""

	@abstractmethod
	def render_template(self, messages: list[Message], tools: list[dict[str, Any]] | None, template_args: dict[str, Any]) -> str:
		"""Render messages through the model's chat template and return the prompt text.
		Raises TemplateError if the model has no template or the input does not fit it."""

	def has_chat_template(self) -> bool:
		"""Whether chat-mode requests (messages) can be served. Only valid after load()."""
		return True

	def context_length(self) -> int | None:
		"""Total tokens (prompt + completion) the loaded model accepts, or None if unknown. Only valid after load()."""
		return None

	def is_alive(self) -> bool:
		"""False once the loaded model can no longer serve requests (e.g. its process died)."""
		return True


def read_model_config(model_dir: Path) -> dict[str, Any]:
	"""Read and minimally validate models/<name>/model.json."""
	path = model_dir / MODEL_CONFIG_FILE
	if not path.is_file():
		raise ModelConfigError(f"missing {MODEL_CONFIG_FILE}")
	try:
		config = json.loads(path.read_text())
	except (OSError, ValueError) as e:  # ValueError covers JSONDecodeError and UnicodeDecodeError
		raise ModelConfigError(f"unreadable {MODEL_CONFIG_FILE}: {e}")
	if not isinstance(config, dict) or not isinstance(config.get("backend"), str):
		raise ModelConfigError(f"{MODEL_CONFIG_FILE} must be an object with a string 'backend' key")
	if config["backend"] not in BACKENDS:
		raise ModelConfigError(f"unknown backend '{config['backend']}' (known: {', '.join(sorted(BACKENDS))})")
	if not isinstance(config.get("template_defaults", {}), dict):
		raise ModelConfigError("'template_defaults' must be an object")
	return config


def create_backend(name: str, model_dir: Path) -> Backend:
	config = read_model_config(model_dir)
	return BACKENDS[config["backend"]](name, model_dir, config)


# Imported last to avoid circular imports; each module registers itself here.
from . import transformers_backend, llama_cpp  # noqa: E402

BACKENDS: dict[str, type[Backend]] = {
	"transformers": transformers_backend.TransformersBackend,
	"llama-cpp": llama_cpp.LlamaCppBackend,
}
