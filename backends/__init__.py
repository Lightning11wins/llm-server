"""
Model backends.

Every model directory under models/ must contain a `model.json` describing how to
run it. The only required key is `backend`; the rest is backend-specific:

    {"backend": "transformers"}

A backend is a plain object with blocking `load()`, `unload()` and `generate()`
methods. The server calls them from a thread pool so the event loop stays free.
"""
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

MODEL_CONFIG_FILE = "model.json"


@dataclass
class GenParams:
	prompt: str
	max_tokens: int
	temperature: float
	top_p: float
	repetition_penalty: float


class ModelConfigError(Exception):
	"""Raised when a model directory has a missing or invalid model.json."""


class Backend(ABC):
	def __init__(self, name: str, model_dir: Path, config: dict[str, Any]) -> None:
		self.name = name
		self.model_dir = model_dir
		self.config = config

	@abstractmethod
	def load(self) -> None:
		"""Load the model into memory. Blocking. Raises on failure."""

	@abstractmethod
	def unload(self) -> None:
		"""Release all resources held by the model. Blocking. Must not raise."""

	@abstractmethod
	def generate(self, params: GenParams) -> Iterator[str]:
		"""Yield generated text fragments. Blocking iterator. Raises on failure."""


def read_model_config(model_dir: Path) -> dict[str, Any]:
	"""Read and minimally validate models/<name>/model.json."""
	path = model_dir / MODEL_CONFIG_FILE
	if not path.is_file():
		raise ModelConfigError(f"missing {MODEL_CONFIG_FILE}")
	try:
		config = json.loads(path.read_text())
	except (OSError, json.JSONDecodeError) as e:
		raise ModelConfigError(f"unreadable {MODEL_CONFIG_FILE}: {e}")
	if not isinstance(config, dict) or not isinstance(config.get("backend"), str):
		raise ModelConfigError(f"{MODEL_CONFIG_FILE} must be an object with a string 'backend' key")
	if config["backend"] not in BACKENDS:
		raise ModelConfigError(f"unknown backend '{config['backend']}' (known: {', '.join(sorted(BACKENDS))})")
	return config


def create_backend(name: str, model_dir: Path) -> Backend:
	config = read_model_config(model_dir)
	return BACKENDS[config["backend"]](name, model_dir, config)


# Imported last to avoid circular imports; each module registers itself here.
from . import transformers_backend  # noqa: E402

BACKENDS: dict[str, type[Backend]] = {
	"transformers": transformers_backend.TransformersBackend,
}
