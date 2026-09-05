"""
HuggingFace transformers backend. Loads a standard HF model directory (config.json,
tokenizer files, safetensors weights) with device_map="auto".

model.json: {"backend": "transformers"}
"""
import gc
import threading
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from . import Backend, GenParams

STREAMER_TIMEOUT = 60  # seconds to wait for the next token before giving up


# Ends generation at the next token once `event` is set, so unload() can stop a running generate.
class _StopOnEvent(StoppingCriteria):
	def __init__(self, event: threading.Event) -> None:
		self.event = event

	def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
		return torch.full((input_ids.shape[0],), self.event.is_set(), device=input_ids.device, dtype=torch.bool)  # type: ignore[return-value]


class TransformersBackend(Backend):
	model: PreTrainedModel | None = None
	tokenizer: PreTrainedTokenizerBase | None = None
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self._stop = threading.Event()
		self._lock = threading.Lock()  # guards _threads and the start-vs-unload decision
		self._threads: set[threading.Thread] = set()  # generate threads that may still be running

	def load(self) -> None:
		path = str(self.model_dir)
		tok = AutoTokenizer.from_pretrained(path)
		if tok.pad_token_id is None:
			tok.pad_token_id = tok.eos_token_id
		self.tokenizer = tok
		self.model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")

	def unload(self) -> None:
		# Generate threads hold references to the model, so nothing is freed until they all exit.
		# Stop them at their next token and wait, so the VRAM is actually returned before we report
		# done. The event is set before taking the lock, so no new thread can start after the snapshot.
		self._stop.set()
		with self._lock:
			threads = list(self._threads)
		for thread in threads:
			thread.join()
		self._threads.clear()
		self.model = None
		self.tokenizer = None
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	def generate(self, params: GenParams) -> Iterator[str]:
		model, tok = self.model, self.tokenizer
		assert model is not None and tok is not None, "generate() called before load()"

		inputs = tok(params.prompt, return_tensors="pt").to(model.device)
		streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=STREAMER_TIMEOUT)  # type: ignore[arg-type]
		thread = threading.Thread(
			target=model.generate,  # type: ignore[arg-type]
			kwargs=dict(**inputs, streamer=streamer, max_new_tokens=params.max_tokens,
						temperature=params.temperature, top_p=params.top_p,
						repetition_penalty=params.repetition_penalty, do_sample=True,
						stopping_criteria=StoppingCriteriaList([_StopOnEvent(self._stop)])),
			daemon=True,
		)
		with self._lock:
			if self._stop.is_set():
				raise RuntimeError("model unloaded")
			self._threads = {t for t in self._threads if t.is_alive()}
			self._threads.add(thread)
			thread.start()
		# This suspended frame must not keep the weights alive: unload() frees them right after the
		# generate thread exits, possibly before the server gets around to closing this generator.
		del model, inputs
		yield from streamer
		if self._stop.is_set():
			raise RuntimeError("model unloaded")  # do not pass off a stopped generation as complete
