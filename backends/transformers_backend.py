"""
HuggingFace transformers backend. Loads a standard HF model directory (config.json,
tokenizer files, safetensors weights) with device_map="auto".

model.json: {"backend": "transformers"}

Chat requests are rendered with the tokenizer's `apply_chat_template`; models whose
tokenizer has no chat template (e.g. GPT-2) accept raw prompts only. Reasoning and
tool calls are not parsed out of the output on this backend: whatever the model
prints is streamed as `token` events.
"""
import gc
import threading
from typing import Any, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from . import Backend, GenEvent, GenParams, Message, TemplateError

STREAMER_TIMEOUT = 60  # seconds to wait for the next token before giving up


# Ends generation at the next token once `event` is set, so unload() can stop a running generate.
class _StopOnEvent(StoppingCriteria):
	def __init__(self, event: threading.Event) -> None:
		self.event = event

	def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
		return torch.full((input_ids.shape[0],), self.event.is_set(), device=input_ids.device, dtype=torch.bool)  # type: ignore[return-value]


# Runs model.generate on a thread and keeps its output, so the caller can inspect how it ended.
def _generate_into(result: dict[str, Any], model: PreTrainedModel, kwargs: dict[str, Any]) -> None:
	try:
		result["ids"] = model.generate(**kwargs)
	except BaseException as e:  # surfaced by the streamer's timeout otherwise; keep the real cause
		result["error"] = e
		raise


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

	def has_chat_template(self) -> bool:
		return self.tokenizer is not None and self.tokenizer.chat_template is not None

	def context_length(self) -> int | None:
		if self.model is None:
			return None
		n = getattr(self.model.config, "max_position_embeddings", None)
		return n if isinstance(n, int) else None

	def render_template(self, messages: list[Message], tools: list[dict[str, Any]] | None, template_args: dict[str, Any]) -> str:
		tok = self.tokenizer
		assert tok is not None, "render_template() called before load()"
		if tok.chat_template is None:
			raise TemplateError("model has no chat template; send a raw 'prompt' instead of 'messages'")
		try:
			prompt = tok.apply_chat_template(messages, tools=tools or None, add_generation_prompt=True,
											 tokenize=False, **template_args)
		except Exception as e:  # jinja errors, missing keys, bad roles: all client-side problems
			raise TemplateError(f"chat template rejected the request: {e}")
		assert isinstance(prompt, str)
		return prompt

	def generate(self, params: GenParams) -> Iterator[GenEvent]:
		model, tok = self.model, self.tokenizer
		assert model is not None and tok is not None, "generate() called before load()"

		prompt = params.prompt if params.messages is None else self.render_template(params.messages, params.tools, params.template_args)
		inputs = tok(prompt, return_tensors="pt").to(model.device)
		prompt_tokens = int(inputs["input_ids"].shape[1])
		eos_ids = model.generation_config.eos_token_id
		eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids]) - {None}
		if tok.eos_token_id is not None:
			eos_ids.add(tok.eos_token_id)

		streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True, timeout=STREAMER_TIMEOUT)  # type: ignore[arg-type]
		kwargs: dict[str, Any] = dict(**inputs, streamer=streamer, max_new_tokens=params.max_tokens,
									  temperature=params.temperature, top_p=params.top_p,
									  repetition_penalty=params.repetition_penalty, do_sample=True,
									  stopping_criteria=StoppingCriteriaList([_StopOnEvent(self._stop)]))
		if params.stop:
			kwargs.update(stop_strings=params.stop, tokenizer=tok)
		result: dict[str, Any] = {}
		thread = threading.Thread(target=_generate_into, args=(result, model, kwargs), daemon=True)
		with self._lock:
			if self._stop.is_set():
				raise RuntimeError("model unloaded")
			self._threads = {t for t in self._threads if t.is_alive()}
			self._threads.add(thread)
			thread.start()
		# This suspended frame must not keep the weights alive: unload() frees them right after the
		# generate thread exits, possibly before the server gets around to closing this generator.
		del model, inputs, kwargs
		text = ""
		for fragment in streamer:
			if fragment:  # the streamer flushes empty strings at token boundaries it cannot decode yet
				text += fragment
				yield {"token": fragment}
		thread.join()
		if self._stop.is_set():
			raise RuntimeError("model unloaded")  # do not pass off a stopped generation as complete
		if "error" in result:
			raise RuntimeError(f"generate failed: {result['error']}")

		ids = result["ids"][0]
		completion_tokens = int(ids.shape[0]) - prompt_tokens
		if completion_tokens > 0 and int(ids[-1]) in eos_ids:
			stop_reason = "eos"
		elif params.stop and any(text.endswith(s) for s in params.stop):
			stop_reason = "stop"
		elif completion_tokens >= params.max_tokens:
			stop_reason = "length"
		else:
			stop_reason = "eos"
		del ids, result
		yield {"stop_reason": stop_reason}
		yield {"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
						 "context_length": self.context_length()}}
