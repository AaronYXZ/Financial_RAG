"""Minimal OpenAI-compatible adapter for local generation servers."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class AdapterResult:
    text: str
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    attempts: int = 1


class GenerationRequestError(RuntimeError):
    def __init__(self, message: str, *, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class GenerationAdapter(Protocol):
    model_id: str

    def count_tokens(self, system_prompt: str, user_prompt: str) -> int: ...

    def generate(self, system_prompt: str, user_prompt: str) -> AdapterResult: ...


class OpenAICompatibleAdapter:
    """Call a local `/chat/completions` endpoint and count with its HF tokenizer."""

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        tokenizer_id: str,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
        timeout_seconds: float = 300.0,
        runtime_retries: int = 1,
    ) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local generation requires transformers: pip install -e '.[generation]'"
            ) from exc

        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        if runtime_retries < 0:
            raise ValueError("runtime_retries cannot be negative")
        self.runtime_retries = runtime_retries
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    def _messages(self, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def count_tokens(self, system_prompt: str, user_prompt: str) -> int:
        tokenized = self.tokenizer.apply_chat_template(
            self._messages(system_prompt, user_prompt),
            add_generation_prompt=True,
            tokenize=True,
        )
        token_ids = tokenized["input_ids"] if isinstance(tokenized, Mapping) else tokenized
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError(
                    "Expected one tokenized prompt, received a batch with multiple rows"
                )
            token_ids = token_ids[0]
        return len(token_ids)

    def generate(self, system_prompt: str, user_prompt: str) -> AdapterResult:
        payload = {
            "model": self.model_id,
            "messages": self._messages(system_prompt, user_prompt),
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        response_payload = None
        last_error: Exception | None = None
        for attempt in range(self.runtime_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == self.runtime_retries:
                    raise GenerationRequestError(
                        f"Generation request failed: {exc}", attempts=attempt + 1
                    ) from exc
        if response_payload is None:
            raise GenerationRequestError(
                f"Generation request failed: {last_error}",
                attempts=self.runtime_retries + 1,
            )
        latency = time.perf_counter() - started

        try:
            text = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Generation server returned an unexpected response") from exc
        usage = response_payload.get("usage") or {}
        return AdapterResult(
            text=str(text),
            latency_seconds=latency,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            attempts=attempt + 1,
        )
