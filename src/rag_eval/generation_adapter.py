"""Generation adapters for local, OpenAI, and OpenRouter model providers."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


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


GENERATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "abstain": {"type": "boolean"},
        "citations": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "required": ["answer", "abstain", "citations", "confidence"],
    "additionalProperties": False,
}


class OpenAICompatibleAdapter:
    """Call a local `/chat/completions` endpoint and count with its HF tokenizer."""

    provider = "local"

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        tokenizer_id: str,
        max_output_tokens: int = 1024,
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


class OpenAIResponsesAdapter:
    """Call the OpenAI Responses API with structured output."""

    provider = "openai"

    def __init__(
        self,
        *,
        model_id: str,
        env_file: Path = Path(".env"),
        api_key_env: str = "OPENAI_API_KEY",
        max_output_tokens: int = 1024,
        reasoning_effort: str = "low",
        timeout_seconds: float = 300.0,
        runtime_retries: int = 1,
    ) -> None:
        if runtime_retries < 0:
            raise ValueError("runtime_retries cannot be negative")
        try:
            import openai
            import tiktoken
            from dotenv import load_dotenv
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI generation requires the openai extra: "
                "pip install -e '.[generation,openai]'"
            ) from exc

        load_dotenv(dotenv_path=env_file, override=False)
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} is not set. Add it to {env_file} or the environment."
            )

        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.runtime_retries = runtime_retries
        self.client = OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self.retryable_errors = (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        )
        try:
            self.encoding = tiktoken.encoding_for_model(model_id)
        except KeyError:
            self.encoding = tiktoken.get_encoding("o200k_base")

    def count_tokens(self, system_prompt: str, user_prompt: str) -> int:
        # Responses framing adds a small amount beyond the two text fields.
        return (
            len(self.encoding.encode(system_prompt))
            + len(self.encoding.encode(user_prompt))
            + 16
        )

    @staticmethod
    def _usage_value(usage: Any, field: str) -> int | None:
        value = getattr(usage, field, None) if usage is not None else None
        return int(value) if isinstance(value, int) else None

    def generate(self, system_prompt: str, user_prompt: str) -> AdapterResult:
        request: dict[str, Any] = {
            "model": self.model_id,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": self.max_output_tokens,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "generation_response",
                    "strict": True,
                    "schema": GENERATION_RESPONSE_SCHEMA,
                }
            },
        }
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}

        started = time.perf_counter()
        response = None
        last_error: Exception | None = None
        for attempt in range(self.runtime_retries + 1):
            try:
                response = self.client.responses.create(**request)
                break
            except self.retryable_errors as exc:
                last_error = exc
                if attempt == self.runtime_retries:
                    raise GenerationRequestError(
                        f"OpenAI generation request failed: {exc}",
                        attempts=attempt + 1,
                    ) from exc
        if response is None:
            raise GenerationRequestError(
                f"OpenAI generation request failed: {last_error}",
                attempts=self.runtime_retries + 1,
            )

        text = str(getattr(response, "output_text", "") or "").strip()
        if not text:
            raise RuntimeError("OpenAI Responses API returned no output text")
        usage = getattr(response, "usage", None)
        return AdapterResult(
            text=text,
            latency_seconds=time.perf_counter() - started,
            input_tokens=self._usage_value(usage, "input_tokens"),
            output_tokens=self._usage_value(usage, "output_tokens"),
            attempts=attempt + 1,
        )


class OpenRouterChatAdapter:
    """Call OpenRouter chat completions with a user-selected model slug."""

    provider = "openrouter"

    def __init__(
        self,
        *,
        model_id: str,
        env_file: Path = Path(".env"),
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = "https://openrouter.ai/api/v1",
        http_referer: str | None = None,
        app_title: str | None = None,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
        timeout_seconds: float = 300.0,
        runtime_retries: int = 1,
    ) -> None:
        if runtime_retries < 0:
            raise ValueError("runtime_retries cannot be negative")
        if not model_id.strip():
            raise ValueError("model_id cannot be empty")
        try:
            import tiktoken
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError(
                "OpenRouter generation requires the openrouter extra: "
                "pip install -e '.[generation,openrouter]'"
            ) from exc

        load_dotenv(dotenv_path=env_file, override=False)
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} is not set. Add it to {env_file} or the environment."
            )

        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.http_referer = http_referer
        self.app_title = app_title
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.runtime_retries = runtime_retries
        self.encoding = tiktoken.get_encoding("o200k_base")

    def count_tokens(self, system_prompt: str, user_prompt: str) -> int:
        # OpenRouter spans tokenizer families. This is a stable preflight estimate;
        # authoritative provider token counts are recorded from response usage.
        return (
            len(self.encoding.encode(system_prompt))
            + len(self.encoding.encode(user_prompt))
            + 16
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title
        return headers

    @staticmethod
    def _error_detail(exc: urllib.error.HTTPError) -> str:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            error = payload.get("error")
            if isinstance(error, Mapping):
                message = error.get("message")
                if message:
                    return str(message)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        return str(exc)

    @staticmethod
    def _response_text(response_payload: Mapping[str, Any]) -> str:
        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenRouter returned an unexpected response") from exc
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, Mapping) and item.get("type") == "text"
            ]
            return "".join(str(part) for part in parts).strip()
        return ""

    def generate(self, system_prompt: str, user_prompt: str) -> AdapterResult:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "generation_response",
                    "strict": True,
                    "schema": GENERATION_RESPONSE_SCHEMA,
                },
            },
            "provider": {"require_parameters": True},
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        started = time.perf_counter()
        response_payload: Mapping[str, Any] | None = None
        last_error: Exception | None = None
        for attempt in range(self.runtime_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                if not isinstance(decoded, Mapping):
                    raise RuntimeError("OpenRouter returned a non-object response")
                response_payload = decoded
                break
            except urllib.error.HTTPError as exc:
                last_error = exc
                detail = self._error_detail(exc)
                retryable = exc.code in {408, 429, 500, 502, 503, 504}
                if not retryable or attempt == self.runtime_retries:
                    raise GenerationRequestError(
                        f"OpenRouter generation request failed ({exc.code}): {detail}",
                        attempts=attempt + 1,
                    ) from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == self.runtime_retries:
                    raise GenerationRequestError(
                        f"OpenRouter generation request failed: {exc}",
                        attempts=attempt + 1,
                    ) from exc
        if response_payload is None:
            raise GenerationRequestError(
                f"OpenRouter generation request failed: {last_error}",
                attempts=self.runtime_retries + 1,
            )

        text = self._response_text(response_payload)
        if not text:
            raise RuntimeError("OpenRouter returned no output text")
        usage = response_payload.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        return AdapterResult(
            text=text,
            latency_seconds=time.perf_counter() - started,
            input_tokens=(
                int(usage["prompt_tokens"])
                if isinstance(usage.get("prompt_tokens"), int)
                else None
            ),
            output_tokens=(
                int(usage["completion_tokens"])
                if isinstance(usage.get("completion_tokens"), int)
                else None
            ),
            attempts=attempt + 1,
        )
