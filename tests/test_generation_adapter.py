import json
import urllib.error
from types import SimpleNamespace

import pytest

from rag_eval.generation.adapter import (
    GenerationRequestError,
    OpenAICompatibleAdapter,
    OpenAIResponsesAdapter,
    OpenRouterChatAdapter,
)


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(
            {
                "model": "qwen/qwen3.7-plus",
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
        ).encode()


def make_adapter() -> OpenAICompatibleAdapter:
    adapter = OpenAICompatibleAdapter.__new__(OpenAICompatibleAdapter)
    adapter.base_url = "http://127.0.0.1:8080/v1"
    adapter.model_id = "local-model"
    adapter.max_output_tokens = 512
    adapter.temperature = 0.0
    adapter.timeout_seconds = 300.0
    adapter.runtime_retries = 1
    return adapter


def test_count_tokens_uses_input_ids_from_batch_encoding_shape():
    class FakeTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return {
                "input_ids": [10, 20, 30],
                "attention_mask": [1, 1, 1],
            }

    adapter = make_adapter()
    adapter.tokenizer = FakeTokenizer()

    assert adapter.count_tokens("system", "user") == 3


def test_openai_compatible_adapter_retries_once(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            raise urllib.error.URLError("temporary")
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = make_adapter()

    result = adapter.generate("system", "user")

    request_payload = json.loads(calls[-1][0].data)
    assert calls[-1][0].full_url.endswith("/v1/chat/completions")
    assert request_payload["model"] == "local-model"
    assert request_payload["temperature"] == 0.0
    assert result.attempts == 2
    assert result.input_tokens == 10


def test_openai_compatible_adapter_records_terminal_attempt_count(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("still unavailable")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = make_adapter()

    with pytest.raises(GenerationRequestError) as caught:
        adapter.generate("system", "user")

    assert caught.value.attempts == 2


def make_openai_adapter(responses) -> OpenAIResponsesAdapter:
    adapter = OpenAIResponsesAdapter.__new__(OpenAIResponsesAdapter)
    adapter.model_id = "gpt-test"
    adapter.max_output_tokens = 512
    adapter.reasoning_effort = "none"
    adapter.runtime_retries = 1
    adapter.retryable_errors = (RuntimeError,)
    adapter.client = SimpleNamespace(responses=responses)
    return adapter


def test_openai_responses_adapter_counts_prompt_tokens_with_framing_reserve():
    class FakeEncoding:
        def encode(self, text):
            return text.split()

    adapter = make_openai_adapter(None)
    adapter.encoding = FakeEncoding()

    assert adapter.count_tokens("two words", "three more words") == 21


def test_openai_responses_adapter_uses_structured_responses_api():
    calls = []

    class FakeResponses:
        def create(self, **request):
            calls.append(request)
            return SimpleNamespace(
                output_text='{"answer":"","abstain":true,"citations":[],"confidence":0.5}',
                usage=SimpleNamespace(input_tokens=100, output_tokens=20),
            )

    adapter = make_openai_adapter(FakeResponses())
    result = adapter.generate("system", "user")

    request = calls[0]
    assert request["model"] == "gpt-test"
    assert request["instructions"] == "system"
    assert request["input"] == "user"
    assert request["store"] is False
    assert request["reasoning"] == {"effort": "none"}
    assert request["text"]["format"]["type"] == "json_schema"
    assert request["text"]["format"]["strict"] is True
    assert result.input_tokens == 100
    assert result.output_tokens == 20


def test_openai_responses_adapter_retries_transient_errors():
    calls = []

    class FakeResponses:
        def create(self, **request):
            calls.append(request)
            if len(calls) == 1:
                raise RuntimeError("temporary")
            return SimpleNamespace(output_text="{}", usage=None)

    result = make_openai_adapter(FakeResponses()).generate("system", "user")

    assert result.attempts == 2


def test_openai_responses_adapter_records_terminal_attempt_count():
    class FakeResponses:
        def create(self, **request):
            raise RuntimeError("still unavailable")

    with pytest.raises(GenerationRequestError) as caught:
        make_openai_adapter(FakeResponses()).generate("system", "user")

    assert caught.value.attempts == 2


def make_openrouter_adapter() -> OpenRouterChatAdapter:
    adapter = OpenRouterChatAdapter.__new__(OpenRouterChatAdapter)
    adapter.model_id = "anthropic/claude-sonnet-4.5"
    adapter.base_url = "https://openrouter.ai/api/v1"
    adapter.api_key = "test-key"
    adapter.http_referer = "https://example.com"
    adapter.app_title = "Local RAG Test"
    adapter.fallback_model_ids = (
        "qwen/qwen3.7-plus",
        "deepseek/deepseek-v4-flash",
    )
    adapter.max_output_tokens = 512
    adapter.temperature = 0.0
    adapter.timeout_seconds = 300.0
    adapter.runtime_retries = 1
    return adapter


def test_openrouter_adapter_counts_prompt_tokens_with_framing_reserve():
    class FakeEncoding:
        def encode(self, text):
            return text.split()

    adapter = make_openrouter_adapter()
    adapter.encoding = FakeEncoding()

    assert adapter.count_tokens("two words", "three more words") == 21


def test_openrouter_adapter_sends_selected_model_and_strict_schema(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = make_openrouter_adapter().generate("system", "user")

    request = calls[0][0]
    payload = json.loads(request.data)
    headers = {key.lower(): value for key, value in request.header_items()}
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert headers["authorization"] == "Bearer test-key"
    assert headers["http-referer"] == "https://example.com"
    assert headers["x-openrouter-title"] == "Local RAG Test"
    assert payload["model"] == "anthropic/claude-sonnet-4.5"
    assert payload["models"] == [
        "qwen/qwen3.7-plus",
        "deepseek/deepseek-v4-flash",
    ]
    assert payload["max_tokens"] == 512
    assert "temperature" not in payload
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["provider"]["require_parameters"] is True
    assert result.input_tokens == 10
    assert result.output_tokens == 2
    assert result.resolved_model_id == "qwen/qwen3.7-plus"


def test_openrouter_adapter_retries_transient_transport_errors(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise urllib.error.URLError("temporary")
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = make_openrouter_adapter().generate("system", "user")

    assert result.attempts == 2
