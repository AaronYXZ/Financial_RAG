import json
import urllib.error

import pytest

from rag_eval.generation_adapter import GenerationRequestError, OpenAICompatibleAdapter


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(
            {
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
