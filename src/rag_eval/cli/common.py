"""Shared CLI arguments and provider construction."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..generation.adapter import (
    OpenAICompatibleAdapter,
    OpenAIResponsesAdapter,
    OpenRouterChatAdapter,
)

LOCAL_MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
OPENAI_MODEL_ID = "gpt-5"
OPENROUTER_MODEL_ID = "openai/gpt-5.6-luna-pro"
OPENROUTER_FALLBACK_MODEL_IDS = (
    "qwen/qwen3.7-plus",
    "deepseek/deepseek-v4-flash",
)
OPENAI_MODEL_IDS = ("gpt-5", "gpt-5.6-sol", "gpt-5.6-luna")
OPENAI_REASONING_EFFORTS = {
    "gpt-5": ("minimal", "low", "medium", "high"),
    "gpt-5.6-sol": ("none", "low", "medium", "high", "xhigh", "max"),
    "gpt-5.6-luna": ("none", "low", "medium", "high", "xhigh", "max"),
}


def _openrouter_fallback_models(args: argparse.Namespace) -> tuple[str, ...]:
    configured = args.openrouter_fallback_model
    if configured is None:
        return OPENROUTER_FALLBACK_MODEL_IDS
    return tuple(configured)

def _build_adapter(args: argparse.Namespace):
    if args.provider == "openai":
        allowed_efforts = OPENAI_REASONING_EFFORTS[args.openai_model]
        if args.openai_reasoning_effort not in allowed_efforts:
            raise ValueError(
                f"{args.openai_model} supports reasoning efforts {allowed_efforts}, "
                f"not {args.openai_reasoning_effort!r}"
            )
        return OpenAIResponsesAdapter(
            model_id=args.openai_model,
            env_file=Path(args.env_file),
            api_key_env=args.openai_api_key_env,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.openai_reasoning_effort,
            timeout_seconds=args.timeout,
            runtime_retries=args.retries,
        )
    elif args.provider == "openrouter":
        return OpenRouterChatAdapter(
            model_id=args.openrouter_model,
            env_file=Path(args.env_file),
            api_key_env=args.openrouter_api_key_env,
            base_url=args.openrouter_base_url,
            http_referer=args.openrouter_http_referer,
            app_title=args.openrouter_app_title,
            fallback_model_ids=_openrouter_fallback_models(args),
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            timeout_seconds=args.timeout,
            runtime_retries=args.retries,
        )
    else:
        return OpenAICompatibleAdapter(
            base_url=args.base_url,
            model_id=args.model,
            tokenizer_id=args.tokenizer,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            timeout_seconds=args.timeout,
            runtime_retries=args.retries,
        )

def _add_generation_arguments(
    command: argparse.ArgumentParser,
    *,
    default_output_file: str,
    include_max_cases: bool,
) -> None:
    command.add_argument(
        "--cases-file",
        default="data/generation/qasper-v1/validation.cases.jsonl",
    )
    command.add_argument("--output-file", default=default_output_file)
    _add_provider_arguments(command)
    command.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    command.add_argument(
        "--model",
        default=LOCAL_MODEL_ID,
        help="Model ID exposed by the local OpenAI-compatible server.",
    )
    command.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Instruct-2507")
    command.add_argument("--max-context-tokens", type=int, default=32_768)
    command.add_argument("--max-output-tokens", type=int, default=1024)
    if include_max_cases:
        command.add_argument("--max-cases", type=int, default=25)
        command.add_argument(
            "--all-cases",
            action="store_const",
            const=None,
            dest="max_cases",
            help="Run every eligible case instead of the 25-case smoke default.",
        )
    command.add_argument("--temperature", type=float, default=0.0)
    command.add_argument("--timeout", type=float, default=300.0)
    command.add_argument("--retries", type=int, default=1)
    command.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Overwrite the prediction file instead of resuming it",
    )
    command.set_defaults(resume=True)

def _add_provider_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--provider",
        choices=("local", "openai", "openrouter"),
        default="local",
    )
    command.add_argument(
        "--openai-model",
        choices=OPENAI_MODEL_IDS,
        default=OPENAI_MODEL_ID,
        help="OpenAI model used when --provider openai is selected.",
    )
    command.add_argument(
        "--env-file",
        default=".env",
        help="Environment file containing provider API keys.",
    )
    command.add_argument(
        "--openai-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable that contains the OpenAI API key.",
    )
    command.add_argument(
        "--openai-reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        default="low",
    )
    command.add_argument(
        "--openrouter-model",
        default=OPENROUTER_MODEL_ID,
        help=(
            "OpenRouter model slug used when --provider openrouter is selected. "
            "Any model with structured-output support may be specified."
        ),
    )
    command.add_argument(
        "--openrouter-api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable that contains the OpenRouter API key.",
    )
    command.add_argument(
        "--openrouter-base-url",
        default="https://openrouter.ai/api/v1",
        help="OpenRouter-compatible API base URL.",
    )
    command.add_argument(
        "--openrouter-http-referer",
        help="Optional site URL sent in OpenRouter's HTTP-Referer attribution header.",
    )
    command.add_argument(
        "--openrouter-app-title",
        default="Project Local RAG",
        help="Optional app name sent in OpenRouter's X-OpenRouter-Title header.",
    )
    command.add_argument(
        "--openrouter-fallback-model",
        action="append",
        default=None,
        help=(
            "Fallback model slug, tried in order after the primary. Repeat to add "
            "models. Defaults to Qwen3.7 Plus, then DeepSeek V4 Flash."
        ),
    )
    command.add_argument(
        "--no-openrouter-fallbacks",
        action="store_const",
        const=[],
        dest="openrouter_fallback_model",
        help="Disable the default OpenRouter model fallback chain.",
    )
