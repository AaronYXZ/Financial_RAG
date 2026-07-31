import json
from argparse import Namespace
from pathlib import Path

import pytest

import rag_eval.generation_cli as generation_cli
import rag_eval.end_to_end.cli as end_to_end_cli
import rag_eval.generation.cli as generation_commands
from rag_eval.generation_cli import build_parser


def test_generation_run_defaults_to_local_qwen_smoke_test():
    args = build_parser().parse_args(["run"])

    assert args.track == "oracle-evidence"
    assert args.provider == "local"
    assert args.model == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    assert args.max_context_tokens == 32_768
    assert args.max_output_tokens == 1024
    assert args.max_cases == 25
    assert args.retries == 1
    assert args.resume is True


def test_all_cases_overrides_smoke_limit_for_full_validation():
    run_args = build_parser().parse_args(["run", "--all-cases"])
    oracle_args = build_parser().parse_args(["generate-oracle", "--all-cases"])

    assert run_args.max_cases is None
    assert oracle_args.max_cases is None


def test_prepare_records_registered_source_checksum(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        generation_commands,
        "load_qasper_cases",
        lambda *args, **kwargs: [],
    )
    args = Namespace(
        split="validation",
        cache_dir=None,
        revision=generation_cli.QASPER_PARQUET_REVISION,
        limit_papers=None,
        output_dir=str(tmp_path),
    )

    assert generation_commands._prepare(args) == 0

    manifest = json.loads((tmp_path / "validation.manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["source_parquet_sha256"] == (
        "089781b91c337d348dd9e8b57cc8adc100ed2d9cab84a6127402bcccf1559222"
    )


def test_generation_commands_accept_openai_provider_configuration():
    args = build_parser().parse_args(
        [
            "generate-oracle",
            "--provider",
            "openai",
            "--openai-model",
            "gpt-5.6-sol",
            "--env-file",
            "secrets.env",
            "--openai-reasoning-effort",
            "low",
        ]
    )

    assert args.provider == "openai"
    assert args.openai_model == "gpt-5.6-sol"
    assert args.env_file == "secrets.env"
    assert args.openai_api_key_env == "OPENAI_API_KEY"
    assert args.openai_reasoning_effort == "low"


def test_generation_commands_accept_openrouter_model_selection():
    args = build_parser().parse_args(
        [
            "generate-retrieved",
            "--provider",
            "openrouter",
            "--openrouter-model",
            "google/gemini-2.5-pro",
            "--openrouter-http-referer",
            "https://example.com",
            "--openrouter-app-title",
            "RAG Benchmark",
            "--context-manifest",
            "frozen.json",
        ]
    )

    assert args.provider == "openrouter"
    assert args.openrouter_model == "google/gemini-2.5-pro"
    assert args.openrouter_api_key_env == "OPENROUTER_API_KEY"
    assert args.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert args.openrouter_http_referer == "https://example.com"
    assert args.openrouter_app_title == "RAG Benchmark"


def test_openrouter_model_defaults_to_luna_pro():
    args = build_parser().parse_args(
        [
            "generate-retrieved",
            "--provider",
            "openrouter",
            "--context-manifest",
            "frozen.json",
        ]
    )

    assert args.openrouter_model == "openai/gpt-5.6-luna-pro"
    assert args.openrouter_fallback_model is None
    assert generation_cli._openrouter_fallback_models(args) == (
        "qwen/qwen3.7-plus",
        "deepseek/deepseek-v4-flash",
    )


def test_openrouter_default_fallbacks_can_be_disabled():
    args = build_parser().parse_args(
        [
            "generate-retrieved",
            "--provider",
            "openrouter",
            "--no-openrouter-fallbacks",
            "--context-manifest",
            "frozen.json",
        ]
    )

    assert args.openrouter_fallback_model == []
    assert generation_cli._openrouter_fallback_models(args) == ()


def test_openai_model_defaults_and_choices_are_explicit():
    parser = build_parser()
    default_args = parser.parse_args(["generate-oracle", "--provider", "openai"])
    luna_args = parser.parse_args(
        [
            "generate-oracle",
            "--provider",
            "openai",
            "--openai-model",
            "gpt-5.6-luna",
        ]
    )

    assert default_args.openai_model == "gpt-5"
    assert default_args.openai_reasoning_effort == "low"
    assert luna_args.openai_model == "gpt-5.6-luna"


def test_openai_reasoning_effort_is_validated_for_selected_model(tmp_path: Path):
    args = build_parser().parse_args(
        [
            "generate-oracle",
            "--provider",
            "openai",
            "--openai-model",
            "gpt-5",
            "--openai-reasoning-effort",
            "none",
            "--cases-file",
            str(tmp_path / "cases.jsonl"),
        ]
    )
    (tmp_path / "cases.jsonl").write_text("")

    with pytest.raises(ValueError, match="gpt-5 supports reasoning efforts"):
        generation_commands._generate_oracle(args)


def test_generation_metrics_defaults_to_smoke_test_artifacts():
    args = build_parser().parse_args(["metrics"])

    assert args.track == "oracle-evidence"
    assert args.model == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    assert args.predictions_file.endswith("qwen3-4b-track-a-v2.jsonl")
    assert args.eligibility_file is None
    assert args.output_dir.endswith("metrics/qwen3-4b-track-a-v2")


def test_freeze_context_defaults_to_bm25_top_five():
    args = build_parser().parse_args(
        ["freeze-context", "--eligibility-file", "eligible.json"]
    )

    assert args.top_k == 5
    assert args.retriever == "bm25"
    assert args.retrieval_scope == "paper"
    assert args.dense_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert args.output_file.endswith("bm25-paper-top5.json")


def test_compare_metrics_requires_explicit_matched_run_inputs():
    args = build_parser().parse_args(
        [
            "compare-metrics",
            "--baseline-per-case-file",
            "baseline.jsonl",
            "--candidate-per-case-file",
            "candidate.jsonl",
            "--baseline-label",
            "baseline",
            "--candidate-label",
            "candidate",
            "--track",
            "oracle-evidence",
        ]
    )

    assert args.bootstrap_resamples == 10_000
    assert args.bootstrap_seed == 42
    assert args.track == "oracle-evidence"


def test_intersect_eligibility_accepts_multiple_source_manifests():
    args = build_parser().parse_args(
        [
            "intersect-eligibility",
            "--eligibility-file",
            "qwen.json",
            "--eligibility-file",
            "gpt.json",
            "--output-file",
            "common.json",
        ]
    )

    assert args.eligibility_file == ["qwen.json", "gpt.json"]
    assert args.output_file == "common.json"


def test_compare_retrieval_accepts_multiple_context_manifests():
    args = build_parser().parse_args(
        [
            "compare-retrieval",
            "--context-manifest",
            "bm25.json",
            "--context-manifest",
            "dense.json",
            "--output-file",
            "comparison.json",
        ]
    )

    assert args.context_manifest == ["bm25.json", "dense.json"]
    assert args.output_file == "comparison.json"


def test_estimate_cost_defaults_to_conservative_budget_basis():
    args = build_parser().parse_args(
        ["estimate-cost", "--predictions-file", "pilot.jsonl"]
    )

    assert args.model == "gpt-5"
    assert args.max_output_tokens == 1024
    assert args.retries == 1
    assert args.budget_basis == "ceiling_with_retries"


def test_retrieved_context_is_a_supported_track():
    args = build_parser().parse_args(
        ["run", "--track", "retrieved-context", "--context-manifest", "frozen.json"]
    )

    assert args.track == "retrieved-context"
    assert args.context_manifest == "frozen.json"


def test_generate_oracle_has_task_specific_defaults():
    args = build_parser().parse_args(["generate-oracle"])

    assert args.command == "generate-oracle"
    assert args.max_cases == 25
    assert args.output_file.endswith("qwen3-4b-oracle-v2.jsonl")
    assert not hasattr(args, "track")
    assert not hasattr(args, "context_manifest")


def test_generate_retrieved_requires_a_frozen_context_manifest():
    args = build_parser().parse_args(
        ["generate-retrieved", "--context-manifest", "frozen.json"]
    )

    assert args.command == "generate-retrieved"
    assert args.context_manifest == "frozen.json"
    assert args.output_file.endswith("qwen3-4b-retrieved-v3.jsonl")
    assert args.max_cases is None
    assert not hasattr(args, "retriever")
    assert not hasattr(args, "retrieval_scope")


def test_generate_retrieved_accepts_an_ordered_pilot_limit():
    args = build_parser().parse_args(
        [
            "generate-retrieved",
            "--context-manifest",
            "frozen.json",
            "--max-cases",
            "25",
        ]
    )

    assert args.max_cases == 25


def test_freeze_context_accepts_dense_model_and_corpus_scope():
    args = build_parser().parse_args(
        [
            "freeze-context",
            "--eligibility-file",
            "eligible.json",
            "--retriever",
            "dense",
            "--retrieval-scope",
            "corpus",
            "--dense-model",
            "sentence-transformers/all-mpnet-base-v2",
        ]
    )

    assert args.retriever == "dense"
    assert args.retrieval_scope == "corpus"
    assert args.dense_model == "sentence-transformers/all-mpnet-base-v2"


def test_generate_end_to_end_freezes_retrieval_before_generation():
    args = build_parser().parse_args(
        ["generate-end-to-end", "--eligibility-file", "eligible.json"]
    )

    assert args.command == "generate-end-to-end"
    assert args.eligibility_file == "eligible.json"
    assert args.top_k == 5
    assert args.context_manifest.endswith("end-to-end-retrieval-v2.json")
    assert args.output_file.endswith("qwen3-4b-end-to-end-v3.jsonl")


def test_end_to_end_handler_passes_new_manifest_to_generation(
    monkeypatch,
    tmp_path: Path,
):
    calls = []
    context_manifest = tmp_path / "retrieval.json"
    args = Namespace(
        cases_file=str(tmp_path / "cases.jsonl"),
        eligibility_file=str(tmp_path / "eligible.json"),
        context_manifest=str(context_manifest),
        top_k=7,
        retriever="hybrid",
        retrieval_scope="paper",
        dense_model="dense-model",
        dense_batch_size=16,
        hybrid_rrf_k=40,
        hybrid_candidate_k=20,
        output_file=str(tmp_path / "predictions.jsonl"),
        max_context_tokens=32_768,
        max_output_tokens=1024,
        max_cases=None,
        resume=True,
    )

    def fake_workflow(**kwargs):
        calls.append(kwargs)
        return (
            {
                "eligible_case_count": 3,
                "retriever": {"name": "hybrid", "parameters": {"top_k": 7}},
            },
            {
                "selected": 3,
                "completed": 3,
                "skipped": 0,
                "ineligible": 0,
                "errors": 0,
            },
        )

    adapter = object()
    monkeypatch.setattr(end_to_end_cli, "_build_adapter", lambda received: adapter)
    monkeypatch.setattr(
        end_to_end_cli,
        "run_retrieve_then_generate",
        fake_workflow,
    )

    assert end_to_end_cli._generate_end_to_end(args) == 0
    assert calls[0]["adapter"] is adapter
    assert calls[0]["context_manifest_file"] == context_manifest
    assert calls[0]["top_k"] == 7
    assert calls[0]["retriever"] == "hybrid"
    assert calls[0]["retrieval_scope"] == "paper"
    assert calls[0]["dense_model"] == "dense-model"
