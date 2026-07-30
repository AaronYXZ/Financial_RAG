import json
from argparse import Namespace
from pathlib import Path

import pytest

import rag_eval.generation_cli as generation_cli
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
    monkeypatch.setattr(generation_cli, "load_qasper_cases", lambda *args, **kwargs: [])
    args = Namespace(
        split="validation",
        cache_dir=None,
        revision=generation_cli.QASPER_PARQUET_REVISION,
        limit_papers=None,
        output_dir=str(tmp_path),
    )

    assert generation_cli._prepare(args) == 0

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
        generation_cli._generate_oracle(args)


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
    assert not hasattr(args, "max_cases")
    assert not hasattr(args, "retriever")
    assert not hasattr(args, "retrieval_scope")


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
    )

    def fake_freeze(**kwargs):
        calls.append(("freeze", kwargs))
        return {
            "eligible_case_count": 3,
            "retriever": {"name": "bm25", "parameters": {"top_k": 7}},
        }

    def fake_execute(received_args, *, track, context_manifest):
        calls.append(
            (
                "generate",
                {
                    "args": received_args,
                    "track": track,
                    "context_manifest": context_manifest,
                },
            )
        )
        return 0

    monkeypatch.setattr(generation_cli, "_freeze_context_payload", fake_freeze)
    monkeypatch.setattr(generation_cli, "_execute_generation", fake_execute)

    assert generation_cli._generate_end_to_end(args) == 0
    assert calls[0][0] == "freeze"
    assert calls[0][1]["top_k"] == 7
    assert calls[0][1]["retriever"] == "hybrid"
    assert calls[0][1]["retrieval_scope"] == "paper"
    assert calls[0][1]["dense_model"] == "dense-model"
    assert calls[1] == (
        "generate",
        {
            "args": args,
            "track": "retrieved-context",
            "context_manifest": str(context_manifest),
        },
    )
