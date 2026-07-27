from rag_eval.generation_cli import build_parser


def test_generation_run_defaults_to_local_qwen_smoke_test():
    args = build_parser().parse_args(["run"])

    assert args.track == "oracle-evidence"
    assert args.model == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    assert args.max_context_tokens == 32_768
    assert args.max_output_tokens == 512
    assert args.max_cases == 25
    assert args.retries == 1
    assert args.resume is True


def test_generation_metrics_defaults_to_smoke_test_artifacts():
    args = build_parser().parse_args(["metrics"])

    assert args.track == "oracle-evidence"
    assert args.model == "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    assert args.predictions_file.endswith("qwen3-4b-smoke.jsonl")
    assert args.eligibility_file is None
    assert args.output_dir.endswith("metrics/qwen3-4b-smoke")
