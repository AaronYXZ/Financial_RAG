from rag_eval.semantic_cli import DEFAULT_OPENROUTER_JUDGE, build_parser


def test_semantic_run_defaults_to_openrouter_independent_judge():
    args = build_parser().parse_args(
        [
            "run",
            "--inputs-file",
            "inputs.jsonl",
            "--output-file",
            "judgments.jsonl",
        ]
    )

    assert args.provider == "openrouter"
    assert args.judge_model == DEFAULT_OPENROUTER_JUDGE
    assert args.fallback_model == []
    assert args.resume is True


def test_semantic_prepare_requires_generator_identity_for_independence_check():
    args = build_parser().parse_args(
        [
            "prepare",
            "--cases-file",
            "cases.jsonl",
            "--predictions-file",
            "predictions.jsonl",
            "--output-file",
            "inputs.jsonl",
            "--track",
            "retrieved-context",
            "--generator-model",
            "generator-model",
        ]
    )

    assert args.generator_model == "generator-model"
