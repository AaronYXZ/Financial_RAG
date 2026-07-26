from rag_eval.benchmark_cli import build_parser


def test_benchmark_cli_defaults_to_scidocs_matrix():
    args = build_parser().parse_args([])

    assert args.dataset == "scidocs"
    assert args.chunker == "both"
    assert args.chunk_size == 256
    assert args.chunk_overlap == 32
    assert args.repetitions == 3
