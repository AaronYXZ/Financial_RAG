"""Compatibility façade for the rag-generation command."""

from __future__ import annotations

from typing import Sequence

from .cli.common import (
    OPENROUTER_FALLBACK_MODEL_IDS,
    _openrouter_fallback_models,
)
from .cli.parser import build_parser
from .generation.data import QASPER_PARQUET_REVISION


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
