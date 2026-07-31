"""Compatibility dispatch for commands that span benchmark domains."""

from __future__ import annotations

import argparse

from ..end_to_end.cli import _generate_retrieved
from ..generation.cli import _run as _run_component_generation


def _run(args: argparse.Namespace) -> int:
    """Dispatch the legacy run command without reversing domain dependencies."""

    if args.track == "retrieved-context":
        if not args.context_manifest:
            raise ValueError("--context-manifest is required for retrieved-context")
        return _generate_retrieved(args)
    return _run_component_generation(args)
