"""Preflight generation cost estimates from pilot prediction artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


OPENAI_TEXT_PRICES_PER_MILLION = {
    "gpt-5": {
        "input": 1.25,
        "cached_input": 0.125,
        "output": 10.0,
    },
    "gpt-5.6-sol": {
        "input": 5.0,
        "cached_input": 0.5,
        "output": 30.0,
    },
    "gpt-5.6-luna": {
        "input": 1.0,
        "cached_input": 0.1,
        "output": 6.0,
    },
}
PRICING_SOURCES = {
    "gpt-5": "https://developers.openai.com/api/docs/models/gpt-5",
    "gpt-5.6-sol": "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
    "gpt-5.6-luna": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
}
PRICING_VERIFIED_ON = "2026-07-29"


def _percentile(values: Sequence[int], quantile: float) -> int:
    if not values:
        raise ValueError("Cannot calculate output-token percentile without observations")
    ordered = sorted(values)
    index = int((len(ordered) - 1) * quantile)
    return ordered[index]


def _cost(
    *,
    input_tokens: float,
    output_tokens: float,
    input_rate: float,
    output_rate: float,
) -> float:
    return (
        input_tokens * input_rate / 1_000_000
        + output_tokens * output_rate / 1_000_000
    )


def estimate_openai_cost(
    *,
    predictions_file: Path,
    output_usage_file: Path | None = None,
    model: str,
    target_case_count: int | None,
    max_output_tokens: int,
    retries: int,
    budget_usd: float | None = None,
    budget_basis: str = "ceiling_with_retries",
) -> dict[str, Any]:
    """Project a full run from observed pilot usage without making API calls."""

    if model not in OPENAI_TEXT_PRICES_PER_MILLION:
        raise ValueError(f"No frozen price table for model {model!r}")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    if retries < 0:
        raise ValueError("retries cannot be negative")
    if budget_usd is not None and budget_usd < 0:
        raise ValueError("budget_usd cannot be negative")

    rows = [
        json.loads(line)
        for line in predictions_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("Predictions file must contain at least one pilot row")
    case_ids = [row.get("case_id") for row in rows]
    if not all(isinstance(case_id, str) and case_id for case_id in case_ids):
        raise ValueError("Every pilot row must contain a case_id")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Pilot predictions contain duplicate case IDs")

    observed_input = [
        int(row["server_input_tokens"])
        for row in rows
        if isinstance(row.get("server_input_tokens"), int)
    ]
    counted_input = [
        int(row["counted_input_tokens"])
        for row in rows
        if isinstance(row.get("counted_input_tokens"), int)
    ]
    input_values = observed_input or counted_input
    input_basis = (
        "observed_server_input_tokens"
        if observed_input
        else "counted_input_tokens"
    )
    if not input_values:
        raise ValueError("Pilot predictions contain no input-token measurements")
    output_source = output_usage_file or predictions_file
    output_rows = (
        rows
        if output_source == predictions_file
        else [
            json.loads(line)
            for line in output_source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )
    output_values = [
        int(row["server_output_tokens"])
        for row in output_rows
        if isinstance(row.get("server_output_tokens"), int)
    ]
    if not output_values:
        raise ValueError("Pilot predictions contain no observed output-token usage")

    target = target_case_count or len(rows)
    if target <= 0:
        raise ValueError("target_case_count must be positive")
    input_mean = sum(input_values) / len(input_values)
    output_mean = sum(output_values) / len(output_values)
    output_p95 = _percentile(output_values, 0.95)
    projected_input = input_mean * target
    rates = OPENAI_TEXT_PRICES_PER_MILLION[model]
    scenarios = {
        "expected": _cost(
            input_tokens=projected_input,
            output_tokens=output_mean * target,
            input_rate=rates["input"],
            output_rate=rates["output"],
        ),
        "observed_p95_output": _cost(
            input_tokens=projected_input,
            output_tokens=output_p95 * target,
            input_rate=rates["input"],
            output_rate=rates["output"],
        ),
        "ceiling": _cost(
            input_tokens=projected_input,
            output_tokens=max_output_tokens * target,
            input_rate=rates["input"],
            output_rate=rates["output"],
        ),
    }
    scenarios["ceiling_with_retries"] = scenarios["ceiling"] * (retries + 1)
    if budget_basis not in scenarios:
        raise ValueError(f"Unsupported budget basis {budget_basis!r}")
    approved = (
        None if budget_usd is None else scenarios[budget_basis] <= budget_usd
    )

    return {
        "schema_version": 1,
        "estimate": "openai_generation_cost_preflight",
        "model": model,
        "pricing": {
            "currency": "USD",
            "per_million_tokens": rates,
            "source": PRICING_SOURCES[model],
            "verified_on": PRICING_VERIFIED_ON,
        },
        "pilot": {
            "predictions_file": str(predictions_file),
            "output_usage_file": str(output_source),
            "row_count": len(rows),
            "input_measurement_count": len(input_values),
            "output_measurement_count": len(output_values),
            "input_basis": input_basis,
            "mean_input_tokens": input_mean,
            "mean_output_tokens": output_mean,
            "p95_output_tokens": output_p95,
        },
        "projection": {
            "target_case_count": target,
            "max_output_tokens": max_output_tokens,
            "retries": retries,
            "projected_input_tokens": projected_input,
            "projected_mean_output_tokens": output_mean * target,
            "projected_p95_output_tokens": output_p95 * target,
            "projected_ceiling_output_tokens": max_output_tokens * target,
            "cost_usd": scenarios,
        },
        "budget_gate": {
            "budget_usd": budget_usd,
            "basis": budget_basis,
            "estimated_cost_usd": scenarios[budget_basis],
            "approved": approved,
        },
    }
