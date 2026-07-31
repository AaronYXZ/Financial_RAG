import json
from pathlib import Path

from rag_eval.generation.cost import estimate_openai_cost


def _write_rows(path: Path):
    rows = [
        {
            "case_id": "q1",
            "counted_input_tokens": 90,
            "server_input_tokens": 100,
            "server_output_tokens": 20,
        },
        {
            "case_id": "q2",
            "counted_input_tokens": 100,
            "server_input_tokens": 120,
            "server_output_tokens": 40,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_cost_estimate_projects_observed_usage_and_retry_ceiling(tmp_path: Path):
    predictions = tmp_path / "pilot.jsonl"
    _write_rows(predictions)

    result = estimate_openai_cost(
        predictions_file=predictions,
        model="gpt-5",
        target_case_count=10,
        max_output_tokens=100,
        retries=1,
    )

    assert result["pilot"]["input_basis"] == "observed_server_input_tokens"
    assert result["projection"]["projected_input_tokens"] == 1100
    assert result["projection"]["projected_mean_output_tokens"] == 300
    assert result["projection"]["cost_usd"]["ceiling_with_retries"] == (
        result["projection"]["cost_usd"]["ceiling"] * 2
    )


def test_cost_budget_gate_can_block_full_run(tmp_path: Path):
    predictions = tmp_path / "pilot.jsonl"
    _write_rows(predictions)

    result = estimate_openai_cost(
        predictions_file=predictions,
        model="gpt-5",
        target_case_count=1000,
        max_output_tokens=1024,
        retries=1,
        budget_usd=0.01,
        budget_basis="expected",
    )

    assert result["budget_gate"]["approved"] is False


def test_cost_budget_gate_requires_explicit_budget_for_approval(tmp_path: Path):
    predictions = tmp_path / "pilot.jsonl"
    _write_rows(predictions)

    result = estimate_openai_cost(
        predictions_file=predictions,
        model="gpt-5",
        target_case_count=10,
        max_output_tokens=100,
        retries=1,
    )

    assert result["budget_gate"]["approved"] is None


def test_cost_estimate_can_use_separate_output_usage_file(tmp_path: Path):
    predictions = tmp_path / "input-pilot.jsonl"
    predictions.write_text(
        json.dumps({"case_id": "q1", "counted_input_tokens": 200}) + "\n"
    )
    output_usage = tmp_path / "output-pilot.jsonl"
    _write_rows(output_usage)

    result = estimate_openai_cost(
        predictions_file=predictions,
        output_usage_file=output_usage,
        model="gpt-5",
        target_case_count=10,
        max_output_tokens=100,
        retries=1,
    )

    assert result["pilot"]["input_basis"] == "counted_input_tokens"
    assert result["pilot"]["mean_input_tokens"] == 200
    assert result["pilot"]["mean_output_tokens"] == 30
    assert result["pilot"]["output_usage_file"] == str(output_usage)
