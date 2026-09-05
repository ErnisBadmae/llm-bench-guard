from llm_bench_guard.compare import compare
from llm_bench_guard.guards import (
    check_contention,
    check_reasoning_mode,
    check_served_model,
    check_thread_parity,
)


def test_contention_blocks_when_a_prompt_varies_wildly():
    finding = check_contention([{"name": "short_qa", "latency_spread_max_min": 7.4}])
    assert finding is not None and finding.blocking


def test_contention_silent_on_a_quiet_endpoint():
    assert check_contention([{"name": "short_qa", "latency_spread_max_min": 1.2}]) is None


def test_served_model_warns_on_alias_drift():
    finding = check_served_model("qwen-q5", "qwen-q4")
    assert finding is not None and finding.severity == "warn"
    assert "qwen-q4" in finding.message


def test_served_model_silent_when_names_match():
    assert check_served_model("qwen-q5", "qwen-q5") is None


def test_reasoning_detected_from_reasoning_field():
    finding = check_reasoning_mode(completion_tokens=4, reasoning_chars=514)
    assert finding is not None and "thinking enabled" in finding.message


def test_reasoning_inferred_from_token_count_alone():
    """A server that hides the reasoning field still spends the tokens."""
    finding = check_reasoning_mode(completion_tokens=131, reasoning_chars=0)
    assert finding is not None and finding.guard == "reasoning_mode"


def test_reasoning_silent_on_a_short_answer():
    assert check_reasoning_mode(completion_tokens=4, reasoning_chars=0) is None


def test_thread_parity_blocks_on_different_budgets():
    finding = check_thread_parity({"threads": 8}, {"threads": 12})
    assert finding is not None and finding.blocking


def test_thread_parity_warns_when_unrecorded():
    finding = check_thread_parity({"threads": None}, {"threads": 8})
    assert finding is not None and finding.severity == "warn"


def _artifact(**over):
    base = {
        "requested_model": "m",
        "served_model": "m",
        "threads": 8,
        "reasoning_observed": False,
        "guards": {"reportable": True, "findings": []},
        "overall": {"latency_ms_p50": 800.0, "latency_ms_p95": 8000.0, "tokens_per_s_mean": 45.0},
    }
    base.update(over)
    return base


def test_compare_reports_deltas_and_always_pairs_with_quality():
    result = compare(
        _artifact(),
        _artifact(overall={"latency_ms_p50": 600.0, "latency_ms_p95": 7600.0,
                           "tokens_per_s_mean": 49.5}),
    )
    assert result["latency_p50_delta_pct"] == -25.0
    assert result["throughput_delta_pct"] == 10.0
    guards = [f["guard"] for f in result["guards"]["findings"]]
    assert "quality_pairing" in guards


def test_compare_refuses_across_reasoning_modes():
    result = compare(_artifact(), _artifact(reasoning_observed=True))
    assert result["guards"]["reportable"] is False


def test_compare_refuses_an_artifact_that_failed_its_own_guards():
    bad = _artifact(guards={"reportable": False, "findings": []})
    assert compare(_artifact(), bad)["guards"]["reportable"] is False
