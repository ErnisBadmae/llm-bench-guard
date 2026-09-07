"""Compare two artifacts and say plainly whether the comparison is allowed at all."""

from __future__ import annotations

from .guards import GuardReport, check_load_parity, check_thread_parity, quality_pairing_note


def _pct_delta(baseline: float | None, candidate: float | None) -> float | None:
    if not baseline or candidate is None:
        return None
    return round((candidate - baseline) / baseline * 100.0, 1)


def _was_blocked(artifact: dict) -> bool:
    return not (artifact.get("guards") or {}).get("reportable", True)


def compare(baseline: dict, candidate: dict) -> dict:
    """Deltas plus the reasons this comparison may be meaningless.

    Sign convention is reported as measured: a negative latency delta means the candidate is
    faster, a positive throughput delta means the candidate is faster.
    """
    report = GuardReport()

    for name, artifact in (("baseline", baseline), ("candidate", candidate)):
        if _was_blocked(artifact):
            report.add(
                "source_artifact",
                "invalid",
                f"the {name} artifact failed its own guards and must not be compared.",
            )

    finding = check_thread_parity(baseline, candidate)
    if finding:
        report.findings.append(finding)

    finding = check_load_parity(baseline, candidate)
    if finding:
        report.findings.append(finding)

    if bool(baseline.get("reasoning_observed")) != bool(candidate.get("reasoning_observed")):
        report.add(
            "reasoning_mode",
            "invalid",
            "one run was measured with reasoning enabled and the other without. "
            "The difference between them is the mode, not the model.",
        )

    report.findings.append(quality_pairing_note())

    b = baseline.get("overall") or {}
    c = candidate.get("overall") or {}
    return {
        "baseline_model": baseline.get("served_model") or baseline.get("requested_model"),
        "candidate_model": candidate.get("served_model") or candidate.get("requested_model"),
        "latency_p50_delta_pct": _pct_delta(b.get("latency_ms_p50"), c.get("latency_ms_p50")),
        "latency_p95_delta_pct": _pct_delta(b.get("latency_ms_p95"), c.get("latency_ms_p95")),
        "throughput_delta_pct": _pct_delta(b.get("tokens_per_s_mean"), c.get("tokens_per_s_mean")),
        "system_throughput_delta_pct": _pct_delta(
            b.get("system_tokens_per_s"), c.get("system_tokens_per_s")
        ),
        "ttft_p50_delta_pct": _pct_delta(b.get("ttft_ms_p50"), c.get("ttft_ms_p50")),
        "ttft_p95_delta_pct": _pct_delta(b.get("ttft_ms_p95"), c.get("ttft_ms_p95")),
        "guards": report.as_dict(),
    }
