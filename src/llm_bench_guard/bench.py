"""The measurement itself: latency percentiles and decode throughput over a fixed prompt set."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from .guards import GuardReport, check_contention, check_reasoning_mode, check_served_model
from .prompts import CONTROL, DEFAULT_PROMPTS, Prompt


@dataclass
class Endpoint:
    """An OpenAI-compatible endpoint.

    ``trust_env`` is off by default on purpose: an ambient HTTP proxy will happily intercept a
    request to a private address and answer it itself, which looks like an authentication error
    from the model server and costs an afternoon to find.
    """

    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 180.0
    trust_env: bool = False

    def client(self) -> httpx.Client:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return httpx.Client(
            base_url=self.base_url.rstrip("/"),
            headers=headers,
            timeout=self.timeout,
            trust_env=self.trust_env,
        )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def served_model_name(client: httpx.Client) -> str | None:
    """Ask the endpoint what it actually loaded, rather than trusting the configured alias."""
    try:
        response = client.get("/models")
        response.raise_for_status()
        data = response.json().get("data") or []
        return str(data[0]["id"]) if data else None
    except Exception:
        return None


def _one_call(client: httpx.Client, model: str, prompt: Prompt, extra: dict | None) -> dict:
    body: dict[str, Any] = {
        "model": model,
        "messages": prompt["messages"],
        "max_tokens": prompt.get("max_tokens", 256),
        "temperature": 0,
    }
    if extra:
        body.update(extra)
    started = time.perf_counter()
    response = client.post("/chat/completions", json=body)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.raise_for_status()
    payload = response.json()
    message = payload["choices"][0].get("message") or {}
    completion_tokens = int((payload.get("usage") or {}).get("completion_tokens") or 0)
    return {
        "latency_ms": elapsed_ms,
        "completion_tokens": completion_tokens,
        "tokens_per_s": (completion_tokens / elapsed_ms * 1000.0) if elapsed_ms else 0.0,
        "reasoning_chars": len(message.get("reasoning_content") or ""),
        "content": message.get("content") or "",
    }


def run_benchmark(
    endpoint: Endpoint,
    *,
    repeats: int = 5,
    prompts: list[Prompt] | None = None,
    threads: int | None = None,
    extra_body: dict | None = None,
    label: str | None = None,
) -> dict:
    """Measure the endpoint and return an artifact with a guard report attached.

    ``threads`` is not used by the measurement — it is recorded so that a later CPU comparison
    can refuse to compare runs made with different budgets.
    """
    prompts = prompts or DEFAULT_PROMPTS
    report = GuardReport()

    with endpoint.client() as client:
        served = served_model_name(client)
        finding = check_served_model(endpoint.model, served)
        if finding:
            report.findings.append(finding)

        # Warm-up. The first call after load pays for page-ins and is never representative.
        _one_call(client, endpoint.model, prompts[0], extra_body)

        control = _one_call(client, endpoint.model, CONTROL, extra_body)
        finding = check_reasoning_mode(control["completion_tokens"], control["reasoning_chars"])
        if finding:
            report.findings.append(finding)

        per_prompt: list[dict] = []
        all_latencies: list[float] = []
        all_throughputs: list[float] = []

        for prompt in prompts:
            latencies: list[float] = []
            throughputs: list[float] = []
            tokens: list[int] = []
            for _ in range(repeats):
                measured = _one_call(client, endpoint.model, prompt, extra_body)
                latencies.append(measured["latency_ms"])
                throughputs.append(measured["tokens_per_s"])
                tokens.append(measured["completion_tokens"])
            spread = (max(latencies) / min(latencies)) if min(latencies) else 0.0
            per_prompt.append(
                {
                    "name": prompt["name"],
                    "latency_ms_mean": round(statistics.fmean(latencies), 1),
                    "latency_ms_max": round(max(latencies), 1),
                    "latency_spread_max_min": round(spread, 2),
                    "tokens_per_s_mean": round(statistics.fmean(throughputs), 2),
                    "completion_tokens_mean": round(statistics.fmean(tokens), 1),
                }
            )
            all_latencies.extend(latencies)
            all_throughputs.extend(throughputs)

    finding = check_contention(per_prompt)
    if finding:
        report.findings.append(finding)

    return {
        "schema": "llm-bench-guard/1",
        "label": label,
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested_model": endpoint.model,
        "served_model": served,
        "repeats": repeats,
        "threads": threads,
        "reasoning_observed": control["reasoning_chars"] > 0,
        "control_completion_tokens": control["completion_tokens"],
        "overall": {
            "latency_ms_p50": round(_percentile(all_latencies, 0.50), 1),
            "latency_ms_p95": round(_percentile(all_latencies, 0.95), 1),
            "tokens_per_s_mean": round(statistics.fmean(all_throughputs), 2),
        },
        "per_prompt": per_prompt,
        "guards": report.as_dict(),
    }
