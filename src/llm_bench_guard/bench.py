"""The measurement itself: latency percentiles and decode throughput over a fixed prompt set."""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
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


def _request_body(model: str, prompt: Prompt, extra: dict | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": prompt["messages"],
        "max_tokens": prompt.get("max_tokens", 256),
        "temperature": 0,
    }
    if extra:
        body.update(extra)
    return body


def _one_call(client: httpx.Client, model: str, prompt: Prompt, extra: dict | None) -> dict:
    """One non-streaming call. Total latency only — no time-to-first-token."""
    started = time.perf_counter()
    response = client.post("/chat/completions", json=_request_body(model, prompt, extra))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.raise_for_status()
    payload = response.json()
    message = payload["choices"][0].get("message") or {}
    completion_tokens = int((payload.get("usage") or {}).get("completion_tokens") or 0)
    return {
        "latency_ms": elapsed_ms,
        "ttft_ms": None,
        "completion_tokens": completion_tokens,
        "tokens_per_s": (completion_tokens / elapsed_ms * 1000.0) if elapsed_ms else 0.0,
        "reasoning_chars": len(message.get("reasoning_content") or ""),
        "content": message.get("content") or "",
    }


def _one_streaming_call(
    client: httpx.Client, model: str, prompt: Prompt, extra: dict | None
) -> dict:
    """One streaming call, so that time-to-first-token is measurable.

    TTFT and total latency answer different questions: TTFT is what the user feels before
    anything appears, total latency is bounded by ``max_tokens`` and therefore says as much
    about your output limit as about the model. Reporting only one of them hides that.
    """
    body = _request_body(model, prompt, extra)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}

    started = time.perf_counter()
    ttft_ms: float | None = None
    completion_tokens = 0
    chunks = 0
    reasoning_chars = 0
    content: list[str] = []

    with client.stream("POST", "/chat/completions", json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            payload_text = line[5:].strip()
            if not payload_text or payload_text == "[DONE]":
                continue
            payload = json.loads(payload_text)
            usage = payload.get("usage")
            if usage and usage.get("completion_tokens"):
                completion_tokens = int(usage["completion_tokens"])
            for choice in payload.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                reasoning_chars += len(delta.get("reasoning_content") or "")
                if piece or delta.get("reasoning_content"):
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000.0
                    chunks += 1
                content.append(piece)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not completion_tokens:
        # The server did not report usage for the stream; chunks are the honest fallback.
        completion_tokens = chunks
    return {
        "latency_ms": elapsed_ms,
        "ttft_ms": ttft_ms,
        "completion_tokens": completion_tokens,
        "tokens_per_s": (completion_tokens / elapsed_ms * 1000.0) if elapsed_ms else 0.0,
        "reasoning_chars": reasoning_chars,
        "content": "".join(content),
    }


def run_benchmark(
    endpoint: Endpoint,
    *,
    repeats: int = 5,
    prompts: list[Prompt] | None = None,
    threads: int | None = None,
    concurrency: int = 1,
    extra_body: dict | None = None,
    label: str | None = None,
) -> dict:
    """Measure the endpoint and return an artifact with a guard report attached.

    ``threads`` is not used by the measurement — it is recorded so that a later CPU comparison
    can refuse to compare runs made with different budgets.

    ``concurrency`` above 1 issues that many requests at once. Latency then includes queueing,
    which is the point: a number measured on a quiet endpoint does not tell you how the service
    behaves when several people use it. Total work is ``repeats * concurrency`` per prompt.
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
        all_ttft: list[float] = []
        total_completion_tokens = 0
        measured_started = time.perf_counter()

        for prompt in prompts:
            latencies: list[float] = []
            throughputs: list[float] = []
            tokens: list[int] = []
            ttfts: list[float] = []
            if concurrency == 1:
                results = [
                    _one_streaming_call(client, endpoint.model, prompt, extra_body)
                    for _ in range(repeats)
                ]
            else:
                # Requests are issued together on purpose: this measures the endpoint under
                # load, so queueing here is the signal, not contamination.
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [
                        pool.submit(
                            _one_streaming_call,
                            client,
                            endpoint.model,
                            prompt,
                            extra_body,
                        )
                        for _ in range(repeats * concurrency)
                    ]
                    results = [f.result() for f in futures]
            for measured in results:
                latencies.append(measured["latency_ms"])
                throughputs.append(measured["tokens_per_s"])
                tokens.append(measured["completion_tokens"])
                if measured["ttft_ms"] is not None:
                    ttfts.append(measured["ttft_ms"])
                    all_ttft.append(measured["ttft_ms"])
            spread = (max(latencies) / min(latencies)) if min(latencies) else 0.0
            per_prompt.append(
                {
                    "name": prompt["name"],
                    "latency_ms_mean": round(statistics.fmean(latencies), 1),
                    "latency_ms_max": round(max(latencies), 1),
                    "latency_spread_max_min": round(spread, 2),
                    "tokens_per_s_mean": round(statistics.fmean(throughputs), 2),
                    "completion_tokens_mean": round(statistics.fmean(tokens), 1),
                    "ttft_ms_mean": round(statistics.fmean(ttfts), 1) if ttfts else None,
                }
            )
            all_latencies.extend(latencies)
            all_throughputs.extend(throughputs)
            total_completion_tokens += sum(tokens)

        measured_wall_ms = (time.perf_counter() - measured_started) * 1000.0

    if concurrency == 1:
        # Under deliberate load the spread IS the measurement, so the guard would fire on
        # exactly the thing we asked for. It only guards the quiet case.
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
        "concurrency": concurrency,
        "reasoning_observed": control["reasoning_chars"] > 0,
        "control_completion_tokens": control["completion_tokens"],
        "overall": {
            "latency_ms_p50": round(_percentile(all_latencies, 0.50), 1),
            "latency_ms_p95": round(_percentile(all_latencies, 0.95), 1),
            "tokens_per_s_mean": round(statistics.fmean(all_throughputs), 2),
            "ttft_ms_p50": round(_percentile(all_ttft, 0.50), 1) if all_ttft else None,
            "ttft_ms_p95": round(_percentile(all_ttft, 0.95), 1) if all_ttft else None,
            "system_tokens_per_s": round(total_completion_tokens / measured_wall_ms * 1000.0, 2)
            if measured_wall_ms
            else 0.0,
            "wall_ms": round(measured_wall_ms, 1),
        },
        "per_prompt": per_prompt,
        "guards": report.as_dict(),
    }
