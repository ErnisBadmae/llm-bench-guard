# llm-bench-guard

A latency/throughput benchmark for local OpenAI-compatible LLM endpoints that **refuses to
report a number it believes is misleading**.

Measuring a local model is easy. Measuring it in a way that survives someone asking "are you
sure?" is not. Every guard in this package exists because the corresponding mistake was made on
a running system, produced a plausible number, and was wrong.

## The problem, in one run

Here is the same model, on the same host, measured twice. The only difference is one request
parameter:

```console
$ llm-bench-guard compare --baseline artifacts/thinking_on.json \
                          --candidate artifacts/thinking_off.json
{
  "latency_p50_delta_pct": -85.5,
  "latency_p95_delta_pct": -0.3,
  "throughput_delta_pct": -1.3
}
  [BLOCKING] reasoning_mode: one run was measured with reasoning enabled and the other
  without. The difference between them is the mode, not the model.

These numbers are not reportable. Fix the blocking conditions and measure again.
```

An 85% latency drop is the kind of result that ends up in a slide. But p95 did not move and
throughput went slightly *down* — nothing got faster. One run let the model think before
answering and the other did not, and no HTTP status code anywhere tells you that happened.

The tool exits non-zero, so a comparison like this fails a CI step instead of becoming a claim.

## What it guards against

| Guard | What it catches | Severity |
|---|---|---|
| `contention` | The endpoint was serving something else. One prompt's slowest repeat is several times its fastest — models do not vary like that, queues do. | blocking |
| `reasoning_mode` | The model was thinking when you believed it was not, or the two runs you are comparing were in different modes. Detected with a control question whose correct answer is a few tokens. | warn / blocking on compare |
| `thread_parity` | A CPU comparison where each side got a different thread budget. Runtimes do not share a thread setting, so this measures your configuration and calls it the backend. | blocking |
| `served_model` | The endpoint ignored the model name you sent and served whatever it had loaded, so your config alias describes a different build than the one measured. | warn |
| `load_parity` | A quiet run compared against a loaded one. Latency under concurrency includes queueing, so the difference is the setup, not the model. | blocking |
| `quality_pairing` | Emitted on every comparison: speed alone is not a decision. A candidate that is faster and worse is not a win. | warn |

`contention` and `thread_parity` are blocking because they make the number wrong.
`served_model` is a warning because the run is still valid — you just have to record what was
actually measured.

## Install

```bash
pip install git+https://github.com/ErnisBadmae/llm-bench-guard
```

Not on PyPI yet.

## Use

```bash
# measure
llm-bench-guard run \
  --base-url http://127.0.0.1:8000/v1 \
  --model my-model \
  --repeats 5 --threads 8 \
  --out artifacts/baseline.json

# measure a candidate, then compare
llm-bench-guard compare --baseline artifacts/baseline.json \
                        --candidate artifacts/candidate.json
```

Anything the server accepts can be passed through, which is how you control reasoning mode:

```bash
llm-bench-guard run ... --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
```

As a library:

```python
from llm_bench_guard import Endpoint, run_benchmark

artifact = run_benchmark(Endpoint(base_url="http://127.0.0.1:8000/v1", model="my-model"))
assert artifact["guards"]["reportable"]
```

## What it measures

Three prompt shapes, because one average hides the thing you care about:

- **short_qa** — a few tokens out, dominated by per-request overhead. This is where a faster
  runtime actually shows up.
- **json_extraction** — structured output of moderate length; the shape most services run.
- **long_explanation** — bounded by `max_tokens`. This one measures your output limit as much
  as the model, which is why p95 often barely moves between runtimes while p50 halves.

Reported: **time to first token** (p50/p95), latency p50/p95 across all calls, mean decode
throughput, and per-prompt means with the spread used by the contention guard. A warm-up call
is discarded — the first call after load pays for page-ins and is never representative.

TTFT and total latency answer different questions. TTFT is what a user waits before anything
appears; total latency is bounded by `max_tokens` and so describes your output limit as much as
the model. Reporting only one of them hides that — which is why p95 often barely moves between
runtimes while p50 halves.

### Under load

```bash
llm-bench-guard run ... --concurrency 4
```

Requests are issued together, so latency includes queueing — that is the measurement, not
contamination. The contention guard steps aside above concurrency 1, because otherwise it would
fire on exactly what you asked for. Comparisons across different concurrency levels are refused
outright.

Under load, two throughput numbers say different things and only one of them answers the
capacity question:

- `tokens_per_s_mean` is **per request**, and it includes the time that request spent waiting.
  It falls as concurrency rises even on a server that is doing exactly as much work as before.
- `system_tokens_per_s` is **aggregate**: all completion tokens divided by the wall clock of the
  measured phase. This is the one that tells you whether extra concurrency buys anything.

The difference is not academic. On a llama.cpp server started with a single slot:

| concurrency | per-request t/s | system t/s | wall | TTFT p50 |
|---|---|---|---|---|
| 1 | 44.9 | 48.5 | 18.4 s | 111 ms |
| 2 | 28.0 | 49.5 | 36.0 s | 590 ms |
| 4 | 17.5 | 49.5 | 72.1 s | 1601 ms |
| 8 | 10.7 | 49.9 | 143.1 s | 3599 ms |

Per-request throughput collapses by 4x while the aggregate does not move at all. The obvious
way to get a system number from per-request ones — multiply the mean by the concurrency —
gives 44.9, 56.0, 69.9, 85.5 t/s here: a clean rising curve, and wrong. Each per-request
figure already has that request's queue wait in its denominator, so scaling it by the number
of waiters counts the waiting as work. Total tokens over elapsed wall time cannot be inflated
that way, which is why `system_tokens_per_s` is measured and not derived.

A flat aggregate does not by itself prove serialisation — a saturated resource under genuine
parallelism looks the same. It is a hypothesis the numbers make cheap to test, and here
`/props` reporting a single slot confirmed it.

A number from a quiet endpoint does not tell you how the service behaves when several people
use it, and that is usually the number someone is about to put in a slide.

### What the aggregate number is for

The same server, restarted with four slots and continuous batching, with nothing else changed.
The context window was split four ways to pay for it — that trade was safe here only because
the window was measured to be unused, not because splitting is free:

| concurrency | system t/s, 1 slot | system t/s, 4 slots | TTFT p95, 1 slot | TTFT p95, 4 slots |
|---|---|---|---|---|
| 1 | 48.5 | 50.5 | 391 ms | 90 ms |
| 2 | 49.5 | 51.2 | 8.1 s | 0.86 s |
| 4 | 49.5 | **78.8** | 23.9 s | **1.12 s** |
| 8 | 49.9 | 79.4 | 55.6 s | 20.5 s |

Three things this table says that a single average would hide. Throughput only moves at four
concurrent requests — at two the gain is 3% and the entire benefit is latency. The ceiling is
the same at four and eight, because there are four slots and the fifth request queues again.
And per-request throughput *fell* while the server did more total work, which is the cost
batching charges an individual request, not a regression.

Acceptance was re-measured on the promoted server rather than carried over from the candidate:
+55.7% aggregate and −94.5% TTFT p95, within noise of the numbers above.

## What it does not do

- **No quality measurement.** This is the performance half. Pair it with your own gate; the
  comparison output says so every time.
- **No host-side metrics.** VRAM and RAM live on the inference host, not in an HTTP response.
  Record them next to the artifact by hand.
- **No capacity planning.** Concurrency levels are measured, but the tool does not search for
  the knee of the curve or model arrival rates — it reports what the levels you asked for did.
- **It cannot turn reasoning off for you.** How to disable thinking is server- and
  template-specific. The guard tells you which mode you measured; you pass the right parameter.

## Notes from the field

**Proxies answer for private addresses.** `trust_env` is off by default. An ambient
`HTTPS_PROXY` will intercept a request to `192.168.x.x`, answer it itself, and the failure
arrives as an authentication error that looks like it came from the model server. Pass
`--trust-env` if you actually need the proxy.

**Aliases drift.** Some servers load one model at startup and ignore the `model` field
entirely. The artifact records the served name separately from the requested one, because a
month later that difference is the only thing standing between you and a wrong conclusion.

**The first call is a lie.** Discarded. So is any run where the endpoint was busy.

## License

MIT
