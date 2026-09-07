"""The fixed prompt set.

Three shapes, because they stress different things and a single average hides that:

* ``short_qa``      — a few tokens out. Dominated by per-request overhead, so this is where a
                      faster runtime shows up.
* ``json_extraction`` — structured output of moderate length, the shape most services run.
* ``long_explanation`` — bounded by ``max_tokens``. This one measures the output limit as much
                      as the model, which is why p95 barely moves between runtimes.

``CONTROL`` is not part of the measurement. It is a question whose correct answer is a handful
of tokens, used by the reasoning-mode guard: if the model spends a hundred tokens on it, it is
thinking, and the run is not comparable with a run where it is not.
"""

from __future__ import annotations

from typing import Any

Prompt = dict[str, Any]

DEFAULT_PROMPTS: list[Prompt] = [
    {
        "name": "short_qa",
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": "Answer in one short sentence."},
            {
                "role": "user",
                "content": "What is a write-ahead log used for in a database?",
            },
        ],
    },
    {
        "name": "json_extraction",
        "max_tokens": 200,
        "messages": [
            {
                "role": "system",
                "content": (
                    'Return strictly JSON {"item": str, "quantity": number, "unit": str} '
                    "for the fragment below."
                ),
            },
            {
                "role": "user",
                "content": "Fragment: 'Delivered 15 steel beams, length 6 metres, to bay 3.'",
            },
        ],
    },
    {
        "name": "long_explanation",
        "max_tokens": 400,
        "messages": [
            {"role": "system", "content": "You are a precise technical writer."},
            {
                "role": "user",
                "content": (
                    "Explain the difference between latency and throughput for an inference "
                    "server, and how batching changes each. Give a detailed answer."
                ),
            },
        ],
    },
]

CONTROL: Prompt = {
    "name": "control_reasoning_probe",
    "max_tokens": 128,
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France? Answer in one word.",
        }
    ],
}
