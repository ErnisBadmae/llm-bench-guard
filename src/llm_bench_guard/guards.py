"""Guards: the checks that decide whether a benchmark number may be reported at all.

Every guard here exists because the corresponding mistake was actually made on a running
system, and each one produced a number that looked plausible and was wrong. The point of
the package is not the timing loop — that is twenty lines — but these checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

Severity = Literal["ok", "warn", "invalid"]

#: A per-prompt spread (slowest run / fastest run) above this means the endpoint was busy.
#: Local servers expose a small number of slots; a concurrent job makes requests queue and
#: inflates latency by a factor that has nothing to do with the model.
CONTENTION_SPREAD = 3.0

#: A control question whose correct answer is a handful of tokens. If the model spends far
#: more than this, it is reasoning — see :func:`check_reasoning_mode`.
REASONING_TOKEN_BUDGET = 24


@dataclass(frozen=True)
class Finding:
    guard: str
    severity: Severity
    message: str

    @property
    def blocking(self) -> bool:
        return self.severity == "invalid"


@dataclass
class GuardReport:
    findings: list[Finding] = field(default_factory=list)

    def add(self, guard: str, severity: Severity, message: str) -> None:
        self.findings.append(Finding(guard, severity, message))

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.blocking]

    @property
    def reportable(self) -> bool:
        """False when at least one guard says the numbers must not be published."""
        return not self.blocking

    def as_dict(self) -> dict:
        return {
            "reportable": self.reportable,
            "findings": [
                {"guard": f.guard, "severity": f.severity, "message": f.message}
                for f in self.findings
            ],
        }


def check_contention(per_prompt: Iterable[dict]) -> Finding | None:
    """Detect that the endpoint was not idle.

    Symptom: one prompt's slowest repeat is several times its fastest. Model speed does not
    vary like that; queueing does.
    """
    worst_name, worst_spread = None, 0.0
    for row in per_prompt:
        spread = float(row.get("latency_spread_max_min") or 0.0)
        if spread > worst_spread:
            worst_name, worst_spread = row.get("name"), spread
    if worst_spread > CONTENTION_SPREAD:
        return Finding(
            "contention",
            "invalid",
            f"prompt {worst_name!r} varied {worst_spread:.1f}x between repeats "
            f"(limit {CONTENTION_SPREAD}x). The endpoint was serving something else; "
            "these numbers measure the queue, not the model.",
        )
    return None


def check_served_model(requested: str, served: str | None) -> Finding | None:
    """Detect that the endpoint served a different model than the one you asked for.

    Some servers load one model at startup and ignore the model field of the request, so the
    alias in your config can quietly describe a different quantisation than the one measured.
    """
    if not served:
        return Finding(
            "served_model",
            "warn",
            "the endpoint did not report which model it serves; record it by hand on the host, "
            "otherwise the artifact cannot be compared later.",
        )
    if requested and served != requested:
        return Finding(
            "served_model",
            "warn",
            f"requested {requested!r} but the endpoint serves {served!r}. The artifact records "
            "the served name; make sure that is the one you meant to measure.",
        )
    return None


def check_reasoning_mode(completion_tokens: int, reasoning_chars: int) -> Finding | None:
    """Detect that the model was thinking while you believed it was not.

    A reasoning model answers a trivial control question with a visible block of reasoning,
    which can be an order of magnitude more tokens than the answer needs. If a run is compared
    against another run in a different reasoning mode, the comparison is meaningless — and
    nothing in the response status tells you this happened.

    Turning reasoning off is server- and template-specific; this guard only tells you which
    mode you actually measured.
    """
    if reasoning_chars > 0:
        return Finding(
            "reasoning_mode",
            "warn",
            f"the control answer carried {reasoning_chars} characters of reasoning: this run "
            "measures the model WITH thinking enabled. Compare it only against runs in the "
            "same mode.",
        )
    if completion_tokens > REASONING_TOKEN_BUDGET:
        return Finding(
            "reasoning_mode",
            "warn",
            f"the control question was answered in {completion_tokens} tokens "
            f"(expected under {REASONING_TOKEN_BUDGET}). The model is probably thinking "
            "without exposing a reasoning field. Verify before comparing.",
        )
    return None


def check_thread_parity(baseline: dict, candidate: dict) -> Finding | None:
    """Refuse to compare two runs that were given different CPU thread budgets.

    Backends do not share a thread setting: setting the framework's own knob on one side
    leaves the other on its automatic default. The resulting difference is then reported as a
    property of the backend when it is a property of the configuration.
    """
    b, c = baseline.get("threads"), candidate.get("threads")
    if b is None or c is None:
        return Finding(
            "thread_parity",
            "warn",
            "at least one artifact does not record a thread budget; a CPU comparison without "
            "it cannot be trusted.",
        )
    if b != c:
        return Finding(
            "thread_parity",
            "invalid",
            f"thread budgets differ ({b} vs {c}). This comparison measures the configuration, "
            "not the backend.",
        )
    return None


def check_load_parity(baseline: dict, candidate: dict) -> Finding | None:
    """Refuse to compare a quiet run against a loaded one.

    Latency under concurrency includes queueing. Comparing it with a single-request number
    reads as a regression of the model when it is a difference in how the two were measured.
    """
    b, c = baseline.get("concurrency"), candidate.get("concurrency")
    if b is None or c is None:
        return Finding(
            "load_parity",
            "warn",
            "at least one artifact predates concurrency reporting; assume it was measured "
            "with a single request at a time.",
        )
    if b != c:
        return Finding(
            "load_parity",
            "invalid",
            f"concurrency differs ({b} vs {c}). Latency under load includes queueing, so this "
            "compares the measurement setups, not the models.",
        )
    return None


def quality_pairing_note() -> Finding:
    """Always emitted by a comparison: speed alone is not a decision."""
    return Finding(
        "quality_pairing",
        "warn",
        "this is the performance half only. A candidate that is faster and worse is not a win: "
        "record a quality gate result for the same commit and model before switching anything.",
    )
