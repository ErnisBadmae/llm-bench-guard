"""Benchmark local OpenAI-compatible LLM endpoints, with guards against misleading numbers."""

from .bench import Endpoint, run_benchmark
from .compare import compare
from .guards import Finding, GuardReport

__version__ = "0.1.0"
__all__ = ["Endpoint", "run_benchmark", "compare", "Finding", "GuardReport"]
