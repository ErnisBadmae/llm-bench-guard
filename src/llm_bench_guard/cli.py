"""Command line: ``run`` measures an endpoint, ``compare`` diffs two artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bench import Endpoint, run_benchmark
from .compare import compare


def _print_guards(guards: dict) -> None:
    for finding in guards.get("findings", []):
        marker = "BLOCKING" if finding["severity"] == "invalid" else finding["severity"].upper()
        print(f"  [{marker}] {finding['guard']}: {finding['message']}", file=sys.stderr)
    if not guards.get("reportable", True):
        print(
            "\nThese numbers are not reportable. Fix the blocking conditions and measure again.",
            file=sys.stderr,
        )


def _cmd_run(args: argparse.Namespace) -> int:
    endpoint = Endpoint(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        trust_env=args.trust_env,
    )
    extra = json.loads(args.extra_body) if args.extra_body else None
    artifact = run_benchmark(
        endpoint,
        repeats=args.repeats,
        threads=args.threads,
        extra_body=extra,
        label=args.label,
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), "utf-8")
        print(f"artifact: {args.out}", file=sys.stderr)
    print(json.dumps(artifact["overall"], indent=2))
    _print_guards(artifact["guards"])
    return 0 if artifact["guards"]["reportable"] else 1


def _cmd_compare(args: argparse.Namespace) -> int:
    baseline = json.loads(Path(args.baseline).read_text("utf-8"))
    candidate = json.loads(Path(args.candidate).read_text("utf-8"))
    result = compare(baseline, candidate)
    printable = {k: v for k, v in result.items() if k != "guards"}
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    _print_guards(result["guards"])
    return 0 if result["guards"]["reportable"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-bench-guard", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="measure an endpoint")
    run.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000/v1")
    run.add_argument("--model", required=True)
    run.add_argument("--api-key", default=None)
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--timeout", type=float, default=180.0)
    run.add_argument("--out", default=None, help="where to write the JSON artifact")
    run.add_argument("--label", default=None, help="free-form note stored in the artifact")
    run.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CPU thread budget given to the runtime; recorded so comparisons can check parity",
    )
    run.add_argument(
        "--extra-body",
        default=None,
        help='JSON merged into the request body, e.g. \'{"chat_template_kwargs":'
        ' {"enable_thinking": false}}\'',
    )
    run.add_argument(
        "--trust-env",
        action="store_true",
        help="honour HTTP_PROXY/HTTPS_PROXY (off by default: a proxy will answer for a private "
        "address and the failure looks like an auth error from the model server)",
    )
    run.set_defaults(func=_cmd_run)

    cmp_ = sub.add_parser("compare", help="diff two artifacts")
    cmp_.add_argument("--baseline", required=True)
    cmp_.add_argument("--candidate", required=True)
    cmp_.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
