"""Command-line entry point: `cfa demo --out outputs/`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from cfa import __version__
from cfa.pipeline import print_summary, run_demo


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfa", description="Climate fairness audit toolkit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run the end-to-end pipeline on SYNTHETIC data")
    demo.add_argument("--out", default="outputs", help="output directory (default: outputs)")
    demo.add_argument("--n-districts", type=_positive_int, default=300)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--n-boot", type=_positive_int, default=500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        print_summary(run_demo(args.out, args.n_districts, args.seed, args.n_boot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
