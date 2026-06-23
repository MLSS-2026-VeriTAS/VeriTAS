#!/usr/bin/env python3
"""File-based verifier validation entry point.

This CLI intentionally does not launch MLRC-Bench. It reads an existing ledger
directory, writes verifier outputs, and can create a deterministic synthetic
demo ledger.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from veritas.verifier.io import to_jsonable
from veritas.verifier.pipeline import run_file_verifier
from veritas.verifier.synthetic import run_synthetic_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the file-based VeriTAS verifier.")
    parser.add_argument(
        "--ledger-dir",
        required=True,
        help="Directory containing candidates.jsonl and optional scheduler_state.json.",
    )
    parser.add_argument(
        "--incumbent-id",
        default="baseline",
        help="Candidate ID or logical candidate ID for the incumbent.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Minimum meaningful positive delta.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of paired bootstrap samples for per-unit mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for bootstrap sampling.",
    )
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Create and run a deterministic synthetic ledger in --ledger-dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ledger_dir = Path(args.ledger_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic_demo:
        summary = run_synthetic_demo(
            ledger_dir,
            epsilon=args.epsilon,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
        return 0

    report, action = run_file_verifier(
        ledger_dir,
        incumbent_id=args.incumbent_id,
        epsilon=args.epsilon,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        write_outputs=True,
    )
    print(
        json.dumps(
            {
                "recommended_action": report["recommended_action"],
                "scheduler_next_action": to_jsonable(action),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
