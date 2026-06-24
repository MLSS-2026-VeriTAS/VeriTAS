#!/usr/bin/env python3
"""Audit a completed MLRC product-recommendation run with VeriTAS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from veritas.verifier.io import to_jsonable
from veritas.verifier.mlrc_workflows import audit_completed_product_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate MLRC product predictions and run a VeriTAS audit."
    )
    parser.add_argument(
        "--mlrc-dir",
        default="~/work/MLRC-Bench",
        help="MLRC-Bench checkout root.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Completed MLRC run directory under logs/product-recommendation/...",
    )
    parser.add_argument(
        "--trial-dir",
        help="Output directory for regenerated predictions, ledger, and summary.",
    )
    parser.add_argument(
        "--task-python",
        default="~/miniconda3/envs/product-recommendation/bin/python",
        help="Python executable for the product-recommendation conda env.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.005,
        help="Minimum meaningful positive delta.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=200,
        help="Paired bootstrap samples.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--metric",
        choices=["parsed_mrr", "mlrc_exact"],
        default="parsed_mrr",
        help="Product-recommendation metric variant.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if args.trial_dir:
        trial_dir = Path(args.trial_dir).expanduser()
    else:
        trial_dir = Path("~/work/trials/product-recommendation").expanduser() / run_dir.name

    summary = audit_completed_product_run(
        mlrc_dir=args.mlrc_dir,
        run_dir=run_dir,
        trial_dir=trial_dir,
        task_python=args.task_python,
        epsilon=args.epsilon,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        metric_variant=args.metric,
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
