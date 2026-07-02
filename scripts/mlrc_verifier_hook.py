#!/usr/bin/env python3
"""Run a VeriTAS decision from materialized MLRC prediction files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from veritas.verifier.io import to_jsonable, write_json
from veritas.verifier.mlrc_workflows import (
    build_agent_feedback,
    run_product_recommendation_verifier,
    write_product_candidate_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a product-recommendation verifier hook from pred.csv files."
    )
    parser.add_argument("--ledger-dir", required=True)
    parser.add_argument("--baseline-pred", required=True)
    parser.add_argument("--candidate-pred", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--candidate-id", default="candidate")
    parser.add_argument("--candidate-method", default="candidate")
    parser.add_argument("--candidate-step", type=int)
    parser.add_argument("--incumbent-id", default="baseline")
    parser.add_argument(
        "--metric",
        choices=["parsed_mrr", "mlrc_exact"],
        default="parsed_mrr",
    )
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run-id",
        help="Optional MLRC run id to store in candidate specs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ledger_dir = Path(args.ledger_dir).expanduser()
    ledger_dir.mkdir(parents=True, exist_ok=True)
    specs_path = ledger_dir / "candidate_specs.jsonl"

    write_product_candidate_specs(
        specs_path,
        baseline_prediction_path=args.baseline_pred,
        candidate_prediction_path=args.candidate_pred,
        candidate_method_name=args.candidate_method,
        candidate_step=args.candidate_step,
        candidate_id=args.candidate_id,
        baseline_id=args.incumbent_id,
        run_id=args.run_id,
    )
    report, action, prepared = run_product_recommendation_verifier(
        ledger_dir=ledger_dir,
        candidate_specs_path=specs_path,
        labels_path=args.labels_path,
        incumbent_id=args.incumbent_id,
        metric_variant=args.metric,
        epsilon=args.epsilon,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    feedback = build_agent_feedback(report, action)
    (ledger_dir / "agent_feedback.txt").write_text(feedback, encoding="utf-8")
    write_json(
        ledger_dir / "hook_summary.json",
        {
            "prepared_task": prepared,
            "recommended_action": report["recommended_action"],
            "scheduler_next_action": action,
            "agent_feedback_path": str(ledger_dir / "agent_feedback.txt"),
        },
    )
    print(
        json.dumps(
            {
                "prepared_task": to_jsonable(prepared),
                "recommended_action": report["recommended_action"],
                "scheduler_next_action": action,
                "agent_feedback": feedback,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
