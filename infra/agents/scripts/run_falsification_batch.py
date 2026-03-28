#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from datetime import datetime, UTC
import argparse
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.probe_library import run_experiment, save_json


def compare_numeric(observed: float, operator: str, target: float) -> bool:
    if operator == ">":
        return observed > target
    if operator == ">=":
        return observed >= target
    if operator == "<":
        return observed < target
    if operator == "<=":
        return observed <= target
    raise ValueError(f"unsupported operator {operator!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theory-id", required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional path to a torch checkpoint such as final_model.pt")
    parser.add_argument(
        "--experiment-spec",
        default=None,
        help="Optional JSON file with experiments. Without it, a default architecture+quantization batch runs.",
    )
    parser.add_argument("--outcome", choices=["refuted", "survived", "inconclusive"], default=None)
    args = parser.parse_args()

    repo_root = REPO_ROOT
    out_dir = repo_root / "research" / "falsification" / args.theory_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    if args.experiment_spec:
        spec_path = Path(args.experiment_spec).expanduser().resolve()
        spec = json.loads(spec_path.read_text())
    else:
        spec = {
            "experiments": [
                {"name": "architecture_profile"},
                {"name": "quantization_profile"},
            ]
        }

    results = []
    any_refuted = False
    for experiment in spec.get("experiments", []):
        name = experiment["name"]
        payload = run_experiment(name, checkpoint_path=args.checkpoint, repo_root=repo_root)
        refutes = False
        threshold = experiment.get("refutation_threshold")
        observed = None
        if threshold:
            key_path = threshold["metric_path"].split(".")
            value = payload
            for key in key_path:
                value = value[key]
            observed = float(value)
            refutes = compare_numeric(observed, threshold["operator"], float(threshold["value"]))
            any_refuted = any_refuted or refutes
        results.append(
            {
                "name": name,
                "payload": payload,
                "refutes": refutes,
                "observed": observed,
                "threshold": threshold,
            }
        )

    outcome = args.outcome
    if outcome is None:
        outcome = "refuted" if any_refuted else "survived"

    verdict = {
        "theory_id": args.theory_id,
        "created_at": timestamp,
        "checkpoint_path": args.checkpoint,
        "outcome": outcome,
        "decision": "block" if outcome == "refuted" else "promote" if outcome == "survived" else "rewrite",
        "reasoning": "auto-generated from executed probe results and optional thresholds",
        "supporting_results": [{"name": result["name"], "refutes": result["refutes"]} for result in results],
    }
    results_path = out_dir / f"results_{timestamp}.json"
    verdict_path = out_dir / f"verdict_{timestamp}.json"
    summary_path = out_dir / f"summary_{timestamp}.md"
    save_json(results_path, {"theory_id": args.theory_id, "created_at": timestamp, "results": results})
    save_json(verdict_path, verdict)
    summary_path.write_text(
        "# Falsifier Summary\n\n"
        f"- theory_id: `{args.theory_id}`\n"
        f"- outcome: `{outcome}`\n"
        f"- created_at: `{timestamp}`\n"
        f"- results_file: `{results_path}`\n"
    )
    print(verdict_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
