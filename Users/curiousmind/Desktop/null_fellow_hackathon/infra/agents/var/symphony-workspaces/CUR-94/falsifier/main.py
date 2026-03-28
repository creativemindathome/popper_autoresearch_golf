from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from falsifier.stage1.orchestrator import run_stage1
from falsifier.types import CandidatePackage, TheoryHistoryRecord


def load_input(path: str) -> CandidatePackage:
    payload = json.loads(Path(path).read_text())
    return CandidatePackage(
        theory_id=payload["theory_id"],
        train_gpt_path=payload["train_gpt_path"],
        what_and_why=payload["what_and_why"],
        reference_theories=payload.get("reference_theories", []),
        theory_history=[
            TheoryHistoryRecord(
                theory_id=record["theory_id"],
                verdict=record["verdict"],
                theory_text=record["theory_text"],
                failure_context=record.get("failure_context", ""),
                mechanism_tags=record.get("mechanism_tags", []),
                related_theory_ids=record.get("related_theory_ids", []),
            )
            for record in payload.get("theory_history", [])
        ],
        env_overrides=payload.get("env_overrides", {}),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    inp = load_input(args.input)
    out = run_stage1(inp)
    Path(args.output).write_text(json.dumps(asdict(out), indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
