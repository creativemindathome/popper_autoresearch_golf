#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys


VALID_OUTCOME = "survived"
VALID_DECISION = "promote"


def find_latest_verdict(theory_dir: Path) -> Path | None:
    verdicts = sorted(theory_dir.glob("verdict_*.json"))
    return verdicts[-1] if verdicts else None


def validate_verdict_payload(payload: dict[str, object], theory_id: str) -> list[str]:
    errors: list[str] = []
    if payload.get("theory_id") != theory_id:
        errors.append(
            f"theory_id mismatch: expected {theory_id!r}, got {payload.get('theory_id')!r}"
        )
    if payload.get("outcome") != VALID_OUTCOME:
        errors.append(
            f"outcome must be {VALID_OUTCOME!r}, got {payload.get('outcome')!r}"
        )
    if payload.get("decision") != VALID_DECISION:
        errors.append(
            f"decision must be {VALID_DECISION!r}, got {payload.get('decision')!r}"
        )
    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or not created_at.strip():
        errors.append("created_at is required")
    supporting_results = payload.get("supporting_results")
    if not isinstance(supporting_results, list):
        errors.append("supporting_results must be a list")
    elif not supporting_results:
        errors.append("supporting_results must not be empty")
    return errors


def resolve_verdict_path(repo_root: Path, theory_id: str, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    theory_dir = repo_root / "research" / "falsification" / theory_id
    verdict_path = find_latest_verdict(theory_dir)
    if verdict_path is None:
        raise FileNotFoundError(
            f"no verdict artifacts found for theory {theory_id!r} under {theory_dir}"
        )
    return verdict_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theory-id", required=True)
    parser.add_argument("--verdict-path", default=None)
    parser.add_argument("--json", action="store_true", dest="emit_json")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    try:
        verdict_path = resolve_verdict_path(repo_root, args.theory_id, args.verdict_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    payload = json.loads(verdict_path.read_text())
    errors = validate_verdict_payload(payload, args.theory_id)
    if errors:
        if args.emit_json:
            print(
                json.dumps(
                    {
                        "eligible": False,
                        "theory_id": args.theory_id,
                        "verdict_path": str(verdict_path),
                        "errors": errors,
                    },
                    indent=2,
                )
            )
        else:
            print("; ".join(errors), file=sys.stderr)
        return 1

    if args.emit_json:
        print(
            json.dumps(
                {
                    "eligible": True,
                    "theory_id": args.theory_id,
                    "verdict_path": str(verdict_path),
                },
                indent=2,
            )
        )
    else:
        print(f"execution admission passed for {args.theory_id} using {verdict_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
