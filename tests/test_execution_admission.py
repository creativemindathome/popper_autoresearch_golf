from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "infra" / "agents" / "scripts" / "check_execution_admission.py"


def write_verdict(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_execution_admission_rejects_missing_verdict(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--theory-id", "missing", "--verdict-path", str(tmp_path / "nope.json")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "No such file or directory" in result.stderr or "no verdict artifacts found" in result.stderr


def test_execution_admission_rejects_non_promote_verdict(tmp_path):
    verdict_path = tmp_path / "verdict.json"
    write_verdict(
        verdict_path,
        {
            "theory_id": "theory_bad",
            "created_at": "20260328T120000Z",
            "outcome": "inconclusive",
            "decision": "rewrite",
            "supporting_results": [{"name": "probe_a", "refutes": False}],
        },
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--theory-id", "theory_bad", "--verdict-path", str(verdict_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "outcome must be 'survived'" in result.stderr


def test_execution_admission_accepts_promote_verdict(tmp_path):
    verdict_path = tmp_path / "verdict.json"
    write_verdict(
        verdict_path,
        {
            "theory_id": "theory_good",
            "created_at": "20260328T120000Z",
            "outcome": "survived",
            "decision": "promote",
            "supporting_results": [{"name": "probe_a", "refutes": False}],
        },
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--theory-id", "theory_good", "--verdict-path", str(verdict_path), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["eligible"] is True
    assert payload["theory_id"] == "theory_good"
