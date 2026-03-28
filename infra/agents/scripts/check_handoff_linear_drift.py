#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import requests


LINEAR_API_URL = "https://api.linear.app/graphql"
STATUS_TO_LINEAR = {
    "todo": "Todo",
    "in_progress": "In Progress",
    "backlog": "Backlog",
    "done": "Done",
}


def load_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def load_queue(repo_root: Path) -> dict:
    script = repo_root / "infra" / "agents" / "scripts" / "resolve_handoff_queue.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def query_linear(api_key: str, query: str, variables: dict) -> dict:
    response = requests.post(
        LINEAR_API_URL,
        headers={"Authorization": api_key, "Content-Type": "application/json"},
        json={"query": query, "variables": variables},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"]))
    return payload["data"]


def get_issue_statuses(api_key: str, issue_ids: list[str]) -> dict[str, dict[str, str]]:
    query = """
    query IssueStatus($id: String!) {
      issue(id: $id) {
        identifier
        state {
          name
        }
      }
    }
    """
    statuses: dict[str, dict[str, str]] = {}
    for issue_id in issue_ids:
        data = query_linear(api_key, query, {"id": issue_id})
        issue = data["issue"]
        statuses[issue["identifier"]] = {"state": issue["state"]["name"]}
    return statuses


def update_issue_status(api_key: str, issue_id: str, state_name: str) -> None:
    status_query = """
    query IssueStatusByName($id: String!, $name: String!) {
      issue(id: $id) {
        team {
          states(filter: { name: { eq: $name } }) {
            nodes {
              id
              name
            }
          }
        }
      }
    }
    """
    data = query_linear(api_key, status_query, {"id": issue_id, "name": state_name})
    nodes = data["issue"]["team"]["states"]["nodes"]
    if not nodes:
        raise RuntimeError(f"Could not find Linear state named {state_name!r}")
    state_id = nodes[0]["id"]

    mutation = """
    mutation UpdateIssueState($id: String!, $stateId: String!) {
      issueUpdate(id: $id, input: { stateId: $stateId }) {
        success
      }
    }
    """
    query_linear(api_key, mutation, {"id": issue_id, "stateId": state_id})


def build_expected_statuses(queue: dict, issue_map: dict[str, str]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for status_key, linear_state in STATUS_TO_LINEAR.items():
        for handoff_name in queue.get(status_key, []):
            issue_id = issue_map.get(handoff_name)
            if issue_id:
                expected[issue_id] = linear_state
    return expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fix-status", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    infra_root = repo_root / "infra" / "agents"
    env_data = load_env_file(infra_root / "env" / ".env.symphony")
    api_key = os.environ.get("LINEAR_API_KEY") or env_data.get("LINEAR_API_KEY")
    if not api_key:
        raise SystemExit("LINEAR_API_KEY is required")

    issue_map_path = infra_root / "handoffs" / "buildout_issue_map.json"
    issue_map = json.loads(issue_map_path.read_text())
    queue = load_queue(repo_root)
    expected = build_expected_statuses(queue, issue_map)
    actual = get_issue_statuses(api_key, sorted(set(issue_map.values())))

    drift = []
    for issue_id, expected_state in expected.items():
        actual_state = actual.get(issue_id, {}).get("state")
        if actual_state != expected_state:
            drift.append(
                {
                    "issue_id": issue_id,
                    "expected_state": expected_state,
                    "actual_state": actual_state,
                }
            )

    if args.fix_status:
        for entry in drift:
            update_issue_status(api_key, entry["issue_id"], entry["expected_state"])
        actual = get_issue_statuses(api_key, sorted(set(issue_map.values())))
        drift = [
            {
                "issue_id": issue_id,
                "expected_state": expected_state,
                "actual_state": actual.get(issue_id, {}).get("state"),
            }
            for issue_id, expected_state in expected.items()
            if actual.get(issue_id, {}).get("state") != expected_state
        ]

    payload = {
        "drift": drift,
        "expected": expected,
        "actual": actual,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if drift:
            for entry in drift:
                print(
                    f"{entry['issue_id']}: expected {entry['expected_state']}, "
                    f"found {entry['actual_state']}"
                )
        else:
            print("handoff/Linear status drift check passed")
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
