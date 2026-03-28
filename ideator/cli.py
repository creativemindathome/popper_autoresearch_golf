from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .gemini import GeminiClient, GeminiError
from .knowledge import choose_knowledge_dir, load_knowledge_context
from .prompts import build_ideator_prompts, ideator_response_schema


def _env_api_key() -> Optional[str]:
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_API_KEY"):
        value = os.getenv(key)
        if value:
            return value
    return None


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ideator", add_help=True)

    subparsers = parser.add_subparsers(dest="cmd")

    p_idea = subparsers.add_parser("idea", help="Generate one new idea (default)")
    p_idea.add_argument("--api-key", default=None, help="Gemini API key (or set GEMINI_API_KEY)")
    p_idea.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model name (defaults to gemini-2.5-flash; override via GEMINI_MODEL)",
    )
    p_idea.add_argument("--temperature", type=float, default=1.3)
    p_idea.add_argument("--top-p", type=float, default=0.95)
    p_idea.add_argument("--max-output-tokens", type=int, default=2048)
    p_idea.add_argument("--seed", type=int, default=None)
    p_idea.add_argument(
        "--knowledge-dir",
        default=None,
        help="Path to knowledge directory (defaults to ./knowledge_graph if present)",
    )
    p_idea.add_argument("--out", default=None, help="Write the JSON idea to a file as well")
    p_idea.add_argument(
        "--save-dir",
        default=os.getenv("IDEATOR_SAVE_DIR"),
        help="Directory to save ideas (default: knowledge_graph/outbox/ideator)",
    )
    p_idea.add_argument("--no-save", action="store_true", help="Do not save idea JSON to disk")
    p_idea.add_argument("--dry-run", action="store_true", help="Print prompt JSON and exit")

    p_list = subparsers.add_parser("list-models", help="List models available to your API key")
    p_list.add_argument("--api-key", default=None, help="Gemini API key (or set GEMINI_API_KEY)")

    # Default to `idea` if no explicit subcommand.
    if argv and argv[0] in ("idea", "list-models"):
        return parser.parse_args(list(argv))
    return parser.parse_args(["idea", *argv])


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _print_json(obj: Any) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")


def cmd_list_models(api_key: str) -> int:
    client = GeminiClient(api_key=api_key)
    models = client.list_models()
    for m in models:
        name = m.get("name", "")
        display = m.get("displayName") or m.get("display_name") or ""
        methods = m.get("supportedGenerationMethods") or m.get("supported_generation_methods") or []
        if display:
            sys.stdout.write(f"{name}\t{display}\t{','.join(methods)}\n")
        else:
            sys.stdout.write(f"{name}\t{','.join(methods)}\n")
    return 0


def cmd_idea(
    *,
    api_key: str,
    model: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: Optional[int],
    knowledge_dir: Optional[str],
    out: Optional[str],
    save_dir: Optional[str],
    no_save: bool,
    dry_run: bool,
) -> int:
    kdir = choose_knowledge_dir(Path(knowledge_dir) if knowledge_dir else None, cwd=Path.cwd())
    knowledge_context = load_knowledge_context(kdir) if kdir else ""
    system_prompt, user_prompt = build_ideator_prompts(knowledge_context=knowledge_context)

    request_debug = {
        "model": model,
        "system": system_prompt,
        "user": user_prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
        "seed": seed,
        "knowledge_dir": str(kdir) if kdir else None,
    }
    if dry_run:
        _print_json(request_debug)
        return 0

    client = GeminiClient(api_key=api_key)
    try:
        idea = client.generate_json(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=ideator_response_schema(),
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            seed=seed,
        )
    except GeminiError as e:
        sys.stderr.write(str(e).rstrip() + "\n")
        sys.stderr.write("Tip: run `python3 -m ideator list-models` to verify model availability.\n")
        return 2

    if isinstance(idea, dict):
        idea = _finalize_idea(idea, model=model, knowledge_dir=str(kdir) if kdir else None)
        issues = _validate_idea_minimal(idea)
        for msg in issues:
            sys.stderr.write(f"warning: {msg}\n")

    _print_json(idea)

    saved_paths: List[Path] = []
    if out:
        p = Path(out)
        _write_json(p, idea)
        saved_paths.append(p)

    if not no_save:
        save_root = Path(save_dir) if save_dir else _default_save_dir(kdir)
        saved_paths.extend(_save_idea_bundle(save_root, idea))

    for p in saved_paths:
        sys.stderr.write(f"saved: {p}\n")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.cmd == "list-models":
        api_key = args.api_key or _env_api_key()
        if not api_key:
            sys.stderr.write(
                "Missing API key. Set GEMINI_API_KEY (recommended) or pass --api-key.\n"
            )
            return 2
        return cmd_list_models(api_key)

    if args.cmd == "idea":
        api_key = args.api_key or _env_api_key() or ""
        if not api_key and not args.dry_run:
            sys.stderr.write(
                "Missing API key. Set GEMINI_API_KEY (recommended) or pass --api-key.\n"
            )
            return 2
        return cmd_idea(
            api_key=api_key,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_output_tokens=args.max_output_tokens,
            seed=args.seed,
            knowledge_dir=args.knowledge_dir,
            out=args.out,
            save_dir=args.save_dir,
            no_save=args.no_save,
            dry_run=args.dry_run,
        )

    sys.stderr.write("Unknown command.\n")
    return 2


def _default_save_dir(knowledge_dir: Optional[Path]) -> Path:
    if knowledge_dir is not None:
        return knowledge_dir / "outbox" / "ideator"
    return Path.cwd() / "ideator_outbox"


def _sanitize_slug(value: str) -> str:
    slug = value.strip().lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug[:80] or "idea"


def _save_idea_bundle(save_root: Path, idea: Any) -> List[Path]:
    if not isinstance(idea, dict):
        return []
    idea_id = _sanitize_slug(str(idea.get("idea_id") or "idea"))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = save_root
    out_dir.mkdir(parents=True, exist_ok=True)

    stamped = out_dir / f"{ts}_{idea_id}.json"
    latest = out_dir / "latest.json"

    _write_json(stamped, idea)
    _write_json(latest, idea)
    return [stamped, latest]


def _finalize_idea(idea: Dict[str, Any], *, model: str, knowledge_dir: Optional[str]) -> Dict[str, Any]:
    # Normalize a few fields to keep the saved artifact consistent.
    idea = dict(idea)
    idea["schema_version"] = "ideator.idea.v1"
    parent = idea.get("parent_implementation")
    if isinstance(parent, dict):
        parent = dict(parent)
        parent["repo_url"] = "https://github.com/openai/parameter-golf"
        parent["primary_file"] = "train_gpt.py"
        idea["parent_implementation"] = parent
    idea["meta"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "knowledge_dir": knowledge_dir,
    }
    return idea


def _validate_idea_minimal(idea: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    required_top = [
        "schema_version",
        "idea_id",
        "title",
        "novelty_summary",
        "parent_implementation",
        "implementation_steps",
        "falsifier_smoke_tests",
        "expected_metric_change",
    ]
    for k in required_top:
        if k not in idea:
            issues.append(f"missing field '{k}'")

    parent = idea.get("parent_implementation")
    if isinstance(parent, dict):
        for k in ("repo_url", "primary_file", "run_command", "code_search_hints"):
            if k not in parent:
                issues.append(f"missing parent_implementation.{k}")
    else:
        issues.append("parent_implementation is not an object")

    steps = idea.get("implementation_steps")
    if isinstance(steps, list):
        if not steps:
            issues.append("implementation_steps is empty")
        else:
            for i, s in enumerate(steps[:3]):
                if not isinstance(s, dict):
                    issues.append(f"implementation_steps[{i}] is not an object")
                    continue
                for k in ("step_id", "file", "locate", "change", "done_when"):
                    if k not in s:
                        issues.append(f"missing implementation_steps[{i}].{k}")
    else:
        issues.append("implementation_steps is not an array")

    return issues
