from __future__ import annotations

import argparse
import json
import os
import sys
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

    _print_json(idea)
    if out:
        _write_json(Path(out), idea)
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
            dry_run=args.dry_run,
        )

    sys.stderr.write("Unknown command.\n")
    return 2
