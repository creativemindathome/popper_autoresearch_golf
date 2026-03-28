from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .gemini import GeminiClient, GeminiError
from .knowledge import choose_knowledge_dir, load_knowledge_context
from .openai_client import OpenAIClient, OpenAIError
from .parent_code import (
    ParentCode,
    ParentCodeError,
    load_parent_code_from_file,
    load_parent_code_from_github,
    load_parent_code_from_run,
)
from .prompts import (
    build_ideator_prompts,
    build_ideator_revision_prompts,
    build_patch_prompts,
    build_reviewer_prompts,
    ideator_response_schema,
    patch_response_schema,
    reviewer_response_schema,
)


def _env_api_key() -> Optional[str]:
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_API_KEY"):
        value = os.getenv(key)
        if value:
            return value
    return None


def _env_openai_api_key() -> Optional[str]:
    for key in ("OPENAI_API_KEY",):
        value = os.getenv(key)
        if value:
            return value
    return None


def _env_gemini_timeout_s() -> float:
    value = os.getenv("GEMINI_TIMEOUT_S", "").strip()
    if not value:
        return 180.0
    try:
        return float(value)
    except Exception:
        return 180.0


def _env_gemini_max_retries() -> int:
    value = os.getenv("GEMINI_MAX_RETRIES", "").strip()
    if not value:
        return 2
    try:
        return int(value)
    except Exception:
        return 2


def _env_reviewer_min_score() -> int:
    value = os.getenv("IDEATOR_REVIEWER_MIN_SCORE", "").strip()
    if not value:
        return 6
    try:
        return int(value)
    except Exception:
        return 6


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
    p_idea.add_argument("--max-output-tokens", type=int, default=16384)
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
    p_idea.add_argument(
        "--reviewer-api-key",
        default=None,
        help="OpenAI API key for the novelty reviewer (or set OPENAI_API_KEY)",
    )
    p_idea.add_argument(
        "--reviewer-model",
        default=os.getenv("OPENAI_REVIEWER_MODEL", "gpt-4o-mini"),
        help="OpenAI model for the novelty reviewer (default: gpt-4o-mini; override via OPENAI_REVIEWER_MODEL)",
    )
    p_idea.add_argument(
        "--reviewer-base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
        help="OpenAI API base URL (default: https://api.openai.com; override via OPENAI_BASE_URL)",
    )
    p_idea.add_argument("--reviewer-temperature", type=float, default=0.0)
    p_idea.add_argument("--reviewer-max-tokens", type=int, default=700)
    p_idea.add_argument(
        "--reviewer-min-score",
        type=int,
        default=_env_reviewer_min_score(),
        help="Auto-pass if novelty_score >= this value (default: 6; override via IDEATOR_REVIEWER_MIN_SCORE)",
    )
    p_idea.add_argument(
        "--max-review-rounds",
        type=int,
        default=int(os.getenv("IDEATOR_MAX_REVIEW_ROUNDS", "4")),
        help="Maximum ideator↔reviewer revision rounds before failing (default: 4; override via IDEATOR_MAX_REVIEW_ROUNDS)",
    )
    p_idea.add_argument(
        "--no-reviewer",
        action="store_true",
        help="Disable the novelty reviewer loop (not recommended)",
    )
    p_idea.add_argument(
        "--parent-run",
        default=None,
        help="Run id to use as parent (defaults to latest generated train_gpt if present)",
    )
    p_idea.add_argument(
        "--parent-train-gpt",
        default=None,
        help="Path to a local parent train_gpt.py (overrides --parent-run/--parent-repo-url)",
    )
    p_idea.add_argument(
        "--parent-repo-url",
        default=os.getenv("PARAM_GOLF_PARENT_REPO_URL", "https://github.com/openai/parameter-golf"),
        help="GitHub repo URL for the parent implementation (default: openai/parameter-golf)",
    )
    p_idea.add_argument(
        "--parent-git-ref",
        default=os.getenv("PARAM_GOLF_PARENT_GIT_REF", "main"),
        help="Git ref (branch or commit) for the parent implementation (default: main)",
    )
    p_idea.add_argument(
        "--parent-file-path",
        default=os.getenv("PARAM_GOLF_PARENT_FILE_PATH", "train_gpt.py"),
        help="Path to train_gpt.py inside the parent repo (default: train_gpt.py)",
    )
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
    client = GeminiClient(
        api_key=api_key,
        timeout_s=_env_gemini_timeout_s(),
        max_retries=_env_gemini_max_retries(),
    )
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
    reviewer_api_key: Optional[str],
    reviewer_model: str,
    reviewer_base_url: str,
    reviewer_temperature: float,
    reviewer_max_tokens: int,
    reviewer_min_score: int,
    max_review_rounds: int,
    no_reviewer: bool,
    parent_run: Optional[str],
    parent_train_gpt: Optional[str],
    parent_repo_url: str,
    parent_git_ref: str,
    parent_file_path: str,
    dry_run: bool,
) -> int:
    kdir = choose_knowledge_dir(Path(knowledge_dir) if knowledge_dir else None, cwd=Path.cwd())
    knowledge_context = load_knowledge_context(kdir) if kdir else ""

    save_root = Path(save_dir) if save_dir else _default_save_dir(kdir)
    runs_root = save_root / "runs"

    try:
        parent = _load_parent_train_gpt(
            runs_root=runs_root,
            parent_run=parent_run,
            parent_train_gpt=parent_train_gpt,
            parent_repo_url=parent_repo_url,
            parent_git_ref=parent_git_ref,
            parent_file_path=parent_file_path,
            save_root=save_root,
        )
    except ParentCodeError as e:
        sys.stderr.write(str(e).rstrip() + "\n")
        sys.stderr.write(
            "Tip: clone Parameter Golf to ./parameter-golf or pass --parent-train-gpt /path/to/train_gpt.py.\n"
        )
        return 2

    parent_code_ref = {
        "kind": parent.ref.kind,
        "repo_url": parent_repo_url,
        "git_ref": parent_git_ref,
        "file_path": parent_file_path,
        "parent_run_id": parent.ref.run_id,
        "parent_sha256": parent.sha256,
    }

    system_prompt, user_prompt = build_ideator_prompts(
        knowledge_context=knowledge_context,
        parent_code_ref=parent_code_ref,
    )

    request_debug = {
        "model": model,
        "system": system_prompt,
        "user": user_prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
        "seed": seed,
        "knowledge_dir": _relpath_or_none(kdir),
        "parent_code_ref": parent_code_ref,
    }
    if dry_run:
        _print_json(request_debug)
        return 0

    reviewer_enabled = not no_reviewer
    reviewer_client: Optional[OpenAIClient] = None
    if reviewer_enabled:
        if not reviewer_api_key:
            sys.stderr.write(
                "Missing OpenAI reviewer API key. Set OPENAI_API_KEY (recommended) or pass --reviewer-api-key.\n"
            )
            return 2
        reviewer_client = OpenAIClient(api_key=reviewer_api_key, base_url=reviewer_base_url)

    ideator_client = GeminiClient(
        api_key=api_key,
        timeout_s=_env_gemini_timeout_s(),
        max_retries=_env_gemini_max_retries(),
    )

    accepted_idea_raw: Optional[Dict[str, Any]] = None
    accepted_review: Optional[Dict[str, Any]] = None

    prev_thin_idea: Optional[Dict[str, Any]] = None
    prev_review: Optional[Dict[str, Any]] = None

    rounds = max(1, int(max_review_rounds))
    for round_idx in range(rounds):
        if round_idx == 0:
            ideator_system = system_prompt
            ideator_user = user_prompt
        else:
            if prev_thin_idea is None or prev_review is None:
                sys.stderr.write("Internal error: missing previous idea/review for revision.\n")
                return 2
            ideator_system, ideator_user = build_ideator_revision_prompts(
                knowledge_context=knowledge_context,
                parent_code_ref=parent_code_ref,
                previous_idea=prev_thin_idea,
                reviewer_feedback=prev_review,
            )

        try:
            idea_raw = ideator_client.generate_json(
                model=model,
                system_prompt=ideator_system,
                user_prompt=ideator_user,
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

        if not isinstance(idea_raw, dict):
            if not reviewer_enabled:
                _print_json(idea_raw)
                return 0
            prev_thin_idea = {"note": "non-object model output; cannot review", "raw": idea_raw}  # type: ignore[assignment]
            prev_review = _synthetic_review(
                decision="revise",
                novelty_score=0,
                reasons=["Model returned non-object JSON; cannot accept."],
                revision_instructions="Return a single JSON object matching the ideator schema.",
                must_fix_fields=["schema_version", "parent_implementation", "implementation_steps"],
            )
            sys.stderr.write(f"review round {round_idx + 1}/{rounds}: revise (non-object JSON)\n")
            continue

        thin_candidate = _thin_idea_for_review(idea_raw, parent_sha256=parent.sha256)

        if not reviewer_enabled:
            accepted_idea_raw = idea_raw
            break

        reviewer_context = _knowledge_context_for_reviewer(knowledge_context)
        review_system, review_user = build_reviewer_prompts(
            knowledge_context=reviewer_context,
            idea=thin_candidate,
        )

        try:
            assert reviewer_client is not None
            review = reviewer_client.generate_json(
                model=reviewer_model,
                system_prompt=review_system,
                user_prompt=review_user,
                response_schema=reviewer_response_schema(),
                temperature=reviewer_temperature,
                max_tokens=reviewer_max_tokens,
            )
        except OpenAIError as e:
            sys.stderr.write(str(e).rstrip() + "\n")
            return 2

        if not isinstance(review, dict):
            review = _synthetic_review(
                decision="revise",
                novelty_score=0,
                reasons=["Reviewer returned non-object JSON; treating as revise."],
                revision_instructions="Return a valid review JSON object.",
                must_fix_fields=["novelty_summary", "implementation_steps"],
            )

        decision = str(review.get("decision") or "").strip().lower()
        score_raw = review.get("novelty_score")
        score_int: Optional[int] = None
        try:
            score_int = int(score_raw)
        except Exception:
            score_int = None
        score_disp = score_int if score_int is not None else score_raw

        reviewer_decision = decision or "revise"
        effective_decision = reviewer_decision
        note = ""
        if effective_decision != "pass" and score_int is not None and score_int >= int(reviewer_min_score):
            effective_decision = "pass"
            note = f" (auto-pass: score>={int(reviewer_min_score)}, reviewer_decision={reviewer_decision})"
            review = dict(review)
            review["decision"] = "pass"
        sys.stderr.write(
            f"review round {round_idx + 1}/{rounds}: decision={effective_decision} novelty_score={score_disp}{note}\n"
        )
        if effective_decision == "pass":
            accepted_idea_raw = idea_raw
            accepted_review = review
            break

        prev_thin_idea = thin_candidate
        prev_review = review

    if accepted_idea_raw is None:
        sys.stderr.write(
            f"Reviewer did not approve an idea after {rounds} round(s); refusing to emit an unreviewed idea.\n"
        )
        if not no_save and prev_thin_idea is not None and prev_review is not None:
            _save_review_failure(save_root=save_root, previous_idea=prev_thin_idea, reviewer_feedback=prev_review)
        return 3

    idea = dict(accepted_idea_raw)

    # After an idea is accepted, generate a minimal patch separately (more reliable than bundling with ideation JSON).
    train_gpt_text: Optional[str] = None
    last_patch_err: Optional[str] = None
    patch_attempts = 3
    sys.stderr.write("patch: generating train_gpt_patch...\n")
    for attempt_idx in range(patch_attempts):
        if attempt_idx:
            sys.stderr.write(f"patch: retry {attempt_idx + 1}/{patch_attempts}\n")
        patch_system, patch_user = build_patch_prompts(
            parent_train_gpt_py=parent.content,
            accepted_idea=idea,
        )
        if last_patch_err:
            patch_user += (
                "\n\nPrevious patch attempt failed to apply:\n"
                f"{last_patch_err}\n\n"
                "Regenerate a correct minimal unified diff that applies cleanly to the provided Parent train_gpt.py."
            )
        try:
            patch_obj = ideator_client.generate_json(
                model=model,
                system_prompt=patch_system,
                user_prompt=patch_user,
                response_schema=patch_response_schema(),
                temperature=0.2,
                top_p=1.0,
                max_output_tokens=int(max_output_tokens),
                seed=seed,
            )
        except GeminiError as e:
            sys.stderr.write(str(e).rstrip() + "\n")
            return 2

        patch_str = patch_obj.get("train_gpt_patch") if isinstance(patch_obj, dict) else None
        if not isinstance(patch_str, str) or not patch_str.strip():
            last_patch_err = "Patch generator returned missing/empty train_gpt_patch."
            continue
        try:
            train_gpt_text = _apply_unified_diff(parent.content, patch_str)
            break
        except PatchApplyError as e:
            last_patch_err = str(e)
            continue

    if train_gpt_text is None:
        sys.stderr.write("Failed to generate an applicable train_gpt_patch.\n")
        if last_patch_err:
            sys.stderr.write(last_patch_err.rstrip() + "\n")
        return 2

    idea_id = _sanitize_slug(str(idea.get("idea_id") or "idea"))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}_{idea_id}"

    patch_text = _unified_diff_text(parent.content, train_gpt_text)

    idea_final = _finalize_idea_v2(
        idea=idea,
        run_id=run_id,
        model=model,
        parent=parent,
        parent_code_ref=parent_code_ref,
        save_root=save_root,
        train_gpt_text=train_gpt_text,
        patch_text=patch_text,
        reviewer_feedback=accepted_review,
        reviewer_model=reviewer_model if reviewer_enabled else None,
    )

    _print_json(idea_final)

    saved_paths: List[Path] = []
    if out:
        p = Path(out)
        _write_json(p, idea_final)
        saved_paths.append(p)

    if not no_save:
        saved_paths.extend(
            _save_run_bundle(
                save_root=save_root,
                run_id=run_id,
                idea=idea_final,
                train_gpt_text=train_gpt_text,
                patch_text=patch_text,
                parent_train_gpt_text=parent.content,
                reviewer_feedback=accepted_review,
            )
        )

    for p in saved_paths:
        sys.stderr.write(f"saved: {_relpath_for_display(p)}\n")
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
            reviewer_api_key=args.reviewer_api_key or _env_openai_api_key(),
            reviewer_model=args.reviewer_model,
            reviewer_base_url=args.reviewer_base_url,
            reviewer_temperature=args.reviewer_temperature,
            reviewer_max_tokens=args.reviewer_max_tokens,
            reviewer_min_score=args.reviewer_min_score,
            max_review_rounds=args.max_review_rounds,
            no_reviewer=args.no_reviewer,
            parent_run=args.parent_run,
            parent_train_gpt=args.parent_train_gpt,
            parent_repo_url=args.parent_repo_url,
            parent_git_ref=args.parent_git_ref,
            parent_file_path=args.parent_file_path,
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

def _relpath_or_none(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:
        return path.name


def _relpath_for_display(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:
        return str(path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _join_lines(lines: List[str]) -> str:
    for i, line in enumerate(lines[:10]):
        if "\n" in line or "\r" in line:
            raise ValueError(f"train_gpt_py_lines[{i}] contains a newline character")
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    return text


def _unified_diff_text(a_text: str, b_text: str) -> str:
    a_lines = a_text.splitlines()
    b_lines = b_text.splitlines()
    diff_iter = difflib.unified_diff(
        a_lines,
        b_lines,
        fromfile="a/train_gpt.py",
        tofile="b/train_gpt.py",
        lineterm="",
    )
    diff = "\n".join(diff_iter)
    if diff and not diff.endswith("\n"):
        diff += "\n"
    return diff


class PatchApplyError(RuntimeError):
    pass


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _apply_unified_diff(original_text: str, patch_text: str) -> str:
    patch_text = patch_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = patch_text.split("\n")

    # Drop trailing empty entry if patch_text ends with \n.
    if lines and lines[-1] == "":
        lines = lines[:-1]

    # Find first hunk header.
    idx = 0
    while idx < len(lines) and not lines[idx].startswith("@@ "):
        idx += 1
    if idx >= len(lines):
        raise PatchApplyError("Invalid train_gpt_patch: no hunks found (expected lines starting with '@@ ').")

    src_lines = original_text.splitlines()
    out_lines: List[str] = []
    src_i = 0

    while idx < len(lines):
        header = lines[idx]
        if not header.startswith("@@ "):
            idx += 1
            continue

        m = _HUNK_HEADER_RE.match(header)
        if not m:
            raise PatchApplyError(f"Invalid train_gpt_patch hunk header: {header!r}")

        a_start = int(m.group(1))
        # a_len = int(m.group(2) or "1")  # unused (we validate by matching lines)
        target_src_i = a_start - 1
        if target_src_i < src_i:
            raise PatchApplyError("Invalid train_gpt_patch: hunks are overlapping or out of order.")

        out_lines.extend(src_lines[src_i:target_src_i])
        src_i = target_src_i
        idx += 1

        while idx < len(lines) and not lines[idx].startswith("@@ "):
            pline = lines[idx]
            if (
                pline.startswith("diff ")
                or pline.startswith("index ")
                or pline.startswith("--- ")
                or pline.startswith("+++ ")
            ):
                idx += 1
                continue
            if pline.startswith("\\ No newline at end of file"):
                idx += 1
                continue
            if pline == "":
                raise PatchApplyError("Invalid train_gpt_patch: encountered an empty patch line without a prefix.")

            prefix = pline[0]
            content = pline[1:]

            if prefix == " ":
                if src_i >= len(src_lines):
                    raise PatchApplyError("Invalid train_gpt_patch: context line goes past end of file.")
                if src_lines[src_i] != content:
                    got = src_lines[src_i]
                    raise PatchApplyError(
                        f"Invalid train_gpt_patch: context mismatch at source line {src_i + 1}. "
                        f"Expected {content!r}, got {got!r}."
                    )
                out_lines.append(content)
                src_i += 1
            elif prefix == "-":
                if src_i >= len(src_lines):
                    raise PatchApplyError("Invalid train_gpt_patch: removal goes past end of file.")
                if src_lines[src_i] != content:
                    got = src_lines[src_i]
                    raise PatchApplyError(
                        f"Invalid train_gpt_patch: removal mismatch at source line {src_i + 1}. "
                        f"Expected {content!r}, got {got!r}."
                    )
                src_i += 1
            elif prefix == "+":
                out_lines.append(content)
            else:
                raise PatchApplyError(f"Invalid train_gpt_patch: unexpected line prefix {prefix!r} in {pline!r}.")

            idx += 1

    out_lines.extend(src_lines[src_i:])
    text = "\n".join(out_lines)
    if not text.endswith("\n"):
        text += "\n"
    return text


def _diff_stats(patch_text: str) -> Dict[str, int]:
    added = 0
    removed = 0
    hunks = 0
    for line in patch_text.splitlines():
        if line.startswith("@@ "):
            hunks += 1
            continue
        if line.startswith("+++ ") or line.startswith("--- "):
            continue
        if line.startswith("+"):
            added += 1
            continue
        if line.startswith("-"):
            removed += 1
            continue
    return {"added_lines": added, "removed_lines": removed, "hunks": hunks}


def _knowledge_context_for_reviewer(knowledge_context: str, *, max_chars: int = 8000) -> str:
    ctx = (knowledge_context or "").strip()
    if not ctx:
        return ""
    marker = "## Knowledge Graph (raw snippets)"
    if marker in ctx:
        ctx = ctx.split(marker, 1)[0].strip()
    if len(ctx) <= max_chars:
        return ctx
    return (ctx[: max_chars - 20] + "\n...[truncated]").strip()


def _thin_idea_for_review(
    idea_raw: Dict[str, Any],
    *,
    parent_sha256: str,
    train_gpt_text: Optional[str] = None,
    patch_text: Optional[str] = None,
) -> Dict[str, Any]:
    thin = dict(idea_raw)
    train_lines = thin.pop("train_gpt_py_lines", None)
    patch_candidate = thin.pop("train_gpt_patch", None)

    summary: Dict[str, Any] = {
        "parent_sha256": parent_sha256,
        "train_gpt_py_lines_present": isinstance(train_lines, list),
        "train_gpt_py_lines_count": len(train_lines) if isinstance(train_lines, list) else None,
        "train_gpt_patch_present": isinstance(patch_candidate, str) and bool(patch_candidate.strip()),
        "train_gpt_patch_chars": len(patch_candidate) if isinstance(patch_candidate, str) else None,
    }
    if isinstance(patch_candidate, str) and patch_candidate.strip():
        excerpt = patch_candidate.strip()
        if len(excerpt) > 1400:
            excerpt = excerpt[:1400] + "\n...[truncated]"
        summary["train_gpt_patch_excerpt"] = excerpt
    if train_gpt_text is not None:
        train_bytes = train_gpt_text.encode("utf-8")
        summary.update(
            {
                "train_gpt_sha256": _sha256_bytes(train_bytes),
                "train_gpt_bytes": len(train_bytes),
            }
        )
    if patch_text is not None:
        patch_bytes = patch_text.encode("utf-8")
        summary.update(
            {
                "patch_sha256": _sha256_bytes(patch_bytes),
                "patch_bytes": len(patch_bytes),
                "patch_stats": _diff_stats(patch_text),
            }
        )

    thin["_code_summary"] = summary
    return thin


def _synthetic_review(
    *,
    decision: str,
    novelty_score: int,
    reasons: List[str],
    revision_instructions: str,
    must_fix_fields: List[str],
    similar_to_knowledge: Optional[List[str]] = None,
) -> Dict[str, Any]:
    primary_reasons = [str(r).strip() for r in (reasons or []) if str(r).strip()]
    if len(primary_reasons) < 2:
        primary_reasons.append("Needs revision.")
    primary_reasons = primary_reasons[:6]

    return {
        "decision": decision,
        "novelty_score": int(novelty_score),
        "primary_reasons": primary_reasons,
        "revision_instructions": str(revision_instructions).strip(),
        "must_fix_fields": [str(x).strip() for x in (must_fix_fields or []) if str(x).strip()],
        "similar_to_knowledge": [str(x).strip() for x in (similar_to_knowledge or []) if str(x).strip()],
    }


def _save_review_failure(
    *, save_root: Path, previous_idea: Dict[str, Any], reviewer_feedback: Dict[str, Any]
) -> Path:
    out_dir = save_root / "review_failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    idea_id = _sanitize_slug(str(previous_idea.get("idea_id") or "idea"))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{ts}_{idea_id}.json"
    obj = {
        "schema_version": "ideator.review_failure.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "previous_idea": previous_idea,
        "reviewer_feedback": reviewer_feedback,
    }
    _write_json(path, obj)
    sys.stderr.write(f"saved review failure: {_relpath_for_display(path)}\n")
    return path


def _read_latest_run_id(save_root: Path) -> Optional[str]:
    latest = save_root / "latest.json"
    if not latest.exists():
        return None
    try:
        obj = json.loads(latest.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    if isinstance(obj, dict):
        rid = obj.get("run_id")
        if isinstance(rid, str) and rid.strip():
            return rid.strip()
    return None


def _load_parent_train_gpt(
    *,
    runs_root: Path,
    parent_run: Optional[str],
    parent_train_gpt: Optional[str],
    parent_repo_url: str,
    parent_git_ref: str,
    parent_file_path: str,
    save_root: Path,
) -> ParentCode:
    if parent_train_gpt:
        return load_parent_code_from_file(Path(parent_train_gpt))

    if parent_run:
        return load_parent_code_from_run(runs_root, parent_run, file_name=Path(parent_file_path).name)

    latest_run = _read_latest_run_id(save_root)
    if latest_run:
        try:
            return load_parent_code_from_run(
                runs_root, latest_run, file_name=Path(parent_file_path).name
            )
        except ParentCodeError:
            pass

    # As a fallback, if a previous run wrote a convenience copy.
    latest_train_gpt = save_root / "latest_train_gpt.py"
    if latest_train_gpt.exists():
        return load_parent_code_from_file(latest_train_gpt)

    discovered = _discover_parent_train_gpt_path()
    if discovered is not None:
        return load_parent_code_from_file(discovered)

    return load_parent_code_from_github(parent_repo_url, parent_git_ref, parent_file_path)


def _discover_parent_train_gpt_path() -> Optional[Path]:
    candidates = [
        Path.cwd() / "train_gpt.py",
        Path.cwd() / "parameter-golf" / "train_gpt.py",
        Path.cwd() / "parameter_golf" / "train_gpt.py",
        Path.cwd() / "parents" / "train_gpt.py",
        Path.cwd() / "knowledge_graph" / "parents" / "train_gpt.py",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def _finalize_idea_v2(
    *,
    idea: Dict[str, Any],
    run_id: str,
    model: str,
    parent: ParentCode,
    parent_code_ref: Dict[str, Any],
    save_root: Path,
    train_gpt_text: str,
    patch_text: str,
    reviewer_feedback: Optional[Dict[str, Any]] = None,
    reviewer_model: Optional[str] = None,
) -> Dict[str, Any]:
    idea_out = dict(idea)
    idea_out["schema_version"] = "ideator.idea.v2"
    idea_out["run_id"] = run_id
    idea_id = _sanitize_slug(str(idea_out.get("idea_id") or "idea"))

    parent_impl = idea_out.get("parent_implementation")
    if not isinstance(parent_impl, dict):
        parent_impl = {}
    parent_impl = dict(parent_impl)
    parent_impl["repo_url"] = str(parent_code_ref.get("repo_url") or parent_impl.get("repo_url") or "")
    parent_impl["git_ref"] = str(parent_code_ref.get("git_ref") or parent_impl.get("git_ref") or "")
    parent_impl["primary_file"] = "train_gpt.py"
    idea_out["parent_implementation"] = parent_impl

    run_dir = (save_root / "runs" / run_id)
    rel_run_dir = _relpath_for_display(run_dir)

    train_bytes = train_gpt_text.encode("utf-8")
    patch_bytes = patch_text.encode("utf-8")

    idea_out["parents"] = [
        {
            "kind": parent.ref.kind,
            "repo_url": str(parent_code_ref.get("repo_url") or ""),
            "git_ref": str(parent_code_ref.get("git_ref") or ""),
            "file_path": str(parent_code_ref.get("file_path") or "train_gpt.py"),
            "run_id": parent.ref.run_id,
            "sha256": parent.sha256,
        }
    ]
    idea_out["artifacts"] = {
        "run_dir": rel_run_dir,
        "idea_json": f"{rel_run_dir}/idea.json",
        "train_gpt_py": {
            "path": f"{rel_run_dir}/train_gpt.py",
            "sha256": _sha256_bytes(train_bytes),
            "bytes": len(train_bytes),
        },
        "train_gpt_patch": {
            "path": f"{rel_run_dir}/train_gpt.patch",
            "sha256": _sha256_bytes(patch_bytes),
            "bytes": len(patch_bytes),
        },
        "parent_train_gpt_py": {
            "path": f"{rel_run_dir}/parent_train_gpt.py",
            "sha256": parent.sha256,
            "bytes": len(parent.content.encode("utf-8")),
        },
    }
    if reviewer_feedback is not None:
        idea_out["artifacts"]["review_json"] = f"{rel_run_dir}/review.json"

        approved_train = save_root / f"{idea_id}_train_gpt.py"
        approved_patch = save_root / f"{idea_id}_train_gpt.patch"
        approved_parent = save_root / f"{idea_id}_parent_train_gpt.py"
        approved_idea = save_root / f"{idea_id}.json"
        approved_review = save_root / f"{idea_id}_review.json"

        idea_out["artifacts"]["approved_idea_json"] = _relpath_for_display(approved_idea)
        idea_out["artifacts"]["approved_review_json"] = _relpath_for_display(approved_review)
        idea_out["artifacts"]["approved_train_gpt_py"] = {
            "path": _relpath_for_display(approved_train),
            "sha256": _sha256_bytes(train_bytes),
            "bytes": len(train_bytes),
        }
        idea_out["artifacts"]["approved_train_gpt_patch"] = {
            "path": _relpath_for_display(approved_patch),
            "sha256": _sha256_bytes(patch_bytes),
            "bytes": len(patch_bytes),
        }
        idea_out["artifacts"]["approved_parent_train_gpt_py"] = {
            "path": _relpath_for_display(approved_parent),
            "sha256": parent.sha256,
            "bytes": len(parent.content.encode("utf-8")),
        }
    run_command = ""
    if isinstance(parent_impl.get("run_command"), str):
        run_command = parent_impl["run_command"].strip()
    copy_source = f"{rel_run_dir}/train_gpt.py"
    if reviewer_feedback is not None:
        copy_source = _relpath_for_display(save_root / f"{idea_id}_train_gpt.py")
    idea_out["falsifier_instructions"] = {
        "assumed_repo_dir": "parameter-golf",
        "steps": [
            f"cp {copy_source} parameter-golf/train_gpt.py",
            "cd parameter-golf",
            run_command or "torchrun --standalone --nproc_per_node=1 train_gpt.py",
        ],
    }
    idea_out["meta"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
    }
    if reviewer_feedback is not None:
        idea_out["meta"]["reviewer"] = {
            "model": reviewer_model,
            "decision": reviewer_feedback.get("decision"),
            "novelty_score": reviewer_feedback.get("novelty_score"),
            "primary_reasons": reviewer_feedback.get("primary_reasons"),
        }

    _warn_if_user_paths(idea_out)
    return idea_out


def _warn_if_user_paths(obj: Any) -> None:
    try:
        dumped = json.dumps(obj, ensure_ascii=False)
    except Exception:
        return
    if "/Users/" in dumped or "\\Users\\" in dumped:
        sys.stderr.write("warning: output JSON contains an absolute user path; consider removing it.\n")


def _save_run_bundle(
    *,
    save_root: Path,
    run_id: str,
    idea: Dict[str, Any],
    train_gpt_text: str,
    patch_text: str,
    parent_train_gpt_text: str,
    reviewer_feedback: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    saved: List[Path] = []
    run_dir = save_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    idea_path = run_dir / "idea.json"
    train_path = run_dir / "train_gpt.py"
    patch_path = run_dir / "train_gpt.patch"
    parent_path = run_dir / "parent_train_gpt.py"
    review_path = run_dir / "review.json"

    _write_json(idea_path, idea)
    train_path.write_text(train_gpt_text, encoding="utf-8")
    patch_path.write_text(patch_text, encoding="utf-8")
    parent_path.write_text(parent_train_gpt_text, encoding="utf-8")
    if reviewer_feedback is not None:
        _write_json(review_path, reviewer_feedback)

    latest_json = save_root / "latest.json"
    latest_train = save_root / "latest_train_gpt.py"
    latest_patch = save_root / "latest_train_gpt.patch"
    latest_review = save_root / "latest_review.json"

    save_root.mkdir(parents=True, exist_ok=True)
    _write_json(latest_json, idea)
    latest_train.write_text(train_gpt_text, encoding="utf-8")
    latest_patch.write_text(patch_text, encoding="utf-8")
    if reviewer_feedback is not None:
        _write_json(latest_review, reviewer_feedback)

    saved.extend([idea_path, train_path, patch_path, parent_path, latest_json, latest_train, latest_patch])
    if reviewer_feedback is not None:
        saved.extend([review_path, latest_review])

        idea_id = _sanitize_slug(str(idea.get("idea_id") or run_id))
        named_idea = save_root / f"{idea_id}.json"
        named_train = save_root / f"{idea_id}_train_gpt.py"
        named_patch = save_root / f"{idea_id}_train_gpt.patch"
        named_parent = save_root / f"{idea_id}_parent_train_gpt.py"
        named_review = save_root / f"{idea_id}_review.json"

        _write_json(named_idea, idea)
        named_train.write_text(train_gpt_text, encoding="utf-8")
        named_patch.write_text(patch_text, encoding="utf-8")
        named_parent.write_text(parent_train_gpt_text, encoding="utf-8")
        _write_json(named_review, reviewer_feedback)

        saved.extend([named_idea, named_train, named_patch, named_parent, named_review])
    return saved
