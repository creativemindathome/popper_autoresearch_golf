from __future__ import annotations

import hashlib
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


class ParentCodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParentCodeRef:
    kind: str  # "run" | "local" | "github"
    repo_url: Optional[str] = None
    git_ref: Optional[str] = None
    file_path: str = "train_gpt.py"
    run_id: Optional[str] = None
    local_path: Optional[str] = None


@dataclass(frozen=True)
class ParentCode:
    ref: ParentCodeRef
    content: str
    sha256: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_parent_code_from_file(path: Path) -> ParentCode:
    if not path.exists():
        raise ParentCodeError(f"Parent file not found: {path}")
    content = path.read_text(encoding="utf-8", errors="replace")
    ref = ParentCodeRef(kind="local", local_path=str(path), file_path=path.name)
    return ParentCode(ref=ref, content=content, sha256=sha256_text(content))


def load_parent_code_from_run(runs_root: Path, run_id: str, *, file_name: str = "train_gpt.py") -> ParentCode:
    run_dir = runs_root / run_id
    if not run_dir.exists():
        raise ParentCodeError(f"Parent run not found: {run_dir}")
    path = run_dir / file_name
    if not path.exists():
        raise ParentCodeError(f"Parent run missing {file_name}: {path}")
    content = path.read_text(encoding="utf-8", errors="replace")
    ref = ParentCodeRef(kind="run", run_id=run_id, file_path=file_name)
    return ParentCode(ref=ref, content=content, sha256=sha256_text(content))


def load_parent_code_from_github(repo_url: str, git_ref: str, file_path: str) -> ParentCode:
    owner_repo = _parse_github_owner_repo(repo_url)
    if owner_repo is None:
        raise ParentCodeError(f"Unsupported repo_url (expected GitHub HTTPS URL): {repo_url}")
    owner, repo = owner_repo
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{git_ref}/{file_path.lstrip('/')}"
    content = _http_get_text(raw_url)
    ref = ParentCodeRef(kind="github", repo_url=repo_url, git_ref=git_ref, file_path=file_path)
    return ParentCode(ref=ref, content=content, sha256=sha256_text(content))


def _http_get_text(url: str) -> str:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        raise ParentCodeError(f"Failed to fetch parent code (HTTP {e.code}): {url}") from e
    except urllib.error.URLError as e:
        raise ParentCodeError(f"Failed to fetch parent code (network error): {e}") from e
    return data.decode("utf-8", errors="replace")


_GITHUB_RE = re.compile(r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$")


def _parse_github_owner_repo(repo_url: str) -> Optional[Tuple[str, str]]:
    m = _GITHUB_RE.match(repo_url.strip())
    if not m:
        return None
    return m.group(1), m.group(2)
