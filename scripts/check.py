"""Security/hygiene gate: fails (exit 1) if secrets, live credentials or AI/LLM code are present.

python -m scripts.check
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".yaml",
    ".yml",
    ".txt",
    ".toml",
    ".js",
    ".html",
    ".css",
    ".service",
    ".plist",
    ".example",
    ".cfg",
    ".ini",
    "",
}
SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)(token|secret|password|api[_-]?key)\w*\s*[=:]\s*["'][A-Za-z0-9_\-\.]{20,}["']"""
)
KNOWN_TOKEN_SHAPES = re.compile(
    r"(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|eyJ[A-Za-z0-9_\-]{20,}\.)"
)
LLM_IMPORT = re.compile(
    r"^\s*(import|from)\s+(openai|anthropic|google\.generativeai|langchain\w*|llama_index|ollama|"
    r"mistralai|cohere)\b",
    re.MULTILINE,
)
ALLOWED_PLACEHOLDERS = ("YOUR_", "placeholder", "example", "xxxx", "test", "stub", "dummy")


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    return [REPO / f for f in out if (REPO / f).is_file() and (REPO / f).suffix in TEXT_SUFFIXES]


def main() -> int:
    problems: list[str] = []
    for path in tracked_files():
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(("tests/", ".venv/")) or rel == "scripts/check.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            if any(p.lower() in low for p in ALLOWED_PLACEHOLDERS):
                continue
            is_literal = "get_secret_value" not in line and "os.environ" not in line
            if SECRET_ASSIGNMENT.search(line) and is_literal and "(" not in line.split("=")[-1][:3]:
                problems.append(f"{rel}:{n}: possible hard-coded secret")
            if KNOWN_TOKEN_SHAPES.search(line):
                problems.append(f"{rel}:{n}: token-shaped string")
        if path.suffix == ".py" and LLM_IMPORT.search(text):
            problems.append(f"{rel}: imports an AI/LLM library (forbidden)")
    for forbidden in (".env",):
        if (REPO / forbidden).exists() and forbidden in subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=False
        ).stdout.split():
            problems.append(f"{forbidden} is tracked by git")
    env_example = (REPO / ".env.example").read_text(encoding="utf-8")
    if "ALLOW_LIVE=false" not in env_example:
        problems.append(".env.example must ship with ALLOW_LIVE=false")
    for key in ("DERIV_LIVE_TOKEN", "DERIV_LIVE_ACCOUNT_ID"):
        m = re.search(rf"^{key}=(.*)$", env_example, re.MULTILINE)
        if m is None or m.group(1).strip():
            problems.append(f".env.example must leave {key} empty")
    if problems:
        print("SECURITY CHECK FAILED")
        print("\n".join(f"  - {p}" for p in problems))
        return 1
    print("security check passed: no secrets, no live credentials, no AI/LLM imports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
