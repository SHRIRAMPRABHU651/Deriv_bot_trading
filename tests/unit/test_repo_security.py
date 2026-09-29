"""Repository-level safety checks: no secrets, no AI/LLM deps, single buy path, LIVE defaults."""

from __future__ import annotations

import re
import subprocess
from decimal import Decimal
from pathlib import Path

from app.config import Settings
from app.risk.limits import compute_stake

REPO = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ("app", "research", "scripts", "tests")
FORBIDDEN_LIBS = re.compile(
    r"\b(openai|anthropic|google\.generativeai|genai|langchain|llama_index|ollama|mistralai|cohere)\b",
    re.IGNORECASE,
)
SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)(token|secret|password|api[_-]?key)\w*\s*[=:]\s*["'][A-Za-z0-9_\-\.]{20,}["']"""
)


def source_files() -> list[Path]:
    files: list[Path] = []
    for d in SOURCE_DIRS:
        files += [p for p in (REPO / d).rglob("*") if p.suffix in {".py", ".js", ".html"}]
    return files


def test_no_llm_or_ai_dependencies() -> None:
    for name in ("requirements.txt", "requirements-dev.txt", "pyproject.toml"):
        text = (REPO / name).read_text()
        assert not FORBIDDEN_LIBS.search(text), name
    for path in source_files():
        if path.name == "test_repo_security.py":
            continue
        text = path.read_text()
        for line in text.splitlines():
            if line.lstrip().startswith(("import ", "from ")):
                assert not FORBIDDEN_LIBS.search(line), f"{path}: {line}"


def test_no_hardcoded_credentials_in_source() -> None:
    for path in source_files():
        assert not SECRET_ASSIGNMENT.search(path.read_text()), path


def test_secrets_files_are_ignored_and_not_tracked() -> None:
    ignore = (REPO / ".gitignore").read_text().splitlines()
    for required in (
        ".env",
        "*.db",
        "logs/",
        ".venv/",
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        "model/*.joblib",
    ):
        assert required in ignore, required
    env_example = (REPO / ".env.example").read_text()
    assert "ALLOW_LIVE=false" in env_example
    for line in env_example.splitlines():
        if line.startswith(("DERIV_LIVE_TOKEN", "DERIV_LIVE_ACCOUNT_ID")):
            assert line.split("=", 1)[1].strip() == ""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.split()
    assert ".env" not in tracked
    assert not [t for t in tracked if t.endswith(".db")]


def test_live_is_off_by_default() -> None:
    s = Settings(_env_file=None)
    assert s.allow_live is False and not s.live_permitted()


def test_only_the_executor_calls_client_buy() -> None:
    """There must be no code path that sends a Deriv buy without going through RiskManager."""
    offenders = []
    for path in (REPO / "app").rglob("*.py"):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"(?<!protocol)\.buy\(", line) and not line.strip().startswith("#"):
                rel = path.relative_to(REPO).as_posix()
                if rel not in {"app/execution/executor.py"}:
                    offenders.append(f"{rel}:{n}")
    assert offenders == []
    client_src = (REPO / "app/deriv/client.py").read_text()
    assert "permit" in client_src and "PermitError" in client_src


def test_no_martingale_stake_only_shrinks_after_losses() -> None:
    balance = Decimal("1000")
    stakes = []
    for _ in range(6):
        stakes.append(compute_stake(balance, Decimal("0.01"), Decimal("0.01"), 2))
        balance -= stakes[-1]  # a loss
    assert stakes == sorted(stakes, reverse=True) and stakes[0] > stakes[-1]
    assert "martingale" not in (REPO / "app/risk/manager.py").read_text().lower()
