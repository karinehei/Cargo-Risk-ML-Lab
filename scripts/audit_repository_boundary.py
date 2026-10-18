"""Filesystem repository-boundary checks that do not require a Git worktree."""

from __future__ import annotations

import argparse

from src.config import PROJECT_ROOT
from src.pages.schema import unexpected_site_files
from src.pages.verify import VerificationError, verify_site


def gitignore_text() -> str:
    return (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")


def dockerignore_text() -> str:
    return (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")


def required_gitignore_tokens() -> list[str]:
    return [
        ".env",
        "mlruns/",
        "mlartifacts/",
        "data/raw/*",
        "data/processed/*",
        "data/monitoring/*",
        "*.pkl",
        "*.joblib",
        ".venv/",
        "__pycache__/",
        ".ci-work/",
    ]


def required_dockerignore_tokens() -> list[str]:
    return [".env", "data", "artifacts", "mlruns", "tests", ".venv"]


def audit() -> dict[str, object]:
    gi = gitignore_text()
    di = dockerignore_text()
    missing_gi = [token for token in required_gitignore_tokens() if token not in gi]
    missing_di = [token for token in required_dockerignore_tokens() if token not in di]
    env_files = [str(path.relative_to(PROJECT_ROOT)) for path in PROJECT_ROOT.glob(".env")]
    site_dir = PROJECT_ROOT / "site"
    unexpected = unexpected_site_files(site_dir)
    pages_error = ""
    if (site_dir / "index.html").is_file():
        try:
            verify_site(site_dir)
        except VerificationError as exc:
            pages_error = str(exc)
    return {
        "gitignore_missing": missing_gi,
        "dockerignore_missing": missing_di,
        "env_files_present": env_files,
        "git_worktree": (PROJECT_ROOT / ".git").exists(),
        "unexpected_site_files": unexpected,
        "pages_verification_error": pages_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repository boundary filesystem audit")
    parser.parse_args()
    payload = audit()
    if (
        payload["gitignore_missing"]
        or payload["dockerignore_missing"]
        or payload["unexpected_site_files"]
        or payload["pages_verification_error"]
    ):
        raise SystemExit(f"Repository boundary gaps: {payload}")
    print("repository_boundary_ok")
    if not payload["git_worktree"]:
        print(
            "git_worktree_absent: git ls-files and history scanning require the real Git repository"
        )


if __name__ == "__main__":
    main()
