from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def code_provenance() -> dict[str, object]:
    """Capture code/runtime identity without environment values or Git diff text."""
    package_dir = Path(__file__).resolve().parents[1]
    project_dir = package_dir.parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py")):
        digest.update(path.relative_to(package_dir).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")

    git_sha: str | None = os.environ.get("GITHUB_SHA") or None
    dirty: bool | None = None
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        pass

    packages: dict[str, str | None] = {}
    for name in ("anthropic", "fastembed", "pgvector", "psycopg", "fastapi"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    lock_path = project_dir / "uv.lock"
    return {
        "git_sha": git_sha,
        "git_dirty": dirty,
        "source_sha256": digest.hexdigest(),
        "dependency_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest()
        if lock_path.is_file()
        else None,
        "python_version": platform.python_version(),
        "package_versions": packages,
    }
