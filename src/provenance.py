"""
Which version of the pipeline produced a result.

The release number lives in the ``VERSION`` file at the repository root (bumped
by hand at each release, see CONTRIBUTING.md). When the code is a git checkout,
the exact commit and whether it had uncommitted edits are added, so every log,
``pipeline_version.json`` and report can be traced back to the code that wrote
it. Nothing here fails: without git the commit is read from ``.git`` directly,
and without either it says "unknown".

This file is identical in micropattern_pipeline and analysis_pipeline_v2.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def name() -> str:
    """Which pipeline this is, whatever folder a clone or worktree sits in."""
    if (ROOT / "src" / "downstream" / "colony.py").exists():
        return "micropattern_pipeline"
    if (ROOT / "src" / "downstream" / "gating.py").exists():
        return "analysis_pipeline_v2"
    return ROOT.name


def version() -> str:
    try:
        return (ROOT / "VERSION").read_text().strip() or "unknown"
    except OSError:
        return "unknown"


def _git(*args: str) -> str | None:
    # cwd= rather than `git -C`: the cluster login node has git 1.8.3.
    # GIT_OPTIONAL_LOCKS=0 keeps `status` from writing the index of a shared checkout.
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                             timeout=90, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _commit_from_files() -> str | None:
    """HEAD's commit without running git (a compute node may not have it)."""
    git_dir = ROOT / ".git"
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        loose = git_dir / ref
        if loose.exists():
            return loose.read_text().strip()
        for line in (git_dir / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return None


@lru_cache(maxsize=1)
def info() -> dict:
    commit = _git("rev-parse", "HEAD") or _commit_from_files()
    status = _git("status", "--porcelain", "--untracked-files=no") if commit else None
    return {
        "pipeline": name(),
        "version": version(),
        "commit": commit,
        # No --dirty here: it re-checks every file a second time (~45 s over gvfs);
        # uncommitted_changes carries that answer.
        "describe": _git("describe", "--tags", "--always") if commit else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") if commit else None,
        "uncommitted_changes": None if status is None else bool(status),
        "code_dir": str(ROOT),
    }


def summary() -> str:
    """One line, e.g. ``micropattern_pipeline 1.0.0 (v1.0.0-3-g1a2b3c4, uncommitted changes)``."""
    i = info()
    where = i["describe"] or (i["commit"] or "")[:7] or "not a git checkout"
    if i["uncommitted_changes"]:
        where += ", uncommitted changes"
    return f"{i['pipeline']} {i['version']} ({where})"


def write(out_dir: Path) -> Path:
    """Write ``pipeline_version.json`` into a results folder."""
    path = Path(out_dir) / "pipeline_version.json"
    path.write_text(json.dumps({**info(), "run_at": datetime.now().isoformat(timespec="seconds")},
                               indent=2) + "\n")
    return path


__all__ = ["version", "info", "summary", "write"]
