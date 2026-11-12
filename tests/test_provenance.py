"""
Version stamping: VERSION is read, a git checkout's commit is found with or
without the git program, and results folders get pipeline_version.json.
Identical in micropattern_pipeline and analysis_pipeline_v2.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import provenance  # noqa: E402


def test_version_matches_file_and_cli():
    v = (ROOT / "VERSION").read_text().strip()
    assert provenance.version() == v
    out = subprocess.run([sys.executable, str(ROOT / "main.py"), "--version"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and v in out.stdout, out
    import experiment
    import image_processing
    assert experiment.__version__ == image_processing.__version__ == v
    print(f"ok  version {v} from VERSION, --version and the packages")


def test_commit_without_git():
    if not (ROOT / ".git").exists():
        print("skip  not a git checkout")
        return
    with_git = provenance._git("rev-parse", "HEAD")
    if not with_git and not provenance._commit_from_files():
        print("skip  repository has no commits yet")
        return
    if with_git:
        assert provenance._commit_from_files() == with_git
    assert provenance.info()["commit"]
    print("ok  commit read from .git matches git rev-parse")


def test_results_folder_stamp():
    out = Path(tempfile.mkdtemp())
    path = provenance.write(out)
    d = json.loads(path.read_text())
    assert d["version"] == provenance.version() and d["pipeline"] in ("micropattern_pipeline", "analysis_pipeline_v2")
    assert {"commit", "describe", "uncommitted_changes", "run_at"} <= set(d)
    assert provenance.version() in provenance.summary()
    print("ok  pipeline_version.json:", provenance.summary())


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
