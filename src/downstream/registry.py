"""
Step registry: how the downstream pipeline stays extensible without being edited.

A step is a function ``f(ctx) -> None`` registered with a name, the steps it must
run after, and the config keys it depends on::

    @register("composition", after=["gating"], config_keys=["gating", "visualization"])
    def composition(ctx):
        ...

Experiment-specific analyses live **outside this package** and are pulled in by
config::

    downstream:
      extra_modules: [downstream_analysis.custom]
      steps: [assemble, qc, thresholds, distributions, my_fate_calls]

Importing that module executes its ``@register`` calls, after which its steps are
first-class: ordered by their declared dependencies, checkpointed, and selectable
with ``--steps`` / ``--force-step`` exactly like the built-ins. Nothing in the
generic pipeline needs to change to add an analysis, which is the whole point.

**Checkpointing.** Each step writes ``.checkpoints/<name>.json`` recording a hash
of the config subtrees it declared. A step is skipped when its checkpoint exists
and the hash still matches, so re-running the pipeline redoes only what is stale:
change ``thresholds.fpr`` and the threshold-dependent steps re-run while
``assemble`` (the expensive one) does not. ``--force-step`` overrides one step,
``--force`` overrides all.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = ".checkpoints"


@dataclass
class Step:
    name: str
    func: Callable
    after: list[str] = field(default_factory=list)
    config_keys: list[str] | None = None
    description: str = ""
    # Always invoked, even when its checkpoint is current: for steps that keep
    # their own per-item checkpoints (``colony``) and are cheap when nothing is
    # new. Such a step returns False to say "nothing changed", and the runner then
    # leaves its checkpoint alone so dependants are not marked stale.
    always: bool = False


REGISTRY: dict[str, Step] = {}


def register(name: str, after: list[str] | None = None,
             config_keys: list[str] | None = None, description: str = "",
             always: bool = False):
    """Decorator registering an analysis step under *name*."""
    def decorator(func: Callable) -> Callable:
        if name in REGISTRY:
            logger.warning("Step %r re-registered (overriding the previous definition).", name)
        REGISTRY[name] = Step(name=name, func=func, after=list(after or []),
                              config_keys=config_keys, always=always,
                              description=description or (func.__doc__ or "").strip().split("\n")[0])
        return func
    return decorator


def load_extra_modules(module_names, search_paths=()) -> None:
    """Import experiment-specific modules so their ``@register`` calls run."""
    import sys
    for path in search_paths:
        path = str(Path(path).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    for name in module_names or []:
        try:
            importlib.import_module(str(name))
            logger.info("Loaded extra analysis module %s", name)
        except Exception:
            logger.exception("Could not import extra module %r: its steps are unavailable.", name)


def order_steps(names: list[str]) -> list[Step]:
    """Topologically order the requested steps by their declared dependencies."""
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown step(s) {unknown}. Registered: {sorted(REGISTRY)}")

    requested = set(names)
    ordered: list[Step] = []
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(name: str) -> None:
        if name in done:
            return
        if name in visiting:
            raise ValueError(f"Circular dependency involving step {name!r}.")
        visiting.add(name)
        for dep in REGISTRY[name].after:
            # Dependencies not explicitly requested are not run: the step reads
            # their outputs from disk instead (that is what checkpoints are for).
            if dep in requested:
                visit(dep)
        visiting.discard(name)
        done.add(name)
        ordered.append(REGISTRY[name])

    for name in names:
        visit(name)
    # The report collects every other step's figures, so it always runs last:
    # also after experiment-specific steps registered after it.
    last = [s for s in ordered if s.name == "report"]
    return [s for s in ordered if s.name != "report"] + last


def default_step_names() -> list[str]:
    """Every registered step, dependency-ordered."""
    return [s.name for s in order_steps(list(REGISTRY))]


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def config_hash(cfg: dict, keys: list[str] | None) -> str:
    """Stable hash of the config subtrees a step declared it depends on."""
    subset = cfg if keys is None else {k: cfg.get(k) for k in keys}
    blob = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def checkpoint_path(out_dir: Path, name: str) -> Path:
    return Path(out_dir) / CHECKPOINT_DIR / f"{name}.json"


def is_current(out_dir: Path, step: Step, cfg: dict) -> bool:
    path = checkpoint_path(out_dir, step.name)
    if not path.exists():
        return False
    try:
        with open(path) as f:
            stored = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return stored.get("config_hash") == config_hash(cfg, step.config_keys)


def write_checkpoint(out_dir: Path, step: Step, cfg: dict, outputs=()) -> None:
    from datetime import datetime
    path = checkpoint_path(out_dir, step.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "step": step.name,
            "config_hash": config_hash(cfg, step.config_keys),
            "completed": datetime.now().isoformat(timespec="seconds"),
            "outputs": [str(o) for o in outputs if o is not None],
        }, f, indent=2)


__all__ = ["register", "REGISTRY", "Step", "order_steps", "default_step_names",
           "load_extra_modules", "is_current", "write_checkpoint", "config_hash",
           "checkpoint_path", "CHECKPOINT_DIR"]
