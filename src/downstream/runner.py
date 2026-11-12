"""
Runner: resolve the step list, honor checkpoints, execute, keep going on failure.

One failing figure must never cost you the rest of a run, so each step is wrapped
in its own try/except with a full traceback in the log; the runner reports how
many steps succeeded and returns a non-zero exit code if any failed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from . import steps as _builtin_steps  # noqa: F401  (registers the built-in steps)
import provenance

from . import style
from .context import Context
from .registry import (REGISTRY, checkpoint_path, default_step_names, is_current,
                       load_extra_modules, order_steps, write_checkpoint)

logger = logging.getLogger(__name__)


def run(cfg: dict, datasets, out_dir: Path, step_names=None, force: bool = False,
        force_steps=(), experiment_root: Path | None = None) -> int:
    """Execute the requested downstream steps. Returns the number that failed."""
    down_cfg = cfg.get("downstream") or {}

    # Experiment-specific analyses register themselves on import.
    load_extra_modules(down_cfg.get("extra_modules"),
                       search_paths=[experiment_root] if experiment_root else [])

    requested = list(step_names or down_cfg.get("steps") or default_step_names())
    # Optional analyses switched off in the config (GUI step 3, "Analyses to run").
    skipped = _builtin_steps.resolve_skips(down_cfg.get("skip_steps"))
    if skipped:
        logger.info("skipping (downstream.skip_steps): %s", ", ".join(sorted(skipped)))
    requested = [s for s in requested if s not in skipped]
    ordered = order_steps(requested)
    force_steps = set(force_steps or ())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance.write(out_dir)          # which code wrote these results
    ctx = Context(down_cfg, datasets, out_dir, force=force)
    ctx.skipped = skipped
    for name in sorted(skipped):
        _builtin_steps.set_aside(out_dir, name)

    # Install the publication rcParams for the whole run. Without this every
    # figure is drawn at matplotlib's defaults: `visualization.dpi` does nothing,
    # the boxed spines stay on, and text is saved as paths rather than as the
    # Type-42 fonts that keep a PDF editable in Illustrator.
    vis_cfg = down_cfg.get("visualization") or {}
    dpi = int(vis_cfg.get("dpi", 300))
    style.apply_style(dpi=dpi, font_scale=float(vis_cfg.get("font_scale", 1.0)))
    logger.info("figure style applied: %d dpi", dpi)

    logger.info("=== downstream: %d step(s) → %s ===", len(ordered), out_dir)
    n_failed = n_skipped = n_ran = 0

    for step in ordered:
        forced = force or step.name in force_steps
        stale_dep = _newer_dependency(out_dir, step)
        if step.name == "report" and n_ran and not stale_dep:
            stale_dep = "an earlier step"           # any new figure belongs in the report
        if stale_dep and not forced:
            logger.info("[stale] %-13s '%s' was re-run after it: re-running", step.name, stale_dep)
        current = is_current(out_dir, step, down_cfg)
        if not forced and not stale_dep and current and not step.always:
            logger.info("[skip] %-14s up to date (use --force-step %s to redo)",
                        step.name, step.name)
            n_skipped += 1
            continue
        logger.info("[run ] %-14s %s", step.name, step.description)
        start = time.perf_counter()
        # Steps with their own per-item checkpoints (colony) need to know that
        # THIS step was forced, not just whether --force was given.
        ctx.step_forced = forced
        try:
            changed = step.func(ctx)
        except Exception:
            logger.exception("Step %r failed: continuing with the remaining steps.", step.name)
            n_failed += 1
            continue
        if changed is False and current and not stale_dep:
            n_skipped += 1
            logger.info("[same] %-14s nothing new (%.1fs)", step.name, time.perf_counter() - start)
            continue
        write_checkpoint(out_dir, step, down_cfg)
        n_ran += 1
        logger.info("[done] %-14s %.1fs", step.name, time.perf_counter() - start)

    logger.info("=== downstream finished: %d ran, %d skipped, %d failed ===",
                n_ran, n_skipped, n_failed)
    if n_failed == 0 and (out_dir / "report.html").exists():
        logger.info("Open the contact sheet: %s", out_dir / "report.html")
    return n_failed


def _newer_dependency(out_dir: Path, step) -> str | None:
    """Name of a dependency whose checkpoint is newer than *step*'s, if any.

    A step's own config hash cannot see that its INPUT changed: re-running
    ``profiles`` with a new bin width leaves ``aggregate``'s settings untouched,
    yet every table it wrote is now stale. Comparing checkpoint times closes that
    hole, and because each re-run step rewrites its checkpoint, the staleness
    propagates down the whole chain in one run.
    """
    own = checkpoint_path(out_dir, step.name)
    if not own.exists():
        return None
    mine = own.stat().st_mtime
    for dep in step.after:
        path = checkpoint_path(out_dir, dep)
        if path.exists() and path.stat().st_mtime > mine:
            return dep
    return None


def describe_steps() -> str:
    """Human-readable listing of every registered step, in dependency order."""
    lines = ["Registered downstream steps (dependency order):", ""]
    for name in default_step_names():
        step = REGISTRY[name]
        after = f"  [after: {', '.join(step.after)}]" if step.after else ""
        lines.append(f"  {name:<15} {step.description}{after}")
    return "\n".join(lines)


__all__ = ["run", "describe_steps"]
