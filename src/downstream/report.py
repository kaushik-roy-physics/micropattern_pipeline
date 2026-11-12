"""
The contact sheet: one self-contained HTML page holding every figure.

Eyeballing a run means looking at forty PNGs. Opening them one by one in a file
browser loses the grouping and the context; this page keeps them in analysis
order, with the QC section first (thresholds and field flags come before any
figure that depends on them) and the run's headline numbers at the top.

Images are referenced by relative path rather than embedded, so the page stays
small and reflects re-rendered figures on reload: it lives alongside the
``figures/`` tree it points at.
"""

from __future__ import annotations

import html
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

import provenance

logger = logging.getLogger(__name__)

# Section order and the human-readable blurb shown under each heading.
SECTIONS = [
    ("qc", "Quality control: colonies",
     "Is every colony found, whole, round and the right size? Flagged colonies are "
     "named on the overlays; the ones listed in qc.exclude_flags are left out of "
     "everything below. Read this before trusting any profile."),
    ("compare", "Conditions compared",
     "Condition mean profiles ± 95% CI across replicates, on the µm grid (distance "
     "from the edge) and the normalized grid. The global p tests whether the whole "
     "curves differ; the gray bar marks where along the radius they do (family-wise "
     "error controlled). Pairwise tests, profile features and colony similarity "
     "follow."),
    ("signaling_summary", "Signaling readouts",
     "Signaling markers (β-catenin, SMADs, ...) are read by where the protein is: "
     "nuclear intensity with the background subtracted, or the nuclear:cytoplasmic "
     "ratio from a ring around each nucleus. One point per colony for the whole "
     "colony, the edge band and the center, relative to the reference wells "
     "(dashed line at 1) when the plate has them. Their radial profiles are in the "
     "sections above and below, in the same units."),
    ("boundary", "Boundaries and peaks",
     "Per colony, a logistic fitted across each marker's transition gives its "
     "boundary (x50) and width (10-90 %); for two facing markers of one stain set, "
     "where their curves cross is the boundary, with the overlap (or gap) between "
     "them and the double-positive fraction near it (open circles: expected if the "
     "two were independent). For signaling markers, the interior peak and its offset "
     "from each boundary. Points are colonies with accepted fits; the black diamond is "
     "the mean ± 95 % CI across replicates. Per-colony uncertainty (tables) comes from "
     "resampling angular sectors of the colony. Roughness: the border fitted in each "
     "angular sector, its angular SD with decentering and fitting noise removed, and the "
     "transition width after registering every sector to its own border (the "
     "registered_* figures: dashed = angle-averaged, solid = registered)."),
    ("effects", "Effects, trends and power",
     "From every per-colony number. Trends: replicate values against a numeric condition "
     "column (dose, time, size) with the fitted slope, its bootstrap 95 % CI and a "
     "permutation p. Differences from the reference condition ± 95 % CI (hollow: too few "
     "replicates, colonies used). Power: the smallest difference two conditions could be "
     "shown to differ by, against the number of independent experiments, from the "
     "measured colony / well / experiment variances (a lower bound when the experiment "
     "variance cannot be estimated)."),
    ("boundary/borders", "Border maps (QC)",
     "Per colony: nuclei positive (blue) or negative (grey) for each boundary marker, the "
     "border fitted in each angular sector ± SE (vermillion) and the mean border (dashed)."),
    ("signaling", "Signaling rings (QC)",
     "Where the N:C ratio was measured: nucleus cores (orange) and the cytoplasmic "
     "ring inside cell bodies (sky blue), on a crop of each image. Rings should hug "
     "the nuclei and stay off the empty surface beyond the colony edge."),
    ("multimarker_profiles/wells", "All markers together: wells",
     "Every marker of a well on one normalized distance axis, each panel on its own "
     "y-range (shared scale), bold = well mean ± CI across colonies, thin = colonies. "
     "The strip underneath is the local trend (slope / SE): blue falls, red rises, "
     "arrows mark |z| at or above multimarker.z_crit (descriptive, not a test)."),
    ("coexpression/pages/wells", "Co-expression in space: wells",
     "Per marker pair: nuclei of every colony (the same number from each) colored by "
     "normalized distance from the edge, the same split into radial bands, the "
     "within-bin correlation and the double-positive fraction against what "
     "independence predicts, and the path of the bin means from edge to center. Each "
     "readout is computed per colony and then averaged (n = colonies)."),
    ("profiles", "Every colony, by well",
     "Each thin line is one colony; the bold line is the well mean. Dotted lines are "
     "colonies QC flagged but did not exclude."),
    ("thresholds", "Positivity thresholds",
     "The evidence behind each marker's gate, which the % positive profiles use."),
    ("colony_profiles", "Per-colony profiles",
     "One figure per colony on its own bins, in µm from the edge, with the "
     "within-colony confidence interval over nuclei."),
    ("multimarker_profiles/colonies", "All markers together: colonies",
     "Every marker of one colony stacked on the µm grid, each on its own y-range, "
     "with the local-trend strip (CI and SE over the nuclei of each bin)."),
    ("coexpression/pages/colonies", "Co-expression in space: colonies",
     "Per marker pair, in µm from the edge: the colony's nuclei colored by position, "
     "split into radial bands, the within-bin correlation, the double-positive "
     "fraction against independence, the edge-to-center path and, when both markers "
     "have a gate, the map of where each co-expression class sits."),
]

_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0;
       background: #fbfbfc; color: #1a1a1a; }
header { background: #0072B2; color: white; padding: 1.4rem 2rem; }
header h1 { margin: 0 0 .3rem; font-size: 1.35rem; }
header p { margin: 0; opacity: .9; font-size: .85rem; }
header p.version { margin-top: .3rem; opacity: .7; font-size: .75rem; }
header a { color: white; }
main { padding: 1.2rem 2rem 3rem; max-width: 1500px; margin: 0 auto; }
section { margin-bottom: 2.4rem; }
h2 { font-size: 1.05rem; border-bottom: 2px solid #0072B2; padding-bottom: .3rem;
     margin-bottom: .3rem; }
.blurb { font-size: .82rem; color: #555; margin: .3rem 0 1rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
        gap: 1rem; }
figure { margin: 0; background: white; border: 1px solid #e2e2e6; border-radius: 6px;
         padding: .6rem; }
figure img { width: 100%; height: auto; display: block; cursor: zoom-in; }
figcaption { font-size: .72rem; color: #666; margin-top: .4rem; word-break: break-all; }
table { border-collapse: collapse; font-size: .78rem; background: white; }
th, td { border: 1px solid #e2e2e6; padding: .3rem .55rem; text-align: left; }
th { background: #f0f2f5; }
.stats { display: flex; gap: 1.6rem; flex-wrap: wrap; margin: 1rem 0 1.6rem; }
.stat { background: white; border: 1px solid #e2e2e6; border-radius: 6px;
        padding: .7rem 1.1rem; min-width: 110px; }
.stat .value { font-size: 1.35rem; font-weight: 600; color: #0072B2; }
.stat .label { font-size: .72rem; color: #666; text-transform: uppercase;
               letter-spacing: .04em; }
.warn { background: #fff4e5; border-left: 4px solid #E69F00; padding: .7rem 1rem;
        font-size: .82rem; border-radius: 4px; margin-bottom: 1rem; }
a { color: #0072B2; }
@media (prefers-color-scheme: dark) {
  body { background: #16181c; color: #e8e8ea; }
  figure, table, .stat { background: #1e2126; border-color: #2c3038; }
  th { background: #24272e; }
  .blurb, figcaption, .stat .label { color: #a0a4ab; }
  .warn { background: #2e2517; }
}
"""


def _figure_cards(directory: Path, root: Path) -> str:
    if not directory.exists():
        return ""
    cards = []
    for png in sorted(directory.glob("*.png")):
        rel = png.relative_to(root).as_posix()
        cards.append(
            f'<figure><a href="{html.escape(rel)}" target="_blank">'
            f'<img src="{html.escape(rel)}" loading="lazy" alt="{html.escape(png.stem)}"></a>'
            f'<figcaption>{html.escape(png.name)}</figcaption></figure>'
        )
    return f'<div class="grid">{"".join(cards)}</div>' if cards else ""


def _anchor(folder: str) -> str:
    return folder.replace("/", "-")


def _stat_card(value, label) -> str:
    return f'<div class="stat"><div class="value">{value}</div><div class="label">{label}</div></div>'


def _table_html(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "<p><em>none</em></p>"
    shown = df.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in shown.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_fmt(v))}</td>" for v in row) + "</tr>"
        for row in shown.itertuples(index=False)
    )
    more = (f"<p><em>… {len(df) - max_rows} more row(s) in the CSV</em></p>"
            if len(df) > max_rows else "")
    return f"<table><tr>{head}</tr>{body}</table>{more}"


def _fmt(value) -> str:
    if isinstance(value, float):
        return "-" if pd.isna(value) else (f"{value:.4g}")
    return str(value)


def build_report(ctx) -> Path:
    """Render ``report.html`` at the analysis root."""
    root = ctx.out
    parts: list[str] = []

    # Headline numbers.
    stats_html = ""
    warnings: list[str] = []
    try:
        col = ctx.colonies
        kept = col[~col["excluded"].astype(bool)] if "excluded" in col else col
        cells = ctx.data.cells
        stats_html = "".join([
            _stat_card(f"{len(kept):,}", "colonies analyzed"),
            _stat_card(f"{int(col['flag_any'].sum()) if 'flag_any' in col else 0}",
                       "colonies flagged"),
            _stat_card(f"{len(col) - len(kept)}", "colonies excluded"),
            _stat_card(f"{kept['well'].nunique():,}", "wells"),
            _stat_card(f"{len(cells):,}", "nuclei in colonies"),
            _stat_card(f"{len(ctx.markers)}", "markers"),
        ])
        if len(col) - len(kept):
            names = ", ".join(Path(str(r)).name for r in col.loc[col["excluded"].astype(bool),
                                                                   "result_dir"])[:400]
            warnings.append(f"<strong>{len(col) - len(kept)} colony(ies) excluded</strong> "
                            f"by qc.exclude_flags: {html.escape(names)}")
    except Exception:
        logger.debug("report: colony tables unavailable for the summary bar", exc_info=True)

    try:
        import json
        with open(root / "tables" / "replicate_unit.json") as f:
            for label, rec in json.load(f).items():
                prefix = "" if label == "all" else f"[{html.escape(label)}] "
                warnings.append(prefix + html.escape(rec["note"]))
    except Exception:
        pass

    try:
        thr = ctx.thresholds
        graded = thr[thr["verdict"] != "ok"]
        if len(graded):
            names = ", ".join(sorted(set(graded["marker"].astype(str))))
            warnings.append(
                f"<strong>{len(graded)} marker/scope combination(s) have no positivity "
                f"gate</strong> ({html.escape(names)}): their % positive profiles are empty. "
                f"See the threshold evidence panel.")
    except Exception:
        pass
    warnings_html = "".join(f'<div class="warn">{w}</div>' for w in warnings)

    # Known sections in analysis order, then any group a custom step invented.
    known = {name for name, _, _ in SECTIONS}
    parents = {name.split("/")[0] for name in known}      # nested groups' top folders
    figures_root = root / "figures"
    extra = sorted(d.name for d in figures_root.iterdir()
                   if d.is_dir() and d.name not in known | parents) if figures_root.exists() else []
    sections = list(SECTIONS) + [(name, name.replace("_", " ").title(),
                                  "Produced by a custom analysis step.") for name in extra]

    for folder, title, blurb in sections:
        cards = _figure_cards(figures_root / folder, root)
        if not cards:
            continue
        parts.append(f'<section id="{_anchor(folder)}"><h2>{html.escape(title)}</h2>'
                     f'<p class="blurb">{html.escape(blurb)}</p>{cards}</section>')

    # Key tables inline.
    tables_html = ""
    table_specs = (
        ("compare_global.csv", "Global functional tests (one row per marker × grid × readout)",
         ["stratum", "facet", "grid", "readout", "marker", "groups", "unit", "n_replicates",
          "global_p", "exact", "smallest_possible_p", "significant_from", "significant_to"],
         None),
        ("compare_pairwise.csv", "Pairwise tests with q < 0.05",
         ["grid", "readout", "marker", "group_a", "group_b", "p_value", "q_value"],
         lambda d: d[d["significant"].astype(bool)]),
        ("feature_tests.csv", "Profile features that differ (q < 0.05)",
         ["marker", "feature", "kruskal_H", "p_value", "q_value", "n_replicates"],
         lambda d: d[d["q_value"] < 0.05]),
        ("boundary_summary.csv", "Boundaries and peaks: mean ± SE across replicates",
         ["condition", "kind", "subject", "readout", "metric", "unit", "n", "mean", "se",
          "ci_lo", "ci_hi", "n_colonies_rejected"],
         lambda d: d[d["metric"].isin(["crossing_um", "overlap_um", "dp_fraction",
                                       "dp_expected", "x50_um", "width_um", "peak_um",
                                       "peak_ratio"])
                     | d["metric"].astype(str).str.contains(" / ")].round(3)),
        ("effects_trends.csv", "Trends over numeric condition columns (p < 0.05)",
         ["column", "group", "family", "subject", "readout", "metric", "unit", "n_units",
          "scale", "slope", "slope_ci_lo", "slope_ci_hi", "p_perm", "tau", "note"],
         lambda d: d[d["p_perm"] < 0.05].round(4)),
        ("effects_variance.csv", "Variance components (SD at each level)",
         ["family", "subject", "readout", "metric", "sd_colony", "sd_well", "sd_experiment",
          "n_colonies", "n_wells", "n_experiments"],
         lambda d: d[d["family"] != "feature"].round(3)),
        ("similarity_within_between.csv", "Colony similarity: within vs between conditions",
         None, None),
        ("qc_colonies.csv", "Flagged colonies",
         ["field_id", "dataset", "well", "result_dir", "diameter_um", "n_nuclei_colony",
          "border_fraction", "fill_ratio", "flags", "excluded"],
         lambda d: d[d["flag_any"].astype(bool)]),
        ("thresholds.csv", "Positivity thresholds", None, None),
        ("signaling_reference.csv", "Signaling reference levels",
         ["marker", "scope_key", "readout", "reference_values", "reference",
          "n_reference_colonies", "source", "relative"], None),
    )
    for name, caption, cols, filt in table_specs:
        path = root / "tables" / name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except (pd.errors.EmptyDataError, OSError):
            continue
        if filt is not None:
            try:
                df = filt(df)
            except Exception:
                pass
        if cols:
            df = df[[c for c in cols if c in df.columns]]
        tables_html += (f"<h3>{html.escape(caption)} "
                        f'<a href="tables/{name}">({name})</a></h3>' + _table_html(df))
    if tables_html:
        parts.append(f'<section id="tables"><h2>Tables</h2>{tables_html}</section>')

    nav = " · ".join(f'<a href="#{_anchor(f)}">{html.escape(t)}</a>'
                     for f, t, _ in SECTIONS if _figure_cards(root / "figures" / f, root))

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Micropattern analysis: {html.escape(root.parent.name)}</title>
<style>{_CSS}</style></head>
<body>
<header>
  <h1>Micropattern radial analysis: {html.escape(root.parent.name)}</h1>
  <p>Generated {datetime.now():%Y-%m-%d %H:%M} · colonies (or wells) are the replicates ·
     {nav}</p>
  <p class="version">{html.escape(provenance.summary())}</p>
</header>
<main>
<div class="stats">{stats_html}</div>
{warnings_html}
{"".join(parts)}
</main></body></html>"""

    out = root / "report.html"
    out.write_text(page, encoding="utf-8")
    logger.info("report → %s", out)
    return out


__all__ = ["build_report"]
