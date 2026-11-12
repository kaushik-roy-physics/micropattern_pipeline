"""
Micropattern pipeline: point-and-click set-up.

Start it with the launcher for your computer (launch_gui.bat / .command / .sh),
or by hand:

    streamlit run gui/app.py

Five steps, in the sidebar:

1. Experiment      pick the experiment folder; the acquisition folders in it are found
2. Plate maps      say what is in each well: writes <folder>/plate_map.yaml
3. Analysis        choose what the analysis compares: writes config.yaml
4. Check & run     validate everything; optionally run the CPU steps on this computer
5. HPC job         writes hpc_job.env and shows the commands to type on the HPC

The GUI only writes the three text files the pipeline reads. Everything it
writes can be opened again here, or edited by hand.
"""

from __future__ import annotations

import copy
import html
import os
import shutil
import sys
import webbrowser
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

from selection import parse_index_spec, parse_well_spec  # noqa: E402  (core put src on the path)
from downstream.signaling import catalog as sig_catalog  # noqa: E402

st.set_page_config(page_title="Micropattern pipeline", page_icon="🔬", layout="wide")
ss = st.session_state

STEPS = ["1 · Experiment", "2 · Plate maps", "3 · Analysis settings",
         "4 · Check & run", "5 · HPC job"]

# Okabe-Ito: the lab's color-vision-safe palette (same as downstream/style.py).
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00",
             "#F0E442", "#000000"]


# ===========================================================================
# Session state + small UI helpers
# ===========================================================================

def init_state() -> None:
    ss.setdefault("prefs", core.load_prefs())
    ss.setdefault("exp_dir", ss.prefs.get("last_experiment", ""))
    ss.setdefault("loaded", False)
    ss.setdefault("scans", {})          # rel dir -> FolderScan
    ss.setdefault("datasets", [])       # [{include, name, directory, seg_mode, extra}]
    ss.setdefault("pm", {})             # dataset name -> plate-map state
    ss.setdefault("pm_saved", {})       # dataset name -> text last written/read
    ss.setdefault("cfg", None)          # merged config values
    ss.setdefault("template", core.ConfigTemplate())
    ss.setdefault("gen", 0)
    ss.setdefault("_exp_input", ss.exp_dir)


def editor(name: str, make_df, **kwargs) -> pd.DataFrame:
    """A data editor whose starting table changes only on explicit actions.

    Feeding an editor its own output every rerun makes it re-apply edits (and
    duplicate added rows), so the starting table is kept separately and
    replaced, with a fresh widget, only through :func:`reset_editor`.

    Streamlit forgets a table's edits when its page is not shown (another step
    in the sidebar, another folder in a dropdown). The starting table would then
    come back without them and overwrite what was typed. So whenever the
    widget's own state is gone, the starting table is rebuilt from the current
    values (*make_df*), which already hold every edit.
    """
    base, ver = f"_base_{name}", f"_ver_{name}"
    widget = f"_ed_{name}_{ss.get(ver, 0)}"
    keep = ss.pop(f"_keep_{name}", False)
    if base not in ss or (widget not in ss and not keep):
        ss[base] = make_df()
        ss[ver] = ss.get(ver, 0) + 1
    return st.data_editor(ss[base], key=f"_ed_{name}_{ss[ver]}", **kwargs)


def reset_editor(name: str, df: pd.DataFrame | None = None) -> None:
    """Start the table again: from the current values, or from *df* when given."""
    base, ver = f"_base_{name}", f"_ver_{name}"
    if df is None:
        ss.pop(base, None)
    else:
        ss[base] = df
        ss[ver] = ss.get(ver, 0) + 1
        ss[f"_keep_{name}"] = True        # the next run shows *df*, not a rebuild


def reset_editors(prefix: str) -> None:
    for k in [k for k in ss.keys() if str(k).startswith(f"_base_{prefix}")]:
        ss.pop(k, None)


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame → list of dicts with NaN turned into None."""
    return [{k: (None if (isinstance(v, float) and v != v) else v) for k, v in r.items()}
            for r in df.to_dict("records")]


def exp_path() -> Path:
    return Path(ss.exp_dir).expanduser()


def included() -> list[dict]:
    return [d for d in ss.datasets if d.get("include", True)]


def backup(path: Path) -> None:
    """Keep the previous version of a file we are about to overwrite."""
    if path.exists():
        try:
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        except OSError:
            pass


def save_with_confirm(path: Path, text: str, key: str, loaded_text: str | None,
                      label: str, disabled: bool = False) -> bool:
    """A save button that never silently replaces a file already on disk.

    A new file is written at once. An existing one needs a second, explicit
    click, with a warning that names it, says the old version is kept as
    ``.bak``, and says so louder when the file changed outside the GUI since it
    was opened here (hand edits would be lost). Returns True when written.
    """
    confirm = f"_confirm_{key}"
    if st.button(label, type="primary", disabled=disabled, key=f"_save_{key}"):
        if path.exists():
            ss[confirm] = True
        else:
            return _write_file(path, text)
    if not ss.get(confirm):
        return False
    import datetime
    try:
        when = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        on_disk = path.read_text(encoding="utf-8")
    except OSError:
        when, on_disk = "?", None
    msg = (f"**{path.name} already exists** in `{path.parent}` (last changed {when}). "
           f"Saving replaces it; the current file is kept as `{path.name}.bak`.")
    if loaded_text is not None and on_disk is not None and on_disk != loaded_text:
        msg += ("\n\n⚠️ **It was changed outside the GUI since it was opened here.** "
                "Those changes will be lost unless you reload the experiment first "
                "(step 1, *Rescan folders*).")
    elif loaded_text is None and on_disk is not None:
        msg += "\n\n⚠️ The GUI did not load this file, so its content will be replaced."
    st.warning(msg)
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Yes, overwrite", type="primary", key=f"_yes_{key}"):
        ss.pop(confirm, None)
        return _write_file(path, text)
    if c2.button("Cancel", key=f"_no_{key}"):
        ss.pop(confirm, None)
        st.rerun()
    return False


def _write_file(path: Path, text: str) -> bool:
    try:
        backup(path)
        core.write_text(path, text)
    except OSError as exc:
        st.error(f"Could not write {path}: {exc}")
        return False
    st.toast(f"Saved {path}")
    return True


def text_color(hex_color: str) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000000" if lum > 0.45 else "#FFFFFF"


def need_experiment() -> bool:
    if not ss.loaded:
        st.info("Open an experiment folder in step **1 · Experiment** first.")
        return False
    return True


# ===========================================================================
# 1 · Experiment
# ===========================================================================

def folder_browser() -> None:
    """A tiny folder browser: works the same on Windows, macOS and Linux."""
    with st.expander("Browse for the folder", expanded=not ss.exp_dir):
        cur = Path(ss.get("_browse", ss.exp_dir or Path.home())).expanduser()
        if not cur.exists():
            cur = Path.home()
        st.caption(f"Looking in: `{cur}`")
        shortcuts = core.folder_shortcuts()

        def jump() -> None:
            ss["_browse"] = shortcuts[ss["_browse_jump"]]
            ss["_browse_jump"] = "-"

        st.selectbox("Jump to", ["-", *shortcuts], key="_browse_jump", on_change=jump,
                     help="Your home folder and the drives and network shares mounted "
                          "on this computer.")
        try:
            subdirs = sorted([d.name for d in cur.iterdir()
                              if d.is_dir() and not d.name.startswith(".")], key=str.lower)
        except OSError as exc:
            subdirs = []
            st.warning(f"Cannot list this folder: {exc}")
        c1, c2, c3 = st.columns([3, 1, 1])
        pick = c1.selectbox("Sub-folder", ["-"] + subdirs, key="_browse_pick",
                            label_visibility="collapsed")
        if c2.button("Open ↓", disabled=pick == "-", width="stretch"):
            ss["_browse"] = str(cur / pick)
            st.rerun()
        if c3.button("Up ↑", width="stretch"):
            ss["_browse"] = str(cur.parent)
            st.rerun()
        if st.button(f"Use this folder: {cur.name or cur}", type="primary"):
            ss.exp_dir = str(cur)
            ss["_exp_input"] = str(cur)
            with st.spinner("Scanning for images (slow on a network share the first time)…"):
                load_experiment(cur)
            st.rerun()


def load_experiment(path: Path) -> None:
    """Scan the folder and read any config / plate maps already there."""
    cfg_path = path / core.CONFIG_NAME
    user_cfg = core.read_yaml(cfg_path) if cfg_path.exists() else {}
    ss.cfg_loaded_text = cfg_path.read_text(encoding="utf-8") if cfg_path.exists() else None
    ss.cfg = ss.template.merged(user_cfg)
    cfg_input = core.get_in(ss.cfg, ["defaults", "input"], {})

    scans = {s.rel: s for s in core.find_dataset_folders(path, cfg_input)}
    datasets: list[dict] = []
    default_mode = core.get_in(ss.cfg, ["defaults", "segmentation", "mode"]) or "2d"
    for entry in user_cfg.get("datasets") or []:
        entry = dict(entry)
        rel = str(entry.pop("directory"))
        rel = rel if rel.startswith((".", "/")) or ":" in rel else f"./{rel}"
        try:
            rel_norm = os.path.relpath((path / rel).resolve(), path).replace("\\", "/")
            rel = "." if rel_norm == "." else f"./{rel_norm}"
        except ValueError:
            pass
        if rel not in scans and (path / rel).exists():
            scans[rel] = core.scan_folder((path / rel).resolve(), path, cfg_input)
        name = str(entry.pop("name", Path(rel).name))
        mode = core.get_in(entry, ["segmentation", "mode"]) or default_mode
        datasets.append({"include": True, "name": name, "directory": rel,
                         "seg_mode": str(mode).lower(), "extra": entry})
    known = {d["directory"] for d in datasets}
    for rel, s in scans.items():
        if rel not in known:
            datasets.append({"include": not user_cfg.get("datasets"), "name": s.name,
                             "directory": rel, "seg_mode": str(default_mode).lower(),
                             "extra": {}})

    ss.scans, ss.datasets, ss.pm, ss.pm_saved = scans, datasets, {}, {}
    ss["env_loaded_text"] = {}
    reset_editors("")
    for key in [k for k in ss.keys() if str(k).startswith("_confirm_")]:
        ss.pop(key, None)
    ss.gen = ss.get("gen", 0) + 1         # fresh widgets: no values leak between experiments
    ss.loaded = True
    # Read every existing plate map now, so all pages start from what is on disk.
    ss.load_notes = []
    if cfg_path.exists():
        ss.load_notes.append(f"`{core.CONFIG_NAME}`")
    for d in datasets:
        if (path / d["directory"] / core.PLATE_MAP_NAME).exists():
            get_pm_state(d)
            if d["name"] in ss.pm_saved:
                ss.load_notes.append(f"`{d['directory']}/{core.PLATE_MAP_NAME}`")
    ss.prefs["last_experiment"] = str(path)
    core.save_prefs(ss.prefs)


def page_experiment() -> None:
    st.header("1 · Experiment folder")
    st.markdown(
        "Choose the folder that holds **one experiment**: either the images "
        "themselves, or one sub-folder per acquisition (for example one folder per "
        "stain panel, or a MIP folder and a z-stack folder). `config.yaml` and "
        "`hpc_job.env` are saved here; each acquisition folder gets its own "
        "`plate_map.yaml`.")

    folder_browser()
    path_txt = st.text_input("Experiment folder", key="_exp_input",
                             help="Paste a path, or use the browser above. On Windows this "
                                  "is usually something like Z:\\KR\\exp_033.")
    c1, c2 = st.columns([1, 4])
    if c1.button("Open experiment", type="primary", disabled=not path_txt.strip()):
        p = Path(path_txt.strip().strip('"')).expanduser()
        if not p.is_dir():
            st.error(f"Not a folder: {p}")
        else:
            ss.exp_dir = str(p)
            with st.spinner("Scanning for images (slow on a network share the first time)…"):
                load_experiment(p)
            st.rerun()
    if ss.loaded and c2.button("Rescan folders"):
        with st.spinner("Scanning…"):
            load_experiment(exp_path())
        st.rerun()

    if not ss.loaded:
        return

    path = exp_path()
    notes = ss.get("load_notes") or []
    st.success(f"Experiment: `{path}`" + (f"\n\nLoaded from disk: {', '.join(notes)}. Every "
                                         "page starts from these; nothing is re-entered."
                                         if notes else "\n\nNo config.yaml or plate map "
                                                       "yet: the pages start from defaults."))
    if not ss.datasets:
        st.warning("No image files were found in this folder or its sub-folders. "
                   "Check the folder, or the file types under Analysis settings → Advanced.")
        return

    st.subheader("Acquisition folders (datasets)")
    st.caption("Untick folders you do not want analyzed. The name is how the folder is "
               "called in tables and figures. Segmentation: **2d** works on the DAPI "
               "maximum projection (any image); **3d** segments the full z-stack.")
    rows = []
    for d in ss.datasets:
        s = ss.scans.get(d["directory"])
        rows.append({"include": d["include"], "name": d["name"], "folder": d["directory"],
                     "segmentation": d["seg_mode"],
                     "images": s.summary if s else "folder missing",
                     "acquisition": s.mode if s and s.mode else "?",
                     "plate map": "✓ exists" if s and s.has_plate_map else "- not yet"})
    df = editor(f"datasets{ss.gen}", lambda: pd.DataFrame(rows), hide_index=True, width="stretch",
                column_config={
                    "include": st.column_config.CheckboxColumn("use"),
                    "name": st.column_config.TextColumn("name", help="Letters, digits, _ and -"),
                    "folder": st.column_config.TextColumn(disabled=True),
                    "segmentation": st.column_config.SelectboxColumn(options=["2d", "3d"],
                                                                     required=True),
                    "images": st.column_config.TextColumn(disabled=True),
                    "acquisition": st.column_config.TextColumn(disabled=True),
                    "plate map": st.column_config.TextColumn(disabled=True),
                })
    names_seen = set()
    for d, r in zip(ss.datasets, records(df)):
        old = d["name"]
        d["include"] = bool(r["include"])
        d["name"] = str(r["name"] or d["directory"]).strip().replace(" ", "_")
        d["seg_mode"] = r["segmentation"] or "2d"
        if old != d["name"] and old in ss.pm:
            ss.pm[d["name"]] = ss.pm.pop(old)
        if d["include"] and d["name"] in names_seen:
            st.error(f"Two datasets are called '{d['name']}': names must be unique.")
        names_seen.add(d["name"])

    st.info("Next: **2 · Plate maps** (sidebar).")


# ===========================================================================
# 2 · Plate maps
# ===========================================================================

def get_pm_state(ds: dict) -> dict:
    name = ds["name"]
    if name not in ss.pm:
        scan = ss.scans.get(ds["directory"])
        p = exp_path() / ds["directory"] / core.PLATE_MAP_NAME
        if p.exists():
            try:
                ss.pm[name] = core.plate_map_from_yaml(core.read_yaml(p), scan)
                ss.pm_saved[name] = p.read_text(encoding="utf-8")
            except Exception as exc:                    # noqa: BLE001
                st.error(f"Could not read the existing plate map ({exc}). Starting a new one; "
                         "the old file will be kept as plate_map.yaml.bak when you save.")
                ss.pm[name] = core.new_plate_map(scan)
        else:
            ss.pm[name] = core.new_plate_map(scan)
    return ss.pm[name]


def n_img_total(scan) -> int:
    return len(scan.seq) if scan and scan.seq else 0


def plate_grid_html(state: dict, color_by: str, counts: dict[str, int]) -> str:
    """The plate drawn as a grid, wells colored by one column (Okabe-Ito)."""
    rows, cols = int(state["rows"]), int(state["cols"])
    label_mode = state["mode"] == "spinning_disk"
    by_key = {str(r.get(core.WELL_COL)).upper(): r for r in state["wells"]}
    values = []
    for r in state["wells"]:
        v = core.coerce(r.get(color_by))
        if v is not None and str(v) not in values:
            values.append(str(v))
    palette = {v: OKABE_ITO[i % len(OKABE_ITO)] for i, v in enumerate(values)}

    cells = []
    for row in range(1, rows + 1):
        tds = [f"<th>{chr(64 + row) if label_mode or state['well_numbering'] == 'plate' else ''}</th>"]
        for col in range(1, cols + 1):
            num = (row - 1) * cols + col
            key = f"{chr(64 + row)}{col}" if label_mode else str(num)
            r = by_key.get(key)
            if r is None:
                tds.append(f"<td class='empty'><b>{key}</b></td>")
                continue
            v = core.coerce(r.get(color_by))
            bg = palette.get(str(v), "#DDDDDD") if v is not None else "#FFFFFF"
            fg = text_color(bg)
            n = counts.get(key)
            tip = html.escape(", ".join(f"{k}={r[k]}" for k in r if r[k] not in (None, "")))
            tds.append(
                f"<td style='background:{bg};color:{fg}' title='{tip}'><b>{key}</b><br>"
                f"<span>{html.escape(str(v)) if v is not None else '-'}</span>"
                f"{f'<br><small>{n} img</small>' if n is not None else ''}</td>")
        cells.append("<tr>" + "".join(tds) + "</tr>")
    head = "<tr><th></th>" + "".join(f"<th>{c}</th>" for c in range(1, cols + 1)) + "</tr>"
    legend = " ".join(
        f"<span class='lg' style='background:{palette[v]};color:{text_color(palette[v])}'>"
        f"{html.escape(v)}</span>" for v in values)
    style = """
    <style>
    table.plate {border-collapse:separate;border-spacing:4px;font-size:12px}
    table.plate td {min-width:64px;height:52px;text-align:center;border-radius:10px;
                    border:1px solid #888;padding:2px;vertical-align:middle}
    table.plate td.empty {background:transparent;color:#999;border:1px dashed #bbb}
    table.plate th {color:#888;font-weight:normal}
    span.lg {padding:2px 8px;border-radius:6px;margin-right:4px;display:inline-block;
             border:1px solid #888}
    </style>"""
    return (style + f"<table class='plate'>{head}{''.join(cells)}</table>"
            + (f"<div style='margin-top:6px'>{legend}</div>" if legend else ""))


def page_plate_maps() -> None:
    st.header("2 · Plate maps: what is in each well")
    if not need_experiment():
        return
    dss = included()
    if not dss:
        st.warning("No datasets are ticked in step 1.")
        return
    names = [d["name"] for d in dss]
    pick = st.selectbox("Acquisition folder", names, key="_pm_pick",
                        help="Each folder has its own plate map.")
    ds = next(d for d in dss if d["name"] == pick)
    scan = ss.scans.get(ds["directory"])
    state = get_pm_state(ds)
    # widget/editor prefix for this dataset; a copy bumps the version so every
    # widget starts again from the copied values
    ek = f"pm{ss.gen}_{pick}_{ss.get('pm_ver', {}).get(pick, 0)}"

    if scan:
        st.caption(f"`{ds['directory']}`: {scan.summary}; detected acquisition: "
                   f"**{scan.mode or 'unknown'}**"
                   + (f"; channels: {', '.join(scan.channels)}" if scan.channels else
                      " (channel names appear once the 'load' step has run)"))

    sources = [d["name"] for d in dss if d["name"] != pick]
    if sources:
        with st.expander("📋 Copy the plate map from another folder"):
            st.caption("For folders imaged from the same plate, e.g. the max projections "
                       "and the z-stacks. Everything is copied: plate, plate ID, "
                       "image-to-well layout, stain sets and the wells table. This "
                       f"replaces what is entered for **{pick}**; nothing is written to "
                       "disk until you save, and the old file is kept as .bak.")
            c1, c2 = st.columns([3, 1])
            src = c1.selectbox("Copy from", sources, key=f"{ek}_copysrc",
                               label_visibility="collapsed")
            if c2.button("Copy here", width="stretch", key=f"{ek}_copy"):
                src_state = get_pm_state(next(d for d in dss if d["name"] == src))
                ss.pm[pick] = copy.deepcopy(src_state)
                ss.setdefault("pm_ver", {})[pick] = ss.get("pm_ver", {}).get(pick, 0) + 1
                st.toast(f"Copied the plate map of {src}: check it, then save.")
                st.rerun()

    # -- A. plate + acquisition -------------------------------------------
    st.subheader("Plate and acquisition")
    c1, c2, c3 = st.columns(3)
    plate_opts = list(core.PLATES) + ["custom"]
    state["plate"] = c1.selectbox("Plate", plate_opts, index=plate_opts.index(state["plate"]),
                                  key=f"{ek}_plate")
    if state["plate"] == "custom":
        cc1, cc2 = c1.columns(2)
        state["rows"] = int(cc1.number_input("rows", 1, 32, int(state["rows"]), key=f"{ek}_rows"))
        state["cols"] = int(cc2.number_input("columns", 1, 48, int(state["cols"]), key=f"{ek}_cols"))
    else:
        state["rows"], state["cols"] = core.PLATES[state["plate"]]
    state["plate_id"] = c2.text_input(
        "Plate ID", state["plate_id"], key=f"{ek}_pid",
        help="A name for the physical plate. If several folders are different stain "
             "panels of the SAME plate, give them the SAME plate ID: that is what lets "
             "a shared marker be pooled into one threshold.").strip() or pick
    mode_idx = core.MODES.index(state["mode"]) if state["mode"] in core.MODES else 0
    new_mode = c3.selectbox("Acquisition mode", core.MODES, index=mode_idx, key=f"{ek}_mode",
                            format_func=lambda m: {"predefined": "Predefined points (Seq)",
                                                   "generate": "Generate points (Well07)",
                                                   "spinning_disk": "Spinning disk (WellC10)"}[m],
                            help="\n\n".join(f"**{k}**: {v}" for k, v in core.MODE_HELP.items()))
    if new_mode != state["mode"]:
        state["mode"] = new_mode
        state["well_numbering"] = "sequential" if new_mode == "predefined" else "plate"
    if scan and scan.mode and scan.mode != state["mode"]:
        st.warning(f"The filenames look like **{scan.mode}** mode, but **{state['mode']}** "
                   "is selected.")

    others = [d for d in dss if d["name"] != pick and d["name"] in ss.pm]
    same = [d["name"] for d in others if ss.pm[d["name"]]["plate_id"] == state["plate_id"]]
    if others and not same:
        st.caption("ℹ️ Other folders use different plate IDs. That is right if they are "
                   "different plates; if they are panels of this same plate, use one ID.")

    # -- B. Seq -> well (predefined) ----------------------------------------
    if state["mode"] == "predefined":
        st.subheader("Which images belong to which well?")
        st.markdown(
            "Predefined-points filenames only carry the **acquisition order** "
            "(`Seq0000`, `Seq0001`, …). Wells are therefore numbered in the order you "
            "imaged them: **well 1 = the first well imaged**, whatever its position on "
            "the plate.")
        n_img = len(scan.seq) if scan and scan.seq else 0
        mapping_opts = ["variable", "equal", "ranges"]
        mapping_lbl = {"equal": "Same number of images in every well",
                       "variable": "Type the number of images per well (Wells table below)",
                       "ranges": "Explicit Seq ranges per well"}
        old_mapping = state["mapping"]
        state["mapping"] = st.radio("How were images divided among wells?", mapping_opts,
                                    index=mapping_opts.index(state["mapping"]),
                                    format_func=mapping_lbl.get, horizontal=True,
                                    key=f"{ek}_mapping")
        if state["mapping"] == "variable" and old_mapping != "variable" and not state["variable"]:
            # Keep what the previous mapping implied, so switching never loses wells.
            per = {str(r["well"]).upper(): r["images"]
                   for r in core.validate_plate_map({**state, "mapping": old_mapping},
                                                    scan).per_well}
            state["variable"] = [{"well": int(r[core.WELL_COL]),
                                  "count": int(per.get(str(r[core.WELL_COL]).upper(), 0))}
                                 for r in state["wells"] if str(r[core.WELL_COL]).isdigit()]
        if state["mapping"] == "equal":
            c1, c2 = st.columns([1, 3])
            state["points_per_well"] = int(c1.number_input(
                "Images per well", 1, 1000, int(state["points_per_well"] or 1), key=f"{ek}_ppw"))
            if n_img:
                k, rem = divmod(n_img, state["points_per_well"])
                c2.markdown(f"{n_img} images ÷ {state['points_per_well']} per well = "
                            f"**{k} wells**" + (f" and {rem} left over ⚠️" if rem else ""))
        elif state["mapping"] == "variable":
            st.caption("Add one row per well in the **Wells and conditions** table below, "
                       "**in the order you imaged them**, and type its number of images in "
                       "the *images* column. Well numbers are normally just 1, 2, 3 …")
        else:
            st.caption("Seq numbers as in the filenames, both ends included.")
            df = editor(f"{ek}_ranges",
                        lambda: pd.DataFrame(state["ranges"] or [{"well": 1, "first": 0, "last": 0}],
                                             columns=["well", "first", "last"]),
                        num_rows="dynamic", hide_index=True)
            state["ranges"] = [r for r in records(df) if r.get("well") is not None]

        with st.expander("Advanced: several stain sets in this ONE folder, or real plate positions"):
            st.markdown(
                "**Stain sweeps**: use only if you imaged the plate several times into "
                "this one folder with different stain sets (e.g. Seq 0-107 = panel 1, "
                "Seq 108-215 = panel 2). Each block restarts at well 1. Leave empty otherwise.")
            df = editor(f"{ek}_sweeps",
                        lambda: pd.DataFrame(state["sweeps"],
                                             columns=["stain", "first", "last", "points_per_well"]),
                        num_rows="dynamic", hide_index=True,
                        column_config={"stain": st.column_config.SelectboxColumn(
                            "stain set", options=list(state["stains"]))})
            state["sweeps"] = [r for r in records(df) if r.get("stain")]
            plate_pos = st.checkbox(
                "My well numbers are real plate positions (enables plate-layout figures)",
                value=state["well_numbering"] == "plate", key=f"{ek}_wn")
            state["well_numbering"] = "plate" if plate_pos else "sequential"

    # -- C. stains -----------------------------------------------------------
    st.subheader("Stain sets, which marker is on which channel")
    st.caption("Channel 0 is the first channel saved by the microscope. Write the "
               "**antibody / marker** name, not the fluorophore. DAPI is used for "
               "segmentation and is never analyzed as a marker. Add a second set only if "
               "some wells carry a different panel.")
    n_ch = max([len(v) for v in state["stains"].values()] + [len(scan.channels) if scan else 0, 1])
    max_ch = max([max(v) for v in state["stains"].values() if v] + [n_ch - 1])

    def stains_df():
        idx = list(range(max_ch + 1))
        return pd.DataFrame({name: [chs.get(i, "") for i in idx]
                             for name, chs in state["stains"].items()},
                            index=pd.Index(idx, name="channel"))

    sdf = editor(f"{ek}_stains", stains_df, num_rows="dynamic", width="stretch")
    new_stains = {}
    for name in sdf.columns:
        chs = {}
        for i, v in zip(range(len(sdf)), sdf[name].tolist()):
            if v is not None and str(v).strip() and str(v) != "nan":
                chs[i] = str(v).strip()
        new_stains[str(name)] = chs
    state["stains"] = new_stains
    c1, c2, c3 = st.columns([2, 1, 2])
    new_set = c1.text_input("New stain set name", key=f"{ek}_newset",
                            placeholder="e.g. panel2", label_visibility="collapsed")
    if c2.button("Add stain set", disabled=not new_set.strip()):
        base = sdf.copy()
        base[new_set.strip()] = base.iloc[:, 0]
        reset_editor(f"{ek}_stains", base)
        st.rerun()
    removable = [s for s in state["stains"] if s != "default"]
    if removable:
        drop = c3.selectbox("Remove set", ["-"] + removable, key=f"{ek}_dropset")
        if drop != "-" and c3.button(f"Remove '{drop}'"):
            reset_editor(f"{ek}_stains", sdf.drop(columns=[drop]))
            st.rerun()

    # -- D. wells --------------------------------------------------------------
    st.subheader("Wells and conditions")
    st.markdown(
        "One row per well. Add a column for each thing that varies between wells "
        "(cell line, BMP4 dose, pattern diameter, …): **numbers stay numbers** "
        "(`diameter_um: 500`). One image is one colony. Then:\n"
        "- **class**: mark **negative control** wells (`control`) if you have them; "
        "they set the gates for the *% positive* profiles. Intensity profiles never "
        "need them. You may also use `class` for the cell line and compare by it.\n"
        "- **stain**: only if the well uses a stain set other than `default`.\n"
        "- **positive_for / control_for**: rare exceptions, markers this well is a "
        "*positive* control for, or the *negative* control for (comma-separated).")

    # Predefined points with a typed count per well: the images column is the
    # count itself, typed here. Every other layout derives it from the filenames
    # or the mapping above, so there it is shown read-only.
    typed_counts = state["mode"] == "predefined" and state["mapping"] == "variable" \
        and not state.get("sweeps")
    var_counts = {str(r["well"]): r.get("count") for r in state["variable"]}
    counts: dict[str, int] = {}
    report = core.validate_plate_map(state, scan)
    for r in report.per_well:
        counts[str(r["well"]).upper()] = r["images"]
    # The table restarts only when its columns change, never on a typed count
    # (a restart mid-typing would drop the cell being edited).
    wells_sig = abs(hash((tuple(state["columns"]), tuple(state["stains"]), state["mode"],
                          typed_counts) + (() if typed_counts else
                                           tuple(sorted(counts.items()))))) % 10**8

    def wells_df():
        cols = [core.WELL_COL, core.COUNT_COL] + state["columns"] + core.SPECIAL_COLS
        rows = []
        for r in state["wells"]:
            row = {c: r.get(c) for c in cols}
            key = str(r.get(core.WELL_COL))
            row[core.COUNT_COL] = (var_counts.get(key) if typed_counts
                                   else counts.get(key.upper(), 0))
            for c in state["columns"] + core.SPECIAL_COLS:
                v = row.get(c)
                row[c] = "" if v is None else str(v)
            row[core.WELL_COL] = str(r.get(core.WELL_COL))
            rows.append(row)
        return pd.DataFrame(rows, columns=cols)

    c1, c2, c3 = st.columns([2, 1, 2])
    new_col = c1.text_input("New column", key=f"{ek}_newcol", placeholder="e.g. dose_uM",
                            label_visibility="collapsed")
    if c2.button("Add column", disabled=not new_col.strip()):
        col = new_col.strip().replace(" ", "_")
        if col in core.SPECIAL_COLS + [core.WELL_COL, core.COUNT_COL] or col in state["columns"]:
            st.error(f"'{col}' already exists or is reserved.")
        else:
            state["columns"].append(col)
            reset_editor(f"{ek}_wells_{wells_sig}")
            st.rerun()
    if state["columns"]:
        drop = c3.selectbox("Remove column", ["-"] + state["columns"], key=f"{ek}_dropcol")
        if drop != "-" and c3.button(f"Remove '{drop}'"):
            state["columns"].remove(drop)
            for r in state["wells"]:
                r.pop(drop, None)
            reset_editor(f"{ek}_wells_{wells_sig}")
            st.rerun()

    class_opts = [""] + core.CONTROL_CLASSES + sorted(
        {str(r.get(core.CLASS_COL)) for r in state["wells"]
         if core.coerce(r.get(core.CLASS_COL)) is not None} - set(core.CONTROL_CLASSES))
    wdf = editor(f"{ek}_wells_{wells_sig}", wells_df, num_rows="dynamic", hide_index=True,
                 width="stretch",
                 column_config={
                     core.WELL_COL: st.column_config.TextColumn(
                         "well", required=True,
                         help="Well number (or label like C10 for the spinning disk)"),
                     core.COUNT_COL: st.column_config.NumberColumn(
                         "images", disabled=not typed_counts, min_value=0, step=1,
                         help=("Number of images (Seq points) of this well, in acquisition "
                               "order" if typed_counts else
                               "Counted from the image files and the layout above")),
                     core.CLASS_COL: st.column_config.SelectboxColumn(
                         "class", options=class_opts,
                         help="control = negative control well"),
                     core.STAIN_COL: st.column_config.SelectboxColumn(
                         "stain", options=[""] + list(state["stains"]),
                         help="Empty = default"),
                     core.POS_COL: st.column_config.TextColumn(
                         "positive_for", help="Markers this well is a POSITIVE control for, "
                                              "e.g. SOX2: it stays the negative control for "
                                              "everything else"),
                     core.CTRL_COL: st.column_config.TextColumn(
                         "control_for", help="Markers this well IS the negative control for, "
                                             "or 'all'"),
                 })
    rows_now = [r for r in records(wdf) if str(r.get(core.WELL_COL) or "").strip()]
    state["wells"] = [{k: v for k, v in r.items() if k != core.COUNT_COL} for r in rows_now]
    if typed_counts:
        state["variable"] = []
        for r in rows_now:
            w = str(r[core.WELL_COL]).strip()
            if w.isdigit():
                n = r.get(core.COUNT_COL)
                state["variable"].append({"well": int(w),
                                          "count": int(n) if n is not None and n == n else 0})
        total = sum(r["count"] for r in state["variable"])
        if rows_now and n_img_total(scan):
            (st.success if total == n_img_total(scan) else st.warning)(
                f"The images column adds up to {total}; the folder has "
                f"{n_img_total(scan)} images.")
    if not rows_now:
        st.caption("The table starts empty: click the **+** under it to add a row per "
                   "well, or use the helper below.")

    c1, c2, c3 = st.columns([1, 1, 2])
    if typed_counts:
        n_add = c1.number_input("Add wells", 1, 384, 1, key=f"{ek}_addn")
        k_add = c2.number_input("…with images each", 0, 1000, 1, key=f"{ek}_addk")
        if c3.button("Add these rows", key=f"{ek}_addrows"):
            nums = [int(r[core.WELL_COL]) for r in state["wells"]
                    if str(r[core.WELL_COL]).isdigit()]
            start = max(nums, default=0) + 1
            for w in range(start, start + int(n_add)):
                state["wells"].append({core.WELL_COL: str(w)})
                state["variable"].append({"well": w, "count": int(k_add)})
            reset_editor(f"{ek}_wells_{wells_sig}")
            st.rerun()
    elif c1.button("List the wells found in the images (keeps what you typed)",
                   key=f"{ek}_relist"):
        existing = {str(r[core.WELL_COL]).upper(): r for r in state["wells"]}
        fresh = core.default_well_keys(state, scan)
        state["wells"] = [existing.get(k.upper(), {core.WELL_COL: k}) for k in fresh] + \
                         [r for k, r in existing.items() if k not in {f.upper() for f in fresh}]
        reset_editor(f"{ek}_wells_{wells_sig}")
        st.rerun()

    with st.expander("✏️ Fill many wells at once", expanded=False):
        keys = [str(r[core.WELL_COL]) for r in state["wells"]]
        c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
        sel_txt = c1.text_input("Wells", key=f"{ek}_fill_wells",
                                placeholder="e.g. 1-6, 9   or   C1-C4",
                                help="Ranges and lists; labels for the spinning disk.")
        target = c2.selectbox("Column", state["columns"] + core.SPECIAL_COLS, key=f"{ek}_fill_col")
        value = c3.text_input("Value", key=f"{ek}_fill_val")
        if c4.button("Fill", width="stretch"):
            try:
                spec = parse_well_spec(sel_txt)
                if spec is None:
                    chosen = set(keys)
                else:
                    nums, labels = spec
                    chosen = {k for k in keys if (k.isdigit() and int(k) in nums)
                              or k.upper() in labels}
                base = wdf.copy()
                base[target] = base[target].astype(object)
                mask = base[core.WELL_COL].astype(str).isin(chosen)
                base.loc[mask, target] = value
                reset_editor(f"{ek}_wells_{wells_sig}", base)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    # -- plate picture -------------------------------------------------------
    color_opts = state["columns"] + [core.CLASS_COL, core.STAIN_COL]
    color_by = st.selectbox("Color the plate by", color_opts, key=f"{ek}_colorby")
    if state["mode"] == "predefined" and state["well_numbering"] != "plate":
        st.caption("Positions below are acquisition order laid out on the plate grid "
                   "(well 1 top-left), not physical positions.")
    st.markdown(plate_grid_html(state, color_by, counts), unsafe_allow_html=True)

    # -- E. validate + save -----------------------------------------------------
    st.subheader("Check and save")
    report = core.validate_plate_map(state, scan)
    for e in report.errors:
        st.error(e)
    for w in report.warnings:
        st.warning(w)
    if report.ok and not report.warnings:
        st.success("Every image maps to a well with a condition.")
    if report.per_well:
        with st.expander("Image → well table", expanded=not report.ok):
            st.dataframe(pd.DataFrame(report.per_well), hide_index=True,
                         width="stretch")
    target_path = exp_path() / ds["directory"] / core.PLATE_MAP_NAME
    on_disk = target_path.read_text(encoding="utf-8") if target_path.exists() else None
    text = core.plate_map_text(state, Path(ds["directory"]).name, on_disk)
    with st.expander("Preview plate_map.yaml"):
        st.code(text, language="yaml")
    unsaved = on_disk != text
    st.caption(("Unsaved changes. " if unsaved else "Saved. ") + f"`{target_path}`"
               + ("" if report.ok else "  ⚠️ there are errors above: you can save, but "
                                        "`check` will fail until they are fixed."))
    if save_with_confirm(target_path, text, f"pm_{pick}", ss.pm_saved.get(pick),
                         "💾 Save plate map", disabled=not unsaved):
        ss.pm_saved[pick] = text
        if scan:
            scan.has_plate_map = True
        st.rerun()


# ===========================================================================
# 3 · Analysis settings (config.yaml)
# ===========================================================================

def vocabulary() -> dict:
    maps = {}
    for d in included():
        if d["name"] in ss.pm:
            maps[d["name"]] = core.plate_map_dict(ss.pm[d["name"]])
    for name, raw in core.collect_plate_maps(exp_path(), included()).items():
        maps.setdefault(name, raw)
    return core.plate_map_vocabulary(maps)


def is_section(path: tuple) -> bool:
    e = ss.template.entries.get(tuple(path))
    return e is not None and not e.is_leaf


def auto_widget(path: tuple, value, key_prefix="cfg"):
    """A widget chosen from the value's type, with help text from the template."""
    t = ss.template
    label = path[-1]
    helptext = t.help(*path) or None
    key = f"{key_prefix}{ss.gen}::{'.'.join(map(str, path))}"
    if isinstance(value, bool):
        return st.checkbox(label, value=value, help=helptext, key=key)
    if isinstance(value, int):
        return int(st.number_input(label, value=value, step=1, help=helptext, key=key))
    if isinstance(value, float):
        return float(st.number_input(label, value=value, help=helptext, key=key,
                                     format="%g", step=None))
    if isinstance(value, (list, dict)) or value is None or isinstance(value, str):
        shown = "" if value is None else (value if isinstance(value, str) else core.flow(value))
        txt = st.text_input(label, value=shown, help=helptext, key=key,
                            placeholder="(empty = default / none)")
        if isinstance(value, (list, dict)) or (txt.strip()[:1] in "[{"):
            try:
                import yaml
                parsed = yaml.safe_load(txt) if txt.strip() else ([] if isinstance(value, list)
                                                                   else {} if isinstance(value, dict)
                                                                   else None)
                return parsed
            except Exception:                           # noqa: BLE001
                st.error(f"{label}: not valid; use e.g. [a, b] or {{key: value}}")
                return value
        return core.coerce(txt)
    return value


def auto_section(cfg: dict, path: tuple, skip: set = frozenset(), ncols: int = 3) -> None:
    node = core.get_in(cfg, path, {})
    leaves = [(k, v) for k, v in node.items()
              if not is_section(path + (k,)) and path + (k,) not in skip]
    cols = st.columns(ncols)
    for i, (k, v) in enumerate(leaves):
        with cols[i % ncols]:
            core.set_in(cfg, path + (k,), auto_widget(path + (k,), v))
    for k, v in node.items():
        if is_section(path + (k,)) and isinstance(v, dict) and path + (k,) not in skip:
            with st.expander(f"{'.'.join(path[1:] + (k,))}"):
                auto_section(cfg, path + (k,), skip, ncols)


@st.cache_data(ttl=120, show_spinner=False)
def _segmented_dirs(root: str) -> list[str]:
    """Result folders holding a segmentation (cached: a share walk is slow)."""
    return [str(m.parent) for m in sorted(Path(root).rglob("nuclear_masks.tif"))]


def colony_preview(cfg: dict) -> None:
    """Run colony detection on one segmented image with the settings on screen."""
    from downstream import colony as colony_mod
    with st.expander("🔍 Preview colony detection on one image (needs segmentation done)"):
        segmented = []
        for d in included():
            root = (exp_path() / d["directory"]).resolve()
            segmented += [(d["name"], Path(m)) for m in _segmented_dirs(str(root))]
        if not segmented:
            st.caption("No segmented images yet. Once the GPU job has run, pick any image "
                       "here to see where the colony edge, circle and outliers land with the "
                       "current settings: before running the whole analysis.")
            return
        labels = [f"{n} / {p.name}" for n, p in segmented]
        pick = st.selectbox("Image", range(len(segmented)), format_func=lambda i: labels[i],
                            key=f"preview_pick{ss.gen}")
        if not st.button("Detect colony", key=f"preview_go{ss.gen}"):
            return
        name, res = segmented[pick]
        ds_cfg = next((d for d in included() if d["name"] == name), {})
        dapi = int(core.get_in(cfg, ("defaults", "channels", "dapi_index"), 0))
        try:
            import tifffile
            params = colony_mod.settings(core.get_in(cfg, ("downstream",), {}))
            pixel = colony_mod.pixel_size_for(res, params)
            masks = tifffile.imread(res / colony_mod.MASKS_NAME)
            with st.spinner("Detecting…"):
                geo = colony_mod.detect_colony(masks, pixel, params)
                masks_xy = masks.max(axis=0) if masks.ndim == 3 else masks
                fig = colony_mod.render_overlay(colony_mod._dapi_image(res, dapi, masks_xy.shape),
                                                masks_xy, geo, 1200)
        except Exception as exc:                          # noqa: BLE001
            st.error(f"{type(exc).__name__}: {exc}")
            return
        meta = geo["meta"]
        c1, c2 = st.columns([3, 2])
        c1.pyplot(fig, width="stretch")
        import matplotlib.pyplot as plt
        plt.close(fig)
        if meta.get("status") != "ok":
            c2.error("No colony found in this image.")
            return
        exp_d = params.get("expected_diameter_um")
        c2.metric("Fitted diameter", f"{meta['diameter_um']:.0f} µm",
                  delta=f"{meta['diameter_um'] - exp_d:+.0f} µm vs expected" if exp_d else None,
                  delta_color="off")
        c2.metric("Nuclei in colony", f"{meta['n_nuclei_colony']:,}",
                  delta=f"{meta['n_nuclei_outside']} outside", delta_color="off")
        c2.write(f"- pixel size **{pixel:.4g} µm/px**\n"
                 f"- edge distance from **{meta['edge_method']}** "
                 f"({100 * meta['border_fraction']:.0f}% of the rim on the image border)\n"
                 f"- mask fills **{100 * (meta['fill_ratio'] or 0):.0f}%** of the fitted circle; "
                 f"rim deviates {100 * (meta['fit_rms_rel'] or 0):.1f}% of R\n"
                 f"- {meta['n_neighbour_colonies']} neighboring colony(ies); "
                 f"{100 * meta['trimmed_fraction']:.1f}% of the mask trimmed outside the circle")
        c2.caption("Nothing is written. If the orange edge swallows a neighbor, raise "
                   "open_radius_um or lower close_radius_um (Colonies & QC tab).")


def channel_markers() -> dict[int, list[str]]:
    """Channel index → every marker name it carries, over all stain sets and folders."""
    out: dict[int, list[str]] = {}
    maps = {}
    for d in included():
        if d["name"] in ss.pm:
            maps[d["name"]] = core.plate_map_dict(ss.pm[d["name"]])
    for name, raw in core.collect_plate_maps(exp_path(), included()).items():
        maps.setdefault(name, raw)
    for raw in maps.values():
        for chs in (raw.get("stains") or {}).values():
            for c, m in (chs or {}).items():
                try:
                    c = int(c)
                except (TypeError, ValueError):
                    continue
                names = out.setdefault(c, [])
                if str(m) not in names:
                    names.append(str(m))
    return dict(sorted(out.items()))


def signaling_settings(cfg: dict, sig_cfg: dict, columns: list[str], voc: dict,
                       markers: list[str]) -> None:
    """Reference wells and measurement settings for the signaling markers."""
    D = ("downstream",)
    cond = core.get_in(cfg, D + ("conditions",), {}) or {}
    with st.expander(f"Signaling markers: {', '.join(markers)}", expanded=True):
        st.caption("Each signaling readout is shown relative to a **reference** population "
                   "(1 = the reference), per plate × panel: the median over the reference "
                   "colonies. Usually the pluripotent wells. Without reference wells, N:C is "
                   "shown as the plain ratio and a nuclear readout on the shared scale.")
        c1, c2 = st.columns(2)
        col_opts = sorted(set(columns) | {"class"})
        ref_col = sig_cfg.get("reference_column") or cond.get("control_column", "class")
        ref_col = c1.selectbox("Reference wells are marked in column", col_opts,
                               index=col_opts.index(ref_col) if ref_col in col_opts else 0,
                               key=f"sig_refcol{ss.gen}")
        vals = voc["columns"].get(ref_col, [])
        cur = sig_cfg.get("reference_values")
        if cur is None:
            cur = [v for v in map(str, cond.get("control_values") or []) if v in vals] \
                if ref_col == cond.get("control_column", "class") else []
        cur = [str(v) for v in ([cur] if isinstance(cur, str) else cur)]
        opts = sorted(set(map(str, vals)) | set(cur))
        ref_vals = c2.multiselect("…with value", opts, default=[v for v in cur if v in opts],
                                  key=f"sig_refval{ss.gen}",
                                  help="e.g. pluripotent. Empty = no reference: readouts "
                                       "unnormalized.")
        sig_cfg["reference_column"] = ref_col
        sig_cfg["reference_values"] = ref_vals
        c1, c2, c3, c4 = st.columns(4)
        ns = ["background_subtracted", "raw", "dapi_norm"]
        sig_cfg["nuclear_space"] = c1.selectbox(
            "Nuclear readout", ns, index=ns.index(sig_cfg.get("nuclear_space",
                                                              "background_subtracted"))
            if sig_cfg.get("nuclear_space", "background_subtracted") in ns else 0,
            format_func={"background_subtracted": "background-subtracted",
                         "raw": "raw", "dapi_norm": "÷ nuclear stain"}.get,
            key=f"sig_ns{ss.gen}")
        sig_cfg["ring_inner_um"] = float(c2.number_input(
            "N:C ring from (µm)", 0.0, 10.0, float(sig_cfg.get("ring_inner_um", 1.0)), 0.5,
            key=f"sig_rin{ss.gen}", help="Gap between the nucleus and its cytoplasmic ring"))
        sig_cfg["ring_outer_um"] = float(c3.number_input(
            "…to (µm)", 0.5, 20.0, float(sig_cfg.get("ring_outer_um", 4.0)), 0.5,
            key=f"sig_rout{ss.gen}", help="Outer edge of the ring, from the nucleus"))
        sig_cfg["gate"] = c4.checkbox("Also fit % positive gates", bool(sig_cfg.get("gate", False)),
                                      key=f"sig_gate{ss.gen}",
                                      help="Off by default: signaling markers are not "
                                           "called positive or negative.")
        if sig_cfg["ring_outer_um"] <= sig_cfg["ring_inner_um"]:
            st.error("The ring's outer edge must be beyond its inner edge.")
        st.caption("More settings (background, cell-body mask): **Everything else** tab, "
                   "section signaling. The ring QC images land in figures/signaling/.")


# Optional analyses (downstream.skip_steps): label, what it gives you.
STEP_INFO = {
    "compare": ("Compare conditions",
                "Permutation tests on whole profiles, where along the radius conditions "
                "differ, pairwise tests, profile features, colony similarity."),
    "boundary": ("Boundaries and peaks",
                 "Per colony: where each marker switches off (x50) and how sharply, where "
                 "two facing domains meet, border roughness, signaling peaks."),
    "effects": ("Effects, dose trends and power",
                "Every per-colony number vs the reference condition, slopes over dose or "
                "time, variance split, experiments needed. Needs Compare or Boundaries."),
    "multimarker": ("All markers together",
                    "Every marker of a colony or well stacked on one distance axis."),
    "coexpression": ("Co-expression in space",
                     "Each marker pair: per-nucleus scatter by distance, correlation and "
                     "double-positive fraction along the radius."),
    "signaling_summary": ("Signaling summary",
                          "One signaling value per colony (whole, edge, center) by condition. "
                          "Only does something with signaling markers."),
}
# Figure sets / sub-analyses inside steps: (config path, label, help, parent step).
SUB_SWITCHES = [
    (("colony", "overlay"), "Colony overlay picture per image",
     "colony_overlay.png next to each image: the QC picture. Off saves time on big runs.", None),
    (("profiles", "colony_figures"), "One profile figure per colony",
     "figures/colony_profiles/: one page per colony.", None),
    (("boundary", "roughness", "enabled"), "Border roughness and angular registration",
     "Border radius around the colony, its roughness, registered profiles.", "boundary"),
    (("boundary", "bin_check"), "Bin check (do boundary positions move with the bins?)",
     "Re-fits every boundary with 3 other binnings. Off saves time.", "boundary"),
    (("multimarker", "colony_figures"), "  …one figure per colony", "", "multimarker"),
    (("multimarker", "well_figures"), "  …one figure per well", "", "multimarker"),
    (("coexpression", "colony_figures"), "  …one page per colony", "", "coexpression"),
    (("coexpression", "well_figures"), "  …one page per well", "", "coexpression"),
    (("signaling", "qc_images"), "Signaling ring QC pictures",
     "signaling_rings.png per image and the figures/signaling/ gallery.", None),
]
BIN_CHECK_DEFAULT = ["fixed:5", "fixed:20", "quantile:20"]


def analyses_to_run(cfg: dict, signaling_markers: list[str]) -> None:
    """downstream.skip_steps and the figure-set switches, as check boxes."""
    D = ("downstream",)
    st.subheader("Analyses to run")
    st.caption("Everything runs by default. The core always runs: colony detection, QC, "
               "positivity thresholds, radial profiles per colony / well / condition, and "
               "the report. Untick what your experiment does not need; a skipped "
               "analysis's earlier results are moved to `_skipped/` in the analysis folder.")
    skip = {str(x) for x in (core.get_in(cfg, D + ("skip_steps",), []) or [])}
    steps = [s for s in STEP_INFO if s in core.optional_steps()]
    cols = st.columns(3)
    new_skip = []
    for i, name in enumerate(steps):
        label, helptext = STEP_INFO[name]
        on = cols[i % 3].checkbox(label, value=name not in skip, help=helptext,
                                  key=f"step_{name}{ss.gen}")
        if not on:
            new_skip.append(name)
    if {"boundary", "compare"} <= set(new_skip) and "effects" not in new_skip:
        st.caption("ℹ️ *Effects* needs *Compare* or *Boundaries*: it is skipped too.")
        new_skip.append("effects")
    if "signaling_summary" not in new_skip and not signaling_markers:
        st.caption("(*Signaling summary* does nothing here: no marker is read as signaling.)")
    core.set_in(cfg, D + ("skip_steps",), new_skip)
    with st.expander("Figure sets and sub-analyses"):
        cols = st.columns(2)
        k = 0
        for path, label, helptext, parent in SUB_SWITCHES:
            if parent in new_skip:
                continue
            full = D + path
            if path == ("boundary", "bin_check"):
                cur = core.get_in(cfg, full, BIN_CHECK_DEFAULT)
                on = cols[k % 2].checkbox(label, value=bool(cur), help=helptext or None,
                                          key=f"sub_{'.'.join(path)}{ss.gen}")
                core.set_in(cfg, full, (cur or BIN_CHECK_DEFAULT) if on else [])
            else:
                cur = core.get_in(cfg, full, True)
                on = cols[k % 2].checkbox(label, value=bool(cur), help=helptext or None,
                                          key=f"sub_{'.'.join(path)}{ss.gen}")
                core.set_in(cfg, full, bool(on))
            k += 1
        if "coexpression" not in new_skip:
            co = core.get_in(cfg, D + ("coexpression",), {})
            co["panels"] = st.multiselect(
                "Co-expression: also each panel as its own slide-sized figure, for",
                ["wells", "colonies"], default=[x for x in (co.get("panels") or [])
                                                if x in ("wells", "colonies")],
                key=f"sub_coex_panels{ss.gen}",
                help=ss.template.help(*D, "coexpression", "panels") or None)


@st.cache_data(ttl=60, show_spinner=False)
def _seg_status(exp: str, datasets_key: str, cfg_json: str) -> list:
    import json
    return core.segmentation_status(Path(exp), json.loads(cfg_json))


def seg_status_now(cfg: dict) -> list:
    """Segmentation results on disk per dataset (cached for a minute: a share walk)."""
    import json
    c = {"defaults": cfg.get("defaults"), "datasets": cfg.get("datasets")}
    return _seg_status(str(exp_path()), json.dumps(cfg.get("datasets"), default=str),
                       json.dumps(c, default=str))


SEG_CHOICES = {"auto": "whatever exists (2-D and 3-D side by side)",
               "2d": "2-D results only", "3d": "3-D results only"}


def output_settings(cfg: dict) -> None:
    """Which segmentation results the analysis uses, and where it is written."""
    D = ("downstream",)
    st.subheader("Segmentation results used, and the analysis folder")
    statuses = seg_status_now({**cfg, "datasets": [
        {"name": d["name"], "directory": d["directory"],
         **({"segmentation": {"mode": d["seg_mode"]}} if d["seg_mode"] else {})}
        for d in included()]})
    if statuses:
        rows = [{"dataset": s.name, "set to": s.configured.upper(), "on disk": s.describe()}
                for s in statuses]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        for s in statuses:
            if s.mismatch:
                st.warning(f"**{s.name}** is set to {s.configured.upper()} but its results "
                           f"are {'/'.join(m.upper() for m in s.modes)}. The analysis uses what "
                           f"is on disk and labels it so; run the {s.configured.upper()} "
                           "segmentation job to replace them.")
        if not any(s.n_ready for s in statuses):
            st.caption("Nothing is segmented yet: that is normal before the GPU job.")
    c1, c2 = st.columns(2)
    cur = str(core.get_in(cfg, D + ("seg_mode",), "auto") or "auto").lower()
    cur = "auto" if cur == "both" else cur
    have = {m for s in statuses for m in s.modes}
    choice = c1.radio("Analyze", list(SEG_CHOICES), format_func=SEG_CHOICES.get,
                      index=list(SEG_CHOICES).index(cur) if cur in SEG_CHOICES else 0,
                      key=f"seg_choice{ss.gen}",
                      help="Each image folder holds one segmentation, 2-D or 3-D. Datasets "
                           "without results are left out. 2-D and 3-D results of the same "
                           "colonies are never pooled: with both, every figure is made per "
                           "mode.")
    if choice in ("2d", "3d") and have and choice not in have:
        c1.warning(f"No dataset has {choice.upper()} results yet.")
    core.set_in(cfg, D + ("seg_mode",), choice)
    out = core.get_in(cfg, D + ("results_dir",), "auto")
    auto = c2.checkbox("Name the analysis folder after the results used",
                       value=str(out).lower() in ("auto", "none", ""),
                       key=f"out_auto{ss.gen}",
                       help="analysis_2d, analysis_3d or analysis_2d_3d, so the folder says "
                            "which segmentation it comes from.")
    if auto:
        core.set_in(cfg, D + ("results_dir",), "auto")
        c2.markdown(f"→ `{core.analysis_folder(cfg, statuses)}`")
    else:
        core.set_in(cfg, D + ("results_dir",), c2.text_input(
            "Analysis folder (relative to the experiment)",
            value=str(out) if str(out).lower() != "auto" else "./analysis",
            key=f"out_dir{ss.gen}", help=ss.template.help(*D, "results_dir")))


DNA_STAINS = {"dapi", "hoechst", "hoechst33342", "draq5", "sytox", "topro3", "nucblue"}


def page_analysis() -> None:
    st.header("3 · Analysis settings")
    if not need_experiment():
        return
    cfg = ss.cfg
    voc = vocabulary()
    columns = list(voc["columns"])
    markers = [m for m in voc["markers"] if m.strip().lower() != "dapi"]
    st.caption("Sensible defaults are already filled in. The **Essentials** tab is all "
               "most experiments need; the other tabs hold every remaining option, each "
               "with an explanation (hover the ⓘ).")

    tabs = st.tabs(["Essentials", "Images & segmentation", "Colonies & QC",
                    "Profiles & statistics", "% positive thresholds", "Everything else"])
    D, S = ("downstream",), ("defaults",)

    with tabs[0]:
        # -- channels
        st.subheader("Channels")
        on_channel = channel_markers()
        n_scan = max([len(s.channels) for d in included()
                      if (s := ss.scans.get(d["directory"])) and s.channels] + [0])
        n_ch = max(list(on_channel) + [n_scan - 1, 3]) + 1
        ch_label = lambda i: f"{i}: {' / '.join(on_channel.get(i, [])) or '?'}"  # noqa: E731
        st.caption("Each channel lists every marker it carries across all stain sets of all "
                   "folders: the channels measured are the same in every well, the marker "
                   "name comes from each well's own stain set.")
        c1, c2, c3 = st.columns(3)
        dapi = c1.selectbox("Nuclear channel (DAPI, H2B, …)", list(range(n_ch)),
                            index=min(int(core.get_in(cfg, S + ("channels", "dapi_index"), 0)), n_ch - 1),
                            format_func=ch_label, help=ss.template.help(*S, "channels", "dapi_index"))
        core.set_in(cfg, S + ("channels", "dapi_index"), int(dapi))
        cur_markers = [int(c) for c in core.get_in(cfg, S + ("channels", "marker_channels"), [])
                       if int(c) < n_ch]
        mch = c2.multiselect("Channels to measure", [i for i in range(n_ch) if i != dapi],
                             default=[c for c in cur_markers if c != dapi],
                             format_func=ch_label,
                             help=ss.template.help(*S, "channels", "marker_channels"))
        core.set_in(cfg, S + ("channels", "marker_channels"), sorted(int(c) for c in mch))
        mode = c3.radio("Default segmentation", ["2d", "3d"], horizontal=True,
                        index=0 if str(core.get_in(cfg, S + ("segmentation", "mode")) or "2d") == "2d" else 1,
                        help="Per-folder choices from step 1 override this.")
        core.set_in(cfg, S + ("segmentation", "mode"), mode)

        # -- subset
        st.subheader("Which images")
        st.caption("Leave empty to analyze every image. Handy for a quick test run on a "
                   "few fields before committing GPU hours.")
        c1, c2 = st.columns(2)
        sel = core.get_in(cfg, S + ("input", "select"), {}) or {}
        seq = c1.text_input("Seq numbers", value=str(sel.get("seq") or ""),
                            placeholder="e.g. 0-9, 15", help="As in the filename: Seq0042 → 42")
        wells = c2.text_input("Wells", value=str(sel.get("wells") or ""),
                              placeholder="e.g. 1-3   or   C1-C4")
        try:
            n = parse_index_spec(seq)
            nw = parse_well_spec(wells)
            if n is not None or nw is not None:
                total = sum(len(s.images) for d in included()
                            if (s := ss.scans.get(d["directory"])))
                st.caption(f"Selected: {('%d Seq numbers' % len(n)) if n is not None else 'all Seq'}"
                           f", {'some wells' if nw is not None else 'all wells'} "
                           f"(of {total} images in total)")
        except ValueError as exc:
            st.error(str(exc))
        core.set_in(cfg, S + ("input", "select"),
                    {"seq": seq.strip() or None, "wells": wells.strip() or None})

        # -- conditions
        st.subheader("What is compared")
        if not columns:
            st.warning("No condition columns yet: fill in the plate maps (step 2) first.")
        cond = core.get_in(cfg, D + ("conditions",), {})
        c1, c2 = st.columns(2)
        ck = c1.multiselect("Groups to compare = combination of", columns,
                            default=[c for c in (cond.get("condition_key") or []) if c in columns]
                            or ([c for c in ["class", "treatment"] if c in columns][:1] or columns[:1]),
                            help=ss.template.help(*D, "conditions", "condition_key")
                            + ": e.g. 'class' when the wells are different cell lines.")
        cond["condition_key"] = ck
        fk = c2.multiselect("Compare separately within each value of (optional)", ck,
                            default=[c for c in (cond.get("facet_key") or []) if c in ck],
                            help=ss.template.help(*D, "conditions", "facet_key"))
        cond["facet_key"] = fk or None
        if fk and len(fk) >= len(ck):
            st.warning("Leave at least one column to compare.")
        with st.expander("Negative controls (only used for the '% positive' profiles)"):
            st.caption("A marker's positivity gate is set from the wells that are its "
                       "negative control. Without one, a two-population fit is tried; if that "
                       "fails the marker simply gets no % positive profile. Intensity "
                       "profiles never depend on this.")
            c1, c2 = st.columns(2)
            ctrl_col_opts = sorted(set(columns) | {"class"})
            ctrl_col = c1.selectbox("Control wells are marked in column", ctrl_col_opts,
                                    index=ctrl_col_opts.index(cond.get("control_column", "class"))
                                    if cond.get("control_column", "class") in ctrl_col_opts else 0,
                                    help=ss.template.help(*D, "conditions", "control_column"))
            cond["control_column"] = ctrl_col
            vals = voc["columns"].get(ctrl_col, [])
            opts = sorted(set(vals) | set(map(str, cond.get("control_values") or [])))
            cond["control_values"] = c2.multiselect(
                "…with value", opts,
                default=[v for v in map(str, cond.get("control_values") or []) if v in opts],
                help=ss.template.help(*D, "conditions", "control_values"))
        with st.expander("More options (condition order, per-marker controls)"):
            auto_section(cfg, D + ("conditions",), {D + ("conditions", k) for k in
                         ("condition_key", "facet_key", "control_column", "control_values")}, 2)

        # -- markers
        st.subheader("Markers")
        nuclear_names = on_channel.get(int(dapi), [])
        measured_names = {m for c in mch for m in on_channel.get(int(c), [])}
        if on_channel:
            markers = [m for m in markers if m in measured_names and m not in nuclear_names]
        non_dna = [m for m in nuclear_names if sig_catalog.name_key(m) not in DNA_STAINS]
        default_space = core.get_in(cfg, D + ("default_space",), "dapi_norm")
        c1, c2 = st.columns([1, 2])
        default_space = c1.radio(
            "Standard markers, by default", ["dapi_norm", "raw"],
            index=0 if default_space == "dapi_norm" else 1, horizontal=True,
            format_func={"dapi_norm": "÷ nuclear stain", "raw": "raw intensity"}.get,
            help="What the 'intensity' column below starts from. Per-marker choices in the "
                 "table override it.")
        core.set_in(cfg, D + ("default_space",), default_space)
        if non_dna:
            c2.warning(f"The nuclear channel is **{' / '.join(non_dna)}**, not a DNA stain. "
                       "Its level varies from cell to cell, so dividing by it adds noise "
                       "instead of removing it: use **raw intensity**.")
        st.caption("**analyze**: untick a marker to leave it out of the whole analysis, "
                   "e.g. E-CAD imaged on the same channel as FOXA2 in other wells: it is "
                   "chosen by name, so the other markers on that channel stay in. "
                   "**intensity**: *÷ nuclear stain* divides each nucleus by its own nuclear-"
                   "stain level (good for nuclear markers with a DNA stain like DAPI); *raw* "
                   "keeps the measured intensity (membrane or cytoplasmic markers, or any "
                   "marker when the nuclear channel is not a DNA stain). **readout**: "
                   "*standard* = shared-scale intensity and % positive; *signaling* markers "
                   "(recognized by name: β-catenin, SMAD2/3, SMAD4, …) are read as nuclear "
                   "level or nuclear:cytoplasmic ratio, relative to a reference, never as "
                   "% positive. **One scale across**: which colonies share one intensity "
                   "scale (and gate); colonies on one scale must have been stained and "
                   "imaged alike.")
        mspec = dict(core.get_in(cfg, D + ("markers",), {}) or {})
        sig_cfg = dict(core.get_in(cfg, D + ("signaling",), {}) or {})
        catalog_only = {"signaling": {"enabled": True, "markers": {}}}
        current = {"signaling": sig_cfg}
        readout_lbl = {"standard": "standard", sig_catalog.NUCLEAR: "signaling: nuclear",
                       sig_catalog.NC: "signaling: N:C"}
        rev_readout = {v: k for k, v in readout_lbl.items()}
        space_lbl = {"dapi_norm": "÷ nuclear stain", "raw": "raw"}
        rev_space = {v: k for k, v in space_lbl.items()}
        pool_lbl = {"panel": "plate × panel", "plate": "whole plate (all panels)",
                    "experiment": "whole experiment"}
        rev_pool = {v: k for k, v in pool_lbl.items()}
        default_scope = core.get_in(cfg, D + ("thresholds", "scope"), "panel")

        def readout_of(m, conf) -> str:
            match = sig_catalog.classify(m, conf)
            return match.readout if match is not None else "standard"

        excluded = [str(x) for x in (core.get_in(cfg, D + ("exclude_markers",), []) or [])]
        excluded_l = {x.strip().lower() for x in excluded}
        mrows = []
        for m in markers:
            auto = sig_catalog.classify(m, catalog_only)
            mrows.append({
                "analyze": m.strip().lower() not in excluded_l,
                "marker": m,
                "readout": readout_lbl.get(readout_of(m, current), "standard"),
                "intensity": space_lbl.get((mspec.get(m) or {}).get("space", default_space),
                                           space_lbl["dapi_norm"]),
                "one scale across": pool_lbl.get((mspec.get(m) or {}).get("scope", default_scope),
                                                 pool_lbl["panel"]),
                "recognized as": (auto.entry + (f" (ignoring '{' '.join(auto.dropped)}')"
                                                if auto.dropped else "")) if auto else "",
            })
        mkey = abs(hash((tuple(markers), default_space))) % 10**8
        mdf = editor(f"markers{ss.gen}_{mkey}",
                     lambda: pd.DataFrame(mrows, columns=["analyze", "marker", "readout",
                                                          "intensity", "one scale across",
                                                          "recognized as"]),
                     hide_index=True, width="stretch",
                     column_config={
                         "analyze": st.column_config.CheckboxColumn(
                             "analyze", help="Untick to leave this marker out of every "
                                             "table, figure and test (downstream."
                                             "exclude_markers). Chosen by name, not by "
                                             "channel."),
                         "marker": st.column_config.TextColumn(disabled=True),
                         "readout": st.column_config.SelectboxColumn(
                             options=list(readout_lbl.values()), required=True,
                             help="standard = intensity on the shared scale (+ % positive); "
                                  "signaling: nuclear = background-subtracted nuclear level; "
                                  "signaling: N:C = nucleus ÷ a cytoplasmic ring"),
                         "intensity": st.column_config.SelectboxColumn(
                             options=list(space_lbl.values()), required=True,
                             help="Standard markers only (signaling markers have their own "
                                  "readout)"),
                         "one scale across": st.column_config.SelectboxColumn(
                             options=list(pool_lbl.values()), required=True),
                         "recognized as": st.column_config.TextColumn(
                             disabled=True, help="The signaling protein this name was matched "
                                                 "to (spelling-insensitive)"),
                     })
        new_spec = {m: v for m, v in mspec.items() if m not in markers}   # keep unknown markers
        sig_markers = {m: v for m, v in (sig_cfg.get("markers") or {}).items()
                       if m not in markers}
        chosen_signaling = []
        in_table = {m.strip().lower() for m in markers}
        new_excluded = ["DAPI"] + [x for x in excluded if x.strip().lower() not in in_table
                                   and x.strip().lower() != "dapi"]
        for r in records(mdf):
            m = r["marker"]
            if not bool(r.get("analyze", True)):
                new_excluded.append(m)
            spec = dict(mspec.get(m) or {})
            sp = rev_space.get(r["intensity"], default_space)
            sc = rev_pool.get(r["one scale across"], default_scope)
            spec.pop("space", None), spec.pop("scope", None)
            if sp != default_space:
                spec["space"] = sp
            if sc != default_scope:
                spec["scope"] = sc
            if spec:
                new_spec[m] = spec
            want = rev_readout.get(r["readout"], "standard")
            auto = readout_of(m, catalog_only)
            if want != auto:
                sig_markers[m] = "none" if want == "standard" else want
            if want != "standard" and bool(r.get("analyze", True)):
                chosen_signaling.append(m)
        core.set_in(cfg, D + ("markers",), new_spec)
        core.set_in(cfg, D + ("exclude_markers",), new_excluded)
        left_out = new_excluded[1:]
        if left_out:
            st.info(f"Left out of the analysis: **{', '.join(left_out)}**.")
        sig_cfg["markers"] = sig_markers
        core.set_in(cfg, D + ("signaling",), sig_cfg)
        if not markers:
            st.caption("(markers appear once the stain sets are filled in, step 2, and their "
                       "channels are ticked under Channels to measure)")
        for m in sig_catalog.near_misses(markers, {"signaling": sig_cfg}):
            st.warning(f"**{m}** looks like a signaling protein but is not in the catalog, so "
                       "it is read as a standard marker. Set its readout above if it is a "
                       "signaling readout.")
        for group in sig_catalog.spelling_groups(markers, {"signaling": sig_cfg}):
            st.warning(f"{' / '.join(group)} look like spellings of one marker but will be "
                       "analyzed as separate markers. Use one spelling in the stain sets.")
        if chosen_signaling:
            signaling_settings(cfg, sig_cfg, columns, voc, chosen_signaling)

        # -- micropattern geometry + profiles
        st.subheader("Colonies and profiles")
        col = core.get_in(cfg, D + ("colony",), {})
        prof = core.get_in(cfg, D + ("profiles",), {})
        c1, c2, c3 = st.columns(3)
        num = voc["numeric"]
        d_opts = ["(same for every well)"] + num
        dcol = c1.selectbox("Pattern diameter comes from", d_opts,
                            index=d_opts.index(col.get("diameter_column"))
                            if col.get("diameter_column") in d_opts else 0,
                            help=ss.template.help(*D, "colony", "diameter_column")
                            + ": pick a plate-map column when wells carry different pattern sizes.")
        col["diameter_column"] = None if dcol == d_opts[0] else dcol
        diam = c2.number_input("Pattern diameter (µm, 0 = not set)", min_value=0.0,
                               value=float(col.get("expected_diameter_um") or 0.0), step=50.0,
                               help=ss.template.help(*D, "colony", "expected_diameter_um"),
                               disabled=col["diameter_column"] is not None)
        col["expected_diameter_um"] = (None if col["diameter_column"] or not diam
                                       else (int(diam) if float(diam).is_integer() else float(diam)))
        px = c3.number_input("Pixel size (µm/px, 0 = from the image files)", min_value=0.0,
                             value=float(col.get("pixel_size_um") or 0.0), step=0.01, format="%.4f",
                             help=ss.template.help(*D, "colony", "pixel_size_um"))
        col["pixel_size_um"] = float(px) or None
        c1, c2, c3 = st.columns(3)
        bw = c1.number_input("µm grid: bin width (µm)", min_value=1.0,
                             value=float(prof.get("bin_um", 10)), step=1.0,
                             help=ss.template.help(*D, "profiles", "bin_um"))
        prof["bin_um"] = int(bw) if float(bw).is_integer() else float(bw)
        prof["n_norm_bins"] = int(c2.number_input(
            "Normalized grid: number of bins", min_value=5, max_value=100,
            value=int(prof.get("n_norm_bins", 25)), step=1,
            help=ss.template.help(*D, "profiles", "n_norm_bins")))
        prof["colony_binning"] = c3.selectbox(
            "Per-colony figures: bins", ["quantile", "equal_area", "fixed"],
            index=["quantile", "equal_area", "fixed"].index(prof.get("colony_binning", "quantile")),
            help=ss.template.help(*D, "profiles", "colony_binning"))
        qc = core.get_in(cfg, D + ("qc",), {})
        st_ = core.get_in(cfg, D + ("stats",), {})
        c1, c2 = st.columns(2)
        flags = ["failed", "clipped", "size", "shape", "neighbour", "outside", "low_count", "field"]
        cur = qc.get("exclude_flags", ["failed"])
        cur = flags if cur == ["all"] or cur == "all" else [f for f in (cur or []) if f in flags]
        qc["exclude_flags"] = c1.multiselect(
            "Leave colonies out of the analysis when flagged", flags, default=cur,
            help="Every flag is always reported (report + overlays). Only these remove a "
                 "colony from profiles, scales, gates and tests. 'failed' = no colony found; "
                 "'clipped' = cut by the image border; 'size' = diameter far from the pattern; "
                 "'shape' = not round; 'neighbour' = a big neighboring colony in the field; "
                 "'outside' = many nuclei outside the colony; 'low_count' = few nuclei; "
                 "'field' = nucleus count / DAPI / size unlike its condition's other colonies.")
        rep = ["auto", "colony", "well"]
        st_["replicate"] = c2.radio(
            "Replicate (the n of every test)", rep, horizontal=True,
            index=rep.index(st_.get("replicate", "auto")) if st_.get("replicate", "auto") in rep else 0,
            help=ss.template.help(*D, "stats", "replicate"))
        colony_preview(cfg)

        analyses_to_run(cfg, chosen_signaling)
        output_settings(cfg)

    essentials = {S + ("channels", "dapi_index"), S + ("channels", "marker_channels"),
                  S + ("segmentation", "mode"), S + ("input", "select"),
                  D + ("conditions",), D + ("markers",), D + ("results_dir",),
                  D + ("default_space",), D + ("exclude_markers",), D + ("skip_steps",),
                  D + ("seg_mode",), D + ("steps",),
                  D + ("colony", "diameter_column"), D + ("colony", "expected_diameter_um"),
                  D + ("colony", "pixel_size_um"), D + ("profiles", "bin_um"),
                  D + ("profiles", "n_norm_bins"), D + ("profiles", "colony_binning"),
                  D + ("qc", "exclude_flags"), D + ("stats", "replicate")} | {
                  D + ("signaling", k) for k in ("markers", "reference_column",
                                                 "reference_values", "nuclear_space",
                                                 "ring_inner_um", "ring_outer_um", "gate")}
    with tabs[1]:
        st.caption("Image discovery, channel layout, Cellpose segmentation and intensity "
                   "extraction. Defaults work for most micropattern data.")
        for sec in ("input", "channels", "segmentation", "intensity"):
            st.markdown(f"**{sec}**")
            auto_section(cfg, S + (sec,), essentials)
    with tabs[2]:
        st.caption("How the main colony is found in each image, and when a colony is "
                   "flagged. Try settings on one image with the preview on the Essentials tab.")
        for sec in ("colony", "qc"):
            st.markdown(f"**{sec}**")
            auto_section(cfg, D + (sec,), essentials)
    with tabs[3]:
        for sec in ("profiles", "stats", "compare", "multimarker", "coexpression", "signaling"):
            st.markdown(f"**{sec}**")
            auto_section(cfg, D + (sec,), essentials)
    with tabs[4]:
        st.caption("Only the '% positive vs distance' readout uses these.")
        auto_section(cfg, D + ("thresholds",), essentials)
    with tabs[5]:
        st.markdown("**downstream (general)**")
        auto_section(cfg, D, essentials | {D + (k,) for k in
                                           ("colony", "qc", "profiles", "stats", "compare",
                                            "multimarker", "coexpression", "signaling",
                                            "thresholds",
                                            "visualization")})
        st.markdown("**downstream figures**")
        auto_section(cfg, D + ("visualization",), essentials)
        st.markdown("**segmentation QC figures**")
        auto_section(cfg, S + ("visualization",), essentials)
        st.markdown("**experiment annotation**")
        auto_section(cfg, S + ("experiment",), essentials)

    # -- datasets from step 1 ------------------------------------------------
    ds_entries = []
    for d in included():
        entry = {"name": d["name"], "directory": d["directory"]}
        extra = dict(d.get("extra") or {})
        if d["seg_mode"] != mode:
            extra.setdefault("segmentation", {})["mode"] = d["seg_mode"]
        elif "segmentation" in extra:
            extra["segmentation"] = {k: v for k, v in extra["segmentation"].items() if k != "mode"}
            if not extra["segmentation"]:
                extra.pop("segmentation")
        entry.update(extra)
        ds_entries.append(entry)
    cfg["datasets"] = ds_entries

    # -- save --------------------------------------------------------------------
    st.divider()
    path = exp_path() / core.CONFIG_NAME
    on_disk = path.read_text(encoding="utf-8") if path.exists() else None
    # Written INTO the file on disk: unchanged lines and every comment typed into
    # it stay as they are; only changed values and missing options are written.
    text = ss.template.render_into(cfg, on_disk)
    with st.expander("Preview config.yaml"):
        st.code(text, language="yaml")
    st.caption(("Saved. " if on_disk == text else "Unsaved changes. ") + f"`{path}`")
    if on_disk is not None and on_disk != text:
        import yaml
        try:
            changes = core.changed_keys(yaml.safe_load(on_disk) or {}, cfg)
        except yaml.YAMLError:
            changes = []
        if changes:
            st.caption("Saving changes: " + ", ".join(f"`{c}`" for c in changes[:25])
                       + (f" and {len(changes) - 25} more" if len(changes) > 25 else "")
                       + ". Everything else in the file, comments included, stays as it is.")
    if save_with_confirm(path, text, "config", ss.get("cfg_loaded_text"),
                         "💾 Save config.yaml", disabled=on_disk == text):
        ss.cfg_loaded_text = text
        st.rerun()


# ===========================================================================
# 4 · Check & run (on this computer)
# ===========================================================================

def run_live(cmd: list[str], title: str) -> int:
    st.markdown(f"**{title}**")
    st.caption("`" + " ".join(Path(c).name if i < 3 else c for i, c in enumerate(cmd)) + "`")
    box = st.empty()
    lines: list[str] = []
    code = 1
    for item in core.stream_command(cmd, exp_path()):
        if isinstance(item, int):
            code = item
            break
        lines.append(item)
        if len(lines) % 5 == 0:
            box.code("\n".join(lines[-400:]), language="log")
    box.code("\n".join(lines[-400:]) or "(no output)", language="log")
    errors = [l for l in lines if "[ERROR]" in l]
    if code == 0:
        st.success(f"{title}: finished without errors.")
    else:
        st.error(f"{title}: finished with problems (exit code {code}).")
        if errors:
            st.markdown("**Errors:**\n" + "\n".join(f"- `{e.split('] ', 1)[-1]}`" for e in errors[:15]))
    return code


def unsaved_items() -> list[str]:
    out = []
    path = exp_path() / core.CONFIG_NAME
    if not path.exists():
        out.append("config.yaml has not been saved (step 3)")
    for d in included():
        p = exp_path() / d["directory"] / core.PLATE_MAP_NAME
        if not p.exists():
            out.append(f"no plate map for '{d['name']}' (step 2)")
        elif d["name"] in ss.pm:
            try:
                same = core.plate_map_equal(core.read_yaml(p), core.plate_map_dict(ss.pm[d["name"]]))
            except Exception:                           # noqa: BLE001
                same = False
            if not same:
                out.append(f"plate map of '{d['name']}' has unsaved changes (step 2)")
    return out


def page_check_run() -> None:
    st.header("4 · Check, and run on this computer")
    if not need_experiment():
        return
    for item in unsaved_items():
        st.warning(item)
    cfg_path = exp_path() / core.CONFIG_NAME
    if not cfg_path.exists():
        return
    base = ["--config", str(cfg_path)]

    st.markdown("**Check** reads the config, the plate maps and the file names: no images "
                "are opened and nothing is written. Run it before every HPC submission.")
    if st.button("🔎 Run check", type="primary"):
        run_live(core.main_command("check", *base), "check")

    st.divider()
    st.subheader("Downstream analysis on this computer (CPU)")
    st.caption("Needs the segmentation results (per_nuclei_intensities.joblib): normally "
               "produced by the HPC GPU job. The downstream analysis itself needs no GPU and "
               "takes minutes; you can also run it on the HPC with run_downstream.lsf.")
    names = [d["name"] for d in included()]
    segmented = [d["name"] for d in included()
                 if core.has_segmentation(exp_path() / d["directory"])]
    picked = st.multiselect(
        "Datasets to analyze", names, default=segmented or names, key=f"_ds_sets{ss.gen}",
        help="Pre-selected: the datasets that have segmentation results. All selected "
             "datasets are analyzed together into one analysis folder; a new run "
             "replaces the previous results there.")
    ds_args = ["--datasets", *picked] if picked and set(picked) != set(names) else []
    steps = core.downstream_step_names()
    c1, c2 = st.columns([3, 1])
    chosen = c1.multiselect("Steps (empty = all)", steps, key="_ds_steps")
    force = c2.checkbox("Recompute (ignore checkpoints)", key="_ds_force")
    c1, c2 = st.columns(2)
    if c1.button("Build manifest", disabled=not picked):
        run_live(core.main_command("manifest", *base, *ds_args), "manifest")
    if c2.button("▶ Run downstream analysis", disabled=not picked):
        args = ["downstream", *base, *ds_args]
        if chosen:
            args += ["--steps", *chosen]
        if force:
            args += ["--force"]
        code = run_live(core.main_command("manifest", *base, *ds_args), "manifest")
        if code == 0:
            run_live(core.main_command(*args), "downstream")

    import yaml
    try:
        saved_cfg = core.read_yaml(cfg_path)
    except (OSError, yaml.YAMLError):
        saved_cfg = ss.cfg
    statuses = core.segmentation_status(exp_path(), saved_cfg)
    results = core.analysis_folder(saved_cfg, statuses)
    candidates = [results] + sorted(
        f"./{d.name}" for d in exp_path().iterdir()
        if d.is_dir() and d.name.startswith("analysis") and (d / "report.html").exists()
        and f"./{d.name}" != results)
    found = [c for c in candidates if (exp_path() / c / "report.html").exists()]
    st.caption(f"This run writes to `{results}`.")
    for c in found:
        report = (exp_path() / c / "report.html").resolve()
        c1, c2 = st.columns([4, 1])
        c1.success(f"Report: `{report}`")
        if c2.button("Open", key=f"open_{c}"):
            webbrowser.open(report.as_uri())

    with st.expander("Segmentation on this computer (needs Cellpose + ideally a GPU)"):
        st.caption("Normally done on the HPC (step 5). Useful for a quick test on a few "
                   "images: set the image subset under Analysis settings first.")
        seg_sets = st.multiselect("Datasets to segment", names, default=names,
                                  key=f"_seg_sets{ss.gen}")
        seg_args = (["--datasets", *seg_sets] if seg_sets and set(seg_sets) != set(names)
                    else [])
        c1, c2 = st.columns(2)
        seg_steps = c1.multiselect("Steps", ["load", "segment", "extract"],
                                   default=["load", "segment", "extract"], key="_seg_steps")
        if c2.button("Run segmentation here", disabled=not seg_sets):
            run_live(core.main_command("process", *base, *seg_args, "--steps", *seg_steps),
                     "process")


# ===========================================================================
# 5 · HPC job
# ===========================================================================

def page_hpc() -> None:
    st.header("5 · HPC job")
    if not need_experiment():
        return
    st.markdown(
        "The HPC job files never change. They read their settings from "
        "**`hpc_job.env` in the experiment folder**, this page writes it, and you "
        "submit from that folder. Segmentation runs on a GPU node; the downstream "
        "analysis on a normal CPU node.")
    prefs = ss.prefs
    env_path = exp_path() / core.ENV_NAME
    env = core.env_defaults()
    if env_path.exists():
        env.update(core.parse_env(env_path.read_text(encoding="utf-8")))
    loaded_env = ss.setdefault("env_loaded_text", {}).setdefault(
        str(env_path), env_path.read_text(encoding="utf-8") if env_path.exists() else None)

    st.subheader("How the HPC sees the lab share")
    c1, c2 = st.columns(2)
    local_root = c1.text_input("Share on this computer", value=prefs.get(
        "local_share_root", core.guess_share_root(exp_path())),
        help="The folder where the lab share is mounted on your laptop, e.g. Z:\\ or "
             "/Volumes/warmflash")
    hpc_root = c2.text_input("Same share on the HPC", value=prefs.get("hpc_share_root",
                                                                      "/data/Warmflash"))
    if (local_root, hpc_root) != (prefs.get("local_share_root"), prefs.get("hpc_share_root")):
        prefs.update(local_share_root=local_root, hpc_share_root=hpc_root)
        core.save_prefs(prefs)
    hpc_exp = core.to_hpc_path(exp_path(), local_root, hpc_root)
    st.caption(f"This experiment on the HPC: `{hpc_exp}`")
    if not hpc_exp.startswith(hpc_root.rstrip("/")):
        st.warning("The experiment folder is not inside the share given above, so its HPC "
                   "path could not be worked out. Fix 'Share on this computer'.")

    # The pipeline lives on the share too; translate its own location.
    default_pipe = core.to_hpc_path(core.PIPELINE_DIR, core.guess_share_root(core.PIPELINE_DIR),
                                    hpc_root)
    remembered = prefs.get("hpc_pipeline_dir", "")
    if not remembered.startswith(hpc_root.rstrip("/")):
        remembered = default_pipe
    c1, c2 = st.columns(2)
    env["PIPELINE_DIR"] = c1.text_input(
        "Pipeline folder on the HPC", value=env.get("PIPELINE_DIR") if env_path.exists()
        else remembered, help="The micropattern_pipeline folder as the HPC sees it.")
    for label, val in (("Pipeline folder", env["PIPELINE_DIR"]), ("Experiment", hpc_exp)):
        if not val.startswith("/") or ":" in val:
            st.warning(f"{label} on the HPC looks like a path on this computer (`{val}`). "
                       "HPC paths start with /. Fix the share settings above.")
    env["VENV_DIR"] = c2.text_input("Python environment on the HPC",
                                    value=env.get("VENV_DIR") or prefs.get("hpc_venv", ""),
                                    help="Each job checks it first and installs only what is "
                                         "missing; the first job builds it.")
    prefs.update(hpc_pipeline_dir=env["PIPELINE_DIR"], hpc_venv=env["VENV_DIR"])
    core.save_prefs(prefs)

    st.subheader("Segmentation job (GPU)")
    names2d = [d["name"] for d in included() if d["seg_mode"] == "2d"]
    names3d = [d["name"] for d in included() if d["seg_mode"] == "3d"]
    c1, c2, c3 = st.columns(3)
    env["SEG_STEPS"] = " ".join(c1.multiselect(
        "Steps", ["load", "segment", "extract"],
        default=[s for s in env.get("SEG_STEPS", "load segment extract").split() if s]))
    all_names = [d["name"] for d in included()]
    env["DATASETS_2D"] = " ".join(c2.multiselect("Folders for the 2-D job", all_names,
                                                 default=names2d))
    env["DATASETS_3D"] = " ".join(c3.multiselect("Folders for the 3-D job", all_names,
                                                 default=names3d))
    env["THEN_ANALYSIS"] = "true" if st.checkbox(
        "Start the analysis by itself when segmentation has finished",
        value=env.get("THEN_ANALYSIS", "true") != "false",
        help="The segmentation job submits run_downstream.lsf when it succeeds; with a "
             "2-D and a 3-D job, the one finishing last does it.") else "false"
    c1, c2, c3 = st.columns(3)
    env["FORCE"] = "true" if c1.checkbox("Redo everything (ignore finished images)",
                                         value=env.get("FORCE") == "true") else "false"
    env["SEG_FORCE_STEPS"] = " ".join(c2.multiselect(
        "Redo only these steps", ["load", "segment", "extract"],
        default=[s for s in env.get("SEG_FORCE_STEPS", "").split() if s]))

    st.subheader("Which images (both jobs)")
    st.caption("Empty = whatever config.yaml says (by default: all images). Values here "
               "override config.yaml for the HPC run only.")
    c1, c2 = st.columns(2)
    env["SEQ"] = c1.text_input("Seq numbers", value=env.get("SEQ", ""), placeholder="e.g. 0-9, 15")
    env["WELLS"] = c2.text_input("Wells", value=env.get("WELLS", ""), placeholder="e.g. 1-3")
    try:
        parse_index_spec(env["SEQ"]), parse_well_spec(env["WELLS"])
    except ValueError as exc:
        st.error(str(exc))

    st.subheader("Downstream job (CPU)")
    c1, c2, c3 = st.columns(3)
    env["RUN_CHECK"] = "true" if c1.checkbox("Run check first", env.get("RUN_CHECK", "true") == "true") else "false"
    env["RUN_MANIFEST"] = "true" if c2.checkbox("Rebuild manifest", env.get("RUN_MANIFEST", "true") == "true") else "false"
    env["DOWNSTREAM_FORCE"] = "true" if c3.checkbox("Recompute all steps",
                                                    env.get("DOWNSTREAM_FORCE") == "true") else "false"
    steps = core.downstream_step_names()
    env["DOWNSTREAM_STEPS"] = " ".join(st.multiselect(
        "Steps (empty = all)", steps,
        default=[s for s in env.get("DOWNSTREAM_STEPS", "").split() if s in steps]))

    with st.expander("Cluster modules and logging"):
        c1, c2, c3 = st.columns(3)
        env["PYTHON_MODULE"] = c1.text_input("Python module", env.get("PYTHON_MODULE", ""))
        env["CUDA_MODULE"] = c2.text_input("CUDA module", env.get("CUDA_MODULE", ""),
                                           help="cuda/12.8 for the Blackwell GPUs on gpu-b6000")
        env["LOG_LEVEL"] = c3.selectbox("Log level", ["INFO", "DEBUG", "WARNING"],
                                        index=["INFO", "DEBUG", "WARNING"].index(env.get("LOG_LEVEL", "INFO"))
                                        if env.get("LOG_LEVEL", "INFO") in ["INFO", "DEBUG", "WARNING"] else 0)
        env["CONFIG"] = st.text_input("Config file (relative to the experiment)",
                                      env.get("CONFIG", "config.yaml"))
        c1, c2 = st.columns([3, 1])
        env["TORCH_INDEX"] = c1.text_input(
            "PyTorch wheel index", env.get("TORCH_INDEX") or
            "https://download.pytorch.org/whl/cu128",
            help="cu128 = CUDA 12.8, the build with Blackwell (sm_120) kernels that matches "
                 "cuda/12.8. Do not drop to cu126/cu121: those install but cannot use the "
                 "GPU. The job checks the installed torch against this.")
        fresh = c2.checkbox("Rebuild the environment", value=str(env.get("FRESH_VENV", "false"))
                            .lower() == "true",
                            help="Delete the Python environment and build it again before "
                                 "the next job (slow). Only when it is broken in a way the "
                                 "automatic check does not repair; untick it afterwards.")
        env["FRESH_VENV"] = "true" if fresh else "false"

    text = core.render_env(env)
    on_disk = env_path.read_text(encoding="utf-8") if env_path.exists() else None
    with st.expander("Preview hpc_job.env"):
        st.code(text, language="bash")
    st.caption(("Saved. " if on_disk == text else "Unsaved changes. ") + f"`{env_path}`")
    if save_with_confirm(env_path, text, "env", loaded_env, "💾 Save hpc_job.env",
                         disabled=on_disk == text):
        ss["env_loaded_text"][str(env_path)] = text
        st.rerun()

    for item in unsaved_items():
        st.warning(item)

    st.subheader("Then, on the HPC")
    pipe = env["PIPELINE_DIR"].rstrip("/")
    q = f"'{hpc_exp}'"
    jobs = []
    if env["DATASETS_2D"] or not env["DATASETS_3D"]:
        jobs.append(("2-D", env["DATASETS_2D"] or "every folder",
                     f"{pipe}/hpc/run_pipeline_2dseg.lsf"))
    if env["DATASETS_3D"]:
        jobs.append(("3-D", env["DATASETS_3D"], f"{pipe}/hpc/run_pipeline_3dseg.lsf"))
    chain = env.get("THEN_ANALYSIS", "true") == "true"
    st.markdown("**Run everything.** Log in to the HPC (`ssh …`) and type:")
    lines = [f"cd {q}"]
    for label, which, lsf in jobs:
        lines.append(f"bsub < {lsf}")
    if not chain:
        lines.append(f"bsub < {pipe}/hpc/run_downstream.lsf")
    width = max(len(l) for l in lines) + 3
    notes = ["# go to the experiment folder"]
    notes += [f"# segmentation {label} on a GPU ({which})" for label, which, _ in jobs]
    if not chain:
        notes.append("# the analysis on a CPU: submit when segmentation has finished")
    st.code("\n".join(f"{l:<{width}}{n}" for l, n in zip(lines, notes)), language="bash")
    st.caption(
        ("The analysis (CPU job) **starts by itself** when segmentation has finished "
         "successfully" + (" (when both jobs have)" if len(jobs) > 1 else "") + ". "
         if chain else "Automatic start of the analysis is off (Segmentation job, above). ")
        + "Results: `report.html` in the analysis folder of the experiment. "
          "Check on jobs with `bjobs`; logs (`lsf_*.log`, `pipeline.log`) land in the "
          "experiment folder.")

    run = f"bash {pipe}/hpc/run.sh"
    st.markdown("**Re-running part of it.** Always from the experiment folder (`cd` first). "
                "Everything is checkpointed: finished images and up-to-date steps are skipped, "
                "and a changed setting re-runs what depends on it by itself.")
    rerun = pd.DataFrame([
        ("Changed analysis settings (step 3) or a plate map (step 2)",
         f"bsub < {pipe}/hpc/run_downstream.lsf", "the analysis only; no GPU"),
        ("Redo one analysis step anyway (e.g. fresh colony overlays)",
         f"{run} redo colony", "that step and every step after it"),
        ("Redo the whole analysis", f"{run} redo-all", "ignores every analysis checkpoint"),
        ("Changed segmentation settings, or masks look wrong",
         f"{run} reseg", "segments every image again, then the analysis"),
        ("Try it on a few images first", f'SEQ="0-4" {run}', "any command; also WELLS=, DATASETS="),
        ("Only the 3-D (or 2-D) segmentation", f"SEG=3d {run} seg", "no analysis afterwards"),
        ("See what a job would do, without the GPU", f"{run} test", "dry run, nothing submitted"),
        ("Where are my jobs?", f"{run} status", "bjobs + the end of the newest logs"),
    ], columns=["When", "Type (after cd)", "What it does"])
    st.dataframe(rerun, hide_index=True, width="stretch")
    st.caption(f"`{run} help` lists every option. The options change one submission only; "
               "hpc_job.env stays as saved here.")

    with st.expander("Optional: build or check the Python environment ahead of time"):
        st.caption("Not required: every job checks the environment first (a few seconds) "
                   "and installs only what is missing or wrong, PyTorch from the CUDA 12.8 "
                   "index. Run this on the login node to have it ready before the first job, "
                   "or to see whether it is healthy.")
        st.code(f"bash {pipe}/hpc/setup_env.sh {env['VENV_DIR'] or '/scratch/$USER/micropattern_venv'}",
                language="bash")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    init_state()
    if not ss.loaded and not ss.get("_auto_tried"):
        ss["_auto_tried"] = True          # once per session: a failure never loops
        last = Path(ss.exp_dir).expanduser() if ss.exp_dir else None
        if last is not None and last.is_dir():
            with st.spinner(f"Opening the last experiment, {last.name}…"):
                try:
                    load_experiment(last)
                except Exception as exc:              # noqa: BLE001
                    st.warning(f"Could not reopen {last}: {exc}")
    with st.sidebar:
        st.title("🔬 Micropattern pipeline")
        page = st.radio("Step", STEPS, key="_page", label_visibility="collapsed")
        st.divider()
        if ss.loaded:
            st.caption(f"Experiment:\n\n`{exp_path().name}`")
            st.caption(f"{len(included())} folder(s) selected")
        st.caption(f"Pipeline: `{core.PIPELINE_DIR}`")
    {STEPS[0]: page_experiment, STEPS[1]: page_plate_maps, STEPS[2]: page_analysis,
     STEPS[3]: page_check_run, STEPS[4]: page_hpc}[page]()


main()
