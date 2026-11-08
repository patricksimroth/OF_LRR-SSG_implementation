#!/usr/bin/env python3
"""
profile_validation.py - wall-normal profiles of every run against its
blockMesh counterpart, one folder per x station.

For every station in STATIONS (a subset of the ten `sampleUyPlus` lines) and
every quantity in QUANTITIES it writes

    summary/profiles/x_<x>m/<quantity>.png           (<quantity>_<model>.png
                                                      when a study has several)

Each figure is a grid of panels - one row per prism-stack height h+, one
column per mesh approach - with one line per wall core cell size L_nom/delta_mean
(same colours as the core_refinement summary figures; both normalisations are
the analytical ones of _pp_common.flow_scales) and the same model's blockMesh run as a black
dashed reference in every panel.  Dotted: the row's prism-stack top; dash-dot-dot:
the blockMesh delta99 at that station.  Wall distance y/delta is on the vertical
axis, delta the analytical delta99 at the station, logarithmic over
pp.Y_DELTA_RANGE as the budget profiles, because the near-wall stress peak
sits at y+ ~ 15, inside the first 1 % of the boundary layer.

Quantities: Ux/U_inf, nu_t/nu, k, Rxx, Ryy, Rzz, -Rxy.  R is the transported
`R` where the model solves it (RSM), else `turbulenceProperties:R` - the
Boussinesq stresses of an eddy-viscosity model, whose normal stresses are
isotropic (= 2k/3) by construction.  k is the sampled k, else half the trace
of R; for SA-type models that k is a reconstruction (Bradshaw), not
transported.

TMR curves are added only where the closure matches: SSG/LRR-RSM-w2012
(CFL3D) for R and k of runs that transport R, SA (CFL3D) for nu_t of the SA
family.  Both come from `_validation_data/tmr_*.dat`, extracted by
`_tmr_extract_profiles.py`.

Without a study manifest (run_simulation.sh) every run is one line in a single
panel and there is no blockMesh reference.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import matplotlib

matplotlib.use("Agg")

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# pyrefly: ignore [missing-import]
import numpy as np

import _pp_common as pp

pp.report_style()

RSTRESS_REFERENCE = "tmr_rstress_ssglrrrsm.dat"
NUT_REFERENCE = "tmr_nut_sa.dat"
RSTRESS_LABEL = "TMR SSG/LRR"
NUT_LABEL = "TMR SA"

# Models whose nu_t is comparable with the TMR SA reference.
SA_FAMILY = ("SpalartAllmarasSALSA", "SpalartAllmaras")

STATIONS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 1.95)

APPROACH_ORDER = pp.APPROACHES

# Coarsest to finest, as in studies/core_refinement/analyze.py.
LINE_COLORS = pp.CORE_COLORS
LINE_MARKERS = pp.CORE_MARKERS
INK, MUTED, EDGE = pp.INK, pp.MUTED, pp.EDGE
# Wide translucent band under the black blockMesh dashes: the two often coincide.
TMR_COLOR = (0.0, 0.62, 0.69, 0.4)
D99_STYLE = (0, (6, 1.5, 1, 1.5, 1, 1.5))

# `type face` sampling returns the wall face itself (y ~ 1e-19 m, sometimes
# slightly negative) as the first point.
WALL_POINT_Y = 1e-9

# Shared value axis is capped at this multiple of the reference's maximum, so
# one run far off does not flatten every other panel.
VALUE_CAP = 3.0

PROFILE_DIR = "profiles"


def station_from_tag(tag):
    """0.2 from `x0p2`, 1.0 from `x1`, 1.95 from `x1p95`."""
    try:
        return float(tag.lstrip("x").replace("p", "."))
    except ValueError:
        return None


def load_reference(val_dir, name):
    """{station: {column: ndarray}} from a `_validation_data` profile file."""
    path = os.path.join(val_dir, name)
    if not os.path.exists(path):
        print(f"  [warn] reference not found: {path}")
        return {}
    header = [line for line in open(path) if line.lower().startswith("variables")]
    if not header:
        print(f"  [warn] no variables line in {path}")
        return {}
    columns = [c.strip().strip('"') for c in header[0].split("=", 1)[1].split(",")]
    out = {}
    for zone, arr in pp.parse_tecplot_zones(path).items():
        station = station_from_tag(zone.split("=")[-1].strip())
        if station is None or arr.shape[1] != len(columns):
            continue
        out[station] = dict(zip(columns, arr.T))
    print(f"  reference {name}: {len(out)} stations, columns {columns[1:]}")
    return out


def read_run_profiles(case_dir):
    """{station: {column: ndarray}} for one run, from sampleUyPlus."""
    sample_dir = pp.latest_time_directory(Path(case_dir) / "postProcessing" / "sampleUyPlus")
    if sample_dir is None:
        return {}
    out = {}
    for path in sorted(glob.glob(os.path.join(sample_dir, "*_profile_*.xy"))):
        station = station_from_tag(os.path.basename(path).split("_profile_", 1)[0])
        if station is None:
            continue
        cols, arr = pp.read_profile_set(path)
        if cols is None:
            print(f"  [warn] unreadable column layout: {os.path.basename(path)}")
            continue
        out[station] = dict(zip(cols, arr.T))
    return out


def sampled_delta99(profile):
    """delta99 [m] of one sampled station, same rule as bl_thickness.py:
    U_e = max Ux on the line, first wall distance where Ux reaches 0.99 U_e."""
    y, ux = profile.get("y"), profile.get("Ux")
    if y is None or ux is None or len(y) < 2:
        return None
    order = np.argsort(y)
    y, ux = y[order], ux[order]
    target = 0.99 * float(ux.max())
    above = np.flatnonzero(ux >= target)
    if target <= 0 or above.size == 0:
        return None
    i = int(above[0])
    if i == 0:
        return float(y[0])
    y0, y1, u0, u1 = y[i - 1], y[i], ux[i - 1], ux[i]
    return float(y0 + (0.0 if u1 == u0 else (target - u0) / (u1 - u0)) * (y1 - y0))


def _column(profile, *names):
    for name in names:
        if name in profile:
            return profile[name]
    return None


def _k(profile):
    """The sampled k, else half the trace of R."""
    k = _column(profile, "k")
    if k is not None:
        return k
    for prefix in ("R_", "R_tp_"):
        parts = [profile.get(f"{prefix}{c}") for c in ("xx", "yy", "zz")]
        if all(p is not None for p in parts):
            return 0.5 * (parts[0] + parts[1] + parts[2])
    return None


def _scaled(values, factor):
    return None if values is None else values * factor


def _stress(component, sign=1.0):
    return lambda p, u_inf, nu: _scaled(_column(p, f"R_{component}", f"R_tp_{component}"), sign)


def _tmr_k(ref):
    return 0.5 * (ref["Rxx"] + ref["Ryy"] + ref["Rzz"])


# (file stem, axis label, value(profile, u_inf, nu), TMR source, TMR value(ref, nu))
#   TMR source: "rstress" (only for runs that transport R), "nut" (SA family)
QUANTITIES = (
    ("Ux", r"$U_x\,/\,U_\infty$",
     lambda p, u_inf, nu: _scaled(_column(p, "Ux"), 1.0 / u_inf), None, None),
    ("nut", r"$\nu_t\,/\,\nu$",
     lambda p, u_inf, nu: _scaled(_column(p, "nut"), 1.0 / nu),
     "nut", lambda ref, nu: ref["nut"] / nu),
    ("k", r"$k$ [m$^2$/s$^2$]", lambda p, u_inf, nu: _k(p), "rstress",
     lambda ref, nu: _tmr_k(ref)),
    ("Rxx", r"$\overline{u'u'}$ [m$^2$/s$^2$]", _stress("xx"), "rstress",
     lambda ref, nu: ref["Rxx"]),
    ("Ryy", r"$\overline{v'v'}$ [m$^2$/s$^2$]", _stress("yy"), "rstress",
     lambda ref, nu: ref["Ryy"]),
    ("Rzz", r"$\overline{w'w'}$ [m$^2$/s$^2$]", _stress("zz"), "rstress",
     lambda ref, nu: ref["Rzz"]),
    ("Rxy", r"$-\overline{u'v'}$ [m$^2$/s$^2$]", _stress("xy", -1.0), "rstress",
     lambda ref, nu: -ref["Rxy"]),
)

# k and R of an eddy-viscosity model are a modelling construct, not a
# transported field: the caption says so (_captions.py), not the figure.


def transports_r(profiles):
    return any("R_xx" in prof for prof in profiles.values())


def layout(names, index):
    """One group per model: panels (h+ x approach), lines (L_nom/delta_mean).

    Returns [{model, reference, rows, cols, cells{(row, col): [run]},
    label{run: str}, order[str]}].  The reference is the model's blockMesh
    run; a study of blockMesh runs only is plotted as is, without one.
    """
    by_model = {}
    for name in names:
        by_model.setdefault(index.get(name, {}).get("model"), []).append(name)

    groups = []
    for model, members in sorted(by_model.items(), key=lambda kv: str(kv[0])):
        base = sorted(n for n in members if index.get(n, {}).get("approach") == pp.ERROR_BASELINE)
        others = sorted(n for n in members if n not in base)
        reference = None
        if base and others:
            reference = base[0]
            if len(base) > 1:
                print(f"  [warn] {len(base)} {pp.ERROR_BASELINE} runs for {model}; "
                      f"reference = {reference}")
        else:
            others = sorted(members)

        def prisms(name):
            value = pp.h_plus(index.get(name)) if name in index else None
            return None if value is None else int(round(value))

        rows = sorted({prisms(n) for n in others}, key=lambda v: (v is None, v or 0))
        present = {index.get(n, {}).get("approach") for n in others}
        cols = [a for a in APPROACH_ORDER if a in present] + sorted(
            (a for a in present if a not in APPROACH_ORDER), key=str)

        sizes = {n: pp.dx_norm(index[n]) if n in index else None for n in others}
        if others and all(sizes.values()):
            label = {n: pp.fmt_dx(sizes[n]) for n in others}
            order = [label[n] for n in sorted(others, key=lambda n: -sizes[n])]
        else:
            keys = [k for k in pp.swept_keys([index[n] for n in others if n in index])
                    if k not in pp.STACK_KEYS]
            label = {n: pp.variant_label(index.get(n, {}), keys) or n for n in others}
            order = sorted(set(label.values()), key=pp._label_order)
        order = list(dict.fromkeys(order))

        cells = {}
        for name in others:
            key = (prisms(name), index.get(name, {}).get("approach"))
            cells.setdefault(key, []).append(name)
        groups.append(dict(model=model, reference=reference, rows=rows, cols=cols,
                           cells=cells, label=label, order=order))
    return groups


def mark_positions(y, slot, n_slots, n=6):
    """Marker indices spread evenly in log y, staggered per line."""
    u = np.log10(np.maximum(y, pp.Y_DELTA_RANGE[0]))
    lo, hi = np.log10(pp.Y_DELTA_RANGE)
    targets = lo + (hi - lo) * (np.arange(n) + (slot + 1) / (n_slots + 1)) / n
    return sorted({int(np.argmin(np.abs(u - t))) for t in targets})


def curve(profile, value_fn, u_inf, nu, delta):
    """(y/delta, value) of one sampled station without the wall face, or None."""
    if profile is None:
        return None
    values = value_fn(profile, u_inf, nu)
    if values is None:
        return None
    y = profile["y"]
    keep = y > WALL_POINT_Y
    return y[keep] / delta, values[keep]


def plot_station(group, runs, index, station, quantity, out_path, u_inf, nu, tmr, study):
    stem, axis_label, value_fn, tmr_source, tmr_fn = quantity
    rows, cols, cells = group["rows"], group["cols"], group["cells"]
    style = {lab: (LINE_COLORS[i % len(LINE_COLORS)], LINE_MARKERS[i % len(LINE_MARKERS)])
             for i, lab in enumerate(group["order"])}

    delta = pp.flow_scales(u_inf, nu, station)["delta_L"]
    reference = group["reference"]
    ref_curve = d99 = None
    if reference:
        ref_profile = runs[reference].get(station)
        ref_curve = curve(ref_profile, value_fn, u_inf, nu, delta)
        d99 = sampled_delta99(ref_profile) if ref_profile is not None else None

    members = [n for names in cells.values() for n in names]
    model = group["model"]
    rsm = any(transports_r(runs[n]) for n in members)
    tmr_curve, tmr_label = None, None
    if tmr_source == "rstress" and rsm:
        ref = tmr["rstress"].get(station)
        if ref is not None:
            tmr_curve, tmr_label = (ref["y"] / delta, tmr_fn(ref, nu)), RSTRESS_LABEL
    elif tmr_source == "nut" and str(model).startswith(SA_FAMILY):
        ref = tmr["nut"].get(station)
        if ref is not None:
            tmr_curve, tmr_label = (ref["y"] / delta, tmr_fn(ref, nu)), NUT_LABEL
    if tmr_curve is not None:
        keep = tmr_curve[0] > 0
        tmr_curve = (tmr_curve[0][keep], tmr_curve[1][keep])

    n_rows, n_cols = len(rows), len(cols)
    fig, axes = pp.grid_figure(n_rows, n_cols, row_height=2.0, sharex=True, sharey=True)
    plotted, extremes = set(), []
    for i, n_prisms in enumerate(rows):
        for j, approach in enumerate(cols):
            ax = axes[i][j]
            names = sorted(cells.get((n_prisms, approach), []),
                           key=lambda n: group["order"].index(group["label"][n]))
            for name in names:
                data = curve(runs[name].get(station), value_fn, u_inf, nu, delta)
                if data is None:
                    continue
                lab = group["label"][name]
                color, marker = style[lab]
                slot = group["order"].index(lab)
                ax.plot(data[1], data[0], color=color, marker=marker, markersize=3.5,
                        markeredgecolor="white", markeredgewidth=0.4, linewidth=1.0,
                        markevery=mark_positions(data[0], slot, len(group["order"])))
                plotted.add(lab)
                lo, hi = pp.Y_DELTA_RANGE
                extremes.append(data[1][(data[0] >= lo) & (data[0] <= hi)])
            if ref_curve is not None:
                ax.plot(ref_curve[1], ref_curve[0], **pp.BASELINE_STYLE)
            if tmr_curve is not None:
                ax.plot(tmr_curve[1], tmr_curve[0], color=TMR_COLOR, linewidth=4.0,
                        solid_capstyle="butt", zorder=4)
            h = next(((index.get(n, {}).get("mesh") or {}).get("totalLayerThickness")
                      for n in names), None)
            if h:
                ax.axhline(h / delta, **pp.STACK_TOP_STYLE)
            if d99:
                ax.axhline(d99 / delta, color=EDGE, linestyle=D99_STYLE, linewidth=0.9)
            if not names:
                ax.text(0.5, 0.5, "no run", transform=ax.transAxes, ha="center",
                        va="center", color=EDGE)
            pp.log_y_delta_axis(ax)
    pp.label_grid(axes, [pp.fmt_hplus(r) if r is not None else "" for r in rows],
                  cols, r"$y/\delta$", axis_label)

    if not plotted:
        plt.close(fig)
        return False

    finite = np.concatenate([e[np.isfinite(e)] for e in extremes] + [np.zeros(1)])
    top, bottom = float(finite.max()), float(min(0.0, finite.min()))
    if ref_curve is not None:
        ref_vals = ref_curve[1][(ref_curve[0] >= pp.Y_DELTA_RANGE[0])
                                & (ref_curve[0] <= pp.Y_DELTA_RANGE[1])]
        if ref_vals.size and ref_vals.max() > 0:
            top = min(max(top, float(ref_vals.max())), VALUE_CAP * float(ref_vals.max()))
            bottom = max(bottom, -VALUE_CAP * float(ref_vals.max()))
    if top > bottom:
        pad = 0.05 * (top - bottom)
        axes[0][0].set_xlim(bottom - (pad if bottom < 0 else 0.0), top + pad)

    handles = [pp.line_handle(lab, style[lab][0], "-", style[lab][1]) for lab in group["order"] if lab in plotted]
    if ref_curve is not None:
        handles.append(pp.line_handle(pp.ERROR_BASELINE, INK, "--"))
    if tmr_curve is not None:
        handles.append(pp.line_handle(tmr_label, TMR_COLOR, linewidth=4.0))
    handles.append(pp.line_handle(pp.STACK_TOP_LABEL, MUTED, ":"))
    if d99:
        handles.append(pp.line_handle(rf"$\delta_{{99}}$, {pp.ERROR_BASELINE}", EDGE, D99_STYLE))
    rect = pp.legend_below(fig, handles)
    fig.tight_layout(rect=rect, h_pad=0.6, w_pad=0.4)
    fig.savefig(out_path)
    plt.close(fig)
    return True


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    _, case_name, run_case_dir, source_case_dir, results_root = \
        pp.resolve_case_paths(script_dir)
    print("=" * 64)
    print(f"{case_name} - wall-normal profiles per station")
    print("=" * 64)

    val_dir = os.path.join(source_case_dir, "_validation_data")
    tmr = {"rstress": load_reference(val_dir, RSTRESS_REFERENCE),
           "nut": load_reference(val_dir, NUT_REFERENCE)}

    runs = {}
    for run, _latest in pp.solved_run_dirs(run_case_dir):
        profiles = read_run_profiles(run)
        if profiles:
            runs[run.name] = profiles
        else:
            print(f"  [warn] {run.name}: no sampleUyPlus profiles")
    if not runs:
        print(f"[warn] no runs with sampled profiles under {run_case_dir}")
        return 0
    print(f"Runs: {len(runs)}")

    u_inf, nu, _rho = pp.detect_case_parameters(os.path.join(run_case_dir, sorted(runs)[0]))
    index = pp.combo_index(pp.load_manifest(run_case_dir))
    groups = layout(sorted(runs), index)
    study = Path(results_root).name

    out_root = Path(results_root) / pp.SUMMARY_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    # Outputs of a previous call with other stations.  Station subfolders
    # (budget/ of profile_budget.py) are not ours.
    for stale in (out_root / PROFILE_DIR).glob("x_*m/*.png"):
        stale.unlink()

    available = sorted({s for profiles in runs.values() for s in profiles})
    tasks = []
    for station in STATIONS:
        match = [s for s in available if np.isclose(s, station)]
        if not match:
            print(f"  [warn] station x = {station:g} m not sampled - skipped")
            continue
        out_dir = out_root / PROFILE_DIR / f"x_{station:.2f}m"
        out_dir.mkdir(parents=True, exist_ok=True)
        for quantity in QUANTITIES:
            for group in groups:
                suffix = f"_{group['model']}" if len(groups) > 1 else ""
                tasks.append((group, match[0], quantity, out_dir / f"{quantity[0]}{suffix}.png"))
        print(f"  x = {station:g} m -> {out_dir}")

    def draw(task):
        group, station, quantity, path = task
        return plot_station(group, runs, index, station, quantity, path, u_inf, nu, tmr, study)

    written = sum(pp.map_processes(draw, tasks, pp.plot_jobs()))
    print(f"\nProfiles complete ({written} figure(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
