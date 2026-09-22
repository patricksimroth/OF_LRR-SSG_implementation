"""
prism_coverage.py – accuracy once the boundary layer outgrows the prism stack.

Reference is the baseline approach (blockMesh: structured, prism-free).  Every
figure uses the baseline's wall units and delta99, so the axes are the same for
every run and do not move with a run's own error.  Into
results/<case>/<study>/validation/:

  prism_coverage_outgrown.png
      delta99(x) per run, one panel per prism approach, with the stack height h
      (dotted) and x_crit (open symbol), the station where the baseline delta99
      reaches h.
  prism_coverage_history.png
      Signed error vs the baseline of Cf, H, delta*, delta99 over the distance
      since the stack was outgrown, x - x_crit in metres (unscaled, so every
      run shares the same axis).
  prism_coverage_region_history.png
      RMS velocity error per boundary-layer region (sublayer y+ < 5, buffer
      < 30, log < 0.2 delta99+, wake < delta99+, outer < 1.5 delta99+) on the
      same axis, plus a top row over all regions together; drawn bold where
      the prism top lies in that region.
  prism_coverage_region_history_<field>.png
      The same for every turbulence field the runs share (nut, nuTilda, k,
      omega, epsilon - see QUANTITIES).  nut and nuTilda are normalised by the
      effective viscosity nu + q_bM, k by its local profile peak.
  prism_coverage_prism_top.png
      Prism-top height h+ along the plate over the region bands.

Only runs whose stack covers the boundary layer at X_RANGE[0] and is outgrown
before X_RANGE[1] have an x_crit; the history figures show only those.

Profiles are cell centroids (_pv_cell_profiles.py via pvbatch), not ParaView's
point data: the reader's cell-to-point filter averages neighbouring cells
unweighted, which on differently stretched meshes faked +2.8 % at the wall
between two runs whose wall shear agrees to 0.04 %.  The extraction is cached
per run in results/<run>/boundary_layer/cell_profiles.npz.

U_e is the freestream Ux at Y_FREE, not max(Ux): on coarse polygonal cores the
profile overshoots the freestream by 1-2 % at the edge once the stack is
outgrown, and max(Ux) then inflates the theta error from ~5 % to ~85 %.
"""
from __future__ import annotations

import subprocess
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
from mesh_study import cf_curve
from profile_validation import ERROR_BASELINE, LEGEND_WIDTH_IN, variant_colors

EXTRACT_SCRIPT = "_pv_cell_profiles.py"
CACHE_NAME = "cell_profiles.npz"
X_GRID = (0.02, 1.98, 197)
# Leading edge singular, last faces influenced by the outlet BC.
X_RANGE = (0.1, 1.95)
# Freestream probe height: ~2x delta99 at the outlet and below the top of the
# extraction grid in _pv_cell_profiles.py (0.06 m).
Y_FREE = 0.055
TIMEOUT_S = 1800
# Measured drift of Cf at the stop criterion - differences inside are noise.
CF_BAND_PCT = 0.15

YP = np.logspace(-1, 4, 260)
LOG_TOP = 0.2
REGIONS = ("sublayer", "buffer", "log", "wake", "outer")
REGION_COLOR = dict(zip(REGIONS, ("#4c72b0", "#55a868", "#c4a000", "#dd8452", "#8c8c8c")))
HISTORY_LABEL = r"$x - x_{crit}$  [m]"
# (field in cell_profiles.npz, symbol, normalisation, output-file suffix).
# "local": (q - q_bM) / q_bM pointwise.
# "visc":  (q - q_bM) / (nu + q_bM), i.e. relative to the effective viscosity.
#          nu_t ~ 0 at the boundary-layer edge, so a pointwise error there is
#          hundreds of % even on fully covered runs and swamps every RMS.
# "peak":  (q - q_bM) / max_y q_bM at the same x.
QUANTITIES = (("u", r"U_x", "local", ""),
              ("nut", r"\nu_t", "visc", "_nut"),
              ("nuTilda", r"\tilde{\nu}", "visc", "_nuTilda"),
              ("k", r"k", "peak", "_k"),
              ("omega", r"\omega", "local", "_omega"),
              ("epsilon", r"\varepsilon", "local", "_epsilon"))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def extract_cells(run_dir, out_path, pvbatch, env, script_dir):
    """Cell-centroid profiles of one run, cached; True when available."""
    latest = pp.latest_time_directory(run_dir)
    script = script_dir / EXTRACT_SCRIPT
    if (out_path.is_file() and latest is not None
            and out_path.stat().st_mtime > max(script.stat().st_mtime, latest.stat().st_mtime)):
        return True
    if pvbatch is None:
        return False
    foam = run_dir / "p.foam"
    foam.touch(exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [pvbatch, "--force-offscreen-rendering", str(script), str(foam), str(out_path),
           *map(str, X_GRID)]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
        ok = proc.returncode == 0 and out_path.is_file()
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        ok, output = False, f"timed out after {TIMEOUT_S} s"
    if not ok:
        print(f"  [FAILED] {run_dir.name}:\n" + "\n".join(output.splitlines()[-12:]))
    return ok


def station_profile(field, i):
    """(y, Ux, nut) of station i: the wall point plus every cell centroid."""
    o = field["offsets"]
    s = slice(o[i], o[i + 1])
    return field["y"][s], field["u"][s], field["nut"][s]


def integrals(field):
    """(n_x, 4) array of delta99, delta*, theta, H with U_e = Ux(Y_FREE)."""
    trap = getattr(np, "trapezoid", None) or np.trapz
    x = field["x"]
    out = np.full((len(x), 4), np.nan)
    # Fine log grid: the centroids are sparse in the core, and a trapezoid over
    # them would weight a 5 mm cell like a 1 um one.
    yq = np.concatenate([[0.0], np.logspace(-7, np.log10(Y_FREE), 4000)])
    for i in range(len(x)):
        y, u, _ = station_profile(field, i)
        if len(y) < 10:
            continue
        ue = float(np.interp(Y_FREE, y, u))
        k = int(np.argmax(u >= 0.99 * ue))
        d99 = y[k] if k == 0 else \
            y[k - 1] + (0.99 * ue - u[k - 1]) * (y[k] - y[k - 1]) / (u[k] - u[k - 1])
        r = np.interp(yq, y, u) / ue
        dstar, theta = trap(1 - r, yq), trap(r * (1 - r), yq)
        out[i] = (d99, dstar, theta, dstar / theta)
    return out


def rel_err(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / b * 100.0


def load(run_case_dir, results_root, runs, index, script_dir):
    """Per-run dict of everything the figures need, or None without a baseline."""
    pvbatch = pp.find_pvbatch()
    if pvbatch is None:
        print("[warn] ParaView pvbatch not found - only cached cell profiles are used")
    env = pp.pvbatch_env()
    data = {}
    for name in runs:
        run_dir = Path(run_case_dir) / name
        cache = Path(results_root) / name / "boundary_layer" / CACHE_NAME
        print(f"[prism_coverage] {name}")
        if not extract_cells(run_dir, cache, pvbatch, env, script_dir):
            print(f"  [warn] no cell profiles for {name} - skipped")
            continue
        field = dict(np.load(cache))
        cx, cf = cf_curve(str(run_dir))
        if cx is None:
            print(f"  [warn] no Cf for {name} - skipped")
            continue
        u_inf, nu, _ = pp.detect_case_parameters(run_dir)
        x = field["x"]
        I = integrals(field)
        data[name] = {"approach": index[name].get("approach"), "field": field, "x": x,
                      "cf": np.interp(x, cx, cf), "d99": I[:, 0], "dstar": I[:, 1],
                      "theta": I[:, 2], "H": I[:, 3], "u_inf": u_inf, "nu": nu,
                      "h": (index[name].get("mesh") or {}).get("totalLayerThickness")}
    base = sorted(n for n in data if data[n]["approach"] == ERROR_BASELINE)
    if not base:
        return None, None
    B = data[base[0]]
    X = B["x"]
    B["utau"] = B["u_inf"] * np.sqrt(np.clip(B["cf"], 0, None) / 2.0)
    B["d99p"] = B["d99"] * B["utau"] / B["nu"]
    base_maps = {q: on_yplus(B, B, q) for q, *_ in QUANTITIES if q in B["field"]}
    for name, r in data.items():
        if not np.allclose(r["x"], X):
            print(f"  [warn] {name}: different x grid - skipped")
            r["h"] = None
            continue
        for q in ("cf", "H", "theta", "dstar", "d99"):
            r[f"e_{q}"] = rel_err(r[q], B[q])
        if not r["h"] or r["approach"] == ERROR_BASELINE:
            continue
        r["hplus"] = r["h"] * B["utau"] / B["nu"]
        r["x_crit"] = x_crit(X, B["d99"], r["h"])
        r["maps"] = {}
        for q, _sym, norm, _suffix in QUANTITIES:
            if q not in base_maps or q not in r["field"]:
                continue
            ref = base_maps[q]
            if norm in ("peak", "visc"):
                denom = (np.nanmax(np.abs(ref), axis=0)[None, :] if norm == "peak"
                         else ref + B["nu"])
                with np.errstate(divide="ignore", invalid="ignore"):
                    r["maps"][q] = (on_yplus(r, B, q) - ref) / denom * 100.0
            else:
                r["maps"][q] = rel_err(on_yplus(r, B, q), ref)
    return data, B


def x_crit(X, d99, h):
    """Station where the baseline delta99 first reaches h, or None when the
    stack is already outgrown at X_RANGE[0] or still covers it at X_RANGE[1]."""
    inside = (X >= X_RANGE[0]) & (X <= X_RANGE[1])
    xs, ds = X[inside], d99[inside]
    if not np.all(np.isfinite(ds)) or ds[0] > h or ds[-1] <= h:
        return None
    k = int(np.argmax(ds > h))
    return float(xs[k - 1] + (h - ds[k - 1]) * (xs[k] - xs[k - 1]) / (ds[k] - ds[k - 1]))


def on_yplus(r, B, q="u"):
    """(len(YP), n_x) field q of run r on the baseline's y+ grid.

    Turbulence fields are interpolated in log-log: near the wall they are power
    laws (nu_t ~ y^4 for SA), which linear interpolation between centroids gets
    wrong by a mesh-dependent ~8 % in the sublayer.  Ux is linear there.
    """
    out = np.full((len(YP), len(B["x"])), np.nan)
    f = r["field"]
    o = f["offsets"]
    for i in range(len(B["x"])):
        s = slice(o[i], o[i + 1])
        y, v = f["y"][s], f[q][s]
        if len(y) < 10 or not np.isfinite(B["utau"][i]) or B["utau"][i] <= 0:
            continue
        yq = YP * B["nu"] / B["utau"][i]
        if q == "u":
            out[:, i] = np.interp(yq, y, v)
            continue
        pos = (y > 0) & (v > 0) & np.isfinite(v)
        if pos.sum() >= 10:
            out[:, i] = np.exp(np.interp(np.log(yq), np.log(y[pos]), np.log(v[pos])))
    return out


def region_masks(d99p):
    return {"sublayer": YP < 5, "buffer": (YP >= 5) & (YP < 30),
            "log": (YP >= 30) & (YP < LOG_TOP * d99p),
            "wake": (YP >= LOG_TOP * d99p) & (YP < d99p),
            "outer": (YP >= d99p) & (YP < 1.5 * d99p)}


def region_of(yplus, d99p):
    for k, top in (("sublayer", 5), ("buffer", 30), ("log", LOG_TOP * d99p), ("wake", d99p)):
        if yplus < top:
            return k
    return "outer"


def region_rms(r, B, q="u"):
    """(region -> RMS |error| along x, region of the prism top along x).

    "total" is the RMS over every region at once, wall to 1.5 delta99+.
    """
    n = len(B["x"])
    rms = {k: np.full(n, np.nan) for k in ("total",) + REGIONS}
    for i in range(n):
        col = np.abs(r["maps"][q][:, i])
        masks = region_masks(B["d99p"][i])
        masks["total"] = np.logical_or.reduce(list(masks.values()))
        for k, m in masks.items():
            v = col[m]
            v = v[np.isfinite(v)]
            if v.size:
                rms[k][i] = np.sqrt(np.mean(v ** 2))
    top = np.array([region_of(r["hplus"][i], B["d99p"][i]) for i in range(n)])
    return rms, top


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def styles(data, index, names):
    # Colours over EVERY prism run, so a layer count keeps its colour in the
    # figures that show only a subset.
    every = [n for n in data if data[n]["h"] and data[n]["approach"] != ERROR_BASELINE]
    colors, label, ordered, shade = variant_colors(every, index)
    title = ", ".join(pp.swept_keys([index[n] for n in names])) or "run"
    return colors, label, ordered, shade, title


def legend(fig, title, ordered, shade, used, extra=()):
    entries = [(v, shade[v], None, "-") for v in ordered if v in used]
    sections = [(title, entries)]
    if extra:
        sections.append(("", list(extra)))
    return pp.legend_right(fig, sections, width_in=LEGEND_WIDTH_IN)


def plot_outgrown(data, B, index, approaches, out_path):
    names = [n for n in data if data[n]["approach"] in approaches and data[n]["h"]]
    colors, label, ordered, shade, title = styles(data, index, names)
    X = B["x"]
    inside = (X >= X_RANGE[0]) & (X <= X_RANGE[1])
    n_cols = len(approaches)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.6 * n_cols + LEGEND_WIDTH_IN, 4.6),
                             squeeze=False, sharey=True)
    ymax = 1.05e3 * max(np.nanmax(data[n]["d99"][inside]) for n in names + [next(
        n for n in data if data[n]["approach"] == ERROR_BASELINE)])
    for ax, approach in zip(axes[0], approaches):
        ax.plot(X[inside], B["d99"][inside] * 1e3, color="black", lw=2)
        for n in names:
            r = data[n]
            if r["approach"] != approach:
                continue
            ax.plot(X[inside], r["d99"][inside] * 1e3, color=colors[n], lw=1.2)
            ax.axhline(r["h"] * 1e3, color=colors[n], lw=0.8, ls=":")
            if r.get("x_crit"):
                ax.plot(r["x_crit"], r["h"] * 1e3, marker="o", ms=9, color=colors[n], mfc="none",
                        mew=1.8)
        ax.set_title(approach)
        ax.set_xlabel("x [m]")
        ax.set_xlim(*X_RANGE)
        ax.set_ylim(0, ymax)
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel(r"$\delta_{99}$ [mm]")
    fig.suptitle(r"Where each stack is outgrown: $\delta_{99}(x)$ per run, dotted = stack height h, "
                 rf"circle = $x_{{crit}}$ where {ERROR_BASELINE} $\delta_{{99}}$ reaches h",
                 fontsize=10)
    rect = legend(fig, title, ordered, shade, set(label[n] for n in names),
                  [(ERROR_BASELINE, "black", None, "-")])
    fig.tight_layout(rect=(rect[0], 0, rect[2], 0.93))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  outgrown -> {out_path}")


def history_runs(data, approaches):
    return [n for n in data if data[n]["approach"] in approaches and data[n].get("x_crit")]


def plot_history(data, B, index, approaches, out_path):
    names = history_runs(data, approaches)
    if not names:
        print("  [info] no run is outgrown inside the plate range - skipping the history")
        return
    colors, label, ordered, shade, title = styles(data, index, names)
    X = B["x"]
    inside = (X >= X_RANGE[0]) & (X <= X_RANGE[1])
    qs = (("cf", "Cf"), ("H", "H"), ("dstar", r"$\delta^*$"), ("d99", r"$\delta_{99}$"))
    n_cols = len(approaches)
    fig, axes = plt.subplots(len(qs), n_cols, figsize=(4.6 * n_cols + LEGEND_WIDTH_IN,
                                                       2.6 * len(qs) + 0.8),
                             sharex=True, sharey="row", squeeze=False)
    for col, approach in enumerate(approaches):
        for n in names:
            r = data[n]
            if r["approach"] != approach:
                continue
            xi = X - r["x_crit"]
            for row, (q, _) in enumerate(qs):
                axes[row][col].plot(xi[inside], r[f"e_{q}"][inside], color=colors[n], lw=1.3)
        axes[0][col].set_title(approach)
        axes[-1][col].set_xlabel(HISTORY_LABEL)
        for row, (q, sym) in enumerate(qs):
            ax = axes[row][col]
            ax.axvline(0, color="black", lw=0.8)
            ax.axhline(0, color="black", lw=0.8)
            ax.grid(True, alpha=0.3)
            if q == "cf":
                ax.axhspan(-CF_BAND_PCT, CF_BAND_PCT, color="0.88", zorder=0)
            if col == 0:
                ax.set_ylabel(f"{sym} error [%]")
    fig.suptitle(rf"Signed error vs {ERROR_BASELINE} after the stack is outgrown (0 = $x_{{crit}}$, "
                 rf"where {ERROR_BASELINE} $\delta_{{99}}$ = h). $U_e$ = freestream. "
                 rf"Grey = Cf convergence band ±{CF_BAND_PCT:g} %.", fontsize=10)
    rect = legend(fig, title, ordered, shade, set(label[n] for n in names))
    fig.tight_layout(rect=(rect[0], 0, rect[2], 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  history -> {out_path}")


def plot_region_history(data, B, index, approaches, out_path, q="u", sym=r"U_x", norm="local"):
    names = [n for n in history_runs(data, approaches) if q in data[n]["maps"]]
    if not names:
        print(f"  [info] no outgrown run with {q} on it and the baseline - skipping its "
              "region history")
        return
    colors, label, ordered, shade, title = styles(data, index, names)
    X = B["x"]
    inside = (X >= X_RANGE[0]) & (X <= X_RANGE[1])
    n_cols = len(approaches)
    rows = ("total",) + REGIONS
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(4.6 * n_cols + LEGEND_WIDTH_IN,
                                                         2.3 * len(rows) + 0.8),
                             sharex=True, sharey=True, squeeze=False)
    peak = 0.0
    for col, approach in enumerate(approaches):
        for n in names:
            r = data[n]
            if r["approach"] != approach:
                continue
            xi = X - r["x_crit"]
            rms, top = region_rms(r, B, q)
            peak = max([peak] + [np.nanmax(v[inside]) for v in rms.values()
                                 if np.isfinite(v[inside]).any()])
            for row, k in enumerate(rows):
                ax = axes[row][col]
                if k == "total":
                    ax.plot(xi[inside], rms[k][inside], color=colors[n], lw=1.6)
                    continue
                ax.plot(xi[inside], rms[k][inside], color=colors[n], lw=0.9, alpha=0.6)
                here = inside & (top == k)
                ax.plot(xi[here], rms[k][here], color=colors[n], lw=3.2, solid_capstyle="butt")
        axes[0][col].set_title(approach)
        axes[-1][col].set_xlabel(HISTORY_LABEL)
        for row, k in enumerate(rows):
            ax = axes[row][col]
            ax.axvline(0, color="black", lw=0.8)
            ax.set_yscale("log")
            ax.set_ylim(1e-3, min(1e4, max(3e2, 2.0 * peak)))
            ax.grid(True, alpha=0.3)
            if col == 0:
                name = "total (all regions)" if k == "total" else k
                ax.set_ylabel(f"{name}\n" + rf"RMS of $\epsilon_{{{sym}}}$ [%]")
    base_sym, _, sub = sym.partition("_")
    ref = rf"{base_sym}_{{{sub + ',' if sub else ''}\mathrm{{{ERROR_BASELINE}}}}}"
    denom = {"peak": rf"\max_y {ref}", "visc": rf"(\nu + {ref})"}.get(norm, ref)
    what = "Velocity" if q == "u" else f"${sym}$"
    peak_note = {"peak": f" (local peak of the {ERROR_BASELINE} profile)",
                 "visc": " (relative to the effective viscosity)"}.get(norm, "")
    fig.suptitle(rf"{what} error per boundary-layer region over the distance since the stack was "
                 rf"outgrown. $\epsilon_{{{sym}}} = ({sym} - {ref})\,/\,{denom}$ at equal wall "
                 f"distance{peak_note};\nRMS = root mean square of $\\epsilon_{{{sym}}}$ over the "
                 "region at each x (log-spaced in $y^+$, so every decade of $y^+$ counts equally). "
                 rf"Regions in {ERROR_BASELINE} wall units. BOLD = the prism top lies in this region.",
                 fontsize=10)
    rect = legend(fig, title, ordered, shade, set(label[n] for n in names),
                  [("prism top in this region", "0.3", None, "-")])
    legend_obj = fig.legends[-1]
    for handle, text in zip(legend_obj.legend_handles, legend_obj.get_texts()):
        if text.get_text() == "prism top in this region":
            handle.set_linewidth(3.2)
    fig.tight_layout(rect=(rect[0], 0, rect[2], 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  region history -> {out_path}")


def plot_prism_top(data, B, index, approaches, out_path):
    names = [n for n in data if data[n]["approach"] in approaches and data[n]["h"]]
    colors, label, ordered, shade, title = styles(data, index, names)
    X = B["x"]
    inside = (X >= X_RANGE[0]) & (X <= X_RANGE[1])
    xs, d99p = X[inside], B["d99p"][inside]
    fig, ax = plt.subplots(figsize=(10 + LEGEND_WIDTH_IN, 4.6))
    bands = (("sublayer", 0.05, 5), ("buffer", 5, 30), ("log", 30, LOG_TOP * d99p),
             ("wake", LOG_TOP * d99p, d99p), ("outer", d99p, 3e4))
    for k, lo, hi in bands:
        ax.fill_between(xs, lo, hi, color=REGION_COLOR[k], alpha=0.5, lw=0)
    # h+ uses the baseline u_tau, so runs sharing a stack height share a line.
    drawn = {}
    for n in sorted(names, key=lambda n: ordered.index(label[n])):
        key = (label[n], round(data[n]["h"], 12))
        if key in drawn:
            continue
        drawn[key] = n
        ax.plot(xs, data[n]["hplus"][inside], color=colors[n], lw=2)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1e4)
    ax.set_xlim(*X_RANGE)
    ax.set_xlabel("x [m]")
    ax.set_ylabel(rf"$h^+$  ({ERROR_BASELINE} $u_\tau$)")
    ax.grid(True, alpha=0.3)
    fig.suptitle("Prism-top height in wall units over the boundary-layer regions: the stack is "
                 "fixed, the boundary layer grows past it", fontsize=10)
    rect = legend(fig, title, ordered, shade, {k[0] for k in drawn},
                  [(k, REGION_COLOR[k], "s", "none") for k in REGIONS])
    fig.tight_layout(rect=(rect[0], 0, rect[2], 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  prism top -> {out_path}")


def main():
    script_dir = Path(__file__).resolve().parent
    _, _, run_case_dir, _, results_root = pp.resolve_case_paths(script_dir)
    manifest = pp.load_manifest(run_case_dir)
    if manifest is None:
        print(f"[info] no study manifest in {run_case_dir} - nothing to do")
        return 0
    index = pp.combo_index(manifest)
    completed, failed = pp.split_completed(run_case_dir, sorted(index))
    pp.warn_failed(failed, "the prism-coverage figures")
    approaches = sorted({index[n].get("approach") for n in completed
                         if (index[n].get("mesh") or {}).get("totalLayerThickness")
                         and index[n].get("approach") != ERROR_BASELINE})
    if not approaches or not any(index[n].get("approach") == ERROR_BASELINE for n in completed):
        print(f"[info] need a {ERROR_BASELINE} run and at least one prism approach - nothing to do")
        return 0

    data, B = load(run_case_dir, results_root, completed, index, script_dir)
    if data is None:
        print(f"[warn] no usable {ERROR_BASELINE} run - skipping the prism-coverage figures")
        return 0
    for n, r in sorted(data.items()):
        if r["h"] and r["approach"] != ERROR_BASELINE:
            xc = r.get("x_crit")
            print(f"  {n}: h = {r['h'] * 1e3:.4g} mm, x_crit = "
                  + (f"{xc:.3f} m" if xc else "none (outgrown or covered along the whole range)"))

    out_dir = Path(results_root) / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_outgrown(data, B, index, approaches, out_dir / "prism_coverage_outgrown.png")
    plot_history(data, B, index, approaches, out_dir / "prism_coverage_history.png")
    for q, sym, norm, suffix in QUANTITIES:
        plot_region_history(data, B, index, approaches,
                            out_dir / f"prism_coverage_region_history{suffix}.png", q, sym, norm)
    plot_prism_top(data, B, index, approaches, out_dir / "prism_coverage_prism_top.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
