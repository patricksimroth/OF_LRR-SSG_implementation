"""Summary figures + table for the core_refinement study and its variants
(core_refinement_SSGLRR, core_refinement_*_upwind: same grid, other model or
schemes).

    .venv/bin/python3 turbulentFlatPlate/studies/core_refinement/analyze.py \
        [results/turbulentFlatPlate/<study>]

Reads the archived _rundata/ and the per-run boundary_layer/ data the regular
post-processing wrote, and writes into <results>/summary/ (next to the
figures of run_postpro.sh):

    tables/summary.csv               one row per run
    along_plate/<q>.png              q(x) per approach x h+, one line per core size,
                                     blockMesh dashed (q = cf, yplus, delta99, dstar,
                                     theta, H)
    error_along_plate/<q>.png        signed q error vs blockMesh over x/delta_mean
                                     (q = cf, delta99, dstar, theta, H)
    error_vs_core/<q>.png            q error vs blockMesh at x = 0.97 m and RMS over
                                     the plate, over the normalised core cell size L*
    cost/cf_error_vs_cells.png       plate-RMS Cf error over the cell count

and then redraws the dashboard and summary/CAPTIONS.md.  Figures follow the
report style of _pp_common (no titles; the captions in _captions.py carry the
details).

Runs are described by the prism-stack height in wall units h+ = h u_tau / nu (not
nPrisms) and the nominal wall core cell size L_nom/delta_mean, with u_tau the analytical
composite-profile value (_bl_theory) at the trailing edge x = 2 m and delta_mean the
analytical delta99 averaged over the plate: constants for every run and study
(_pp_common.flow_scales).  error_vs_core puts every stack on one axis with
L* = L_core/delta_mean * phi, phi = <(delta - h)+>/delta_mean, L_core the core cell
size measured next to the stack (_core_size, cached in
results/<study>/<run>/boundary_layer/cell_sizes.npz).  Along-plate figures use
x/delta_mean as abscissa.  x_crit is where the
analytical delta99 reaches the stack height h.

delta*, theta and H are integrated from the cell-centroid profiles prism_coverage.py
caches in results/<study>/<run>/boundary_layer/cell_profiles.npz (same integration,
U_e = Ux at y = 55 mm); delta99 comes from boundary_layer/delta99.csv.
"""
import csv
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "turbulentFlatPlate" / "_postProcessing"))
import _bl_theory as theory  # noqa: E402
import _core_size as core_size  # noqa: E402
import _pp_common as pp  # noqa: E402
import cf_validation as cfv  # noqa: E402
import prism_coverage as pcov  # noqa: E402

STATION_X = 0.97008
PLATE = (0.1, 1.95)
BASE_SIZE = 0.005
END_TIME = 8000
APPROACHES = pp.APPROACHES
APPROACH_COLORS = pp.APPROACH_COLORS
N_APP = len(APPROACHES)
NPRISMS_MARKERS = {20: "o", 40: "s", 50: "D"}
INK, MUTED, GRID = pp.INK, pp.MUTED, "#e4e3df"
MODEL_LABELS = pp.MODEL_LABELS
RE_PER_M = 69.4 / 1.38801e-5
XCRIT = r"$x_\mathrm{crit}$"
DELTA_MEAN = None
SCALES = None
HSTACK = {}
HPLUS = {}
XI_LABEL = pp.X_PLATE_LABEL
LEGACY = ("fig_*.png", "summary.csv")

pp.report_style()


def solver_stats(case):
    log = (case / "log.rhoSimpleFoam").read_text(errors="replace")
    times = re.findall(r"^Time = (\d+)", log, re.M)
    execs = re.findall(r"ExecutionTime = ([\d.]+) s", log)
    clocks = re.findall(r"ClockTime = ([\d.]+) s", log)
    nprocs = re.search(r"^nProcs\s*:\s*(\d+)", log, re.M)
    host = re.search(r"^Host\s*:\s*(\S+)", log, re.M)
    host = host.group(1) if host else None
    if host and re.fullmatch(r"n\d{4}", host):
        host = "Barnard"
    return {"iters": int(times[-1]) if times else None,
            "exec_s": float(execs[-1]) if execs else None,
            "clock_s": float(clocks[-1]) if clocks else None,
            "nprocs": int(nprocs.group(1)) if nprocs else 1,
            "host": host,
            "ended": bool(re.search(r"^End\s*$", log, re.M))}


def final_cd(case):
    path = pp.latest_time_directory(case / "postProcessing" / "integralCf")
    rows = pp.read_numeric_rows(Path(path) / "coefficient.dat")
    return float(rows[-1, 1]) if rows is not None else None


def final_p_residual(case):
    path = pp.latest_time_directory(case / "postProcessing" / "solverInfo")
    file = Path(path) / "solverInfo.dat"
    header = [l for l in file.read_text().splitlines() if l.startswith("# Time")]
    cols = header[-1].lstrip("# ").split("\t") if header else []
    cols = [c.strip() for c in cols]
    rows = [l.split() for l in file.read_text().splitlines() if l and not l.startswith("#")]
    if "p_initial" not in cols or not rows:
        return None
    return float(rows[-1][cols.index("p_initial")])


def hp(n):
    """Row label of the nPrisms = n runs: their h+."""
    return pp.fmt_hplus(HPLUS[n]) if n in HPLUS else f"nPrisms = {n}"


def dx_label(run):
    return pp.fmt_dx(run["dx_norm"])


def dx_ticks(runs):
    return sorted({round(r["dx_norm"], 6) for r in prism_runs(runs) if r["dx_norm"]})


def load_runs(results):
    global SCALES, DELTA_MEAN
    manifest = json.loads((results / "study_manifest.json").read_text())
    runs = []
    for combo in manifest["combos"]:
        case = results / "_rundata" / combo["run_dir"]
        combo["_case_dir"] = str(case)
        if not (case / "log.rhoSimpleFoam").exists():
            print(f"  [skip] {combo['run_dir']}: no solver log")
            continue
        if not pp.run_completed(case):
            print(f"  [skip] {combo['run_dir']}: solver did not complete (crashed)")
            continue
        record = cfv.process_model(combo["run_dir"], str(case), None)
        if record is None:
            continue
        stats = solver_stats(case)
        iters = stats["iters"]
        mesh = combo.get("mesh") or {}
        size = pp.wall_cell_size(combo)
        scales = pp.combo_scales(combo)
        if SCALES is None:
            SCALES, DELTA_MEAN = scales, scales["delta_mean"]
        n = combo["params"].get("nPrisms")
        if n is not None and pp.stack_height(combo):
            HSTACK.setdefault(n, pp.stack_height(combo))
            HPLUS.setdefault(n, pp.h_plus(combo, scales))
        delta_file = results / combo["run_dir"] / "boundary_layer" / "delta99.csv"
        delta = np.loadtxt(delta_file, delimiter=",", skiprows=1) if delta_file.exists() else None
        cells = results / combo["run_dir"] / "boundary_layer" / pcov.CACHE_NAME
        bl = None
        if cells.exists():
            field = dict(np.load(cells))
            ints = pcov.integrals(field)
            bl = {"x": field["x"], "dstar": ints[:, 1], "theta": ints[:, 2], "H": ints[:, 3]}
        else:
            print(f"  [warn] {combo['run_dir']}: no {pcov.CACHE_NAME}, run prism_coverage.py")
        runs.append({
            "name": combo["run_dir"], "approach": combo["approach"],
            "nPrisms": n, "h_plus": pp.h_plus(combo, scales),
            "dx_norm": pp.dx_norm(combo, scales), "h_stack": pp.stack_height(combo),
            "phi": (pp.outgrown_fraction(pp.stack_height(combo), scales)
                    if pp.stack_height(combo) else None),
            "size": size, "scale": size / BASE_SIZE if size else None,
            "h": mesh.get("totalLayerThickness"), "nCells": mesh.get("nCells"),
            "x": record["x"], "cf": record["cf"],
            "yp_x": record.get("yp_x"), "yp": record.get("yp"),
            "iters": iters, "exec_s": stats["exec_s"], "clock_s": stats["clock_s"],
            "cpu_h": (stats["clock_s"] * stats["nprocs"] / 3600.0
                      if stats["clock_s"] is not None else None),
            "host": stats["host"],
            "converged": stats["ended"] and iters is not None and iters < END_TIME,
            "cd": final_cd(case), "p_res": final_p_residual(case),
            "delta": delta, "bl": bl, "case": case,
        })
    measure_core(runs, results)
    return runs


def measure_core(runs, results):
    """Measured core cell size next to the stack (_core_size) and L* from it."""
    prism = [r for r in runs if r["h_stack"] and r["size"]]
    caches = {r["name"]: results / r["name"] / "boundary_layer" / core_size.CACHE_NAME
              for r in prism}
    core_size.extract_all([(r["case"], caches[r["name"]]) for r in prism])
    for r in runs:
        m = (core_size.measure(caches[r["name"]], r["h_stack"], r["size"])
             if r["name"] in caches else None)
        if r["name"] in caches and m is None:
            print(f"  [warn] {r['name']}: no measured core cell size, L* left empty")
        r["l_core"] = m["l_core"] if m else None
        r["l_core_norm"] = m["l_core"] / SCALES["delta_mean"] if m else None
        r["l_star"] = r["l_core_norm"] * r["phi"] if m else None


def plate_error(run, base):
    grid = np.linspace(*PLATE, 400)
    cf = np.interp(grid, run["x"], run["cf"])
    cf_b = np.interp(grid, base["x"], base["cf"])
    return grid, 100.0 * (cf - cf_b) / cf_b


def delta99_error(run, base):
    if run["delta"] is None or base["delta"] is None:
        return None
    grid = np.linspace(*PLATE, 400)
    d = np.interp(grid, run["delta"][:, 0], run["delta"][:, 1])
    d_b = np.interp(grid, base["delta"][:, 0], base["delta"][:, 1])
    return grid, 100.0 * (d - d_b) / d_b


def bl_error(key):
    def error(run, base):
        if run["bl"] is None or base["bl"] is None:
            return None
        grid = np.linspace(*PLATE, 400)
        values = []
        for r in (run, base):
            ok = np.isfinite(r["bl"][key])
            values.append(np.interp(grid, r["bl"]["x"][ok], r["bl"][key][ok]))
        return grid, 100.0 * (values[0] - values[1]) / values[1]
    return error


ERROR_QUANTITIES = (
    ("cf", "Cf", plate_error),
    ("delta99", "δ99", delta99_error),
    ("dstar", "δ*", bl_error("dstar")),
    ("theta", "θ", bl_error("theta")),
    ("H", "H", bl_error("H")),
)


def delta99_an(x):
    return theory.delta99_composite(np.asarray(x, dtype=float), RE_PER_M)


def x_crit_an(h):
    xs = np.geomspace(1e-5, 2.0, 4000)
    d = delta99_an(xs)
    return float(np.interp(h, d, xs)) if d[0] < h < d[-1] else None


def x_crit(h, base):
    if not h or base["delta"] is None:
        return None
    x, d = base["delta"][:, 0], base["delta"][:, 1]
    above = np.nonzero(d >= h)[0]
    if len(above) == 0 or above[0] == 0:
        return None
    i = above[0]
    return float(np.interp(h, d[i - 1:i + 1], x[i - 1:i + 1]))


def metrics(runs, base, tmr):
    for run in runs:
        grid, err = plate_error(run, base)
        run["err_station"] = float(np.interp(STATION_X, grid, err))
        run["err_rms"] = float(np.sqrt(np.mean(err ** 2)))
        cf = np.interp(grid, run["x"], run["cf"])
        cf_b = np.interp(grid, base["x"], base["cf"])
        run["err_l1"] = float(100.0 * np.trapezoid(np.abs(cf - cf_b), grid)
                              / np.trapezoid(cf_b, grid))
        run["err_max"] = float(err[np.argmax(np.abs(err))])
        cf_station = float(np.interp(STATION_X, run["x"], run["cf"]))
        run["cf_station"] = cf_station
        run["err_tmr"] = 100.0 * (cf_station - tmr) / tmr
        run["cd_err"] = (100.0 * (run["cd"] - base["cd"]) / base["cd"]
                         if run["cd"] and base["cd"] else None)
        run["x_crit"] = x_crit(run["h"], base)
        run["x_crit_an"] = x_crit_an(run["h"]) if run["h"] else None
        run["delta99_x1"] = (float(np.interp(1.0, run["delta"][:, 0], run["delta"][:, 1]))
                             if run["delta"] is not None else None)
        for key, _symbol, error in ERROR_QUANTITIES[1:]:
            data = error(run, base)
            run[f"{key}_err_station"] = (float(np.interp(STATION_X, *data))
                                         if data is not None else None)
            run[f"{key}_err_rms"] = (float(np.sqrt(np.mean(data[1] ** 2)))
                                     if data is not None else None)


def write_csv(runs, path):
    keys = ["name", "approach", "nPrisms", "h_plus", "size", "dx_norm", "l_core", "l_core_norm",
            "phi", "l_star", "scale", "nCells", "h", "x_crit",
            "x_crit_an", "iters", "converged", "exec_s", "clock_s", "cpu_h", "p_res", "cd", "cd_err", "cf_station",
            "err_station", "err_rms", "err_l1", "err_max", "err_tmr", "delta99_x1"]
    keys += [f"{q}_err_{m}" for q, *_ in ERROR_QUANTITIES[1:] for m in ("station", "rms")]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for run in runs:
            writer.writerow([run.get(k) for k in keys])


def prism_runs(runs):
    return [r for r in runs if r["approach"] in APPROACHES]


def core_sizes(runs):
    """Nominal wall core cell sizes L_nom/delta_mean of the prism runs, coarsest first."""
    return sorted({round(r["dx_norm"], 6) for r in prism_runs(runs) if r["dx_norm"]},
                  reverse=True)


CORE_ORDER = []


def core_style(run, xs=None, log=False):
    key = round(run["dx_norm"], 6)
    i = CORE_ORDER.index(key) if key in CORE_ORDER else 0
    style = dict(color=pp.CORE_COLORS[i % len(pp.CORE_COLORS)],
                 marker=pp.CORE_MARKERS[i % len(pp.CORE_MARKERS)], markersize=3.5,
                 markeredgecolor="white", markeredgewidth=0.4, linewidth=1.0)
    if xs is not None:
        style["markevery"] = mark_positions(xs, i, log)
    return style


def core_handles(runs):
    return [pp.line_handle(pp.fmt_dx(d), **{k: v for k, v in core_style(
        {"dx_norm": d}).items() if k in ("color", "marker")}) for d in CORE_ORDER]


def mark_positions(xs, slot, log, n=6):
    u = np.log(xs) if log else np.asarray(xs, dtype=float)
    lo, hi = np.nanmin(u), np.nanmax(u)
    targets = lo + (hi - lo) * (np.arange(n) + (slot + 1) / (len(CORE_ORDER) + 1)) / n
    return sorted({int(np.argmin(np.abs(u - t))) for t in targets})


def error_metrics(run, base, error):
    data = error(run, base)
    if data is None:
        return None, None
    grid, err = data
    return float(np.interp(STATION_X, grid, err)), float(np.sqrt(np.mean(err ** 2)))


def stack(n, first=1e-6, ratio=1.17):
    if n in HSTACK:
        return HSTACK[n]
    return first * (ratio ** n - 1) / (ratio - 1)


def dx_axis(ax, runs, rotate=False):
    ticks = dx_ticks(runs)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.2g}" for t in ticks], rotation=90 if rotate else 0)
    ax.minorticks_off()
    ax.set_xlim(1.25 * ticks[-1], 0.8 * ticks[0])


def cell_runs(runs, approach, n):
    return sorted((r for r in prism_runs(runs) if r["approach"] == approach
                   and r["nPrisms"] == n), key=lambda r: -r["size"])


def core_grid(runs, draw, ylabel, path, extras=(), panel=None, sharey=True):
    """Rows h+, columns approach, one line per core size (draw(ax, run) -> bool)."""
    ns = sorted(NPRISMS_MARKERS)
    fig, axes = pp.grid_figure(len(ns), N_APP, sharex=True, sharey=sharey)
    for i, n in enumerate(ns):
        for j, approach in enumerate(APPROACHES):
            ax = axes[i][j]
            for r in cell_runs(runs, approach, n):
                draw(ax, r)
            if panel:
                panel(ax, n)
    pp.label_grid(axes, [hp(n) for n in ns], APPROACHES, ylabel, XI_LABEL)
    rect = pp.legend_below(fig, core_handles(runs) + list(extras))
    fig.tight_layout(rect=rect, h_pad=0.6, w_pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def fig_error_vs_core(runs, base, error, key, path):
    fig, axes = plt.subplots(1, 2, figsize=(pp.REPORT_WIDTH_IN, 3.0))
    values = {r["name"]: error_metrics(r, base, error) for r in prism_runs(runs)}
    for col, ax in enumerate(axes):
        for approach in APPROACHES:
            for n, marker in NPRISMS_MARKERS.items():
                sel = [r for r in cell_runs(runs, approach, n)
                       if r["l_star"] and values[r["name"]][col] is not None]
                if sel:
                    ax.plot([r["l_star"] for r in sel], [values[r["name"]][col] for r in sel],
                            color=APPROACH_COLORS[approach], marker=marker, markersize=4,
                            markeredgecolor="white", markeredgewidth=0.4, linewidth=1.0)
        ax.set_xscale("log")
        pp.plain_log_axis(ax, "x")
        ax.set_xlabel(pp.LSTAR_SYMBOL)
        if col == 0:
            ax.set_yscale("symlog", linthresh=0.1)
            pp.plain_ticks(ax)
            ax.axhline(0, color=MUTED, linewidth=0.7)
        else:
            ax.set_yscale("log")
            pp.plain_log_axis(ax)
    axes[0].set_ylabel(pp.error_label(key).replace("}$", r"}(x_\mathrm{ref})$", 1))
    axes[1].set_ylabel(pp.error_label(key, "RMS"))
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.set_title(tag, loc="left")
    handles = [pp.line_handle(a, APPROACH_COLORS[a]) for a in APPROACHES]
    handles += [pp.line_handle(hp(n), MUTED, "none", m) for n, m in NPRISMS_MARKERS.items()]
    rect = pp.legend_below(fig, handles, ncol=N_APP)
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def fig_error_along_plate(runs, base, error, key, path):
    def draw(ax, r):
        data = error(r, base)
        if data is not None:
            xi = data[0] / DELTA_MEAN
            ax.plot(xi, data[1], **core_style(r, xi))

    def panel(ax, n):
        ax.axhline(0, color=MUTED, linewidth=0.7)
        xc = x_crit_an(stack(n))
        if xc and PLATE[0] <= xc <= PLATE[1]:
            ax.axvline(xc / DELTA_MEAN, color=INK, linestyle=":", linewidth=1.0)
        ax.set_xlim(PLATE[0] / DELTA_MEAN, PLATE[1] / DELTA_MEAN)

    core_grid(runs, draw, pp.error_label(key), path, panel=panel,
              extras=[pp.line_handle(XCRIT, INK, ":")], sharey=False)


def fig_cost(runs, base, path):
    fig, ax = plt.subplots(figsize=(0.62 * pp.REPORT_WIDTH_IN, 3.2))
    for approach in APPROACHES:
        for n, marker in NPRISMS_MARKERS.items():
            sel = [r for r in cell_runs(runs, approach, n) if r["nCells"]]
            if sel:
                ax.plot([r["nCells"] for r in sel], [r["err_rms"] for r in sel],
                        color=APPROACH_COLORS[approach], marker=marker, markersize=4,
                        markeredgecolor="white", markeredgewidth=0.4, linewidth=1.0)
    if base["nCells"]:
        ax.axvline(base["nCells"], **pp.BASELINE_STYLE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    pp.plain_log_axis(ax)
    pp.plain_log_axis(ax, "x")
    ax.set_xlabel("number of cells")
    ax.set_ylabel(pp.error_label("cf", "RMS"))
    handles = [pp.line_handle(a, APPROACH_COLORS[a]) for a in APPROACHES]
    handles += [pp.line_handle(hp(n), MUTED, "none", m) for n, m in NPRISMS_MARKERS.items()]
    handles.append(pp.line_handle(pp.ERROR_BASELINE, **{k: v for k, v in pp.BASELINE_STYLE.items()
                                                       if k in ("color", "linestyle")}))
    rect = pp.legend_below(fig, handles, ncol=4)
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def series_cf(run):
    return run["x"], run["cf"]


def series_yplus(run):
    if run.get("yp_x") is None:
        return None
    return run["yp_x"], run["yp"]


def series_delta99(run):
    return None if run["delta"] is None else (run["delta"][:, 0], 1e3 * run["delta"][:, 1])


def series_bl(key, scale):
    def series(run):
        if run["bl"] is None:
            return None
        ok = np.isfinite(run["bl"][key])
        return run["bl"]["x"][ok], scale * run["bl"][key][ok]
    return series


ALONG_PLATE = (
    ("cf", series_cf, r"$C_f$", False),
    ("yplus", series_yplus, r"$y^+$", False),
    ("delta99", series_delta99, r"$\delta_{99}$ [mm]", True),
    ("dstar", series_bl("dstar", 1e3), r"$\delta^*$ [mm]", False),
    ("theta", series_bl("theta", 1e3), r"$\theta$ [mm]", False),
    ("H", series_bl("H", 1.0), r"$H$", False),
)


def plate_limits(runs, series, bottom_zero):
    """y range of the curves on the plate (0.1-1.95 m): the leading-edge
    singularity and the outlet faces would otherwise set it."""
    values = []
    for r in runs:
        data = series(r)
        if data is not None:
            x, v = data
            v = v[(x >= PLATE[0]) & (x <= PLATE[1]) & np.isfinite(v)]
            if v.size:
                values.append(v)
    if not values:
        return None
    allv = np.concatenate(values)
    lo, hi = float(allv.min()), float(allv.max())
    pad = 0.08 * (hi - lo)
    return (0.0 if bottom_zero else lo - pad, hi + pad)


def fig_along_plate(runs, base, key, series, ylabel, path, stack_line):
    ylim = plate_limits(runs, series, key in ("cf", "delta99", "dstar", "theta"))
    if ylim is None:
        return
    ref = series(base)

    def draw(ax, r):
        data = series(r)
        if data is not None:
            xi = data[0] / DELTA_MEAN
            ax.plot(xi, data[1], **core_style(r, xi))

    def panel(ax, n):
        if ref is not None:
            ax.plot(ref[0] / DELTA_MEAN, ref[1], **pp.BASELINE_STYLE)
        if stack_line:
            ax.axhline(1e3 * stack(n), **pp.STACK_TOP_STYLE)
        ax.set_xlim(0, 2.0 / DELTA_MEAN)
        ax.set_ylim(*ylim)

    extras = [pp.line_handle(pp.ERROR_BASELINE, INK, "--")]
    if stack_line:
        extras.append(pp.line_handle(pp.STACK_TOP_LABEL, MUTED, ":"))
    core_grid(runs, draw, ylabel, path, extras=extras, panel=panel)


def clean_legacy(out):
    for pattern in LEGACY:
        for stale in out.glob(pattern):
            stale.unlink()


def redraw_dashboard(results):
    os.environ[pp.RUN_DIR_ENV] = str(results / "_rundata")
    import dashboard
    dashboard.main()


def main():
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "results" / "turbulentFlatPlate" / "core_refinement")
    results = results.resolve()
    out = results / pp.SUMMARY_DIR
    out.mkdir(exist_ok=True)
    clean_legacy(out)
    dirs = {name: out / name for name in ("tables", "along_plate", "error_along_plate",
                                          "error_vs_core", "cost")}
    for d in dirs.values():
        d.mkdir(exist_ok=True)
    global RE_PER_M
    runs = load_runs(results)
    base = next(r for r in runs if r["approach"] == "blockMesh")
    CORE_ORDER[:] = core_sizes(runs)
    u_inf, nu = pp.detect_case_parameters(base["case"])[:2]
    RE_PER_M = u_inf / nu
    reference = cfv.load_reference(str(ROOT / "turbulentFlatPlate" / "_validation_data"))
    tmr = float(np.interp(STATION_X, reference[:, 0], reference[:, 1]))
    metrics(runs, base, tmr)
    write_csv(runs, dirs["tables"] / "summary.csv")
    for key, _symbol, error in ERROR_QUANTITIES:
        fig_error_vs_core(runs, base, error, key, dirs["error_vs_core"] / f"{key}.png")
        fig_error_along_plate(runs, base, error, key, dirs["error_along_plate"] / f"{key}.png")
    fig_cost(runs, base, dirs["cost"] / "cf_error_vs_cells.png")
    for key, series, ylabel, stack_line in ALONG_PLATE:
        fig_along_plate(runs, base, key, series, ylabel, dirs["along_plate"] / f"{key}.png",
                        stack_line)
    print(f"wrote {out}")
    redraw_dashboard(results)


if __name__ == "__main__":
    main()
