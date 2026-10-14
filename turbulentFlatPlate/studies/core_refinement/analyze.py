"""Summary figures + table for the core_refinement study.

    .venv/bin/python3 turbulentFlatPlate/studies/core_refinement/analyze.py \
        [results/turbulentFlatPlate/core_refinement]

Reads the archived _rundata/ and the per-run boundary_layer/delta99.csv the
regular post-processing wrote, and writes <results>/summary/:

    summary.csv                      one row per run
    fig_cf_error_vs_core.png         Cf error vs blockMesh at x = 0.97 m over core size
    fig_cf_error_along_plate.png     signed Cf error over x, per approach x nPrisms
    fig_cost_accuracy.png            plate-RMS Cf error over cell count
    fig_velocity_x1.png              Ux(y) at x = 1 m vs blockMesh, absolute units
    fig_delta99_by_core.png          delta99(x) per approach x nPrisms, colour = core size
    fig_cf_by_core.png               Cf(x) per approach x nPrisms, colour = core size
"""
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "turbulentFlatPlate" / "_postProcessing"))
import _pp_common as pp  # noqa: E402
import cf_validation as cfv  # noqa: E402

STATION_X = 0.97008
PLATE = (0.1, 1.95)
BASE_SIZE = 0.005
END_TIME = 8000
APPROACHES = ("snappy", "gmshTri", "gmshHex")
APPROACH_COLORS = {"snappy": "#2a78d6", "gmshTri": "#eb6834", "gmshHex": "#1baf7a"}
NPRISMS_MARKERS = {20: "o", 40: "s", 50: "D"}
SCALE_COLORS = {1.0: "#9ec5f0", 0.5: "#5c9be0", 0.25: "#2a6fc2", 0.125: "#0d3a75"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "font.size": 10, "legend.frameon": False,
})


def wall_cell_size(combo):
    params = combo["params"]
    if combo["approach"] == "snappy":
        return params["backgroundCellSize"] / 2 ** params["wallLevel"]
    return params.get("wallCellSize")


def solver_stats(case):
    log = (case / "log.rhoSimpleFoam").read_text(errors="replace")
    times = re.findall(r"^Time = (\d+)", log, re.M)
    execs = re.findall(r"ExecutionTime = ([\d.]+) s", log)
    return (int(times[-1]) if times else None,
            float(execs[-1]) if execs else None,
            bool(re.search(r"^End\s*$", log, re.M)))


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


def load_runs(results):
    manifest = json.loads((results / "study_manifest.json").read_text())
    runs = []
    for combo in manifest["combos"]:
        case = results / "_rundata" / combo["run_dir"]
        if not (case / "log.rhoSimpleFoam").exists():
            print(f"  [skip] {combo['run_dir']}: no solver log")
            continue
        record = cfv.process_model(combo["run_dir"], str(case), None)
        if record is None:
            continue
        iters, exec_s, ended = solver_stats(case)
        mesh = combo.get("mesh") or {}
        size = wall_cell_size(combo)
        delta_file = results / combo["run_dir"] / "boundary_layer" / "delta99.csv"
        delta = np.loadtxt(delta_file, delimiter=",", skiprows=1) if delta_file.exists() else None
        runs.append({
            "name": combo["run_dir"], "approach": combo["approach"],
            "nPrisms": combo["params"].get("nPrisms"),
            "size": size, "scale": size / BASE_SIZE if size else None,
            "h": mesh.get("totalLayerThickness"), "nCells": mesh.get("nCells"),
            "x": record["x"], "cf": record["cf"],
            "iters": iters, "exec_s": exec_s,
            "converged": ended and iters is not None and iters < END_TIME,
            "cd": final_cd(case), "p_res": final_p_residual(case),
            "delta": delta, "case": case,
        })
    return runs


def plate_error(run, base):
    grid = np.linspace(*PLATE, 400)
    cf = np.interp(grid, run["x"], run["cf"])
    cf_b = np.interp(grid, base["x"], base["cf"])
    return grid, 100.0 * (cf - cf_b) / cf_b


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
        run["err_max"] = float(err[np.argmax(np.abs(err))])
        cf_station = float(np.interp(STATION_X, run["x"], run["cf"]))
        run["cf_station"] = cf_station
        run["err_tmr"] = 100.0 * (cf_station - tmr) / tmr
        run["cd_err"] = (100.0 * (run["cd"] - base["cd"]) / base["cd"]
                         if run["cd"] and base["cd"] else None)
        run["x_crit"] = x_crit(run["h"], base)
        run["delta99_x1"] = (float(np.interp(1.0, run["delta"][:, 0], run["delta"][:, 1]))
                             if run["delta"] is not None else None)


def write_csv(runs, path):
    keys = ["name", "approach", "nPrisms", "size", "scale", "nCells", "h", "x_crit",
            "iters", "converged", "exec_s", "p_res", "cd", "cd_err", "cf_station",
            "err_station", "err_rms", "err_max", "err_tmr", "delta99_x1"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for run in runs:
            writer.writerow([run.get(k) for k in keys])


def prism_runs(runs):
    return [r for r in runs if r["approach"] in APPROACHES]


def fig_error_vs_core(runs, path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    for col, n in enumerate(sorted(NPRISMS_MARKERS)):
        for row, (key, label) in enumerate(
                [("err_station", "Cf error vs blockMesh at x = 0.97 m [%]"),
                 ("err_rms", "RMS Cf error vs blockMesh, x = 0.1–1.95 m [%]")]):
            ax = axes[row, col]
            for approach in APPROACHES:
                sel = sorted((r for r in prism_runs(runs)
                              if r["approach"] == approach and r["nPrisms"] == n),
                             key=lambda r: -r["size"])
                if not sel:
                    continue
                ax.plot([1e3 * r["size"] for r in sel], [r[key] for r in sel],
                        color=APPROACH_COLORS[approach], marker=NPRISMS_MARKERS[n],
                        markersize=7, label=approach)
                for r in sel:
                    if not r["converged"]:
                        ax.plot(1e3 * r["size"], r[key], marker="x", color=INK,
                                markersize=10, linestyle="none")
            if row == 0:
                ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.set_xscale("log", base=2)
            ax.set_xticks([5, 2.5, 1.25, 0.625])
            ax.set_xticklabels(["5", "2.5", "1.25", "0.625"])
            ax.invert_xaxis()
            if row == 0:
                ax.set_title(f"nPrisms = {n}  (stack {1e3 * stack(n):.2f} mm)")
            if row == 1:
                ax.set_xlabel("wall core cell size Δx [mm]  (finer →)")
                ax.set_yscale("log")
            if col == 0:
                ax.set_ylabel(label)
    axes[0, 0].legend(loc="best")
    fig.suptitle("SALSA: Cf error against the blockMesh baseline over the core cell size "
                 "(× = not converged, hit 8000 iterations)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stack(n, first=1e-6, ratio=1.17):
    return first * (ratio ** n - 1) / (ratio - 1)


def fig_error_along_plate(runs, base, path):
    ns = sorted(NPRISMS_MARKERS)
    fig, axes = plt.subplots(len(ns), 3, figsize=(14, 3.2 * len(ns)), sharex=True)
    for i, n in enumerate(ns):
        for j, approach in enumerate(APPROACHES):
            ax = axes[i, j]
            sel = sorted((r for r in prism_runs(runs)
                          if r["approach"] == approach and r["nPrisms"] == n),
                         key=lambda r: -r["size"])
            for r in sel:
                grid, err = plate_error(r, base)
                ax.plot(grid, err, color=SCALE_COLORS.get(round(r["scale"], 4), INK),
                        linewidth=1.6, label=f"Δx = {1e3 * r['size']:g} mm")
            xc = sel[0]["x_crit"] if sel else None
            if xc:
                ax.axvline(xc, color=MUTED, linestyle=":", linewidth=1.2)
                ax.text(xc, 0.97, " stack outgrown", transform=ax.get_xaxis_transform(),
                        color=MUTED, fontsize=8, va="top")
            ax.axhline(0, color=MUTED, linewidth=0.8)
            if i == 0:
                ax.set_title(approach)
            if j == 0:
                ax.set_ylabel(f"nPrisms = {n}\nΔCf / Cf,blockMesh [%]")
            if i == len(ns) - 1:
                ax.set_xlabel("x [m]")
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Signed Cf error along the plate (dotted: blockMesh δ99 reaches the stack top)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_cost(runs, base, path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for approach in APPROACHES:
        for n, marker in NPRISMS_MARKERS.items():
            sel = sorted((r for r in prism_runs(runs)
                          if r["approach"] == approach and r["nPrisms"] == n),
                         key=lambda r: r["nCells"] or 0)
            if not sel:
                continue
            ax.plot([r["nCells"] for r in sel], [r["err_rms"] for r in sel],
                    color=APPROACH_COLORS[approach], marker=marker, markersize=7,
                    linewidth=1.2, alpha=0.9)
    ax.axvline(base["nCells"], color=INK, linestyle="--", linewidth=1)
    ax.text(base["nCells"], 0.98, " blockMesh cells", transform=ax.get_xaxis_transform(),
            fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cells")
    ax.set_ylabel("RMS Cf error vs blockMesh, x = 0.1–1.95 m [%]")
    handles = [plt.Line2D([], [], color=APPROACH_COLORS[a], label=a) for a in APPROACHES]
    handles += [plt.Line2D([], [], color=MUTED, marker=m, linestyle="none",
                           label=f"nPrisms = {n}") for n, m in NPRISMS_MARKERS.items()]
    ax.legend(handles=handles, loc="upper right")
    ax.set_title("Cost vs accuracy: each line runs from Δx = 5 mm (left) to 0.625 mm (right)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def delta99_series(run):
    return None if run["delta"] is None else (run["delta"][:, 0], 1e3 * run["delta"][:, 1])


def cf_series(run):
    return run["x"], run["cf"]


def fig_x_by_core(runs, base, series, ylabel, title, ylim, path, stack_line=False):
    ns = sorted(NPRISMS_MARKERS)
    fig, axes = plt.subplots(len(ns), 3, figsize=(14, 3.4 * len(ns)), sharex=True, sharey=True)
    ref = series(base)
    for i, n in enumerate(ns):
        for j, approach in enumerate(APPROACHES):
            ax = axes[i, j]
            for r in sorted((r for r in prism_runs(runs)
                             if r["approach"] == approach and r["nPrisms"] == n),
                            key=lambda r: -r["size"]):
                data = series(r)
                if data is None:
                    continue
                label = f"Δx = {1e3 * r['size']:g} mm"
                if not r["converged"]:
                    label += " (not converged)"
                ax.plot(*data, color=SCALE_COLORS.get(round(r["scale"], 4), INK),
                        linewidth=1.5, label=label)
            if ref is not None:
                ax.plot(*ref, color=INK, linestyle="--", linewidth=1.3, label="blockMesh",
                        zorder=5)
            if stack_line:
                ax.axhline(1e3 * stack(n), color=MUTED, linestyle=":", linewidth=1.2)
                ax.text(0.02, 1e3 * stack(n), " prism stack top", color=MUTED, fontsize=8,
                        va="bottom")
            ax.set_xlim(0, 2)
            ax.set_ylim(*ylim)
            if i == 0:
                ax.set_title(approach)
            if j == 0:
                ax.set_ylabel(f"nPrisms = {n}\n{ylabel}")
            if i == len(ns) - 1:
                ax.set_xlabel("x [m]")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def x1_profile(case):
    tdir = pp.latest_time_directory(case / "postProcessing" / "sampleUyPlus")
    files = sorted(Path(tdir).glob("x1_profile*.xy")) if tdir else []
    for file in files:
        cols, arr = pp.read_profile_set(file)
        if cols is not None and "Ux" in cols:
            return arr[:, 0], arr[:, cols.index("Ux")]
    return None


def fig_velocity(runs, base, path):
    ns = sorted(NPRISMS_MARKERS)
    fig, axes = plt.subplots(len(ns), 3, figsize=(14, 3.4 * len(ns)), sharex=True, sharey=True)
    ref = x1_profile(base["case"])
    for i, n in enumerate(ns):
        for j, approach in enumerate(APPROACHES):
            ax = axes[i, j]
            if ref is not None:
                ax.plot(1e3 * ref[0][1:], ref[1][1:] / 69.4, color=INK, linestyle="--",
                        linewidth=1.4, label="blockMesh", zorder=5)
            for r in sorted((r for r in prism_runs(runs)
                             if r["approach"] == approach and r["nPrisms"] == n),
                            key=lambda r: -r["size"]):
                prof = x1_profile(r["case"])
                if prof is None:
                    continue
                ax.plot(1e3 * prof[0][1:], prof[1][1:] / 69.4,
                        color=SCALE_COLORS.get(round(r["scale"], 4), INK), linewidth=1.5,
                        label=f"Δx = {1e3 * r['size']:g} mm")
            ax.axvline(1e3 * stack(n), color=MUTED, linestyle=":", linewidth=1.2)
            ax.set_xscale("log")
            ax.set_xlim(1e-3, 60)
            if i == 0:
                ax.set_title(approach)
            if j == 0:
                ax.set_ylabel(f"nPrisms = {n}\nUx / U∞")
            if i == len(ns) - 1:
                ax.set_xlabel("wall distance y [mm]")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Velocity profile at x = 1 m in absolute units (dotted: prism stack top)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "results" / "turbulentFlatPlate" / "core_refinement")
    results = results.resolve()
    out = results / "summary"
    out.mkdir(exist_ok=True)
    runs = load_runs(results)
    base = next(r for r in runs if r["approach"] == "blockMesh")
    reference = cfv.load_reference(str(ROOT / "turbulentFlatPlate" / "_validation_data"))
    tmr = float(np.interp(STATION_X, reference[:, 0], reference[:, 1]))
    metrics(runs, base, tmr)
    write_csv(runs, out / "summary.csv")
    fig_error_vs_core(runs, out / "fig_cf_error_vs_core.png")
    fig_error_along_plate(runs, base, out / "fig_cf_error_along_plate.png")
    fig_cost(runs, base, out / "fig_cost_accuracy.png")
    fig_velocity(runs, base, out / "fig_velocity_x1.png")
    d_max = max(np.max(r["delta"][:, 1]) for r in runs if r["delta"] is not None)
    fig_x_by_core(runs, base, delta99_series, "δ99 [mm]",
                  "δ99(x) per approach and nPrisms, colour = wall core cell size "
                  "(dotted: prism stack top)",
                  (0, 1.05e3 * d_max), out / "fig_delta99_by_core.png", stack_line=True)
    cf_max = max(np.max(r["cf"][r["x"] >= PLATE[0]]) for r in runs)
    fig_x_by_core(runs, base, cf_series, "Cf",
                  "Cf(x) per approach and nPrisms, colour = wall core cell size",
                  (0, 1.15 * cf_max), out / "fig_cf_by_core.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
