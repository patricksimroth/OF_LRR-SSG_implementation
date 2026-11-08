"""
prism_coverage.py – where the prism-stack top sits in the boundary layer.

Into results/<case>/<study>/summary/prism_coverage/:

  hplus_regions.png
      Prism-top height in local wall units h u_tau(x) / nu along the plate
      (u_tau of the blockMesh baseline), over the boundary-layer regions
      (sublayer y+ < 5, buffer < 30, log < 0.2 delta99+, wake < delta99+,
      outer).  One line per stack height: the stack is fixed, the boundary
      layer grows past it.

It also extracts the cell-centroid profiles every run needs for the delta*,
theta and H figures of the study's analyze.py (_pv_cell_profiles.py via
pvbatch), cached per run in results/<run>/boundary_layer/cell_profiles.npz.
Cell centroids, not ParaView's point data: the reader's cell-to-point filter
averages neighbouring cells unweighted, which on differently stretched meshes
faked +2.8 % at the wall between two runs whose wall shear agrees to 0.04 %.

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

from matplotlib.patches import Patch

import _pp_common as pp
from cf_validation import cf_curve
from _pp_common import ERROR_BASELINE

pp.report_style()

EXTRACT_SCRIPT = "_pv_cell_profiles.py"
CACHE_NAME = "cell_profiles.npz"
X_GRID = (0.02, 1.98, 197)
# Leading edge singular, last faces influenced by the outlet BC.
X_RANGE = (0.1, 1.95)
# Freestream probe height: ~2x delta99 at the outlet and below the top of the
# extraction grid in _pv_cell_profiles.py (0.06 m).
Y_FREE = 0.055
TIMEOUT_S = 1800
LOG_TOP = 0.2
REGIONS = (("sublayer", "viscous sublayer"), ("buffer", "buffer layer"),
           ("log", "log layer"), ("wake", "wake region"), ("outer", "free stream"))
REGION_COLOR = dict(zip((k for k, _ in REGIONS),
                        ("#4c72b0", "#55a868", "#c4a000", "#dd8452", "#8c8c8c")))


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


def load(run_case_dir, results_root, runs, index, script_dir):
    """Per-run d99 / Cf / stack height, or (None, None) without a baseline."""
    pvbatch = pp.find_pvbatch()
    if pvbatch is None:
        print("[warn] ParaView pvbatch not found - only cached cell profiles are used")
    env = pp.pvbatch_env()

    def cache_of(name):
        return Path(results_root) / name / "boundary_layer" / CACHE_NAME

    available = dict(zip(runs, pp.map_threads(
        lambda name: extract_cells(Path(run_case_dir) / name, cache_of(name), pvbatch, env,
                                   script_dir),
        runs, pp.pv_jobs())))
    data = {}
    for name in runs:
        run_dir = Path(run_case_dir) / name
        if not available[name]:
            print(f"  [warn] no cell profiles for {name} - skipped")
            continue
        field = dict(np.load(cache_of(name)))
        cx, cf = cf_curve(str(run_dir))
        if cx is None:
            print(f"  [warn] no Cf for {name} - skipped")
            continue
        u_inf, nu, _ = pp.detect_case_parameters(run_dir)
        x = field["x"]
        data[name] = {"approach": index[name].get("approach"), "x": x,
                      "cf": np.interp(x, cx, cf), "d99": integrals(field)[:, 0],
                      "u_inf": u_inf, "nu": nu,
                      "h": (index[name].get("mesh") or {}).get("totalLayerThickness")}
    base = sorted(n for n in data if data[n]["approach"] == ERROR_BASELINE)
    if not base:
        return None, None
    B = data[base[0]]
    B["utau"] = B["u_inf"] * np.sqrt(np.clip(B["cf"], 0, None) / 2.0)
    B["d99p"] = B["d99"] * B["utau"] / B["nu"]
    for r in data.values():
        if r["h"] and r["approach"] != ERROR_BASELINE and np.allclose(r["x"], B["x"]):
            r["hplus"] = r["h"] * B["utau"] / B["nu"]
    return data, B


def plot_prism_top(data, B, out_path, scales):
    heights = sorted({round(r["h"], 12) for r in data.values() if "hplus" in r})
    X = B["x"]
    inside = (X >= X_RANGE[0]) & (X <= X_RANGE[1])
    xs, d99p = X[inside] / scales["delta_mean"], B["d99p"][inside]
    fig, ax = plt.subplots(figsize=(pp.REPORT_WIDTH_IN, 2.9))
    bands = ((0.05, 5), (5, 30), (30, LOG_TOP * d99p), (LOG_TOP * d99p, d99p), (d99p, 3e4))
    for (key, _), (lo, hi) in zip(REGIONS, bands):
        ax.fill_between(xs, lo, hi, color=REGION_COLOR[key], alpha=0.45, lw=0)
    for h in heights:
        r = next(r for r in data.values() if "hplus" in r and round(r["h"], 12) == h)
        ax.plot(xs, r["hplus"][inside], color=pp.INK, lw=1.3)
        label = pp.fmt_hplus(h * scales["u_tau"] / scales["nu"])
        ax.annotate(label, (xs[-1], r["hplus"][inside][-1]), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1e4)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_xlabel(pp.X_PLATE_LABEL)
    ax.set_ylabel(r"$y^+$")
    handles = [Patch(color=REGION_COLOR[k], alpha=0.45, label=label) for k, label in REGIONS]
    handles.append(pp.line_handle(pp.STACK_TOP_LABEL, pp.INK))
    rect = pp.legend_below(fig, handles, ncol=3)
    fig.tight_layout(rect=rect)
    fig.savefig(out_path)
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
    pp.warn_failed(failed, "the prism-coverage figure")
    approaches = sorted({index[n].get("approach") for n in completed
                         if (index[n].get("mesh") or {}).get("totalLayerThickness")
                         and index[n].get("approach") != ERROR_BASELINE})
    if not approaches or not any(index[n].get("approach") == ERROR_BASELINE for n in completed):
        print(f"[info] need a {ERROR_BASELINE} run and at least one prism approach - nothing to do")
        return 0
    scales = pp.case_scales(Path(run_case_dir) / completed[0])
    data, B = load(run_case_dir, results_root, completed, index, script_dir)
    if data is None:
        print(f"[warn] no usable {ERROR_BASELINE} run - skipping the prism-coverage figure")
        return 0
    out_dir = Path(results_root) / pp.SUMMARY_DIR / "prism_coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_prism_top(data, B, out_dir / "hplus_regions.png", scales)
    return 0


if __name__ == "__main__":
    sys.exit(main())
