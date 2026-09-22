"""
mesh_study.py – post-processing for config-driven mesh studies.

Figures (into results/<case>/<study>/validation/):

  cf_error_vs_prism_coverage.png
      One panel per mesh approach.  y: |Cf - Cf_blockMesh| / Cf_blockMesh in %,
      x: prism-stack height over the local boundary-layer thickness,
      h / delta99 in %.  Each run is a CURVE, not a point: its stack height h
      is fixed while delta99 grows downstream, so walking along the plate
      sweeps h / delta99 from large (near the leading edge) to small (at the
      trailing edge).  A filled marker picks out x = 1 m on every curve, which
      is the one-point-per-run reading.

Inputs, all already produced by the scripts that run before this one:
  - Cf(x) from the wall-shear samples (cf_validation's readers),
  - delta99(x) from bl_thickness.py's results/<run>/boundary_layer/delta99.csv,
  - h = mesh.totalLayerThickness from the study manifest.

Still planned: a convergence table (cells, Cf at x = 0.97008, wall y+, log-law
RMS) vs the swept parameter.
"""
from __future__ import annotations

import os
import sys

# pyrefly: ignore [missing-import]
import matplotlib

matplotlib.use("Agg")

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# pyrefly: ignore [missing-import]
import numpy as np

import _pp_common as pp
import cf_validation as cv
from profile_validation import ERROR_BASELINE, LEGEND_WIDTH_IN, variant_colors

# Plate range used for the Cf comparison.  The leading edge is a singularity
# (Cf -> infinity, and meshes resolve it very differently), and the last few
# faces before the outlet are influenced by the outlet BC.
X_RANGE = (0.1, 1.95)
# Station highlighted on every curve.
X_MARK = 1.0


def cf_curve(case_dir):
    """(x, Cf) along the plate for one run, sorted by x, or (None, None)."""
    shear_path, _name = cv.find_wall_shear_file(case_dir)
    if shear_path is None:
        return None, None
    shear = pp.read_numeric_rows(shear_path)
    if shear is None:
        return None, None
    u_inf, _nu, rho = pp.detect_case_parameters(case_dir,
                                                fallback_u_inf=cv.U_INF_DEFAULT,
                                                fallback_nu=cv.NU_DEFAULT)
    order = np.argsort(shear[:, 0])
    x = shear[order, 0]
    cf = cv.compute_cf_from_shear(shear[order, 3:6], rho=rho, u_inf=u_inf)
    return x, cf


def delta99_curve(results_root, run):
    """(x, delta99) from bl_thickness.py's CSV, or (None, None)."""
    path = os.path.join(results_root, run, "boundary_layer", "delta99.csv")
    if not os.path.isfile(path):
        return None, None
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.size == 0:
        return None, None
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def plot_cf_error_vs_coverage(runs, index, results_root, run_case_dir, out_path):
    """The figure described in the module docstring."""
    base = sorted(n for n in runs if index[n].get("approach") == ERROR_BASELINE)
    if not base:
        print(f"  [info] no {ERROR_BASELINE} run - skipping the Cf-error figure")
        return False
    bx, bcf = cf_curve(os.path.join(run_case_dir, base[0]))
    if bx is None:
        print(f"  [warn] no Cf for {base[0]} - skipping the Cf-error figure")
        return False

    others = [n for n in runs if index[n].get("approach") != ERROR_BASELINE]
    approaches = sorted({index[n].get("approach", "?") for n in others})
    if not approaches:
        print("  [info] only the baseline approach present - skipping the Cf-error figure")
        return False
    colors, label, ordered, shade = variant_colors(others, index)
    _c, markers, _s = pp.study_styles(runs, {"combos": [index[n] for n in runs]})

    n_cols = len(approaches)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.6 * n_cols + LEGEND_WIDTH_IN, 5.0),
                             squeeze=False, sharex=True, sharey=True)
    plotted = set()
    for col, approach in enumerate(approaches):
        ax = axes[0][col]
        for name in others:
            if index[name].get("approach") != approach:
                continue
            h = (index[name].get("mesh") or {}).get("totalLayerThickness")
            if not h:
                print(f"  [warn] {name}: no totalLayerThickness in the manifest")
                continue
            x, cf = cf_curve(os.path.join(run_case_dir, name))
            dx, d99 = delta99_curve(results_root, name)
            if x is None or dx is None:
                print(f"  [warn] {name}: missing Cf or delta99 - skipped")
                continue
            keep = (x >= X_RANGE[0]) & (x <= X_RANGE[1])
            x, cf = x[keep], cf[keep]
            ref = np.interp(x, bx, bcf)
            delta = np.interp(x, dx, d99)
            with np.errstate(divide="ignore", invalid="ignore"):
                err = np.abs(cf - ref) / ref * 100.0
                cover = h / delta * 100.0
            ok = np.isfinite(err) & np.isfinite(cover) & (err > 0) & (cover > 0)
            if not ok.any():
                continue
            ax.plot(cover[ok], err[ok], color=colors[name], linewidth=1.3)
            i = int(np.argmin(np.abs(x - X_MARK)))
            if ok[i]:
                ax.plot([cover[i]], [err[i]], marker=markers.get(name, "o"),
                        markersize=9, color=colors[name], linestyle="none",
                        markeredgecolor="black", markeredgewidth=0.8, zorder=5)
            plotted.add(label[name])
        ax.axvline(100.0, color="0.4", linestyle="--", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_title(approach, fontsize=11)
        ax.set_xlabel(r"$h_\mathrm{prism}\,/\,\delta_{99}$  [%]")
        if col == 0:
            ax.set_ylabel(rf"$|C_f - C_{{f,\mathrm{{{ERROR_BASELINE}}}}}|\,/\,"
                          rf"C_{{f,\mathrm{{{ERROR_BASELINE}}}}}$  [%]")

    if not plotted:
        plt.close(fig)
        print("  [warn] nothing to plot for the Cf error")
        return False

    fig.suptitle(f"Cf error vs {ERROR_BASELINE} over prism coverage of the boundary layer, "
                 f"x = {X_RANGE[0]:g} … {X_RANGE[1]:g} m  -  filled marker = x = {X_MARK:g} m, "
                 f"dashed line = stack as tall as $\\delta_{{99}}$", fontsize=10)
    title = ", ".join(pp.swept_keys([index[n] for n in others])) or "run"
    entries = [(v, shade[v], markers.get(next(n for n in others if label[n] == v), "o"), "-")
               for v in ordered if v in plotted]
    rect = pp.legend_right(fig, [(title, entries)], width_in=LEGEND_WIDTH_IN)
    fig.tight_layout(rect=(rect[0], 0, rect[2], 0.92))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Cf error vs prism coverage -> {out_path}")
    return True


def main():
    _, case_name, run_case_dir, _, results_root = pp.resolve_case_paths(
        os.path.dirname(os.path.abspath(__file__))
    )
    manifest = pp.load_manifest(run_case_dir)
    if manifest is None:
        print(f"[info] no study manifest in {run_case_dir} - nothing to do")
        return 0
    combos = manifest.get("combos", [])
    print(f"[mesh_study] study '{manifest.get('study')}': {len(combos)} combos")

    index = pp.combo_index(manifest)
    completed, failed = pp.split_completed(run_case_dir, sorted(index))
    pp.warn_failed(failed, "the mesh study")
    runs = {n: index[n] for n in completed}
    if not runs:
        print("[warn] no completed runs")
        return 0

    out_dir = os.path.join(results_root, "validation")
    os.makedirs(out_dir, exist_ok=True)
    plot_cf_error_vs_coverage(runs, index, results_root, run_case_dir,
                              os.path.join(out_dir, "cf_error_vs_prism_coverage.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
