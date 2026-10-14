"""
bl_thickness.py – boundary-layer thickness delta99 over x, all runs compared.

For each run with a reconstructed latest p, _pv_bl_thickness.py (pvbatch)
resamples U near the plate and computes delta99(x) (wall distance where Ux
reaches 99 % of the local edge velocity) into
results/<case>/<run>/boundary_layer/delta99.csv.  One comparison plot,
results/<case>/validation/delta99_comparison.png, overlays
  - delta99(x) of every run,
  - the composite-profile reference (_bl_theory.py: Spalding inner law +
    Coles/Granville wake, momentum integral, turbulent from x = 0) - delta99
    proper and the only reference drawn.  (The classic 0.37 x / Re_x^(1/5) is the
    1/7 power-law thickness where u = U_e EXACTLY, not a 99 % thickness, and
    sits ~20 % above both the composite and TMR's SA solution.)
  - dotted prism-layer heights from the study manifest (totalLayerThickness of
    snappy/gmshTri/gmshHex runs), one line per distinct height; blockMesh and
    runs without a manifest have no prism stack.
Needs ParaView pvbatch (see _pp_common.find_pvbatch); without it the script
warns and exits 0, so a missing optional tool never fails the pipeline.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _bl_theory as theory
import _pp_common as pp

EXTRACT_SCRIPT = "_pv_bl_thickness.py"
PLOT_NAME = "delta99_comparison.png"
X_RANGE = (0.01, 2.0)   # leading edge (x = 0, singular) to the outlet
N_X = 200
Y_TOP = 0.08            # well above delta99 at the outlet (~0.027 m)
N_Y = 4001              # 2e-5 m wall-normal sampling
TIMEOUT_S = 600


def delta99_composite(x, u_inf, nu):
    """delta99 of the composite profile + momentum integral (see _bl_theory)."""
    return theory.delta99_composite(x, u_inf / nu)


def prism_stacks(manifest):
    """run_dir -> (layer height [m], coverage % or None) for runs with prisms."""
    stacks = {}
    for combo in (manifest or {}).get("combos", []):
        # Derived from firstLayerThickness + expansionRatio + nPrisms by the
        # approach, so it lives in the mesh metadata.  Older manifests
        # requested it as a parameter instead - fall back to that.
        mesh = combo.get("mesh") or {}
        height = mesh.get("totalLayerThickness")
        if height is None:
            height = (combo.get("params") or {}).get("totalLayerThickness")
        if height is not None:
            coverage = mesh.get("layer_coverage_pct")
            stacks[combo.get("run_dir")] = (float(height), coverage)
    return stacks


# Extra figure width reserved for the right-hand legend, in inches.
LEGEND_WIDTH_IN = 3.0


def run_label(run_name, all_names):
    """Approach part of <model>__<approach> when every run shares the model."""
    models = {name.split("__", 1)[0] for name in all_names}
    if len(models) == 1 and "__" in run_name:
        return run_name.split("__", 1)[1]
    return run_name


def plot_comparison(records, case_name, out_path, manifest=None):
    """records: dicts with name, data (x, delta99, U_e), u_inf, nu, prism."""
    fig, ax = plt.subplots(figsize=(10.5 + LEGEND_WIDTH_IN, 6.0))
    names = [r["name"] for r in records]
    model_names = {name.split("__", 1)[0] for name in names}
    # Colour = mesh approach, marker = swept parameter, from the manifest; a
    # 4 x 5 sweep is 20 curves and one colour each is unreadable.
    colors, markers, sections = pp.study_styles(names, manifest)

    for record in records:
        data = record["data"]
        ax.plot(data[:, 0], data[:, 1] * 1e3,
                color=colors.get(record["name"], "k"),
                marker=markers.get(record["name"], None), markersize=4,
                markevery=max(1, len(data) // 15), linewidth=1.6)

    x_ref = np.linspace(X_RANGE[0], X_RANGE[1], 400)
    references = sorted({(round(r["u_inf"], 6), float(f"{r['nu']:.6g}")) for r in records})
    refs = []
    for j, (u_inf, nu) in enumerate(references):
        cond = f"  ($U_\\infty$ = {u_inf:g} m/s, $\\nu$ = {nu:.4g} m²/s)"
        ax.plot(x_ref, delta99_composite(x_ref, u_inf, nu) * 1e3,
                color="k", linestyle="-", linewidth=2.0,
                label=r"composite profile + momentum integral, $\delta_{99}$" + cond)
        refs.append((r"composite $\delta_{99}$" + (f" ({u_inf:g} m/s)" if len(references) > 1 else ""),
                     "black", None, "-"))

    heights = {}
    for record in records:
        if record["prism"] is not None:
            heights.setdefault(record["prism"][0], []).append(
                run_label(record["name"], names))
    for k, (height, owners) in enumerate(sorted(heights.items())):
        ax.axhline(height * 1e3, color="0.35", linestyle=":", linewidth=1.6,
                   label=f"prism layer height {height * 1e3:g} mm ({', '.join(owners)})")
    if heights:
        refs.append(("prism stack top", "0.35", None, ":"))

    ax.set_xlabel(r"$x$  [m]")
    ax.set_ylabel(r"$\delta_{99}$  [mm]")
    title_model = f" – {next(iter(model_names))}" if len(model_names) == 1 else ""
    ax.set_title(f"Boundary-layer thickness (99 % rule){title_model}")
    ax.set_xlim(0.0, X_RANGE[1])
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    rect = pp.legend_right(fig, sections + [("Reference", refs)], width_in=LEGEND_WIDTH_IN)
    fig.tight_layout(rect=rect)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    script_dir = Path(__file__).resolve().parent
    _, case_name, run_case_dir, _, results_root = pp.resolve_case_paths(script_dir)
    pvbatch = pp.find_pvbatch()
    if pvbatch is None:
        print("[warn] ParaView pvbatch not found ($PARAVIEW_PVBATCH, PATH, "
              "~/opt/ParaView-*) - skipping boundary-layer thickness")
        return 0
    runs = pp.solved_run_dirs(run_case_dir)
    if not runs:
        print(f"[info] no finished runs with a p field in {run_case_dir}")
        return 0
    stacks = prism_stacks(pp.load_manifest(run_case_dir))

    env = pp.pvbatch_env()
    failed = []
    records = []
    for run, _ in runs:
        foam = run / "p.foam"
        foam.touch(exist_ok=True)
        out_dir = results_root / run.name / "boundary_layer"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "delta99.csv"
        csv_path.unlink(missing_ok=True)
        (out_dir / "delta99.png").unlink(missing_ok=True)   # per-run plot superseded
        cmd = [pvbatch, "--force-offscreen-rendering", str(script_dir / EXTRACT_SCRIPT),
               str(foam), str(csv_path), f"{X_RANGE[0]:g}", f"{X_RANGE[1]:g}",
               str(N_X), f"{Y_TOP:g}", str(N_Y)]
        print(f"[bl_thickness] {run.name} -> {csv_path}")
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                  timeout=TIMEOUT_S)
            ok = proc.returncode == 0 and csv_path.is_file()
            output = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            ok, output = False, f"timed out after {TIMEOUT_S} s"
        if not ok:
            print(f"  [FAILED] {run.name}:\n" + "\n".join(output.splitlines()[-20:]))
            failed.append(run.name)
            continue
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
        u_inf, nu, _ = pp.detect_case_parameters(run)
        prism = stacks.get(run.name)
        records.append({"name": run.name, "data": data, "u_inf": u_inf, "nu": nu,
                        "prism": prism})
        delta_at_1 = np.interp(1.0, data[:, 0], data[:, 1])
        ref_at_1 = delta99_composite(1.0, u_inf, nu)
        prism_note = f", prism layer {prism[0] * 1e3:g} mm" if prism else ", no prism layer"
        print(f"  delta99(x=1) = {delta_at_1 * 1e3:.2f} mm "
              f"(composite {ref_at_1 * 1e3:.2f} mm, ratio {delta_at_1 / ref_at_1:.3f}), "
              f"delta99(x={data[-1, 0]:g}) = {data[-1, 1] * 1e3:.2f} mm{prism_note}")

    if records:
        val_dir = results_root / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(records, case_name, val_dir / PLOT_NAME,
                        manifest=pp.load_manifest(run_case_dir))
        print(f"[bl_thickness] comparison plot -> {val_dir / PLOT_NAME}")
    if failed:
        print(f"[warn] boundary-layer thickness failed for: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
