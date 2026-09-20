"""
bl_thickness.py – boundary-layer thickness delta99 over x, all runs compared.

For each run with a reconstructed latest p, _pv_bl_thickness.py (pvbatch)
resamples U near the plate and computes delta99(x) (wall distance where Ux
reaches 99 % of the local edge velocity) into
results/<case>/<run>/boundary_layer/delta99.csv.  One comparison plot,
results/<case>/validation/delta99_comparison.png, overlays
  - delta99(x) of every run,
  - the analytical turbulent flat-plate reference delta = 0.37 x / Re_x^(1/5),
    Re_x = U_inf x / nu (turbulent from the leading edge, as in these fully
    turbulent RANS runs); one line per distinct (U_inf, nu),
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

import _pp_common as pp

EXTRACT_SCRIPT = "_pv_bl_thickness.py"
PLOT_NAME = "delta99_comparison.png"
X_RANGE = (0.01, 2.0)   # leading edge (x = 0, singular) to the outlet
N_X = 200
Y_TOP = 0.08            # well above delta99 at the outlet (~0.027 m)
N_Y = 4001              # 2e-5 m wall-normal sampling
TIMEOUT_S = 600


def delta_turbulent_plate(x, u_inf, nu):
    """Turbulent flat-plate correlation delta = 0.37 x / Re_x^(1/5)."""
    return 0.37 * x / (u_inf * x / nu) ** 0.2


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


def run_label(run_name, all_names):
    """Approach part of <model>__<approach> when every run shares the model."""
    models = {name.split("__", 1)[0] for name in all_names}
    if len(models) == 1 and "__" in run_name:
        return run_name.split("__", 1)[1]
    return run_name


def plot_comparison(records, case_name, out_path):
    """records: dicts with name, data (x, delta99, U_e), u_inf, nu, prism."""
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    names = [r["name"] for r in records]
    model_names = {name.split("__", 1)[0] for name in names}

    for i, record in enumerate(records):
        data = record["data"]
        ax.plot(data[:, 0], data[:, 1] * 1e3, color=f"C{i}", linewidth=1.6,
                label=rf"$\delta_{{99}}$  {run_label(record['name'], names)}")

    x_ref = np.linspace(X_RANGE[0], X_RANGE[1], 400)
    references = sorted({(round(r["u_inf"], 6), float(f"{r['nu']:.6g}")) for r in records})
    for j, (u_inf, nu) in enumerate(references):
        ax.plot(x_ref, delta_turbulent_plate(x_ref, u_inf, nu) * 1e3,
                color="k", linestyle=("--", "-.", (0, (5, 1, 1, 1)))[j % 3],
                linewidth=1.3,
                label=(r"analytical $0.37\,x\,/\,\mathrm{Re}_x^{1/5}$"
                       f"  ($U_\\infty$ = {u_inf:g} m/s, $\\nu$ = {nu:.4g} m²/s)"))

    heights = {}
    for record in records:
        if record["prism"] is not None:
            heights.setdefault(record["prism"][0], []).append(
                run_label(record["name"], names))
    for k, (height, owners) in enumerate(sorted(heights.items())):
        ax.axhline(height * 1e3, color="0.35", linestyle=":", linewidth=1.6,
                   label=f"prism layer height {height * 1e3:g} mm ({', '.join(owners)})")

    ax.set_xlabel(r"$x$  [m]")
    ax.set_ylabel(r"$\delta_{99}$  [mm]")
    title_model = f" – {next(iter(model_names))}" if len(model_names) == 1 else ""
    ax.set_title(f"Boundary-layer thickness (99 % rule){title_model}")
    ax.set_xlim(0.0, X_RANGE[1])
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
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
        ref_at_1 = delta_turbulent_plate(1.0, u_inf, nu)
        prism_note = f", prism layer {prism[0] * 1e3:g} mm" if prism else ", no prism layer"
        print(f"  delta99(x=1) = {delta_at_1 * 1e3:.2f} mm "
              f"(analytical {ref_at_1 * 1e3:.2f} mm, ratio {delta_at_1 / ref_at_1:.3f}), "
              f"delta99(x={data[-1, 0]:g}) = {data[-1, 1] * 1e3:.2f} mm{prism_note}")

    if records:
        val_dir = results_root / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(records, case_name, val_dir / PLOT_NAME)
        print(f"[bl_thickness] comparison plot -> {val_dir / PLOT_NAME}")
    if failed:
        print(f"[warn] boundary-layer thickness failed for: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
