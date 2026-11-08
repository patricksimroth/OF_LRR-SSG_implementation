"""
bl_thickness.py – boundary-layer thickness delta99 over x, all runs compared.

For each run with a reconstructed latest p, _pv_bl_thickness.py (pvbatch)
resamples U near the plate and computes delta99(x) (wall distance where Ux
reaches 99 % of the local edge velocity) into
results/<case>/<run>/boundary_layer/delta99.csv, the input of the delta99
figures of the study's analyze.py (summary/along_plate/delta99.png, ...).  The
log compares delta99 at x = 1 m with the composite profile (_bl_theory.py:
Spalding inner law + Coles/Granville wake, momentum integral, turbulent from
x = 0).
Needs ParaView pvbatch (see _pp_common.find_pvbatch); without it the script
warns and exits 0, so a missing optional tool never fails the pipeline.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

import _bl_theory as theory
import _pp_common as pp

EXTRACT_SCRIPT = "_pv_bl_thickness.py"
X_RANGE = (0.01, 2.0)   # leading edge (x = 0, singular) to the outlet
N_X = 200
Y_TOP = 0.08            # well above delta99 at the outlet (~0.027 m)
N_Y = 4001              # 2e-5 m wall-normal sampling
TIMEOUT_S = 600


def delta99_composite(x, u_inf, nu):
    """delta99 of the composite profile + momentum integral (see _bl_theory)."""
    return theory.delta99_composite(x, u_inf / nu)


def prism_stacks(manifest):
    """run_dir -> (layer height [m], coverage % or None, h+) for runs with prisms."""
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
            stacks[combo.get("run_dir")] = (float(height), coverage, pp.h_plus(combo))
    return stacks


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
    extract = script_dir / EXTRACT_SCRIPT
    args = [f"{X_RANGE[0]:g}", f"{X_RANGE[1]:g}", str(N_X), f"{Y_TOP:g}", str(N_Y)]
    key = " ".join(args)

    def extract_run(item):
        """(run, csv path, ok, note); reuses an unchanged run's delta99.csv."""
        run, latest = item
        out_dir = results_root / run.name / "boundary_layer"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "delta99.csv"
        stamp = out_dir / "delta99.stamp"
        (out_dir / "delta99.png").unlink(missing_ok=True)   # per-run plot superseded
        if pp.outputs_fresh(stamp, [csv_path], pp.run_data_inputs(run, latest) + [extract], key):
            return run, csv_path, True, "unchanged, reused"
        csv_path.unlink(missing_ok=True)
        stamp.unlink(missing_ok=True)
        foam = run / "p.foam"
        foam.touch(exist_ok=True)
        cmd = [pvbatch, "--force-offscreen-rendering", str(extract),
               str(foam), str(csv_path), *args]
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                  timeout=TIMEOUT_S)
            ok = proc.returncode == 0 and csv_path.is_file()
            output = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired:
            ok, output = False, f"timed out after {TIMEOUT_S} s"
        if ok:
            pp.write_stamp(stamp, key)
        return run, csv_path, ok, "extracted" if ok else output

    failed = []
    for run, csv_path, ok, note in pp.map_threads(extract_run, runs, pp.pv_jobs()):
        if not ok:
            print(f"[bl_thickness] {run.name}  [FAILED]:\n"
                  + "\n".join(note.splitlines()[-20:]))
            failed.append(run.name)
            continue
        print(f"[bl_thickness] {run.name} -> {csv_path} ({note})")
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
        u_inf, nu, _ = pp.detect_case_parameters(run)
        prism = stacks.get(run.name)
        delta_at_1 = np.interp(1.0, data[:, 0], data[:, 1])
        ref_at_1 = delta99_composite(1.0, u_inf, nu)
        prism_note = f", prism layer {prism[0] * 1e3:g} mm" if prism else ", no prism layer"
        print(f"  delta99(x=1) = {delta_at_1 * 1e3:.2f} mm "
              f"(composite {ref_at_1 * 1e3:.2f} mm, ratio {delta_at_1 / ref_at_1:.3f}), "
              f"delta99(x={data[-1, 0]:g}) = {data[-1, 1] * 1e3:.2f} mm{prism_note}")

    if failed:
        print(f"[warn] boundary-layer thickness failed for: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
