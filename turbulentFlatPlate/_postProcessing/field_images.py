"""
field_images.py – field images with the mesh on top for every run.

For each run dir under run/<case>/ with a reconstructed latest time holding p,
renders into results/<case>/<run>/field_plots/:
  - whole domain: p, |U|
  - boundary layer zoom at x = BL_ZOOM_X: |U| and every turbulence quantity the
    run wrote (nut, k, omega, nuTilda, R components; the model's own R, else the
    turbulenceFields-derived turbulenceProperties:R)
through ParaView's pvbatch (_pv_field_images.py, one session per run).
pvbatch lookup: see _pp_common.find_pvbatch.  Without ParaView it warns and
exits 0, so a missing optional tool never fails the pipeline.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import _pp_common as pp

RENDER_SCRIPT = "_pv_field_images.py"
TIMEOUT_S = 900

# Boundary-layer zoom: delta99 ~ 0.015-0.017 m near x = 1, so 0.035 m shows the
# whole layer plus some free stream.  Same window for every mesh -> comparable.
BL_ZOOM_X = 1.0
BL_ZOOM_HEIGHT = 0.035
BL_ZOOM_WIDTH = 0.063          # ~ image aspect, fills the frame

# (field, component, colour-bar title, caption, preset, log scale)
TURBULENCE_FIELDS = [
    ("nut", None, "nut [m2/s]", "turbulent viscosity nut", "Viridis", False),
    ("k", None, "k [m2/s2]", "turbulent kinetic energy k", "Viridis", False),
    ("omega", None, "omega [1/s]", "specific dissipation omega (log)", "Viridis", True),
    ("nuTilda", None, "nuTilda [m2/s]", "SA working variable nuTilda", "Viridis", False),
]
R_COMPONENTS = ("XX", "YY", "ZZ", "XY")


def image_specs(latest, out_dir):
    """Image list for one run; turbulence images only for fields it wrote."""
    have = {p.name for p in latest.iterdir()}
    zoom = [BL_ZOOM_X - 0.5 * BL_ZOOM_WIDTH, BL_ZOOM_X + 0.5 * BL_ZOOM_WIDTH,
            0.0, BL_ZOOM_HEIGHT]
    bl = f"boundary layer at x = {BL_ZOOM_X:g} m"
    entries = [
        ("domain_p", "p", None, "p [Pa]", "pressure, whole domain", None, "Cool to Warm", False),
        ("domain_U", "U", "Magnitude", "|U| [m/s]", "velocity magnitude, whole domain",
         None, "Viridis", False),
        ("bl_U", "U", "Magnitude", "|U| [m/s]", f"velocity magnitude, {bl}",
         zoom, "Viridis", False),
    ]
    for field, component, title, caption, preset, log in TURBULENCE_FIELDS:
        if field in have:
            entries.append((f"bl_{field}", field, component, title,
                            f"{caption}, {bl}", zoom, preset, log))
    if "R" in have:
        r_field, r_note = "R", "Reynolds stress R"
    elif "turbulenceProperties:R" in have:
        r_field, r_note = "turbulenceProperties:R", "Reynolds stress R (derived from the model)"
    else:
        r_field = None
    if r_field:
        for component in R_COMPONENTS:
            entries.append((f"bl_R_{component}", r_field, component,
                            f"R_{component.lower()} [m2/s2]",
                            f"{r_note}, component {component}, {bl}",
                            zoom, "Cool to Warm", False))
    return [
        {"out": str(out_dir / f"{i:02d}_{name}.png"), "field": field,
         "component": component, "title": title, "caption": caption,
         "window": window, "preset": preset, "log": log}
        for i, (name, field, component, title, caption, window, preset, log)
        in enumerate(entries, start=1)
    ]


def main():
    script_dir = Path(__file__).resolve().parent
    _, case_name, run_case_dir, _, results_root = pp.resolve_case_paths(script_dir)
    pvbatch = pp.find_pvbatch()
    if pvbatch is None:
        print("[warn] ParaView pvbatch not found ($PARAVIEW_PVBATCH, PATH, "
              "~/opt/ParaView-*) - skipping field images")
        return 0
    runs = pp.solved_run_dirs(run_case_dir)
    if not runs:
        print(f"[info] no finished runs with a p field in {run_case_dir}")
        return 0

    env = pp.pvbatch_env()
    failed = []
    n_images = 0
    for run, latest in runs:
        foam = run / "p.foam"
        foam.touch(exist_ok=True)
        out_dir = results_root / run.name / "field_plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob("*.png"):
            stale.unlink()
        specs = image_specs(latest, out_dir)
        cmd = [pvbatch, "--force-offscreen-rendering", str(script_dir / RENDER_SCRIPT),
               str(foam), run.name, json.dumps(specs)]
        print(f"[field_images] {run.name}: {len(specs)} image(s) -> {out_dir}")
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                  timeout=TIMEOUT_S)
            missing = [s["out"] for s in specs if not Path(s["out"]).is_file()]
            ok = proc.returncode == 0 and not missing
            output = proc.stdout + proc.stderr
            if missing:
                output += f"\nmissing images: {missing}"
        except subprocess.TimeoutExpired:
            ok, output = False, f"timed out after {TIMEOUT_S} s"
        if ok:
            n_images += len(specs)
        else:
            print(f"  [FAILED] {run.name}:\n" + "\n".join(output.splitlines()[-20:]))
            failed.append(run.name)
    if failed:
        print(f"[warn] field images failed for: {failed}")
        return 1
    print(f"[field_images] {n_images} image(s) written for {len(runs)} run(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
