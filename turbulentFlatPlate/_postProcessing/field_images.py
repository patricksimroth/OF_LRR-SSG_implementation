"""
field_images.py – field images with the mesh on top for every run.

For each run dir under run/<case>/ with a reconstructed latest time holding p,
renders into results/<case>/<run>/field_plots/:
  - whole domain: p, |U|
  - boundary layer zoom at x = BL_ZOOM_X: |U| and every turbulence quantity the
    run wrote (nut, k, omega, nuTilda, R components; the model's own R, else the
    turbulenceFields-derived turbulenceProperties:R)
  - the same zoom images at the leading edge and at the outlet (WINDOWS) into
    field_plots/front/ and field_plots/back/
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
TIMEOUT_S = 2400

# Boundary-layer zoom: delta99 ~ 0.015-0.017 m near x = 1, so 0.035 m shows the
# whole layer plus some free stream.  Same window for every mesh -> comparable.
BL_ZOOM_X = 1.0
BL_ZOOM_HEIGHT = 0.035
BL_ZOOM_WIDTH = 0.063          # ~ image aspect, fills the frame
BL_ZOOM_ASPECT = BL_ZOOM_HEIGHT / BL_ZOOM_WIDTH

# Zoom windows {sub dir: (x0, x1)}, height = aspect x width.  "" = x = 1 m;
# front: leading edge with 2.5 mm of the upstream slip patch; back: ends at the
# outlet x = 2 m, scaled with delta99 (25.8 mm there vs 14.1 mm at x = 1 m) to
# the same height/delta99 as the x = 1 m window.
WINDOWS = {
    "": (BL_ZOOM_X - 0.5 * BL_ZOOM_WIDTH, BL_ZOOM_X + 0.5 * BL_ZOOM_WIDTH),
    "front": (-0.0025, 0.010),
    "back": (1.885, 2.0),
}


def window(name):
    """[x0, x1, y0, y1] of a zoom window."""
    x0, x1 = WINDOWS[name]
    return [x0, x1, 0.0, BL_ZOOM_ASPECT * (x1 - x0)]


def window_text(name):
    """Caption phrase of a zoom window."""
    x0, x1, _, y1 = window(name)
    size = f"window {1e3 * (x1 - x0):.0f} mm wide, {1e3 * y1:.0f} mm high"
    if name == "front":
        return f"at the leading edge, x = {1e3 * x0:g} ... {1e3 * x1:g} mm ({size})"
    if name == "back":
        return f"at the outlet, x = {x0:g} ... {x1:g} m ({size})"
    return f"at x = {BL_ZOOM_X:g} m ({size})"

# (field, component, colour-bar title, caption, preset, log scale)
TURBULENCE_FIELDS = [
    ("nut", None, r"$\nu_t$ [m$^2$/s]", "turbulent viscosity nut", "Viridis", False),
    ("k", None, r"$k$ [m$^2$/s$^2$]", "turbulent kinetic energy k", "Viridis", False),
    ("omega", None, r"$\omega$ [1/s]", "specific dissipation omega (log)", "Viridis", True),
    ("nuTilda", None, r"$\tilde{\nu}$ [m$^2$/s]", "SA working variable nuTilda", "Viridis",
     False),
]
R_COMPONENTS = ("XX", "YY", "ZZ", "XY")


def image_specs(latest, out_dir):
    """Image list for one run, every zoom window."""
    specs = []
    for name in WINDOWS:
        specs += window_specs(latest, out_dir / name, window(name),
                              f"boundary layer {window_text(name)}", domain=not name)
    return specs


def window_specs(latest, out_dir, zoom, bl, domain):
    """Image list of one zoom window; turbulence images only for fields the run wrote."""
    have = {p.name for p in latest.iterdir()}
    entries = [
        ("domain_p", "p", None, r"$p$ [Pa]", "pressure, whole domain", None, "Cool to Warm",
         False),
        ("domain_U", "U", "Magnitude", r"$|U|$ [m/s]", "velocity magnitude, whole domain",
         None, "Viridis", False),
    ] if domain else []
    entries.append(("bl_U", "U", "Magnitude", r"$|U|$ [m/s]", f"velocity magnitude, {bl}",
                    zoom, "Viridis", False))
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
                            rf"$R_{{{component.lower()}}}$ [m$^2$/s$^2$]",
                            f"{r_note}, component {component}, {bl}",
                            zoom, "Cool to Warm", False))
    return [
        {"out": str(out_dir / f"{i:02d}_{name}.png"), "field": field,
         "component": component, "title": title, "caption": caption,
         "window": window, "preset": preset, "log": log}
        for i, (name, field, component, title, caption, window, preset, log)
        in enumerate(entries, start=1 if domain else 3)
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
    render = script_dir / RENDER_SCRIPT

    def render_run(item):
        """(run, out dir, number of images, ok, note); reuses unchanged images."""
        run, latest = item
        out_dir = results_root / run.name / "field_plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        specs = image_specs(latest, out_dir)
        key = json.dumps(specs, sort_keys=True)
        stamp = out_dir / ".stamp"
        if pp.outputs_fresh(stamp, [s["out"] for s in specs],
                            pp.run_data_inputs(run, latest) + [render], key):
            return run, out_dir, len(specs), True, "unchanged, reused"
        stamp.unlink(missing_ok=True)
        for name in WINDOWS:
            (out_dir / name).mkdir(exist_ok=True)
            for stale in (out_dir / name).glob("*.png"):
                stale.unlink()
        foam = run / "p.foam"
        foam.touch(exist_ok=True)
        cmd = [pvbatch, "--force-offscreen-rendering", str(render),
               str(foam), run.name, json.dumps(specs)]
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
            pp.write_stamp(stamp, key)
        return run, out_dir, len(specs), ok, "rendered" if ok else output

    failed = []
    n_images = 0
    for run, out_dir, count, ok, note in pp.map_threads(render_run, runs, pp.pv_jobs()):
        if ok:
            print(f"[field_images] {run.name}: {count} image(s) -> {out_dir} ({note})")
            n_images += count
        else:
            print(f"[field_images] {run.name}  [FAILED]:\n"
                  + "\n".join(note.splitlines()[-20:]))
            failed.append(run.name)
    if failed:
        print(f"[warn] field images failed for: {failed}")
        return 1
    print(f"[field_images] {n_images} image(s) written for {len(runs)} run(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
