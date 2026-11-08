#!/usr/bin/env python3
"""
field_budget.py - field images of every budget term, one run at a time.

For each finished run, renders into results/<case>/<run>/field_plots/budget/
<eq>_<term>.png the terms of the equations of profile_budget.py (Ux, Uy
momentum for every model; SSG/LRR k, Rxx, Ryy, Rzz, -Rxy, omega; SALSA
nuTilda) in the boundary-layer window of field_images.py (x = 1 m), mesh on
top, colour bar only, as field_plots/04_bl_nut.png; the same at the leading
edge and the outlet into field_plots/{front,back}/budget/ (fi.WINDOWS).  Terms are kinematic and
signed as in profile_budget (gain positive); identically zero terms are left
out.  Colour scale symmetric logarithmic (SYMLOG, see _pv_field_images.py):
most terms grow by decades towards the wall (omega terms ~ y^-4), so a
linear scale would show the first cells only.

The terms come from the turbulenceBudget function object, run as
`rhoSimpleFoam -postProcess` with writeObjects in a throw-away shadow case
(profile_budget.shadow_case), which pvbatch (_pv_field_images.py) renders
before it is deleted: nothing but the images is kept.  Images are reused while
their .stamp is newer than the run data, the renderer and the library, and
its key matches.  Without ParaView it warns and exits 0.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import _pp_common as pp
import field_images as fi
import profile_budget as pb

RENDER_SCRIPT = "_pv_field_images.py"
OUT_DIR = "budget"
TIMEOUT_S = 5400
SYMLOG = {"top": 0.995, "ref": 0.5, "decades": 4}

VTK_COMPONENTS = {"x": 0, "y": 1, "z": 2,
                  "xx": 0, "yy": 1, "zz": 2, "xy": 3, "yz": 4, "xz": 5}
UNITS = {"U": "m/s$^2$", "R": "m$^2$/s$^3$", "Omega": "1/s$^2$",
         "NuTilda": "m$^2$/s$^2$"}
SKIP = {("k", "pressureStrain"), ("Rzz", "production"), ("Rxy", "dissipation")}


def out_dirs(field_dir):
    """{zoom window: output dir} of one run's field_plots dir."""
    return {name: Path(field_dir) / name / OUT_DIR for name in fi.WINDOWS}


def image_specs(source_terms, equations, field_dir):
    """Image list of one run from {source equation: [term, ...]} written."""
    specs = []
    for name, out_dir in out_dirs(field_dir).items():
        specs += window_specs(source_terms, equations, out_dir, fi.window(name))
    return specs


def window_specs(source_terms, equations, out_dir, zoom):
    """Image list of one zoom window."""
    labels = {t: lab for t, lab, *_ in pb.TERMS}
    specs = []
    for stem, label, source, comps, sign, _visc in pb.PLOT_EQUATIONS:
        if source not in equations:
            continue
        if comps is None:
            weights = [[None, sign]]
        else:
            weights = [[VTK_COMPONENTS[c], sign * (0.5 if stem == "k" else 1.0)]
                       for c in comps]
        for term, *_ in pb.TERMS:
            if term not in source_terms.get(source, ()) or (stem, term) in SKIP:
                continue
            specs.append({
                "out": str(out_dir / f"{stem}_{term}.png"),
                "field": f"budget{source}:{term}", "component": None,
                "weights": weights, "symlog": SYMLOG,
                "title": f"{labels[term]}\n{label} [{UNITS[source]}]",
                "caption": f"{labels[term]} in the {stem} equation",
                "window": zoom, "preset": "Cool to Warm", "log": False})
    return specs


def written_terms(time_dir):
    """{source equation: [term, ...]} of the budget fields in time_dir."""
    out = {}
    for path in sorted(Path(time_dir).glob("budget*:*")):
        source, _, term = path.name[len("budget"):].partition(":")
        if term != pb.IMBALANCE:
            out.setdefault(source, []).append(term)
    return out


def main():
    script_dir = Path(__file__).resolve().parent
    base_path, case_name, run_case_dir, _, results_root = pp.resolve_case_paths(script_dir)
    pvbatch = pp.find_pvbatch()
    if pvbatch is None:
        print("[warn] ParaView pvbatch not found ($PARAVIEW_PVBATCH, PATH, "
              "~/opt/ParaView-*) - skipping budget field images")
        return 0
    src_dir = Path(base_path) / "src"
    lib = src_dir / "turbulenceBudget" / "lib" / "libturbulenceBudget.so"
    if not lib.is_file():
        print(f"[warn] {lib} missing - build it with `wmake libso` in its src dir")
        return 0
    runs = [(run, latest) for run, latest in pp.solved_run_dirs(run_case_dir)
            if pb.run_model(run)]
    if not runs:
        print(f"[info] no finished RAS runs in {run_case_dir}")
        return 0

    env = pp.pvbatch_env()
    render = script_dir / RENDER_SCRIPT

    def render_run(item):
        """(run, out dir, number of images, ok, note); reuses unchanged images."""
        run, latest = item
        equations = pb.run_equations(run)
        field_dir = results_root / run.name / "field_plots"
        dirs = out_dirs(field_dir)
        out_dir = dirs[""]
        stamp = out_dir / ".stamp"
        key = pb.cache_key(src_dir, equations, ()) + json.dumps(
            [SYMLOG, sorted(SKIP), fi.WINDOWS, fi.BL_ZOOM_ASPECT,
             pp.source_key(image_specs, window_specs, out_dirs, written_terms)])
        old = sorted(p for d in dirs.values() for p in d.glob("*.png"))
        if (all(any(d.glob("*.png")) for d in dirs.values())
                and pp.outputs_fresh(stamp, old, pp.run_data_inputs(run, latest)
                                     + [render, lib, Path(__file__)], key)):
            return run, out_dir, len(old), True, "unchanged, reused"
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        stamp.unlink(missing_ok=True)
        for stale in old:
            stale.unlink()
        with tempfile.TemporaryDirectory(prefix="budget_fields_") as shadow:
            pb.shadow_case(run, latest, shadow, src_dir)
            dict_path = Path(shadow) / "system" / "budgetDict"
            dict_path.write_text(pb.budget_dict(lib, equations, (), write_fields=True))
            proc = subprocess.run(pb.foam_command(shadow, dict_path), capture_output=True,
                                  text=True)
            output = proc.stdout + proc.stderr
            specs = image_specs(written_terms(Path(shadow) / latest.name), equations,
                                field_dir)
            if proc.returncode != 0 or not specs:
                return run, out_dir, 0, False, output
            foam = Path(shadow) / "budget.foam"
            foam.touch()
            cmd = [pvbatch, "--force-offscreen-rendering", str(render),
                   str(foam), run.name, json.dumps(specs)]
            try:
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                      timeout=TIMEOUT_S)
                missing = [s["out"] for s in specs if not Path(s["out"]).is_file()]
                ok = proc.returncode == 0 and not missing
                output = proc.stdout + proc.stderr
                if missing:
                    output += f"\n{len(missing)} image(s) missing, first {missing[0]}"
            except subprocess.TimeoutExpired:
                ok, output = False, f"timed out after {TIMEOUT_S} s"
        if ok:
            pp.write_stamp(stamp, key)
        return run, out_dir, len(specs), ok, "rendered" if ok else output

    print("=" * 64)
    print(f"{case_name} - budget-term field images")
    print("=" * 64)
    failed = []
    n_images = 0
    for run, out_dir, count, ok, note in pp.map_threads(render_run, runs, pp.pv_jobs()):
        if ok:
            print(f"[field_budget] {run.name}: {count} image(s) -> {out_dir} ({note})")
            n_images += count
        else:
            print(f"[field_budget] {run.name}  [FAILED]:\n"
                  + "\n".join(note.splitlines()[-20:]))
            failed.append(run.name)
    if failed:
        print(f"[warn] budget field images failed for: {failed}")
        return 1
    print(f"[field_budget] {n_images} image(s) written for {len(runs)} run(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
