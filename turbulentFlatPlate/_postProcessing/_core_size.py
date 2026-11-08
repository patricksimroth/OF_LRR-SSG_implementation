"""Measured core cell size next to the prism stack.

L_core = sqrt(median 2D cell area) of the core cells whose centroid lies in the
band h + GAP * L_nom <= y <= h + (GAP + WIDTH) * L_nom over X_RANGE, with h the
prism-stack height and L_nom the nominal wall core cell size
(_pp_common.wall_cell_size).  The gap skips the first core row, whose cells are
built on the prism top and have their own shape (half diamonds on snappy45,
triangles on gmsh, cut polygons on gmshHex); the band stays close to the stack,
before the core coarsens away from the wall.  It measures what the nominal
size hides: cell shape (a triangle with edge a has 0.43 a^2, a dual polygon
~0.87 a^2) and stacks whose top lies above the finest refinement band.

Cell centroids and areas come from _pv_cell_sizes.py (pvbatch), cached per run
in results/<study>/<run>/boundary_layer/cell_sizes.npz.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

import _pp_common as pp

EXTRACT_SCRIPT = "_pv_cell_sizes.py"
CACHE_NAME = "cell_sizes.npz"
X_RANGE = (0.1, 1.95)
Y_MAX = 0.1
GAP = 1.0
WIDTH = 2.0
TIMEOUT_S = 900


def extract(run_dir, out_path, pvbatch, env):
    """Cell centroids and areas of one run, cached; True when available."""
    run_dir, out_path = Path(run_dir), Path(out_path)
    script = Path(__file__).resolve().parent / EXTRACT_SCRIPT
    mesh = run_dir / "constant" / "polyMesh" / "points"
    if (out_path.is_file() and mesh.is_file()
            and out_path.stat().st_mtime > max(script.stat().st_mtime, mesh.stat().st_mtime)):
        return True
    if pvbatch is None:
        return False
    foam = run_dir / "p.foam"
    foam.touch(exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [pvbatch, "--force-offscreen-rendering", str(script), str(foam), str(out_path),
           str(X_RANGE[0]), str(X_RANGE[1]), str(Y_MAX)]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
        ok = proc.returncode == 0 and out_path.is_file()
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        ok, output = False, f"timed out after {TIMEOUT_S} s"
    if not ok:
        print(f"  [FAILED] {run_dir.name}:\n" + "\n".join(output.splitlines()[-12:]))
    return ok


def extract_all(pairs):
    """extract() for (run_dir, cache) pairs, PP_PV_JOBS at a time."""
    pvbatch = pp.find_pvbatch()
    if pvbatch is None:
        print("[warn] ParaView pvbatch not found - only cached cell sizes are used")
    env = pp.pvbatch_env()
    return pp.map_threads(lambda p: extract(p[0], p[1], pvbatch, env), pairs, pp.pv_jobs())


def measure(cache, h, l_nom):
    """{l_core, n, area_median, area_mean, band} of one run, or None."""
    cache = Path(cache)
    if not cache.is_file() or not h or not l_nom:
        return None
    cells = np.load(cache)
    band = (h + GAP * l_nom, h + (GAP + WIDTH) * l_nom)
    sel = (cells["y"] >= band[0]) & (cells["y"] <= band[1])
    if not sel.any():
        return None
    area = cells["area"][sel]
    return {"l_core": float(np.sqrt(np.median(area))), "n": int(sel.sum()),
            "area_median": float(np.median(area)), "area_mean": float(area.mean()),
            "band": band}
