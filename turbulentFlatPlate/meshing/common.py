"""
common.py – shared helpers for the config-driven study pipeline.

A study is a TOML config under <case>/studies/<study>/config.toml listing
turbulence models, a meshing approach and mesh parameters.  Scalar params are
fixed, list-valued params are swept (cartesian product), and [[params.sets]]
entries define explicit parameter combinations.  The runner (run_study.py)
expands models x parameter sets, assembles run dirs named
run/<case>/<model>__<tag> and lets the selected approach generate the mesh.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tomllib
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "run"
RESULTS_ROOT = REPO_ROOT / "results"
VENV_PYTHON = REPO_ROOT.parent / ".venv" / "bin" / "python3"


def load_config(study_dir):
    config_path = Path(study_dir) / "config.toml"
    with open(config_path, "rb") as stream:
        return tomllib.load(stream)


def format_value(value):
    if isinstance(value, bool):
        text = str(value).lower()
    elif isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace(".", "p").replace("-", "m")


def expand_params(params):
    """Expand [params] into concrete parameter dicts.

    `sets` (list of dicts) -> one entry per set, merged over the scalar keys.
    Otherwise list-valued keys are swept (cartesian product), scalar keys are
    merged into every combination.
    """
    fixed = {k: v for k, v in params.items() if k != "sets" and not isinstance(v, list)}
    sets = params.get("sets")
    if sets:
        return [{**fixed, **combo} for combo in sets]
    swept_keys = sorted(k for k, v in params.items() if k != "sets" and isinstance(v, list))
    if not swept_keys:
        return [dict(fixed)]
    combos = []
    for values in product(*(params[k] for k in swept_keys)):
        combo = dict(fixed)
        combo.update(dict(zip(swept_keys, values)))
        combos.append(combo)
    return combos


def combo_tag(approach, params):
    """Deterministic run-dir tag: <approach>[_<param><value>]... for values
    that differ from the approach defaults (<approach> if none differ)."""
    overrides = {k: v for k, v in params.items() if approach.defaults.get(k) != v}
    if not overrides:
        return approach.name
    parts = [approach.name]
    for key in sorted(overrides):
        parts.append(f"{key}{format_value(overrides[key])}")
    return "_".join(parts)


def copy_tree(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True)


def run_foam(cmd, cwd, log_name):
    """Run an OpenFOAM application in cwd, stdout/stderr -> log file.
    Requires the OpenFOAM environment (run via run_study.sh)."""
    log_path = Path(cwd) / log_name
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def write_manifest(run_case_dir, study_name, combos):
    path = Path(run_case_dir) / "study_manifest.json"
    path.write_text(json.dumps({"study": study_name, "combos": combos}, indent=2))
    return path


def load_manifest(run_case_dir):
    path = Path(run_case_dir) / "study_manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def stack_expansion_ratio(first, total, n_layers):
    """Expansion ratio r with first * (r**n - 1) / (r - 1) == total.

    Approaches take the prism stack as its overall height, but it is handed to
    snappyHexMesh as firstLayerThickness + expansionRatio: v2512's own
    `thickness` keyword rejects the stack outright (0% extrusion even for a
    stack that inserts at 100% via expansionRatio).
    """
    lo, hi = 1.0 + 1e-12, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if first * (mid ** n_layers - 1.0) / (mid - 1.0) < total:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
