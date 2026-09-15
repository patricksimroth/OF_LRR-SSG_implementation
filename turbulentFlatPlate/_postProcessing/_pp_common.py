"""
_pp_common.py – shared helpers for the turbulentFlatPlate post-processing
scripts (postpro.py, cf_validation.py, dashboard.py).

Not a standalone script: the run_simulation.sh driver copies every *.py from
_postProcessing/ into run/<case>/ but skips execution of files whose name
starts with an underscore.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def resolve_case_paths(script_dir):
    """Return (base_path, case_name, run_case_dir, source_case_dir, results_root).

    script_dir is either the case-local `_postProcessing` source directory or a
    directory directly under `run/<case>/`.
    """
    script_dir = Path(script_dir)
    if script_dir.name == "_postProcessing":
        source_case_dir = script_dir.parent
        case_name = source_case_dir.name
        base_path = source_case_dir.parent
        run_case_dir = base_path / "run" / case_name
    else:
        case_name = script_dir.name
        parent = script_dir.parent
        if parent.name == "run":
            base_path = parent.parent
            run_case_dir = script_dir
            source_case_dir = base_path / case_name
        else:
            base_path = parent
            run_case_dir = base_path / "run" / case_name
            source_case_dir = script_dir
    results_root = base_path / "results" / case_name
    return base_path, case_name, run_case_dir, source_case_dir, results_root


def latest_time_directory(parent):
    """Return the path of the highest-numbered time-step subdirectory (or None)."""
    if parent is None:
        return None
    parent = Path(parent)
    if not parent.is_dir():
        return None
    candidates = []
    for item in parent.iterdir():
        if not item.is_dir():
            continue
        try:
            candidates.append((float(item.name), item))
        except ValueError:
            pass
    return max(candidates, key=lambda x: x[0])[1] if candidates else None


# ---------------------------------------------------------------------------
# Study manifest (config-driven runs, see <case>/meshing/run_study.py)
# ---------------------------------------------------------------------------
def load_manifest(run_case_dir):
    """Parsed study_manifest.json of a run case dir, or None if absent."""
    path = Path(run_case_dir) / "study_manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Data readers
# ---------------------------------------------------------------------------
def read_numeric_rows(file_path, comments="#"):
    """Read whitespace-separated numeric rows into a 2-D float ndarray.

    Blank and comment lines are skipped.  Returns None if no numeric rows were
    found.  A single numeric row is returned as (1, C).
    """
    rows = []
    with open(file_path) as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith(comments):
                continue
            try:
                rows.append([float(v) for v in line.split()])
            except ValueError:
                pass
    if not rows:
        return None
    arr = np.asarray(rows)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


# ---------------------------------------------------------------------------
# Case-parameter detection
# ---------------------------------------------------------------------------
def _read_internal_field(path):
    """Scalar internalField value of a field file, or None."""
    path = Path(path)
    if not path.is_file():
        return None
    match = re.search(r"internalField\s+(?:uniform\s+)?([-\d.eE+]+)", path.read_text())
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def read_dict_scalar(case_dir, relative_path, key):
    """Scalar value of a top-level `key  value;` entry in a case dictionary."""
    path = Path(case_dir) / relative_path
    if not path.is_file():
        return None
    match = re.search(rf"^\s*{re.escape(key)}\s+([-\d.eE+]+)\s*;",
                      path.read_text(), re.M)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def detect_case_parameters(case_dir, fallback_u_inf=69.4, fallback_nu=1.388e-05):
    """Detect (u_inf, nu, rho) for a model case directory.

    u_inf : X-component of the internalField of 0/U (fallback 0.orig/U).
    nu    : `nu` from constant/transportProperties (fallback: default above).
    rho   : for a compressible case (constant/thermophysicalProperties present)
             rho = p/(R_s*T) from the internalField of 0/p, 0/T (fallback
             0.orig) plus molWeight.  Valid to <<1 % here: constant T,
             ~zero wall pressure gradient, Ma ~ 0.13 -> wall rho ~= inlet rho.
             For an incompressible case: `rho` from transportProperties if
             present, else 1.0.
    """
    case_dir = Path(case_dir)

    u_inf = fallback_u_inf
    u_path = case_dir / "0" / "U"
    if not u_path.is_file():
        u_path = case_dir / "0.orig" / "U"
    if u_path.is_file():
        match = re.search(
            r"internalField\s+uniform\s*\(\s*([-\d.eE+]+)\s+[-\d.eE+]+\s+[-\d.eE+]+\s*\)",
            u_path.read_text(),
        )
        if match:
            u_inf = float(match.group(1))

    tp_path = case_dir / "constant" / "transportProperties"
    tp_text = tp_path.read_text() if tp_path.is_file() else ""

    nu = fallback_nu
    match = re.search(r"^\s*nu\s+([-\d.eE+]+)\s*;", tp_text, re.M)
    if match:
        nu = float(match.group(1))

    rho = None
    match = re.search(r"^\s*rho\s+([-\d.eE+]+)\s*;", tp_text, re.M)
    if match:
        rho = float(match.group(1))

    tpm_path = case_dir / "constant" / "thermophysicalProperties"
    if tpm_path.is_file():
        m_match = re.search(r"molWeight\s+([-\d.eE+]+)", tpm_path.read_text())
        p = _read_internal_field(case_dir / "0" / "p")
        if p is None:
            p = _read_internal_field(case_dir / "0.orig" / "p")
        t = _read_internal_field(case_dir / "0" / "T")
        if t is None:
            t = _read_internal_field(case_dir / "0.orig" / "T")
        if m_match and p is not None and t:
            m_kg_per_kmol = float(m_match.group(1))
            if m_kg_per_kmol > 0:
                rho = p / ((8314.462618 / m_kg_per_kmol) * t)

    if rho is None:
        rho = 1.0
    return u_inf, nu, rho
