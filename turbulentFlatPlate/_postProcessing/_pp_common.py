"""
_pp_common.py – shared helpers for the turbulentFlatPlate post-processing
scripts (postpro.py, cf_validation.py, dashboard.py).

Not a standalone script: the run_simulation.sh driver copies every *.py from
_postProcessing/ into run/<case>/ but skips execution of files whose name
starts with an underscore.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def resolve_case_paths(script_dir):
    """Return (base_path, case_name, run_case_dir, source_case_dir, results_root).

    script_dir is either the case-local `_postProcessing` source directory or a
    directory directly under `run/<case>/`.

    For a study run, results_root is `results/<case>/<study>/`: the study name
    comes from run/<case>/study_manifest.json, so every script lands in the
    folder run_study.py created and studies no longer overwrite each other.
    Without a manifest (run_simulation.sh) it stays `results/<case>/`.
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
    manifest = load_manifest(run_case_dir)
    if manifest and manifest.get("study"):
        results_root = results_root / manifest["study"]
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
# Reference data (Tecplot) and sampled line profiles
# ---------------------------------------------------------------------------
def parse_tecplot_zones(path):
    """Parse a Tecplot-style .dat file into {zone name: ndarray(N, C)}.

    Recognises `zone, t="<name>"` / `ZONE T="<name>"` headers.  The column
    count is taken from the first numeric row of each zone; rows of a
    different width are skipped, which drops the `I=..., J=...` and
    `DT=(SINGLE ...)` lines of ordered zones without special-casing them.
    """
    zones, rows, name = {}, None, None
    with open(path) as stream:
        for line in stream:
            match = re.search(r'zone.*t\s*=\s*"([^"]+)"', line, re.I)
            if match:
                name = match.group(1)
                zones[name] = rows = []
                continue
            if name is None:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                values = [float(v) for v in parts]
            except ValueError:
                continue
            if rows and len(values) != len(rows[0]):
                continue
            rows.append(values)
    return {key: np.asarray(value) for key, value in zones.items() if value}


_SYMM_TENSOR = ("xx", "xy", "xz", "yy", "yz", "zz")

# `sets` names its output file after the sampled fields, ordered by type
# (scalars, vectors, symmTensors) and alphabetically within a type, skipping
# fields the model does not have (e.g. k/omega for SA).  The column layout
# therefore differs per model and has to come from the file name.
PROFILE_FIELD_COLUMNS = {
    "rho": ["rho"],
    "yPlus": ["yPlus"],
    "nut": ["nut"],
    "nuTilda": ["nuTilda"],
    "k": ["k"],
    "omega": ["omega"],
    "epsilon": ["epsilon"],
    "U": ["Ux", "Uy", "Uz"],
    "wallShearStress": ["tau_x", "tau_y", "tau_z"],
    "R": [f"R_{c}" for c in _SYMM_TENSOR],                           # transported (RSM)
    "turbulenceProperties:R": [f"R_tp_{c}" for c in _SYMM_TENSOR],   # turbulenceFields output
}


def profile_columns_from_filename(file_path):
    """Column names of a `<set>_profile_<field>_<field>_....xy` file, or None.

    The name is only trustworthy when it expands to the actual column count -
    the caller checks that.
    """
    base = Path(file_path).name
    marker = "_profile_"
    if marker not in base or not base.endswith(".xy"):
        return None
    cols = ["y"]
    for token in base.split(marker, 1)[1][: -len(".xy")].split("_"):
        if token not in PROFILE_FIELD_COLUMNS:
            return None
        cols.extend(PROFILE_FIELD_COLUMNS[token])
    return cols


def read_profile_set(file_path):
    """(column names, ndarray) of one sampled `sets` profile, or (None, None).

    Rows are sorted by wall distance.  Only files whose name expands to the
    column count are accepted, so a layout change cannot silently mislabel
    columns.
    """
    arr = read_numeric_rows(file_path)
    if arr is None:
        return None, None
    cols = profile_columns_from_filename(file_path)
    if cols is None or len(cols) != arr.shape[1]:
        return None, None
    return cols, arr[np.argsort(arr[:, 0])]


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


def _thermo_mu(text, t_inf):
    """Dynamic viscosity from a thermophysicalProperties body, or None.

    `transport const` pins `mu` directly; `transport sutherland` gives it as
    As*sqrt(T)/(1 + Ts/T), evaluated here at the freestream T.
    """
    match = re.search(r"^\s*mu\s+([-\d.eE+]+)\s*;", text, re.M)
    if match:
        return float(match.group(1))
    a_match = re.search(r"^\s*As\s+([-\d.eE+]+)\s*;", text, re.M)
    s_match = re.search(r"^\s*Ts\s+([-\d.eE+]+)\s*;", text, re.M)
    if a_match and s_match and t_inf:
        a_s, t_s = float(a_match.group(1)), float(s_match.group(1))
        return a_s * math.sqrt(t_inf) / (1.0 + t_s / t_inf)
    return None


def detect_case_parameters(case_dir, fallback_u_inf=69.4, fallback_nu=1.388e-05):
    """Detect (u_inf, nu, rho) for a model case directory.

    u_inf : X-component of the internalField of 0/U (fallback 0.orig/U).
    rho   : for a compressible case (constant/thermophysicalProperties present)
             rho = p/(R_s*T) from the internalField of 0/p, 0/T (fallback
             0.orig) plus molWeight.  Valid to <<1 % here: constant T,
             ~zero wall pressure gradient, Ma = 0.2 -> wall rho ~= inlet rho.
             For an incompressible case: `rho` from transportProperties if
             present, else 1.0.
    nu    : for a compressible case mu/rho, with mu taken from the same
             thermophysicalProperties the solver reads (`transport const` or
             `sutherland`).  This is the single source of truth: mu is what
             sets Re here (TMR: M = 0.2, Re/m = 5e6 -> mu = 1.769e-05,
             nu = 1.388e-05).  Only an incompressible case falls back to `nu`
             from transportProperties.
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

    rho = None
    match = re.search(r"^\s*rho\s+([-\d.eE+]+)\s*;", tp_text, re.M)
    if match:
        rho = float(match.group(1))

    t_inf = _read_internal_field(case_dir / "0" / "T")
    if t_inf is None:
        t_inf = _read_internal_field(case_dir / "0.orig" / "T")

    mu = None
    tpm_path = case_dir / "constant" / "thermophysicalProperties"
    if tpm_path.is_file():
        tpm_text = tpm_path.read_text()
        m_match = re.search(r"molWeight\s+([-\d.eE+]+)", tpm_text)
        p = _read_internal_field(case_dir / "0" / "p")
        if p is None:
            p = _read_internal_field(case_dir / "0.orig" / "p")
        if m_match and p is not None and t_inf:
            m_kg_per_kmol = float(m_match.group(1))
            if m_kg_per_kmol > 0:
                rho = p / ((8314.462618 / m_kg_per_kmol) * t_inf)
        mu = _thermo_mu(tpm_text, t_inf)

    if rho is None:
        rho = 1.0

    if mu is not None:
        nu = mu / rho
    else:
        # Incompressible case: nu straight from transportProperties.
        nu = fallback_nu
        match = re.search(r"^\s*nu\s+([-\d.eE+]+)\s*;", tp_text, re.M)
        if match:
            nu = float(match.group(1))
    return u_inf, nu, rho


# ---------------------------------------------------------------------------
# ParaView pvbatch helpers (field_images.py, bl_thickness.py)
# ---------------------------------------------------------------------------
def find_pvbatch():
    """pvbatch executable: $PARAVIEW_PVBATCH, `pvbatch` on PATH, newest
    ~/opt/ParaView-*/bin/pvbatch (Kitware Linux binary); None if absent."""
    env = os.environ.get("PARAVIEW_PVBATCH")
    if env:
        return env if Path(env).is_file() else None
    on_path = shutil.which("pvbatch")
    if on_path:
        return on_path
    installs = sorted(Path.home().glob("opt/ParaView-*/bin/pvbatch"))
    return str(installs[-1]) if installs else None


def pvbatch_env():
    """Environment for pvbatch: it ships its own Python and libraries, so the
    OpenFOAM/venv variables of the drivers must not leak into it."""
    return {k: v for k, v in os.environ.items()
            if k not in ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME")}


# ---------------------------------------------------------------------------
# Run completion (crashed runs must not enter the validation output)
# ---------------------------------------------------------------------------
SOLVER_LOG = "log.rhoSimpleFoam"


def run_completed(case_dir):
    """True if the solver ran to completion in case_dir.

    A diverged run (e.g. rhoSimpleFoam aborting on "Negative initial
    temperature") leaves its 0/ fields and a partial postProcessing/ tree
    behind, which otherwise looks exactly like a result.  Two cheap checks
    separate the two: the solver log ends in OpenFOAM's `End`, and a time
    directory later than 0 was reconstructed.
    """
    case_dir = Path(case_dir)
    log_path = case_dir / SOLVER_LOG
    if not log_path.is_file():
        return False
    try:
        tail = log_path.read_text(errors="replace")[-4096:]
    except OSError:
        return False
    if not re.search(r"^End\s*$", tail, re.M):
        return False
    latest = latest_time_directory(case_dir)
    if latest is None:
        return False
    try:
        return float(latest.name) > 0.0
    except ValueError:
        return False


def split_completed(run_case_dir, names):
    """Split run-dir names into (completed, failed) by run_completed()."""
    run_case_dir = Path(run_case_dir)
    completed, failed = [], []
    for name in names:
        (completed if run_completed(run_case_dir / name) else failed).append(name)
    return completed, failed


def warn_failed(failed, what):
    """Print one warning line naming the runs excluded from `what`."""
    if failed:
        print(f"[warn] excluded from {what} - solver did not reach End: "
              f"{', '.join(sorted(failed))}")


def solved_run_dirs(run_case_dir):
    """(run dir, latest time dir) for completed run dirs under run_case_dir
    with a controlDict and a reconstructed latest time holding p.

    Runs whose solver crashed are dropped (see run_completed).
    """
    run_case_dir = Path(run_case_dir)
    if not run_case_dir.is_dir():
        return []
    found, failed = [], []
    for item in sorted(run_case_dir.iterdir()):
        if not item.is_dir() or item.name.startswith("_"):
            continue
        if not (item / "system" / "controlDict").is_file():
            continue
        latest = latest_time_directory(item)
        if latest is None or not (latest / "p").is_file():
            continue
        if run_completed(item):
            found.append((item, latest))
        else:
            failed.append(item.name)
    warn_failed(failed, "post-processing")
    return found
