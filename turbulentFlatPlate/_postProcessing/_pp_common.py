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
# Set this to an archived study's `_rundata` folder to regenerate its figures
# without re-running the solver (run_study.py writes one per study, with a
# REPRODUCE.md next to it).
RUN_DIR_ENV = "PP_RUN_CASE_DIR"


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
    # Regenerating figures from an archived study: point every script at
    # results/<case>/<study>/_rundata instead of the (wiped) run/ tree.
    # The figures go back into the study folder that holds that _rundata, not
    # the one the archived manifest names, so a renamed study redraws in place.
    override = os.environ.get(RUN_DIR_ENV)
    if override:
        run_case_dir = Path(override).expanduser().resolve()
        if run_case_dir.name == "_rundata":
            return base_path, case_name, run_case_dir, source_case_dir, run_case_dir.parent
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
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    # Each combo carries its run dir, so the analytical scales (case_scales)
    # can be derived from the combo alone, wherever it is labelled.
    for combo in manifest.get("combos", []):
        if combo.get("run_dir"):
            combo.setdefault("_case_dir", str(Path(run_case_dir) / combo["run_dir"]))
    return manifest


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
    u_path = case_dir / "0.orig" / "U"
    if not u_path.is_file():
        u_path = case_dir / "0" / "U"
    if u_path.is_file():
        match = re.search(
            r"internalField\s+uniform\s*\(\s*([-\d.eE+]+)\s+[-\d.eE+]+\s+[-\d.eE+]+\s*\)",
            u_path.read_text(errors="ignore"),
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


# ---------------------------------------------------------------------------
# Reuse of per-run outputs, parallel execution
# ---------------------------------------------------------------------------
# A redraw usually changes labels, not data, so the expensive per-run outputs
# (pvbatch extractions and renders, per-run convergence plots) are reused when
# nothing they depend on changed.  A stamp file next to them holds a key (the
# extraction arguments or the plotting code) and its mtime records when they
# were made: they are reused only if the key matches and the stamp is newer
# than every input.  Outputs from before stamps existed are reused once when
# they are all newer than the inputs, and stamped then.
PV_JOBS_ENV = "PP_PV_JOBS"
JOBS_ENV = "PP_JOBS"


def newest_mtime(paths):
    """Newest mtime over the given files and directory trees (0 if none exist)."""
    newest = 0.0
    for path in paths:
        path = Path(path)
        if path.is_dir():
            for root, _dirs, files in os.walk(path):
                newest = max([newest, os.stat(root).st_mtime]
                             + [os.stat(os.path.join(root, f)).st_mtime for f in files])
        elif path.exists():
            newest = max(newest, path.stat().st_mtime)
    return newest


def run_data_inputs(run_dir, latest):
    """What a run's derived outputs depend on: its latest time and its mesh."""
    return [Path(latest), Path(run_dir) / "constant" / "polyMesh"]


def outputs_fresh(stamp, outputs, inputs, key=""):
    """True when `outputs` can be reused (see above); stamps legacy outputs."""
    stamp = Path(stamp)
    outputs = [Path(o) for o in outputs]
    try:
        if not outputs or not all(o.is_file() and o.stat().st_size > 0 for o in outputs):
            return False
        newest_input = newest_mtime(inputs)
        if stamp.is_file():
            return stamp.read_text() == key and stamp.stat().st_mtime > newest_input
        if min(o.stat().st_mtime for o in outputs) > newest_input:
            write_stamp(stamp, key)
            return True
    except OSError:
        pass
    return False


def write_stamp(stamp, key=""):
    Path(stamp).write_text(key)


def source_key(*functions):
    """Key that changes whenever the source of any of the functions changes."""
    import hashlib
    import inspect
    text = "".join(inspect.getsource(f) for f in functions)
    return hashlib.sha1(text.encode()).hexdigest()


def _env_jobs(name, default):
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


def pv_jobs():
    """Concurrent pvbatch processes ($PP_PV_JOBS, default 3).  Each one holds a
    whole mesh in memory: on this 7 GB machine keep the total over all
    concurrent run_postpro.sh calls at about 3."""
    return _env_jobs(PV_JOBS_ENV, 3)


def plot_jobs():
    """Worker processes for figure-heavy scripts ($PP_JOBS, default min(6, cores))."""
    return _env_jobs(JOBS_ENV, min(6, os.cpu_count() or 1))


def map_threads(fn, items, jobs):
    """[fn(item)] with up to `jobs` threads, in order; for subprocess-bound work."""
    items = list(items)
    if jobs <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(fn, items))


_FORK_JOB = {}


def _fork_call(i):
    fn, items = _FORK_JOB["job"]
    return fn(items[i])


def map_processes(fn, items, jobs):
    """[fn(item)] in up to `jobs` forked processes, in order.  fn and items reach
    the workers through fork, so they need not be picklable; results must be."""
    items = list(items)
    if jobs <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    _FORK_JOB["job"] = (fn, items)
    try:
        with ProcessPoolExecutor(max_workers=jobs,
                                 mp_context=multiprocessing.get_context("fork")) as pool:
            return list(pool.map(_fork_call, range(len(items))))
    finally:
        _FORK_JOB.clear()


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


# ---------------------------------------------------------------------------
# Study manifest access
# ---------------------------------------------------------------------------
# Sweep values come from the study manifest, never from parsing run-dir names
# (combo_tag mangles values - 1.0e-06 becomes "1em06").
def combo_index(manifest):
    """run_dir -> combo dict from a study manifest ({} without one)."""
    if not manifest:
        return {}
    return {c["run_dir"]: c for c in manifest.get("combos", []) if c.get("run_dir")}


def swept_keys(combos):
    """Parameter keys that vary *within* at least one approach.

    Keys that merely differ between approaches (n_y for blockMesh vs
    wallCellSize for gmsh) are not a sweep and must not drive the markers.
    """
    by_approach = {}
    for combo in combos:
        by_approach.setdefault(combo.get("approach"), []).append(
            combo.get("params") or {})
    keys = set()
    for params_list in by_approach.values():
        for key in set().union(*(set(p) for p in params_list)) if params_list else ():
            values = {repr(p.get(key)) for p in params_list}
            if len(values) > 1:
                keys.add(key)
    return sorted(keys)


# ---------------------------------------------------------------------------
# Non-dimensional case description
# ---------------------------------------------------------------------------
# Every figure describes a mesh by two numbers instead of its raw parameters:
#   h+              = h u_tau / nu         prism-stack height in wall units
#                                          (replaces nPrisms)
#   L_nom/delta_m   = L_nom / delta_mean   NOMINAL streamwise wall core cell size
#                                          (replaces wallLevel, wallCellSize, ...)
# and figures that put every stack on one axis by
#   L* = (L_core/delta_mean) * phi,  phi = <(delta - h)+> / delta_mean
# (L_core: core cell size measured next to the stack, _core_size.py;
# phi: outgrown fraction of the mean boundary layer, 0 if it never leaves the stack).
# Along the plate x/delta_mean.  u_tau and delta_mean are ANALYTICAL (_bl_theory
# composite profile + momentum integral): u_tau at NORM_STATION, delta_mean =
# delta99 averaged over 0 <= x <= NORM_STATION, so the normalisation is one
# constant per flow condition, identical for every run, approach, model and study.
NORM_STATION = 2.0
STACK_KEYS = ("nPrisms", "expansionRatio", "firstLayerThickness", "totalLayerThickness")
CORE_KEYS = ("wallCellSize", "wallLevel", "refineDistance", "backgroundCellSize")
HPLUS_SYMBOL = r"$h^+$"
DX_SYMBOL = r"$L_\mathrm{nom}/\bar\delta$"
LSTAR_SYMBOL = r"$L^*$"
DELTA_L_SYMBOL = r"$\delta_L$"
DELTA_MEAN_SYMBOL = r"$\bar\delta$"


def flow_scales(u_inf, nu, station=NORM_STATION):
    """Analytical normalisation of one flow condition (see NORM_STATION)."""
    import _bl_theory as theory

    key = (round(float(u_inf), 9), float(f"{nu:.9g}"), float(station))
    if key not in _FLOW_SCALES:
        re_per_m = u_inf / nu
        cf = float(theory.cf_composite(station, re_per_m))
        x, delta = delta_curve({"u_inf": u_inf, "nu": nu, "station": station})
        _FLOW_SCALES[key] = {
            "u_inf": float(u_inf), "nu": float(nu), "station": float(station),
            "delta_L": float(theory.delta99_composite(station, re_per_m)),
            "delta_mean": float(np.trapezoid(delta, x) / station),
            "cf": cf, "u_tau": float(u_inf * math.sqrt(cf / 2.0)),
        }
    return _FLOW_SCALES[key]


def delta_curve(scales, n=4001):
    """Analytical delta99 over the plate 0 <= x <= station: (x, delta)."""
    import _bl_theory as theory

    x = np.linspace(0.0, scales["station"], n)
    return x, theory.delta99_composite(x, scales["u_inf"] / scales["nu"])


def outgrown_fraction(h, scales):
    """phi = <(delta - h)+> / delta_mean: plate-averaged analytical boundary-layer
    thickness above the stack top h, relative to the mean thickness."""
    x, delta = delta_curve(scales)
    return float(np.trapezoid(np.maximum(delta - h, 0.0), x) / scales["station"]
                 / scales["delta_mean"])


_FLOW_SCALES = {}
_CASE_SCALES = {}
# Scales of the combos labelled last, for the legend note of that figure.
_ACTIVE_SCALES = {}


def case_scales(case_dir=None):
    """flow_scales() of a run case dir; the TMR defaults without one."""
    key = str(case_dir) if case_dir else None
    if key not in _CASE_SCALES:
        if case_dir and Path(case_dir).is_dir():
            u_inf, nu, _rho = detect_case_parameters(case_dir)
        else:
            u_inf, nu, _rho = detect_case_parameters(Path("/nonexistent"))
        _CASE_SCALES[key] = flow_scales(u_inf, nu)
    return _CASE_SCALES[key]


def combo_scales(combo):
    scales = case_scales((combo or {}).get("_case_dir"))
    _ACTIVE_SCALES["scales"] = scales
    return scales


def norm_note(scales=None, sep="\n"):
    """Definition of h+ and L_nom/delta_mean; lines joined by sep."""
    scales = scales or _ACTIVE_SCALES.get("scales") or case_scales()
    return sep.join([
        rf"{HPLUS_SYMBOL} = $h\,u_\tau/\nu$ (h: prism-stack height)",
        rf"{DX_SYMBOL}: wall core cell size",
        f"analytical, composite BL at x = {scales['station']:g} m",
        rf"$u_\tau$ = {scales['u_tau']:.3f} m/s",
        rf"{DELTA_MEAN_SYMBOL} = mean $\delta_{{99}}$ over the plate = "
        f"{1e3 * scales['delta_mean']:.2f} mm",
    ])


def stack_height(combo):
    """Prism-stack height [m] of a combo, or None (blockMesh: no stack)."""
    combo = combo or {}
    mesh = combo.get("mesh") or {}
    params = combo.get("params") or {}
    if mesh.get("totalLayerThickness"):
        return float(mesh["totalLayerThickness"])
    if params.get("totalLayerThickness"):
        return float(params["totalLayerThickness"])
    try:
        n, first, ratio = params["nPrisms"], params["firstLayerThickness"], params["expansionRatio"]
    except KeyError:
        return None
    return first * n if ratio == 1 else first * (ratio ** n - 1) / (ratio - 1)


def h_plus(combo, scales=None):
    h = stack_height(combo)
    if h is None:
        return None
    scales = scales or combo_scales(combo)
    return h * scales["u_tau"] / scales["nu"]


def dx_norm(combo, scales=None):
    dx = wall_cell_size(combo or {})
    if dx is None:
        return None
    return dx / (scales or combo_scales(combo))["delta_mean"]


def fmt_hplus(value):
    return f"{HPLUS_SYMBOL} = {value:.0f}"


def fmt_dx(value):
    return f"{DX_SYMBOL} = {value:.2g}"


def fmt_lstar(value):
    return f"{LSTAR_SYMBOL} = {value:.2g}"


def variant_label(combo, keys, scales=None):
    """Label of one combo for its swept keys: stack parameters collapse to h+,
    core-size parameters to L_nom/delta_mean, anything else stays 'key=value'."""
    params = (combo or {}).get("params") or {}
    parts = []
    if any(k in STACK_KEYS and k in params for k in keys):
        value = h_plus(combo, scales)
        if value is not None:
            parts.append(fmt_hplus(value))
    if any(k in CORE_KEYS and k in params for k in keys):
        value = dx_norm(combo, scales)
        if value is not None:
            parts.append(fmt_dx(value))
    for k in keys:
        if k in params and k not in STACK_KEYS and k not in CORE_KEYS:
            parts.append(f"{k}={params[k]:g}" if isinstance(params[k], float)
                         else f"{k}={params[k]}")
    return ", ".join(parts)


def _label_order(label):
    """Numeric order of variant labels ("h+ = 24" < "h+ = 590")."""
    plain = re.sub(r"\$[^$]*\$", "", label)
    return ([float(t) for t in re.findall(r"\d+(?:\.\d+)?(?:e-?\d+)?", plain)], label)


# Mesh approach every other approach is compared against (structured,
# prism-free, finest wall-normal grid).
ERROR_BASELINE = "blockMesh"


def wall_cell_size(combo):
    """Nominal streamwise wall core cell size [m] of a study combo, or None.

    snappy: backgroundCellSize / 2**wallLevel.  snappy45: the snapped wall
    faces of a tilted square are h*sqrt(2) long, and its backgroundCellSize
    default is only recorded in the manifest's mesh block.  gmsh:
    wallCellSize.  blockMesh grades, so it has none.
    """
    params = combo.get("params") or {}
    approach = combo.get("approach")
    if approach == "snappy" and "wallLevel" in params:
        return params["backgroundCellSize"] / 2 ** params["wallLevel"]
    if approach == "snappy45":
        mesh = (combo.get("mesh") or {}).get("params", params)
        if "wallLevel" in mesh and "backgroundCellSize" in mesh:
            return mesh["backgroundCellSize"] * math.sqrt(2.0) / 2 ** mesh["wallLevel"]
        return None
    return params.get("wallCellSize")


# ---------------------------------------------------------------------------
# Report figure style
# ---------------------------------------------------------------------------
# Every figure goes into the report as is: text width, report-sized fonts, no
# title and no notes.  Model, scheme set, station, normalisation and the
# meaning of the reference lines belong in the caption (CAPTIONS in
# _captions.py, collected into summary/CAPTIONS.md by dashboard.py).
SUMMARY_DIR = "summary"
REPORT_WIDTH_IN = 6.3
REPORT_DPI = 200
REPORT_RC = {"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
             "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
             "lines.linewidth": 1.3, "axes.linewidth": 0.7,
             "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
             "axes.axisbelow": True, "xtick.direction": "in", "ytick.direction": "in",
             "xtick.top": True, "ytick.right": True, "legend.frameon": False,
             "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "savefig.dpi": REPORT_DPI}

INK, MUTED, EDGE = "#0b0b0b", "#52514e", "#9a9a9a"
APPROACHES = ("snappy", "gmshTri", "gmshHex", "snappy45")
APPROACH_COLORS = {"blockMesh": INK, "snappy": "#2a78d6", "gmshTri": "#eb6834",
                   "gmshHex": "#1baf7a", "snappy45": "#9a3fb5"}
# Wall core cell sizes, coarsest to finest.
CORE_COLORS = ("#eda100", "#e34948", "#6250d6", "#008300", "#1f77b4", "#8c564b",
               "#e377c2", "#7f7f7f")
CORE_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
# Prism-stack heights h+, lowest first.
STACK_MARKERS = ("o", "s", "D", "^", "v")
MODEL_LABELS = {"SpalartAllmarasSALSA": "SALSA", "SpalartAllmaras": "SA",
                "SSGLRROmega2019": "SSG/LRR", "kOmegaSST": "SST", "SSG": "SSG", "LRR": "LRR"}

BASELINE_STYLE = dict(color=INK, linestyle="--", linewidth=1.1, zorder=5)
STACK_TOP_STYLE = dict(color=MUTED, linestyle=":", linewidth=1.0)
STACK_TOP_LABEL = "prism-stack top"

SYM = {"cf": r"C_f", "cd": r"C_D", "delta99": r"\delta_{99}", "dstar": r"\delta^*",
       "theta": r"\theta", "H": r"H", "yplus": r"y^+"}
X_PLATE_LABEL = r"$x/\bar\delta$"


def report_style():
    """Apply REPORT_RC globally (scripts that only draw report figures)."""
    import matplotlib

    matplotlib.rcParams.update(REPORT_RC)


def error_label(key, where=""):
    """$\\varepsilon_{C_f}$ [%] etc.; where = e.g. "RMS"."""
    text = rf"$\varepsilon_{{{SYM.get(key, key)}}}$ [%]"
    return f"{where} {text}" if where else text


def plain_log_axis(ax, axis="y", subs=(1.0, 2.0, 5.0)):
    """Log axis with plain tick numbers (0.5, 1, 10) instead of 10^x / 3x10^1."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(LogLocator(base=10, subs=subs))
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    target.set_minor_formatter(NullFormatter())


Y_DELTA_RANGE = (1e-4, 1.1)


def log_y_delta_axis(ax):
    """Wall-normal axis y/delta99 of every profile figure: logarithmic over
    Y_DELTA_RANGE, down to y+ < 0.5 at every station."""
    ax.set_yscale("log")
    ax.set_ylim(*Y_DELTA_RANGE)
    plain_log_axis(ax, subs=(1.0,))


def plain_ticks(ax, axis="y"):
    """Plain tick numbers (-10, -1, 0, 1, 10) on a symlog or log axis."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\u2212")))
    target.set_minor_formatter(NullFormatter())


def line_handle(label, color=None, linestyle="-", marker=None, **kw):
    from matplotlib.lines import Line2D

    return Line2D([], [], color=color or MUTED, linestyle=linestyle, marker=marker,
                  markersize=4 if marker else 0, markeredgecolor="white",
                  markeredgewidth=0.4, label=label, **kw)


def legend_below(fig, handles, ncol=None):
    """Legend centred under the axes; returns the rect for fig.tight_layout()."""
    n = len(handles)
    ncol = ncol or (n if n <= 4 else int(math.ceil(n / 2)))
    rows = int(math.ceil(n / ncol))
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               handlelength=2.4, columnspacing=1.2)
    return (0, (0.17 * rows + 0.06) / fig.get_size_inches()[1], 1, 1)


def grid_figure(n_rows, n_cols, row_height=1.9, **kw):
    """Text-width figure of n_rows x n_cols panels (rows h+, columns approach)."""
    import matplotlib.pyplot as plt

    width = REPORT_WIDTH_IN if n_cols > 1 else 0.62 * REPORT_WIDTH_IN
    return plt.subplots(n_rows, n_cols, figsize=(width, row_height * n_rows + 0.7),
                        squeeze=False, **kw)


def label_grid(axes, row_labels, col_labels, ylabel, xlabel):
    """Approach on top of each column; h+ and the quantity on the left."""
    for j, label in enumerate(col_labels):
        if label:
            axes[0][j].set_title(label)
    for i, label in enumerate(row_labels):
        axes[i][0].set_ylabel(f"{label}\n{ylabel}" if label else ylabel)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)


def study_schemes(results_root):
    """'upwind' for a study with the first-order upwind fvSchemes overlay."""
    root = Path(results_root)
    return ("upwind" if "upwind" in root.name
            and (root / "overlay" / "system" / "fvSchemes").is_file() else "base")
