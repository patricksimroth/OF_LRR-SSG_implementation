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

import hashlib
import json
import shutil
import subprocess
import tomllib
from datetime import datetime
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "run"
RESULTS_ROOT = REPO_ROOT / "results"
SRC_ROOT = REPO_ROOT / "src"
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


# ---------------------------------------------------------------------------
# Custom turbulence-model libraries (src/<name>/ -> src/<name>/lib/*.so)
# ---------------------------------------------------------------------------
def library_sources():
    """Custom OpenFOAM library source dirs under src/ (those with Make/files)."""
    if not SRC_ROOT.is_dir():
        return []
    return sorted(d for d in SRC_ROOT.iterdir() if (d / "Make" / "files").is_file())


def build_libraries(log_dir=None):
    """`wmake libso` every library under src/; returns the names that failed.

    Studies build unconditionally, because the solver loads the .so named in
    controlDict, not the source: an edited-but-not-rebuilt model would produce
    results that the recorded commit does not describe.  Needs the OpenFOAM
    environment, which run_study.sh sources.
    """
    failed = []
    for src in library_sources():
        print(f"  -> wmake libso: src/{src.name}")
        proc = subprocess.run(["wmake", "libso"], cwd=str(src), text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            (Path(log_dir) / f"wmake.{src.name}.log").write_text(proc.stdout)
        if proc.returncode == 0:
            continue
        failed.append(src.name)
        print(f"  [FAILED] wmake libso in src/{src.name} (exit {proc.returncode})")
        for line in proc.stdout.strip().splitlines()[-15:]:
            print(f"      {line}")
    return failed


def library_fingerprints():
    """One record per built src/*/lib/*.so: path, size, mtime and sha256.

    The commit pins the source; this pins the binary the solver actually
    loads, which is the one thing a stale build would otherwise hide.
    """
    records = []
    for src in library_sources():
        for so in sorted((src / "lib").glob("*.so")):
            data = so.read_bytes()
            mtime = datetime.fromtimestamp(so.stat().st_mtime).astimezone()
            records.append({
                "path": str(so.relative_to(REPO_ROOT)),
                "bytes": len(data),
                "built": mtime.isoformat(timespec="seconds"),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
    return records


# ---------------------------------------------------------------------------
# Provenance (results/ is gitignored, so a study folder carries its own)
# ---------------------------------------------------------------------------
def _git(*args):
    """stdout of a git command in REPO_ROOT; None if it fails or git is absent."""
    try:
        proc = subprocess.run(["git", "-C", str(REPO_ROOT), *args], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return None
    return proc.stdout.rstrip("\n") if proc.returncode == 0 else None


def git_provenance():
    """Commit and working-tree state, or None outside a git repo.

    `dirty` carries as much weight as the commit: with uncommitted changes the
    SHA names code that never produced these results, so the diff has to be
    stored beside it (write_provenance).  `changed` is porcelain output, so
    untracked files appear as `?? path` - those are NOT in the diff.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return None
    changed = [line for line in (_git("status", "--porcelain") or "").splitlines()
               if line.strip()]
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(changed),
        "changed": changed,
        "diff": _git("diff", "HEAD") if changed else "",
    }


def write_provenance(dest, case_name, study_name, study_dir, run_case_dir):
    """Write everything needed to interpret a study's results into dest.

    results/ and run/ are gitignored and run/ is wiped by the next study, so
    the study's own folder keeps: the config it ran, the combo manifest, the
    commit it was based on and - when the tree was dirty - the exact diff.
    Returns the provenance record.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for src in (Path(study_dir) / "config.toml",
                Path(run_case_dir) / "study_manifest.json"):
        if src.is_file():
            shutil.copy2(src, dest / src.name)

    git = git_provenance()
    patch_name = None
    if git and git["diff"]:
        patch_name = "uncommitted.patch"
        (dest / patch_name).write_text(git["diff"] + "\n")
    record = {
        "case": case_name,
        "study": study_name,
        "written": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": None if git is None else {
            "commit": git["commit"],
            "branch": git["branch"],
            "dirty": git["dirty"],
            "changed": git["changed"],
            "patch": patch_name,
        },
        "libraries": library_fingerprints(),
    }
    (dest / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def report_provenance(record):
    """Print the provenance one-liner, and warn when the tree was dirty."""
    git = record.get("git")
    if git is None:
        print("  [warn] not a git repository - results carry no commit")
        return
    state = "dirty" if git["dirty"] else "clean"
    print(f"  provenance -> {git['commit'][:7]} ({git['branch']}, {state}), "
          f"{len(record['libraries'])} librar(y/ies)")
    if not git["dirty"]:
        return
    untracked = [line[3:] for line in git["changed"] if line.startswith("??")]
    print(f"  [warn] the working tree has {len(git['changed'])} uncommitted "
          f"change(s); commit {git['commit'][:7]} does NOT describe this run.")
    print(f"         The diff is stored as {git['patch']}, but it does not "
          f"contain untracked files.")
    if untracked:
        print(f"         untracked: {', '.join(untracked)}")
    print("         Commit before the runs that produce report data.")

