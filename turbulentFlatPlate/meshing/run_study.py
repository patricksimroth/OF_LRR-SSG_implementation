"""
run_study.py – config-driven study runner (models x mesh parameter sweeps).

Driven by ./run_study.sh <case>/studies/<study> [--mesh-only] [--dry-run].

Results land in results/<case>/<study>/ and only that study's own folder is
replaced on a re-run, so studies accumulate instead of overwriting each other.
Each folder keeps the config, the manifest and a provenance stamp (commit,
dirty flag, uncommitted.patch, library fingerprints) - results/ is gitignored,
so nothing else records which code produced the data.

Pipeline:
  0. wmake libso every custom library under src/ (--no-build to skip)
  1. load config.toml, expand combos: models x approaches x parameter sets
     (list params = cartesian sweep, [[params.sets]] = explicit combos).
     `approach` may be a single name or a list; [params] is then the union over
     them and each approach is handed only the parameters it declares.
     A [params.<approach>] sub-table overrides [params] for that approach
     (each approach sizes its core with its own parameters).
  2. assemble run/<case>/<model>__<tag>/ from base_case + model overlay
  3. approach.generate() -> constant/polyMesh (blockMesh / snappy / gmsh ...)
  4. run/<case>/study_manifest.json (consumed by the post-processing, which
     reads the study name from it to resolve results/<case>/<study>/)
  5. ./Allrun per combo (Allrun skips blockMesh when polyMesh is pre-generated)
  6. post-processing: postpro -> other scripts -> dashboard last
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_DIR))

from meshing.approaches import ApproachError, get_approach
from meshing.common import (
    RESULTS_ROOT,
    RUN_ROOT,
    VENV_PYTHON,
    build_libraries,
    combo_tag,
    copy_tree,
    expand_params,
    load_config,
    report_provenance,
    write_manifest,
    write_provenance,
)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", help="path to <case>/studies/<study>")
    parser.add_argument("--mesh-only", action="store_true",
                        help="assemble run dirs + generate meshes, skip Allrun and post-processing")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the planned combos and exit")
    parser.add_argument("--models", default=None,
                        help="comma-separated subset of the configured models")
    parser.add_argument("--approaches", default=None,
                        help="comma-separated subset of the configured approaches")
    parser.add_argument("--no-build", action="store_true",
                        help="skip the wmake of the src/ libraries (the study then "
                             "runs against whatever .so is already built)")
    return parser.parse_args(argv)


def run_postproc(case_dir, run_case_dir):
    pp_src = case_dir / "_postProcessing"
    if not pp_src.is_dir():
        return []
    for script in pp_src.glob("*.py"):
        shutil.copy2(script, run_case_dir / script.name)
    names = [p.name for p in sorted(run_case_dir.glob("*.py"))
             if not p.name.startswith("_")]
    ordered = ([n for n in names if n == "postpro.py"]
               + [n for n in names if n not in ("postpro.py", "dashboard.py")]
               + [n for n in names if n == "dashboard.py"])
    failed = []
    for name in ordered:
        log_path = run_case_dir / f"{Path(name).stem}.log"
        print(f"  -> post-processing: {name} (log: {log_path})")
        with open(log_path, "w") as log:
            rc = subprocess.run([str(VENV_PYTHON), str(run_case_dir / name)],
                                cwd=str(run_case_dir), stdout=log,
                                stderr=subprocess.STDOUT).returncode
        if rc != 0:
            print(f"  [FAILED] {name} (exit {rc}) - log: {log_path}")
            failed.append(name)
    return failed


def main(argv=None):
    args = parse_args(argv)
    study_dir = Path(args.study).resolve()
    if not study_dir.is_dir():
        print(f"[error] study dir not found: {study_dir}")
        return 2
    case_name = CASE_DIR.name
    config = load_config(study_dir)
    study_name = study_dir.name
    approach_names = config.get("approach", [])
    if isinstance(approach_names, str):
        approach_names = [approach_names]
    configured = set(approach_names)
    if args.approaches:
        wanted = {a.strip() for a in args.approaches.split(",") if a.strip()}
        approach_names = [a for a in approach_names if a in wanted]
    try:
        approaches = [get_approach(name) for name in approach_names]
    except ApproachError as exc:
        print(f"[error] {exc}")
        return 2
    if not approaches:
        print("[error] no approach selected")
        return 2
    by_name = {a.name: a for a in approaches}

    models = config.get("models", [])
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        models = [m for m in models if m in wanted]
    # [params] is shared across approaches; a [params.<approach>] sub-table
    # overrides it for that one (core sizing differs per approach; the prism
    # stack is typically shared).
    raw_params = config.get("params", {})
    tables = {k for k, v in raw_params.items() if isinstance(v, dict)}
    overrides = {k: raw_params[k] for k in tables if k in by_name}
    shared = {k: v for k, v in raw_params.items() if k not in tables}

    # Sub-tables of approaches that are configured but deselected with
    # --approaches are just unused; a table naming no configured approach is a typo.
    stray = tables - set(by_name) - configured
    if stray:
        print(f"[error] [params.{sorted(stray)[0]}] is not a selected approach "
              f"({', '.join(sorted(by_name))})")
        return 2

    # Shared keys must belong to at least one approach; an override may only
    # name parameters of its own approach.
    known = set().union(*(a.defaults for a in approaches))
    unknown = {key for params in expand_params(shared) for key in params} - known
    if unknown:
        print(f"[error] parameters {sorted(unknown)} belong to no selected "
              f"approach ({', '.join(sorted(by_name))})")
        return 2
    for name, over in overrides.items():
        bad = {k for k in over if k != "sets"} - set(by_name[name].defaults)
        if bad:
            print(f"[error] [params.{name}] has unknown parameters {sorted(bad)}, "
                  f"available: {sorted(by_name[name].defaults)}")
            return 2

    per_approach_sets = {
        a.name: expand_params({**shared, **overrides.get(a.name, {})})
        for a in approaches
    }
    plan = []
    seen = set()
    for model in models:
        for approach in approaches:
            for params in per_approach_sets[approach.name]:
                own = {k: v for k, v in params.items() if k in approach.defaults}
                run_dir = f"{model}__{combo_tag(approach, own)}"
                if run_dir in seen:
                    continue
                seen.add(run_dir)
                plan.append({
                    "run_dir": run_dir,
                    "model": model,
                    "approach": approach.name,
                    "params": own,
                })

    print("==========================================")
    print(f"study     : {study_name}")
    print(f"case      : {case_name}")
    print(f"approaches: {', '.join(a.name for a in approaches)}")
    print(f"combos    : {len(plan)} ({len(models)} model(s) x "
          f"{len(approaches)} approach(es), "
          + ", ".join(f"{a.name}:{len(per_approach_sets[a.name])} set(s)"
                      for a in approaches) + ")")
    for entry in plan:
        print(f"  - {entry['run_dir']}")
    print("==========================================")
    if args.dry_run:
        return 0

    # A study wipes run/<case> on startup, so two of them at once silently
    # delete each other's combos mid-solve (the second run's Allrun then dies
    # on a missing directory).  Refuse instead.
    lock = acquire_lock(case_name)
    if lock is None:
        return 2
    try:
        return _run_study(args, case_name, study_name, study_dir, approaches,
                          by_name, plan)
    finally:
        lock.unlink(missing_ok=True)


def acquire_lock(case_name):
    """Lockfile outside run/<case> (which gets wiped). None if another study
    holds it; a stale lock from a dead process is taken over."""
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    lock = RUN_ROOT / f".{case_name}.study.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().split()[0])
        except (ValueError, IndexError, OSError):
            pid = None
        if pid is not None and pid != os.getpid():
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, PermissionError) as exc:
                if isinstance(exc, PermissionError):
                    print(f"[error] another study (pid {pid}) is running on "
                          f"{case_name} - wait for it or remove {lock}")
                    return None
            else:
                print(f"[error] another study (pid {pid}) is already running on "
                      f"{case_name}.  Studies wipe run/{case_name} on startup, "
                      f"so running two at once destroys both.  Wait for it to "
                      f"finish, or remove {lock} if that process is gone.")
                return None
        print(f"  [warn] removing stale study lock from pid {pid}")
    lock.write_text(f"{os.getpid()}\n")
    return lock


def _run_study(args, case_name, study_name, study_dir, approaches, by_name, plan):
    run_case_dir = RUN_ROOT / case_name
    if run_case_dir.exists():
        shutil.rmtree(run_case_dir)
    run_case_dir.mkdir(parents=True)
    # Only this study's own results are replaced.  Sibling studies survive,
    # each described by the config, manifest and provenance in its own folder.
    results_study_dir = RESULTS_ROOT / case_name / study_name
    if results_study_dir.exists():
        shutil.rmtree(results_study_dir)
    results_study_dir.mkdir(parents=True)

    if args.no_build:
        print("  [warn] --no-build: running against the already-built "
              "src/*/lib/*.so, which may not match its source")
    else:
        build_failed = build_libraries(results_study_dir / "_build")
        if build_failed:
            print(f"[error] wmake libso failed for: {', '.join(build_failed)}")
            print("        Aborting: the solver loads the .so named in "
                  "controlDict, so a stale binary would silently produce "
                  "results that the recorded commit does not describe.")
            return 2

    mesh_failed = []
    for entry in plan:
        target = run_case_dir / entry["run_dir"]
        try:
            target.mkdir(parents=True, exist_ok=True)
            copy_tree(CASE_DIR / "base_case", target)
            model_dir = CASE_DIR / entry["model"]
            if model_dir.is_dir():
                copy_tree(model_dir, target)
            entry["mesh"] = by_name[entry["approach"]].generate(
                target, entry["params"])
            entry["status"] = "ok"
            # Allrun starts with `rm -f log.*`, which would delete the mesh
            # generation logs (snappyHexMesh, gmsh, ...); keep them.
            for log in target.glob("log.*"):
                (target / "meshLogs").mkdir(exist_ok=True)
                shutil.move(str(log), str(target / "meshLogs" / log.name))
            print(f"  [mesh ok]   {entry['run_dir']}")
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
            print(f"  [FAILED]    {entry['run_dir']}: {exc}")
    manifest_path = write_manifest(run_case_dir, study_name, plan)
    print(f"  manifest -> {manifest_path}")
    report_provenance(write_provenance(results_study_dir, case_name, study_name,
                                       study_dir, run_case_dir))
    print(f"  results  -> {results_study_dir}")

    failed_mesh = [e["run_dir"] for e in plan if e["status"] != "ok"]
    if args.mesh_only:
        if failed_mesh:
            print(f"[warn] mesh generation failed for: {failed_mesh}")
            return 1
        return 0

    if failed_mesh:
        print(f"[warn] skipping solver for failed meshes: {failed_mesh}")

    solve_failed = []
    for entry in plan:
        if entry["status"] != "ok":
            continue
        target = run_case_dir / entry["run_dir"]
        # The case dir can disappear underneath us if something else wipes
        # run/<case> mid-study (another run_study.py, or run_simulation.sh,
        # which clears all of run/).  Report it as a failed combo and carry on
        # instead of dying on a raw FileNotFoundError from subprocess.
        if not (target / "Allrun").is_file():
            print(f"  [FAILED] {entry['run_dir']}: case directory is gone - "
                  f"something wiped run/{case_name} while this study was "
                  f"running (another study, or run_simulation.sh)")
            solve_failed.append(entry["run_dir"])
            continue
        print(f"  -> running Allrun: {entry['run_dir']}")
        start = time.time()
        try:
            rc = subprocess.run(["sh", "./Allrun"], cwd=str(target)).returncode
        except OSError as exc:
            print(f"  [FAILED] {entry['run_dir']}: could not start Allrun ({exc})")
            solve_failed.append(entry["run_dir"])
            continue
        elapsed = time.time() - start
        if rc != 0:
            print(f"  [FAILED] {entry['run_dir']} (exit {rc}) after {elapsed:.0f}s")
            solve_failed.append(entry["run_dir"])
        else:
            print(f"  [done] {entry['run_dir']} in {elapsed:.0f}s")

    postproc_failed = run_postproc(CASE_DIR, run_case_dir)

    print("==========================================")
    if not (failed_mesh or solve_failed or postproc_failed):
        print(f"Study '{study_name}' completed successfully!")
    else:
        print(f"Study '{study_name}' completed with failures:")
        if failed_mesh:
            print(f"  mesh failures    : {failed_mesh}")
        if solve_failed:
            print(f"  solver failures  : {solve_failed}")
        if postproc_failed:
            print(f"  postproc failures: {postproc_failed}")
    print("==========================================")
    return 1 if (failed_mesh or solve_failed or postproc_failed) else 0


if __name__ == "__main__":
    sys.exit(main())
