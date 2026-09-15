"""
run_study.py – config-driven study runner (models x mesh parameter sweeps).

Driven by ./run_study.sh <case>/studies/<study> [--mesh-only] [--dry-run].

Pipeline:
  1. load config.toml, expand combos: models x parameter sets
     (list params = cartesian sweep, [[params.sets]] = explicit combos)
  2. assemble run/<case>/<model>__<tag>/ from base_case + model overlay
  3. approach.generate() -> constant/polyMesh (blockMesh / snappy / gmsh ...)
  4. run/<case>/study_manifest.json (consumed by the post-processing)
  5. ./Allrun per combo (Allrun skips blockMesh when polyMesh is pre-generated)
  6. post-processing: postpro -> other scripts -> dashboard last
"""
from __future__ import annotations

import argparse
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
    combo_tag,
    copy_tree,
    expand_params,
    load_config,
    write_manifest,
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
    try:
        approach = get_approach(config["approach"])
    except ApproachError as exc:
        print(f"[error] {exc}")
        return 2

    models = config.get("models", [])
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        models = [m for m in models if m in wanted]
    param_sets = expand_params(config.get("params", {}))

    plan = []
    for model in models:
        for params in param_sets:
            plan.append({
                "run_dir": f"{model}__{combo_tag(approach, params)}",
                "model": model,
                "approach": approach.name,
                "params": params,
            })

    print("==========================================")
    print(f"study     : {study_name}")
    print(f"case      : {case_name}")
    print(f"approach  : {approach.name}")
    print(f"combos    : {len(plan)} ({len(models)} model(s) x {len(param_sets)} param set(s))")
    for entry in plan:
        print(f"  - {entry['run_dir']}")
    print("==========================================")
    if args.dry_run:
        return 0

    run_case_dir = RUN_ROOT / case_name
    if run_case_dir.exists():
        shutil.rmtree(run_case_dir)
    run_case_dir.mkdir(parents=True)
    results_case_dir = RESULTS_ROOT / case_name
    if results_case_dir.exists():
        shutil.rmtree(results_case_dir)

    mesh_failed = []
    for entry in plan:
        target = run_case_dir / entry["run_dir"]
        target.mkdir()
        try:
            copy_tree(CASE_DIR / "base_case", target)
            model_dir = CASE_DIR / entry["model"]
            if model_dir.is_dir():
                copy_tree(model_dir, target)
            entry["mesh"] = approach.generate(target, entry["params"])
            entry["status"] = "ok"
            print(f"  [mesh ok]   {entry['run_dir']}")
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
            print(f"  [FAILED]    {entry['run_dir']}: {exc}")
    manifest_path = write_manifest(run_case_dir, study_name, plan)
    print(f"  manifest -> {manifest_path}")

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
        print(f"  -> running Allrun: {entry['run_dir']}")
        start = time.time()
        rc = subprocess.run(["sh", "./Allrun"], cwd=str(target)).returncode
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
