"""Compute-effort table of the timing reruns (studies/timing_*: every run of the four
core-refinement folders again on Romeo, one exclusive node and 4 core-pinned ranks per
solve, meshing serial on an exclusive node).

    .venv/bin/python3 turbulentFlatPlate/studies/core_refinement/timing.py

Reads results/turbulentFlatPlate/comparison/timing/<study>/ (zih_cluster/fetch_timing.sh)
and writes into comparison/timing/tables/:

    run_times.csv     one row per run: folder, role (prism / baseline_source /
                      baseline_mapped), h+, L_nom/delta_mean, cells, mesh time, iterations,
                      start-up, solver wall time, s per iteration, core-us per cell and
                      iteration, solver core-h, utility times, Allrun and job wall times
    run_times.txt     per folder: solver core-h per approach x h+ (rows) and core size
                      (columns), mesh time per approach, baseline = source + mapped run

and into comparison/timing/ the figures (report style, captions in CAPTIONS, written to
comparison/timing/CAPTIONS.md and by synthesis.py into comparison/CAPTIONS.md):

    cost_vs_cf_l1_error.png       cost over the L1 Cf error, rows h+, columns approach,
                                  one line per study through the core sizes
    cost_vs_cf_l1_error_all.png   the same, one panel per study, every run
    runtime_vs_cf_l1_error_all.png   as _all, over the run time (mesh + solver wall) in min

Solver core-h = ranks x rhoSimpleFoam wall time; the cost of a run in the figures adds
the serial mesh-generation time.  The billed node-hours are the job wall times (job_s),
every solve job had a whole node.  The L1 error err_l1 is read from
comparison/comparison.csv (compare.py, archived runs: v2506 reproduces v2512 to 5-6 digits).
"""
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as cmp  # noqa: E402

ROOT = cmp.an.ROOT
pp = cmp.an.pp

TIMING = ROOT / "results" / "turbulentFlatPlate" / "comparison" / "timing"
FOLDERS = ["SALSA", "SALSA_upwind", "SSGLRR", "SSGLRR_upwind"]
FOLDER_MODEL = {"SALSA": "SpalartAllmarasSALSA", "SSGLRR": "SSGLRROmega2019"}
END_TIME = 8000
UTILITIES = ["decomposePar", "mapFields", "reconstructPar", "checkMesh"]
CLUSTER_TZ = ZoneInfo("Europe/Berlin")
COMPARISON_CSV = ROOT / "results" / "turbulentFlatPlate" / "comparison" / "comparison.csv"
COST_LABEL = "cost [core-h]"
COST_TICKS = (0.03, 0.1, 0.3, 1, 3)
RUNTIME_LABEL = "run time [min]"
L1_LABEL = r"$\varepsilon_{L1}$ [%]"
COST_NOTE = ("Cost: 4 x rhoSimpleFoam wall time plus the serial mesh-generation time, in"
             " core-hours; every solve ran alone on an exclusive ZIH Romeo node (2 x AMD EPYC"
             " 7702) with 4 MPI ranks pinned to physical cores, OpenFOAM v2506.")
L1_NOTE = ("$\\varepsilon_{L1} = \\int |C_f - C_{f,\\mathrm{blockMesh}}|\\,dx /"
           " \\int C_{f,\\mathrm{blockMesh}}\\,dx$ over 0.1 m $\\le x \\le$ 1.95 m, relative to"
           " the blockMesh run of the same study (for SALSA with upwind schemes the 0.56 mm"
           " plate mesh with the coarse upstream block, i.e. the source run).")
BASE_NOTE = ("Vertical lines: cost of the blockMesh reference, source run from uniform fields"
             " plus the run mapped onto the mesh with 0.56 mm cells on both sides of the"
             " leading edge.")
CAPTIONS = {
    "cost_vs_cf_l1_error.png":
        "Computational cost of every prism-mesh run over its L1 skin-friction deviation"
        " from the blockMesh reference. Rows: prism-stack height $h^+$; columns: mesh"
        " approach; colours: model; solid: base schemes, dashed: first-order upwind"
        " convection schemes; markers: nominal wall core cell size $L_\\mathrm{nom}/\\bar\\delta$, joined from"
        " coarse to fine. " + L1_NOTE + " " + COST_NOTE + " " + BASE_NOTE + " " + cmp.NORM,
    "cost_vs_cf_l1_error_all.png":
        "As cost_vs_cf_l1_error.png, one panel per model and scheme set with every run."
        " Colours: mesh approach; markers: prism-stack height $h^+$; lines join the wall"
        " core cell sizes of one approach and $h^+$ from coarse to fine; dashed vertical"
        " line: cost of the blockMesh reference (source plus mapped run). " + L1_NOTE + " "
        + COST_NOTE + " " + cmp.NORM,
    "runtime_vs_cf_l1_error_all.png":
        "As cost_vs_cf_l1_error_all.png, over the run time instead of the cost: wall time of"
        " the serial mesh generation plus the rhoSimpleFoam run with 4 MPI ranks on an"
        " exclusive ZIH Romeo node (2 x AMD EPYC 7702, ranks pinned to physical cores,"
        " OpenFOAM v2506), without queueing, job start-up and the decomposition,"
        " reconstruction and checkMesh steps; dashed vertical line: run time of the"
        " blockMesh reference (source plus mapped run). " + L1_NOTE + " " + cmp.NORM,
}
FIELDS = ["folder", "study", "role", "run_dir", "approach", "nPrisms", "h_plus", "dx_norm",
          "nCells", "mesh_s", "iters", "ended", "capped", "startup_s", "solver_s",
          "s_per_iter", "core_us_per_cell_iter", "solver_core_h"] + \
         [f"{u}_s" for u in UTILITIES] + ["allrun_s", "job_s", "node", "ranks", "pinned"]


def folder_of(study, model):
    if study.startswith("timing_bMsrc"):
        base = "SALSA" if model == FOLDER_MODEL["SALSA"] else "SSGLRR"
        return base + ("_upwind" if study.endswith("_upwind") else ""), "baseline_source"
    if study.startswith("timing_bM_"):
        return study[len("timing_bM_"):], "baseline_mapped"
    return study[len("timing_"):], "prism"


def clock_times(log):
    return [float(v) for v in re.findall(r"ClockTime = ([\d.]+) s", log)]


def solver_record(case):
    path = case / "log.rhoSimpleFoam"
    if not path.exists():
        return None
    log = path.read_text(errors="replace")
    iters = re.findall(r"^Time = (\d+)", log, re.M)
    clocks = clock_times(log)
    nprocs = re.search(r"^nProcs\s*:\s*(\d+)", log, re.M)
    host = re.search(r"^Host\s*:\s*(\S+)", log, re.M)
    n = int(iters[-1]) if iters else 0
    rec = {"iters": n, "ended": bool(re.search(r"^End\s*$", log, re.M)),
           "ranks": int(nprocs.group(1)) if nprocs else 1,
           "node": host.group(1) if host else None}
    if clocks:
        total = clocks[-1]
        per = (clocks[-1] - clocks[0]) / (len(clocks) - 1) if len(clocks) > 1 else None
        rec.update(solver_s=total, s_per_iter=per,
                   startup_s=clocks[0] - per if per is not None else clocks[0])
    rec["capped"] = n >= END_TIME
    return rec


def utility_seconds(case, name):
    """Wall time of a utility: log header start (cluster local time) to the log's mtime,
    which rsync -a keeps.  v2506 utilities print no ClockTime."""
    path = case / f"log.{name}"
    if not path.exists():
        return None
    head = path.read_text(errors="replace")[:4000]
    date = re.search(r"^Date\s*:\s*(.+?)\s*$", head, re.M)
    time = re.search(r"^Time\s*:\s*(\d\d:\d\d:\d\d)\s*$", head, re.M)
    if not (date and time):
        return None
    start = datetime.strptime(f"{date.group(1)} {time.group(1)}", "%b %d %Y %H:%M:%S")
    start = start.replace(tzinfo=CLUSTER_TZ).timestamp()
    return max(path.stat().st_mtime - start, 0.0)


def pinned(case):
    path = case / "binding.txt"
    if not path.exists() or not path.read_text().strip():
        return None
    lists = re.findall(r"affinity list: (\S+)", path.read_text())
    ranks = [l for l in lists if l != "0-255"]
    if not ranks:
        return None
    return all(len(l.split(",")) <= 2 and "-" not in l for l in ranks) \
        and len(set(ranks)) == len(ranks)


def job_times(study_dir):
    allrun, job = {}, {}
    for out in (study_dir / "jobs").glob("*_*.out"):
        text = out.read_text(errors="replace")
        combo = re.search(r" combo (\S+)", text)
        done = re.search(r"Allrun exit \d+ after (\d+) s", text)
        if combo and done:
            allrun[combo.group(1)] = float(done.group(1))
            job[combo.group(1)] = out.stem.split("-")[-1]
    elapsed = {}
    sacct = study_dir / "sacct.txt"
    if sacct.exists():
        for line in sacct.read_text().splitlines():
            parts = line.split("|")
            if len(parts) >= 4 and parts[2] == "COMPLETED":
                elapsed[parts[0]] = seconds(parts[3])
    return allrun, {c: elapsed.get(j) for c, j in job.items()}


def seconds(text):
    days, _, rest = text.rpartition("-")
    h, m, s = (int(v) for v in rest.split(":"))
    return (int(days) if days else 0) * 86400 + h * 3600 + m * 60 + s


def load_rows():
    rows = []
    for study_dir in sorted(TIMING.glob("timing_*")):
        manifest = study_dir / "study_manifest.json"
        if not manifest.exists():
            continue
        allrun, job = job_times(study_dir)
        for combo in json.loads(manifest.read_text())["combos"]:
            case = study_dir / "_rundata" / combo["run_dir"]
            combo["_case_dir"] = str(case)
            rec = solver_record(case)
            if rec is None:
                print(f"  [skip] {study_dir.name}/{combo['run_dir']}: no solver log")
                continue
            folder, role = folder_of(study_dir.name, combo["model"])
            mesh = combo.get("mesh") or {}
            scales = pp.combo_scales(combo)
            cells = mesh.get("nCells")
            row = {"folder": folder, "study": study_dir.name, "role": role,
                   "run_dir": combo["run_dir"], "approach": combo["approach"],
                   "nPrisms": combo["params"].get("nPrisms"),
                   "h_plus": pp.h_plus(combo, scales),
                   "dx_norm": pp.dx_norm(combo, scales),
                   "nCells": cells, "mesh_s": mesh.get("generate_seconds"),
                   "allrun_s": allrun.get(combo["run_dir"]), "job_s": job.get(combo["run_dir"]),
                   "pinned": pinned(case), **rec}
            if rec.get("s_per_iter") and cells:
                row["core_us_per_cell_iter"] = 1e6 * rec["ranks"] * rec["s_per_iter"] / cells
            if rec.get("solver_s") is not None:
                row["solver_core_h"] = rec["ranks"] * rec["solver_s"] / 3600.0
            for u in UTILITIES:
                row[f"{u}_s"] = utility_seconds(case, u)
            rows.append(row)
    return rows


def fmt(value, digits=3):
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k), 6) for k in FIELDS})


def write_text(rows, path):
    lines = []
    for folder in FOLDERS:
        sel = [r for r in rows if r["folder"] == folder]
        if not sel:
            continue
        prism = [r for r in sel if r["role"] == "prism"]
        dxs = sorted({round(r["dx_norm"], 4) for r in prism if r["dx_norm"]}, reverse=True)
        lines.append(f"== {folder}: solver core-h (4 ranks), rows approach x h+, "
                     f"columns L_nom/delta_mean")
        lines.append(f"{'':10s} {'h+':>6s} " + " ".join(f"{d:>9.2g}" for d in dxs) + f" {'sum':>9s}")
        for approach in pp.APPROACHES:
            for n in sorted({r["nPrisms"] for r in prism if r["approach"] == approach}):
                cells = {round(r["dx_norm"], 4): r for r in prism
                         if r["approach"] == approach and r["nPrisms"] == n}
                if not cells:
                    continue
                hp = next(iter(cells.values()))["h_plus"]
                vals = [cells.get(d, {}).get("solver_core_h") for d in dxs]
                lines.append(f"{approach:10s} {hp:6.0f} "
                             + " ".join(f"{v:9.3f}" if v is not None else f"{'-':>9s}" for v in vals)
                             + f" {sum(v for v in vals if v):9.3f}")
        lines.append("mesh time [s] per approach (sum over h+), columns L_nom/delta_mean:")
        for approach in pp.APPROACHES:
            vals = [sum(r["mesh_s"] or 0 for r in prism if r["approach"] == approach
                        and r["dx_norm"] and round(r["dx_norm"], 4) == d) for d in dxs]
            if any(vals):
                lines.append(f"{approach:10s} {'':6s} " + " ".join(f"{v:9.0f}" for v in vals))
        base = {r["role"]: r for r in sel if r["role"].startswith("baseline")}
        parts = [f"{role[9:]} {r['solver_core_h']:.3f} core-h ({r['iters']} it, "
                 f"{'End' if r['ended'] else 'no End'})" for role, r in sorted(base.items(), reverse=True)
                 if r.get("solver_core_h") is not None]
        total = sum(r.get("solver_core_h") or 0 for r in base.values())
        lines.append(f"blockMesh baseline: {' + '.join(parts)} = {total:.3f} core-h")
        lines.append(f"total solver core-h: prism {sum(r.get('solver_core_h') or 0 for r in prism):.2f}"
                     f", with baseline {sum(r.get('solver_core_h') or 0 for r in sel):.2f}; "
                     f"billed node-h (job wall) {sum(r.get('job_s') or 0 for r in sel) / 3600:.2f}")
        not_pinned = [r["run_dir"] for r in sel if r["pinned"] is False]
        if not_pinned:
            lines.append(f"[warn] not pinned: {not_pinned}")
        nodes = sorted({r["node"] for r in sel if r["node"]})
        lines.append(f"nodes: {len(nodes)} ({', '.join(nodes[:6])}{', ...' if len(nodes) > 6 else ''})")
        lines.append("")
    path.write_text("\n".join(lines))


def folder_model(folder):
    return FOLDER_MODEL[folder.split("_")[0]]


def folder_schemes(folder):
    return "upwind" if folder.endswith("_upwind") else "base"


def folder_style(folder):
    return dict(color=cmp.MODEL_COLORS[folder_model(folder)],
                linestyle=cmp.SCHEME_LS[folder_schemes(folder)])


def folder_label(folder):
    return f"{pp.MODEL_LABELS[folder_model(folder)]}, {folder_schemes(folder)}"


def run_cost(row):
    return (row.get("solver_core_h") or 0.0) + (row.get("mesh_s") or 0.0) / 3600.0


def run_time(row):
    return ((row.get("solver_s") or 0.0) + (row.get("mesh_s") or 0.0)) / 60.0


def attach_errors(rows):
    errors = {}
    with open(COMPARISON_CSV, newline="") as fh:
        for rec in csv.DictReader(fh):
            if rec["err_l1"]:
                errors[(rec["study"], rec["name"])] = float(rec["err_l1"])
    for row in rows:
        row["cost_h"] = run_cost(row)
        row["runtime_min"] = run_time(row)
        row["err_l1"] = errors.get((row["folder"], row["run_dir"]))
    missing = [r["run_dir"] for r in rows if r["role"] == "prism" and r["err_l1"] is None]
    if missing:
        print(f"  [warn] no err_l1 in comparison.csv for {len(missing)} runs: {missing[:3]}")


def baseline_costs(rows, key="cost_h"):
    costs = {}
    for row in rows:
        if row["role"].startswith("baseline"):
            costs[row["folder"]] = costs.get(row["folder"], 0.0) + row[key]
    return costs


def core_order(rows):
    return sorted({round(r["dx_norm"], 6) for r in rows if r["role"] == "prism"}, reverse=True)


def stack_labels(rows):
    return {r["nPrisms"]: pp.fmt_hplus(r["h_plus"]) for r in rows if r["role"] == "prism"}


def log_axes(ax, sparse_x=False):
    ax.set_xscale("log")
    ax.set_yscale("log")
    pp.plain_log_axis(ax, "x")
    pp.plain_log_axis(ax, "y")
    if sparse_x:
        ax.xaxis.set_major_locator(FixedLocator(COST_TICKS))


def fig_cost_grid(rows, path):
    prism = [r for r in rows if r["role"] == "prism" and r["err_l1"]]
    cores, hps, bases = core_order(rows), stack_labels(rows), baseline_costs(rows)
    ns = sorted(hps)
    fig, axes = pp.grid_figure(len(ns), len(pp.APPROACHES), sharex=True, sharey=True)
    for i, n in enumerate(ns):
        for j, approach in enumerate(pp.APPROACHES):
            ax = axes[i][j]
            for folder in FOLDERS:
                style = folder_style(folder)
                sel = sorted((r for r in prism if r["folder"] == folder
                              and r["approach"] == approach and r["nPrisms"] == n),
                             key=lambda r: -r["dx_norm"])
                if not sel:
                    continue
                ax.plot([r["cost_h"] for r in sel], [r["err_l1"] for r in sel],
                        linewidth=1.0, **style)
                for r in sel:
                    ax.plot(r["cost_h"], r["err_l1"], linestyle="none", color=style["color"],
                            marker=pp.CORE_MARKERS[cores.index(round(r["dx_norm"], 6))],
                            markersize=3.5, markeredgecolor="white", markeredgewidth=0.4)
                if folder in bases:
                    ax.axvline(bases[folder], linewidth=0.7, alpha=0.6, **style)
            log_axes(ax, sparse_x=True)
    pp.label_grid(axes, [hps[n] for n in ns], pp.APPROACHES, L1_LABEL, COST_LABEL)
    handles = [pp.line_handle(folder_label(f), **folder_style(f)) for f in FOLDERS]
    handles += [pp.line_handle(pp.fmt_dx(d), pp.MUTED, "none", pp.CORE_MARKERS[k])
                for k, d in enumerate(cores)]
    rect = pp.legend_below(fig, handles, ncol=4)
    fig.tight_layout(rect=rect, h_pad=0.6, w_pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def fig_cost_all(rows, path, key="cost_h", xlabel=COST_LABEL):
    prism = [r for r in rows if r["role"] == "prism" and r["err_l1"]]
    hps, bases = stack_labels(rows), baseline_costs(rows, key)
    ns = sorted(hps)
    models = ["SALSA", "SSGLRR"]
    fig, axes = pp.grid_figure(len(models), 2, row_height=2.4, sharex=True, sharey=True)
    for i, model in enumerate(models):
        for j, folder in enumerate([model, f"{model}_upwind"]):
            ax = axes[i][j]
            for approach in pp.APPROACHES:
                for k, n in enumerate(ns):
                    sel = sorted((r for r in prism if r["folder"] == folder
                                  and r["approach"] == approach and r["nPrisms"] == n),
                                 key=lambda r: -r["dx_norm"])
                    if sel:
                        ax.plot([r[key] for r in sel], [r["err_l1"] for r in sel],
                                color=pp.APPROACH_COLORS[approach], linewidth=0.8,
                                marker=pp.STACK_MARKERS[k], markersize=3.5,
                                markeredgecolor="white", markeredgewidth=0.4)
            if folder in bases:
                ax.axvline(bases[folder], **pp.BASELINE_STYLE)
            log_axes(ax)
    pp.label_grid(axes, [pp.MODEL_LABELS[FOLDER_MODEL[m]] for m in models],
                  ["base schemes", "first-order upwind schemes"], L1_LABEL, xlabel)
    handles = [pp.line_handle(a, pp.APPROACH_COLORS[a]) for a in pp.APPROACHES]
    handles += [pp.line_handle(hps[n], pp.MUTED, "none", pp.STACK_MARKERS[k])
                for k, n in enumerate(ns)]
    handles.append(pp.line_handle("blockMesh reference", pp.INK, "--"))
    rect = pp.legend_below(fig, handles, ncol=4)
    fig.tight_layout(rect=rect, h_pad=0.6, w_pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def write_captions(path):
    lines = ["# Figure captions", "",
             "Caption drafts for the timing figures (written by timing.py; the figures carry"
             " no titles).", ""]
    for name, text in CAPTIONS.items():
        lines += [f"## {name}", "", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    rows = load_rows()
    if not rows:
        print(f"[error] no timing data in {TIMING} - run zih_cluster/fetch_timing.sh")
        return 1
    out = TIMING / "tables"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out / "run_times.csv")
    write_text(rows, out / "run_times.txt")
    print(f"{len(rows)} runs -> {out}/run_times.{{csv,txt}}")
    attach_errors(rows)
    fig_cost_grid(rows, TIMING / "cost_vs_cf_l1_error.png")
    fig_cost_all(rows, TIMING / "cost_vs_cf_l1_error_all.png")
    fig_cost_all(rows, TIMING / "runtime_vs_cf_l1_error_all.png", "runtime_min", RUNTIME_LABEL)
    write_captions(TIMING / "CAPTIONS.md")
    print(f"figures -> {TIMING}/{{cost,runtime}}_vs_cf_l1_error*.png, CAPTIONS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
