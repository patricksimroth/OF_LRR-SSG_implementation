"""Compare the core_refinement grid across models and scheme sets.

    .venv/bin/python3 turbulentFlatPlate/studies/core_refinement/compare.py \
        [results/turbulentFlatPlate/<study> ...]

Default studies: SALSA (study core_refinement, base schemes), SSGLRR,
SALSA_upwind, SSGLRR_upwind; missing ones are
skipped.  Writes results/turbulentFlatPlate/comparison/:

    comparison.csv                   one row per run and study
    cd_error_vs_core.png             Cd error vs the study's OWN blockMesh over core size
    cd_error_vs_core_total.png       Cd error vs the model's base-scheme blockMesh
    cf_rms_error_vs_core.png         plate-RMS Cf error vs the own blockMesh
    blockMesh_cf_vs_tmr.png          blockMesh Cf(x) per study against TMR SST

Report style (_pp_common): no titles; caption drafts in CAPTIONS (written to
comparison/CAPTIONS.md by synthesis.py).
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as an  # noqa: E402

RESULTS = an.ROOT / "results" / "turbulentFlatPlate"
DEFAULT = ["SALSA", "SSGLRR",
           "SALSA_upwind", "SSGLRR_upwind"]
# Model colours, distinct from the approach and core-size colours.
MODEL_COLORS = {"SpalartAllmarasSALSA": "#1f4e79", "SSGLRROmega2019": "#b5651d"}
SCHEME_LS = {"base": "-", "upwind": "--"}
NS = sorted(an.NPRISMS_MARKERS)
LEGACY = ("fig_*.png",)
NORM = ("$h^+$ uses the analytical $u_\\tau$ of the composite boundary-layer profile at"
        " x = 2 m; x and the nominal wall core cell size $L_\\mathrm{nom}$ are divided by"
        " $\\bar\\delta$, the analytical $\\delta_{99}$ averaged over the plate.")
CAPTIONS = {
    "cd_error_vs_core.png":
        "Relative deviation $\\varepsilon_{C_D}$ of the drag coefficient of the plate from the"
        " blockMesh run of the same study over the wall core cell size. Rows: prism-stack"
        " height $h^+$; columns: mesh approach; colours: model; solid: base schemes, dashed:"
        " first-order upwind convection schemes. Symmetric-log axis, linear within"
        " $\\pm$0.1 %. " + NORM,
    "cd_error_vs_core_total.png":
        "As cd_error_vs_core.png, but relative to the base-scheme blockMesh run of the same"
        " model, so the upwind runs include the scheme error. " + NORM,
    "cf_rms_error_vs_core.png":
        "RMS of $\\varepsilon_{C_f}$ over 0.1 m $\\le x \\le$ 1.95 m relative to the"
        " blockMesh run of the same study, over the wall core cell size. Layout and line"
        " styles as cd_error_vs_core.png. " + NORM,
    "blockMesh_cf_vs_tmr.png":
        "Skin-friction coefficient of the blockMesh runs (a) and their deviation from the TMR"
        " SST solution (CFL3D/FUN3D) (b) along the plate. Colours: model; solid: base schemes,"
        " dashed: first-order upwind convection schemes; dotted: TMR SST. " + NORM,
}


def load_study(results):
    runs = an.load_runs(results)
    if not runs:
        return None
    base = next((r for r in runs if r["approach"] == "blockMesh"), None)
    if base is None:
        print(f"  [skip] {results.name}: no completed blockMesh baseline")
        return None
    reference = an.cfv.load_reference(str(an.ROOT / "turbulentFlatPlate" / "_validation_data"))
    tmr = float(np.interp(an.STATION_X, reference[:, 0], reference[:, 1]))
    an.metrics(runs, base, tmr)
    model = runs[0]["name"].split("__", 1)[0]
    return {"name": results.name, "model": model, "schemes": an.pp.study_schemes(results),
            "runs": runs, "base": base}


def label(study):
    return f"{an.MODEL_LABELS.get(study['model'], study['model'])}, {study['schemes']}"


def style(study):
    return dict(color=MODEL_COLORS.get(study["model"], an.INK),
                linestyle=SCHEME_LS[study["schemes"]])


def total_error(study, studies):
    ref = next((s for s in studies if s["model"] == study["model"] and s["schemes"] == "base"),
               None)
    if ref is None or not ref["base"]["cd"]:
        return {}
    cd_ref = ref["base"]["cd"]
    return {r["name"]: 100.0 * (r["cd"] - cd_ref) / cd_ref for r in study["runs"] if r["cd"]}


def symlog_axis(ax):
    ax.set_yscale("symlog", linthresh=0.1)
    an.pp.plain_ticks(ax)
    ax.axhline(0, color=an.MUTED, linewidth=0.7)


def grid_figure(studies, value, ylabel, path, log=False):
    fig, axes = an.pp.grid_figure(len(NS), an.N_APP, sharex=True)
    every = [r for study in studies for r in study["runs"]]
    for i, n in enumerate(NS):
        for j, approach in enumerate(an.APPROACHES):
            ax = axes[i][j]
            for study in studies:
                pts = [(r["dx_norm"], value(study, r)) for r in an.cell_runs(study["runs"], approach, n)]
                pts = [p for p in pts if p[1] is not None]
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=an.NPRISMS_MARKERS[n],
                            markersize=3.5, markeredgecolor="white", markeredgewidth=0.4,
                            linewidth=1.0, **style(study))
            an.dx_axis(ax, every, rotate=True)
            if log:
                ax.set_yscale("log")
                an.pp.plain_log_axis(ax)
            else:
                symlog_axis(ax)
    an.pp.label_grid(axes, [an.hp(n) for n in NS], an.APPROACHES, ylabel, an.pp.DX_SYMBOL)
    handles = [an.pp.line_handle(label(s), **style(s)) for s in studies]
    rect = an.pp.legend_below(fig, handles, ncol=len(handles) if len(handles) <= 4 else None)
    fig.tight_layout(rect=rect, h_pad=0.6, w_pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def fig_baselines(studies, path):
    reference = an.cfv.load_reference(str(an.ROOT / "turbulentFlatPlate" / "_validation_data"))
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(an.pp.REPORT_WIDTH_IN, 2.6))
    dl = an.DELTA_MEAN
    ax.plot(reference[:, 0] / dl, reference[:, 1], color=an.INK, linewidth=1.0, linestyle=":")
    grid = np.linspace(*an.PLATE, 400)
    ref_i = np.interp(grid, reference[:, 0], reference[:, 1])
    for study in studies:
        b = study["base"]
        ax.plot(b["x"] / dl, b["cf"], linewidth=1.0, **style(study))
        axd.plot(grid / dl, 100 * (np.interp(grid, b["x"], b["cf"]) - ref_i) / ref_i,
                 linewidth=1.0, **style(study))
    ax.set_xlim(0, 2 / dl)
    ax.set_ylim(0.002, 0.006)
    ax.set_ylabel(r"$C_f$")
    axd.axhline(0, color=an.MUTED, linewidth=0.7)
    axd.set_xlim(an.PLATE[0] / dl, an.PLATE[1] / dl)
    axd.set_ylabel(r"$(C_f - C_{f,\mathrm{TMR}})/C_{f,\mathrm{TMR}}$ [%]")
    for a, tag in ((ax, "(a)"), (axd, "(b)")):
        a.set_xlabel(an.XI_LABEL)
        a.set_title(tag, loc="left")
    handles = [an.pp.line_handle(label(s), **style(s)) for s in studies]
    handles.append(an.pp.line_handle("TMR SST", an.INK, ":"))
    rect = an.pp.legend_below(fig, handles, ncol=3)
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def write_csv(studies, path):
    keys = ["study", "model", "schemes", "name", "approach", "nPrisms", "h_plus", "size",
            "dx_norm", "l_core", "l_core_norm", "phi", "l_star", "nCells",
            "iters", "converged", "exec_s", "p_res", "cd", "cd_err", "cd_err_total",
            "cf_station", "err_station", "err_rms", "err_l1", "err_tmr"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for study in studies:
            totals = total_error(study, studies)
            for r in study["runs"]:
                row = {**r, "study": study["name"], "model": study["model"],
                       "schemes": study["schemes"], "cd_err_total": totals.get(r["name"])}
                writer.writerow([row.get(k) for k in keys])


def main():
    paths = [Path(p).resolve() for p in sys.argv[1:]] or [RESULTS / s for s in DEFAULT]
    studies = []
    for path in paths:
        if not (path / "study_manifest.json").is_file():
            print(f"  [skip] {path.name}: no study_manifest.json")
            continue
        study = load_study(path)
        if study:
            studies.append(study)
    if not studies:
        print("[error] no studies to compare")
        return 1
    out = RESULTS / "comparison"
    out.mkdir(exist_ok=True)
    for pattern in LEGACY:
        for stale in out.glob(pattern):
            stale.unlink()
    an.CORE_ORDER[:] = an.core_sizes([r for s in studies for r in s["runs"]])
    write_csv(studies, out / "comparison.csv")
    totals = {s["name"]: total_error(s, studies) for s in studies}
    grid_figure(studies, lambda s, r: r["cd_err"], an.pp.error_label("cd"),
                out / "cd_error_vs_core.png")
    grid_figure(studies, lambda s, r: totals[s["name"]].get(r["name"]),
                an.pp.error_label("cd"), out / "cd_error_vs_core_total.png")
    grid_figure(studies, lambda s, r: r["err_rms"], an.pp.error_label("cf", "RMS"),
                out / "cf_rms_error_vs_core.png", log=True)
    fig_baselines(studies, out / "blockMesh_cf_vs_tmr.png")
    print(f"wrote {out}  ({', '.join(s['name'] for s in studies)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
