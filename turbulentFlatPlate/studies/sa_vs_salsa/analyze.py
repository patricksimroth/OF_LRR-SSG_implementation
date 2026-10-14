"""SALSA vs stock SA on the base blockMesh, Ux-gated stop criterion.

    .venv/bin/python3 turbulentFlatPlate/studies/sa_vs_salsa/analyze.py \
        [results/turbulentFlatPlate/sa_vs_salsa]

Writes <results>/summary/{summary.csv,fig_delta99_x.png}: delta99(x) of both runs
against the composite-profile reference (the only reference drawn), and per run
the stop iteration, Ux residual at the stop, Cd and the station metrics.
"""
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "delta99_blockMesh"))
import analyze as d99  # noqa: E402

th, pp = d99.th, d99.pp


def stop_info(case):
    log = (case / "log.rhoSimpleFoam").read_text(errors="replace")
    cd = [float(v) for v in re.findall(r"^\s+Cd:\s+([-\d.eE+]+)", log, re.M)]
    ux = [float(v) for v in re.findall(r"Solving for Ux, Initial residual = ([-\d.eE+]+)", log)]
    return len(cd), cd[-1] / 2.0, max(ux[-100:])


def main():
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "results" / "turbulentFlatPlate" / "sa_vs_salsa")
    results = results.resolve()
    out = results / "summary"
    out.mkdir(exist_ok=True)
    manifest = json.loads((results / "study_manifest.json").read_text())
    runs = []
    for combo in manifest["combos"]:
        case = results / "_rundata" / combo["run_dir"]
        if not pp.run_completed(case):
            print(f"  [skip] {combo['run_dir']}: not completed")
            continue
        run = d99.load_case(case, combo["run_dir"], combo["model"], (248, 193))
        run["iters"], run["cd"], run["ux"] = stop_info(case)
        runs.append(run)

    rows = []
    for r in runs:
        for x0, s in r["stations"].items():
            rows.append({"model": d99.MODEL_LABEL[r["model"]], "x": s["x"], "iters": r["iters"],
                         "Ux_res_max_last100": r["ux"], "Cd_Aref0.2": r["cd"],
                         "delta99_mm": 1e3 * s["delta99"],
                         "delta99_composite_mm": 1e3 * th.delta99_composite(s["x"], d99.RE_M),
                         "theta_mm": 1e3 * s["theta"], "H": s["H"], "Cf": s["Cf"], "Pi": s["Pi"]})
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        print("  ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in row.items()))

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    x = np.linspace(0.01, 2.0, 300)
    ax.plot(x, 1e3 * th.delta99_composite(x, d99.RE_M), color=d99.INK, linewidth=2.2,
            label="composite profile + momentum integral")
    for r in runs:
        ax.plot(r["delta_x"][:, 0], 1e3 * r["delta_x"][:, 1], color=d99.MODEL_COLORS[r["model"]],
                linewidth=1.8, label=f"{d99.MODEL_LABEL[r['model']]} (blockMesh 248×193)")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, None)
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$\delta_{99}$ [mm]")
    ax.set_title(r"$\delta_{99}(x)$: SALSA vs SA against the composite-profile reference")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_delta99_x.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
