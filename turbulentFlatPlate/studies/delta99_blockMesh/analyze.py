"""delta99 investigation: blockMesh resolution x model vs TMR SA and the analytical references.

    .venv/bin/python3 turbulentFlatPlate/studies/delta99_blockMesh/analyze.py \
        [results/turbulentFlatPlate/delta99_blockMesh]

Writes <results>/summary/:
    summary.csv               per run and TMR station: delta99 (cell centres), theta,
                              deltastar, H, Re_theta, Cf, fitted wake strength Pi
    fig_delta99_x.png         delta99(x) of every run vs TMR SA and the references
    fig_outer_profile.png     u/U_e over y at x = 0.97 m: edge shape
    fig_delta99_ratio.png     delta99 / delta99_TMR-SA at both TMR stations per mesh
    fig_delta99_by_nx.png     delta99(x) per model x n_y, colour = n_x
    fig_cf_by_nx.png          Cf(x) per model x n_y, colour = n_x
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "turbulentFlatPlate" / "_postProcessing"))
import _pp_common as pp  # noqa: E402
import _bl_theory as th  # noqa: E402

STATIONS = (0.97008, 1.90334)
U_INF = 69.4
NU = 1.38801e-5
RHO = 1.274486
RE_M = U_INF / NU
CONTINUED = "_extra/continued_n_y386"
TOGGLES = "_extra/salsa_toggles"
TMR_FILE = ROOT / "turbulentFlatPlate" / "_validation_data" / "tmr_u_sa.dat"
TMR_CF_FILE = ROOT / "turbulentFlatPlate" / "_validation_data" / "tmr_cf_plate.dat"
TMR_CF = {0.97008: 0.0027055, 1.90334: 0.0024652}
MODEL_COLORS = {"SpalartAllmarasSALSA": "#2a78d6", "SpalartAllmaras": "#eb6834"}
MODEL_LABEL = {"SpalartAllmarasSALSA": "SALSA", "SpalartAllmaras": "SA"}
MESH_STYLE = {(248, 193): ("-", "o"), (496, 193): ("--", "s"), (992, 193): (":", "D"),
              (248, 386): ("-.", "^"), (992, 386): ((0, (6, 2, 1, 2)), "v")}
NX_COLORS = {248: "#9ec5f0", 496: "#2a6fc2", 992: "#0d3a75"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.8, "font.size": 10, "legend.frameon": False,
})


def read_field(path, ncomp=3):
    raw = Path(path).read_bytes()
    m = re.search(rb"internalField\s+nonuniform\s+List<\w+>\s*\n?(\d+)\s*\n?\(", raw)
    n, start = int(m.group(1)), m.end()
    if b"binary" in raw[:800]:
        return np.frombuffer(raw[start:start + 8 * ncomp * n], dtype="<f8").reshape(n, ncomp)
    body = raw[start:].split(b")\n;")[0].decode()
    return np.array(re.findall(r"[-+\d.eE]+", body), float).reshape(n, ncomp)


def delta99_of(y, u, ue):
    j = int(np.argmax(u >= 0.99 * ue))
    return y[j - 1] + (0.99 * ue - u[j - 1]) * (y[j] - y[j - 1]) / (u[j] - u[j - 1])


def integrals(y, u, ue, y_max=0.08):
    yy = np.concatenate([[0.0], y[y < y_max]])
    uu = np.concatenate([[0.0], u[y < y_max]]) / ue
    return np.trapezoid(uu * (1 - uu), yy), np.trapezoid(1 - uu, yy)


def fit_pi(y, u, ue, cf):
    """Fit delta, Pi of the composite profile to y+ > 50, u_tau from the solution's Cf.

    Grid search, then two zoomed passes (no scipy in the venv).
    """
    u_tau = ue * np.sqrt(cf / 2.0)
    m = (y * u_tau / NU > 50) & (u < 0.999 * ue)
    yp, target = y[m] * u_tau / NU, u[m] / u_tau
    inner = th.spalding_uplus(yp)

    def cost(delta, pi):
        eta = np.minimum(y[m][None, None, :] / delta[:, None, None], 1.0)
        wake = (pi[None, :, None] * (1 - np.cos(np.pi * eta)) + eta ** 2 - eta ** 3) / th.KAPPA
        cap = (th.spalding_uplus(delta * u_tau / NU)[:, None, None]
               + 2 * pi[None, :, None] / th.KAPPA)
        up = np.where(y[m][None, None, :] >= delta[:, None, None], cap, inner + wake)
        return ((up - target) ** 2).sum(axis=2)

    d_lo, d_hi, p_lo, p_hi = 0.004, 0.06, 0.0, 1.5
    for _ in range(3):
        deltas, pis = np.linspace(d_lo, d_hi, 121), np.linspace(p_lo, p_hi, 121)
        c = cost(deltas, pis)
        i, j = np.unravel_index(np.argmin(c), c.shape)
        dd, dp = (d_hi - d_lo) / 20, (p_hi - p_lo) / 20
        d_lo, d_hi = max(deltas[i] - dd, 1e-4), deltas[i] + dd
        p_lo, p_hi = max(pis[j] - dp, 0.0), pis[j] + dp
    return deltas[i], pis[j]


def tmr_profiles():
    zones = {}
    for chunk in TMR_FILE.read_text().split("zone")[1:]:
        lines = chunk.splitlines()
        x = float(re.findall(r"[\d.]+", lines[0])[0])
        arr = np.array([[float(v) for v in l.split()] for l in lines[1:] if l.strip()])
        zones[x] = (arr[:, 1], arr[:, 0] * U_INF)
    return zones


def wall_cf_curve(case):
    tdir = pp.latest_time_directory(case / "postProcessing" / "yPlus_reference_2")
    f = next(Path(tdir).glob("wallShearStress*.raw"))
    a = pp.read_numeric_rows(f)
    o = np.argsort(a[:, 0])
    return a[o, 0], 2 * np.linalg.norm(a[o, 3:5], axis=1) / (RHO * U_INF ** 2)


def wall_cf(case, x0):
    return float(np.interp(x0, *wall_cf_curve(case)))


def station_metrics(y, u, cf, label, x):
    ue = float(np.interp(0.05 if x < 1.5 else 0.07, y, u))
    d99 = delta99_of(y, u, ue)
    theta, dstar = integrals(y, u, ue)
    delta, pi = fit_pi(y, u, ue, cf)
    return {"run": label, "x": x, "delta99": d99, "theta": theta, "dstar": dstar,
            "H": dstar / theta, "Re_theta": theta * ue / NU, "d99_theta": d99 / theta,
            "Cf": cf, "Pi": pi, "delta_fit": delta, "Ue": ue}


def load_case(case, name, model, mesh):
    tdir = Path(pp.latest_time_directory(case))
    U, C = read_field(tdir / "U"), read_field(tdir / "C")
    run = {"name": name, "model": model, "mesh": mesh, "stations": {}, "profiles": {},
           "case": case}
    xs = np.unique(np.round(C[:, 0], 9))
    for x0 in STATIONS:
        xc = xs[np.argmin(abs(xs - x0))]
        m = abs(C[:, 0] - xc) < 1e-7
        o = np.argsort(C[m, 1])
        y, u = C[m, 1][o], U[m, 0][o]
        run["profiles"][x0] = (y, u)
        run["stations"][x0] = station_metrics(y, u, wall_cf(case, xc), name, xc)
    # delta99(x) from the cell-centre columns (Ue = column max, as bl_thickness)
    rows = []
    for xc in xs[(xs > 0.01) & (xs < 2.0)][::max(1, len(xs) // 300)]:
        m = (abs(C[:, 0] - xc) < 1e-7) & (C[:, 1] < 0.08)
        o = np.argsort(C[m, 1])
        y, u = C[m, 1][o], U[m, 0][o]
        rows.append((xc, delta99_of(y, u, u.max())))
    run["delta_x"] = np.array(rows)
    run["cf_x"] = wall_cf_curve(case)
    return run


def load_runs(results):
    """Study runs; a prematurely stopped run is replaced by its continuation."""
    manifest = json.loads((results / "study_manifest.json").read_text())
    runs = []
    for combo in manifest["combos"]:
        case = results / "_rundata" / combo["run_dir"]
        continued = results / CONTINUED / combo["run_dir"]
        if continued.is_dir():
            print(f"  [continued] {combo['run_dir']}: using {CONTINUED}")
            case = continued
        if not pp.run_completed(case):
            print(f"  [skip] {combo['run_dir']}: not completed")
            continue
        p = combo["params"]
        runs.append(load_case(case, combo["run_dir"], combo["model"], (p["n_x"], p["n_y"])))
    return runs


def load_toggles(results):
    out = []
    for case in sorted((results / TOGGLES).glob("SALSA_*")):
        if pp.run_completed(case):
            out.append(load_case(case, case.name, "SpalartAllmarasSALSA", (248, 193)))
    return out


def mesh_label(mesh):
    return f"n_x {mesh[0]}, n_y {mesh[1]}"


def fig_delta99_x(runs, tmr, out):
    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = np.linspace(0.01, 2.0, 300)
    sol = th.solve_plate()
    comp = np.interp(x * RE_M, sol["re_x"], sol["re_d99"]) / RE_M
    ax.plot(x, 1e3 * comp, color=INK, linestyle="-", linewidth=2.2,
            label="composite profile + momentum integral")
    for run in runs:
        if run["delta_x"] is None:
            continue
        ls, _ = MESH_STYLE.get(run["mesh"], ("-", "o"))
        ax.plot(run["delta_x"][:, 0], 1e3 * run["delta_x"][:, 1], linestyle=ls,
                color=MODEL_COLORS[run["model"]], linewidth=1.3, alpha=0.9)
    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=c, label=MODEL_LABEL[m]) for m, c in MODEL_COLORS.items()]
    present = {r["mesh"] for r in runs}
    handles += [plt.Line2D([], [], color=MUTED, linestyle=s[0], label=mesh_label(k))
                for k, s in MESH_STYLE.items() if k in present]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$\delta_{99}$ [mm]")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, None)
    ax.set_title(r"$\delta_{99}(x)$: blockMesh runs vs the composite-profile reference")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def tmr_cf_sa():
    zones = pp.parse_tecplot_zones(str(TMR_CF_FILE))
    x = np.unique(np.concatenate([c[:, 0] for c in zones.values()]))
    return x, np.mean([np.interp(x, c[:, 0], c[:, 1]) for c in zones.values()], axis=0)


def fig_x_by_nx(runs, results, series, ylabel, title, ylim, refs, out):
    manifest = json.loads((results / "study_manifest.json").read_text())
    planned = {(c["model"], c["params"]["n_y"], c["params"]["n_x"]) for c in manifest["combos"]}
    n_ys = sorted({k[1] for k in planned})
    fig, axes = plt.subplots(len(MODEL_COLORS), len(n_ys), figsize=(6.5 * len(n_ys), 8),
                             sharex=True, sharey=True, squeeze=False)
    for i, model in enumerate(MODEL_COLORS):
        for j, n_y in enumerate(n_ys):
            ax = axes[i, j]
            for ref in refs:
                ref(ax)
            for n_x in sorted(NX_COLORS):
                r = next((r for r in runs if r["model"] == model and r["mesh"] == (n_x, n_y)),
                         None)
                if r is not None:
                    ax.plot(*series(r), color=NX_COLORS[n_x], linewidth=1.5,
                            label=f"n_x {n_x}")
            missing = sorted(k[2] for k in planned if k[:2] == (model, n_y)
                             and not any(r["model"] == model and r["mesh"] == (k[2], n_y)
                                         for r in runs))
            if missing:
                ax.text(0.98, 0.03, "diverged (leading edge): "
                        + ", ".join(f"n_x {n}" for n in missing),
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                        color=MUTED)
            ax.set_xlim(0, 2)
            ax.set_ylim(*ylim)
            if i == 0:
                ax.set_title(f"n_y = {n_y}")
            if j == 0:
                ax.set_ylabel(f"{MODEL_LABEL[model]}\n{ylabel}")
            if i == len(MODEL_COLORS) - 1:
                ax.set_xlabel("x [m]")
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_ratio(runs, tmr, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    meshes = list(MESH_STYLE)
    for ax, x0 in zip(axes, STATIONS):
        ref = tmr[x0]["delta99"]
        width = 0.38
        for k, model in enumerate(MODEL_COLORS):
            vals = []
            for mesh in meshes:
                r = next((r for r in runs if r["model"] == model and r["mesh"] == mesh), None)
                vals.append(r["stations"][x0]["delta99"] / ref if r else np.nan)
            pos = np.arange(len(meshes)) + (k - 0.5) * width
            ax.bar(pos, vals, width * 0.92, color=MODEL_COLORS[model], label=MODEL_LABEL[model])
        for i, mesh in enumerate(meshes):
            if not any(r["mesh"] == mesh for r in runs):
                ax.text(i, 0.82, "diverged\n(leading edge)", ha="center", va="bottom",
                        fontsize=8, color=MUTED)
        comp = th.delta99_composite(x0, RE_M) / ref
        ax.axhline(1.0, color=INK, linewidth=1)
        ax.axhline(comp, color=INK, linestyle="--", linewidth=1)
        ax.text(len(meshes) - 0.5, comp, " composite", va="bottom", ha="right", fontsize=8)
        ax.axhline(th.delta_power_law(x0, RE_M) / ref, color=MUTED, linestyle=":", linewidth=1)
        ax.text(len(meshes) - 0.5, th.delta_power_law(x0, RE_M) / ref, " 0.37 law",
                va="bottom", ha="right", fontsize=8, color=MUTED)
        ax.set_xticks(range(len(meshes)))
        ax.set_xticklabels([f"{m[0]}×{m[1]}" for m in meshes], fontsize=8)
        ax.set_title(f"x = {x0} m")
        ax.set_ylim(0.8, 1.2)
    axes[0].set_ylabel(r"$\delta_{99}\,/\,\delta_{99,\mathrm{TMR\,SA}}$")
    axes[0].legend(loc="upper left")
    fig.suptitle("δ99 relative to TMR SA, per blockMesh resolution (n_x × n_y)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_profile(runs, tmr_prof, tmr, out):
    x0 = STATIONS[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    y_t, u_t = tmr_prof[x0]
    ue_t = tmr[x0]["Ue"]
    pick = [r for r in runs if r["mesh"] in ((248, 193), (992, 386))]
    for ax, (y_hi, u_lo) in zip(axes, [(0.03, 0.0), (0.03, 0.9)]):
        ax.plot(1e3 * y_t, u_t / ue_t, color=INK, linewidth=2.4, label="TMR SA")
        for r in pick:
            y, u = r["profiles"][x0]
            ue = r["stations"][x0]["Ue"]
            ls, _ = MESH_STYLE[r["mesh"]]
            ax.plot(1e3 * y, u / ue, color=MODEL_COLORS[r["model"]], linestyle=ls,
                    label=f"{MODEL_LABEL[r['model']]}, {mesh_label(r['mesh'])}")
        ax.axhline(0.99, color=MUTED, linestyle=":", linewidth=1)
        ax.set_xlim(0, 1e3 * y_hi)
        ax.set_ylim(u_lo, 1.005)
        ax.set_xlabel("y [mm]")
    axes[0].set_ylabel(r"$u / U_e$")
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[1].set_title("zoom: outer edge (dotted = 99 %)")
    axes[0].set_title(f"Velocity profile at x = {x0} m")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "results" / "turbulentFlatPlate" / "delta99_blockMesh")
    results = results.resolve()
    out = results / "summary"
    out.mkdir(exist_ok=True)
    tmr_prof = tmr_profiles()
    tmr = {x0: station_metrics(*tmr_prof[x0], TMR_CF[x0], "TMR SA", x0) for x0 in STATIONS}
    runs = load_runs(results)
    toggles = load_toggles(results)
    rows = list(tmr.values()) + [s for r in runs + toggles for s in r["stations"].values()]
    for x0 in STATIONS:
        sol = th.solve_plate()
        rt = float(np.interp(x0 * RE_M, sol["re_x"], sol["re_theta"]))
        rows.append({"run": "composite (Cebeci-Smith Pi)", "x": x0,
                     "delta99": th.delta99_composite(x0, RE_M), "theta": rt / RE_M,
                     "Re_theta": rt, "Cf": float(np.interp(x0 * RE_M, sol["re_x"], sol["cf"])),
                     "H": float(np.interp(x0 * RE_M, sol["re_x"], sol["H"])),
                     "Pi": float(np.interp(x0 * RE_M, sol["re_x"], sol["Pi"]))})
        rows.append({"run": "0.37 x Re_x^-0.2", "x": x0, "delta99": th.delta_power_law(x0, RE_M)})
    keys = ["run", "x", "delta99", "theta", "dstar", "H", "Re_theta", "d99_theta", "Cf",
            "Pi", "delta_fit", "Ue"]
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for row in rows:
            w.writerow([row.get(k) for k in keys])
    fig_delta99_x(runs, tmr, out / "fig_delta99_x.png")
    fig_ratio(runs, tmr, out / "fig_delta99_ratio.png")
    fig_profile(runs, tmr_prof, tmr, out / "fig_outer_profile.png")
    sol = th.solve_plate()
    xs = np.linspace(0.01, 2.0, 300)
    comp = np.interp(xs * RE_M, sol["re_x"], sol["re_d99"]) / RE_M

    def ref_delta(ax):
        ax.plot(xs, 1e3 * comp, color=INK, linestyle="--", linewidth=1.2,
                label="composite + momentum integral")
        ax.plot(list(tmr), [1e3 * v["delta99"] for v in tmr.values()], "k*", markersize=12,
                label="TMR SA (CFL3D, 545×385)")

    x_tmr, cf_tmr = tmr_cf_sa()

    def ref_cf(ax):
        ax.plot(x_tmr, cf_tmr, color=INK, linestyle="--", linewidth=1.2,
                label="TMR SA (CFL3D/FUN3D mean)")

    d_max = max(r["delta_x"][:, 1].max() for r in runs)
    fig_x_by_nx(runs, results, lambda r: (r["delta_x"][:, 0], 1e3 * r["delta_x"][:, 1]),
                "δ99 [mm]", "δ99(x) per model and n_y, colour = streamwise resolution n_x",
                (0, 1.08e3 * d_max), [ref_delta], out / "fig_delta99_by_nx.png")
    cf_max = max(r["cf_x"][1][r["cf_x"][0] >= 0.1].max() for r in runs)
    fig_x_by_nx(runs, results, lambda r: r["cf_x"], "Cf",
                "Cf(x) per model and n_y, colour = streamwise resolution n_x",
                (0, 1.25 * cf_max), [ref_cf], out / "fig_cf_by_nx.png")
    for row in rows:
        def g(k, s=1e3, f="{:.3f}"):
            return f.format(row[k] * s) if row.get(k) is not None else "-"
        print(f"{row['run']:58s} x={row['x']:.3f} d99={g('delta99')} theta={g('theta')} "
              f"H={g('H', 1)} Cf={g('Cf', 1, '{:.6f}')} Pi={g('Pi', 1)} d99/th={g('d99_theta', 1, '{:.2f}')}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
