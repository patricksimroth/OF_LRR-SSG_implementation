"""Cross-study synthesis of the four core-refinement studies (model x schemes).

    .venv/bin/python3 turbulentFlatPlate/studies/core_refinement/synthesis.py

Reads results/turbulentFlatPlate/comparison/comparison.csv (written by
compare.py) and the per-run Cf(x) curves, and writes
results/turbulentFlatPlate/comparison/synthesis/:

    cf_l1_error_map.png             L1 Cf error (no cancellation) of every run, one matrix per study
    cf_l1_error_vs_lstar.png        L1 Cf error over L* (every stack on one axis), per study
    cd_error_scheme_effect.png      upwind vs base error, per model
    cd_error_model_effect.png       SSG/LRR vs SALSA error, per scheme set
    cd_error_factor_effects.png     change in error when one factor changes, per regime
    cd_error_refinement_order.png   |Cd error| over core size (log-log) for the h+ = 548 stack
    cf_tmr_deviation.png            Cf(0.97 m) vs TMR SST, absolute accuracy
    cf_error_along_plate_finest.png Cf error along the plate at the finest core
    iterations.png                  iterations to the stop / cap
    factor_effects.csv, regime_table.csv
and comparison/CAPTIONS.md (caption drafts of these, compare.py's and timing.py's figures;
the figures follow the report style of _pp_common: no titles).

Runs are labelled by h+ (prism-stack height in wall units, analytical u_tau at the
trailing edge) and L_nom/delta_mean (nominal wall core cell size over the
plate-averaged analytical delta99), and placed on one axis by L* = L_core/delta_mean * phi
with the core cell size measured next to the stack (analyze.measure_core), read from the h_plus / dx_norm / l_star columns of
comparison.csv.
"""
import csv
import itertools
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as an  # noqa: E402
import compare as cmp  # noqa: E402
import timing as tim  # noqa: E402
import _captions as caps  # noqa: E402

RESULTS = an.ROOT / "results" / "turbulentFlatPlate"
COMP = RESULTS / "comparison"
OUT = COMP / "synthesis"
STUDIES = [
    ("SALSA", "SALSA", "base"),
    ("SALSA_upwind", "SALSA", "upwind"),
    ("SSGLRR", "SSG/LRR", "base"),
    ("SSGLRR_upwind", "SSG/LRR", "upwind"),
]
LABEL = {s: f"{m}, {k}" for s, m, k in STUDIES}
STUDY_COLORS = {s: cmp.MODEL_COLORS["SpalartAllmarasSALSA" if m == "SALSA" else "SSGLRROmega2019"]
                for s, m, _ in STUDIES}
STUDY_LS = cmp.SCHEME_LS
STUDY_MARKER = {"SALSA": "o", "SALSA_upwind": "o",
                "SSGLRR": "s", "SSGLRR_upwind": "s"}
APPROACHES = an.APPROACHES
APP_COLORS = an.APPROACH_COLORS
APP_MARKERS = {"snappy": "o", "gmshTri": "^", "gmshHex": "h", "snappy45": "s"}
N_APP = len(APPROACHES)
NS = (20, 40, 50)
N_MARKERS = {20: "o", 40: "s", 50: "D"}
SIZES = (0.005, 0.0025, 0.00125, 0.000625)
HP = {}
DXN = {}
SCALES = {}
FINEST_DXN = f"{SIZES[-1] / an.pp.case_scales()['delta_mean']:.2g}"
INK, MUTED, GRID = an.INK, an.MUTED, an.GRID
UNSETTLED = {("SSGLRR", "gmshHex", 50)}
W = an.pp.REPORT_WIDTH_IN
EPS_CD = an.pp.error_label("cd")
LEGACY = ("fig*.png",)
STYLE_NOTE = ("Colours: model; solid: base schemes, dashed: first-order upwind convection"
              " schemes.")
CAPTIONS = {
    "cf_l1_error_map.png":
        "L1 deviation $\\varepsilon_{L1} = \\int |C_f - C_{f,\\mathrm{blockMesh}}|\\,dx /"
        " \\int C_{f,\\mathrm{blockMesh}}\\,dx$ over 0.1 m $\\le x \\le$ 1.95 m of every run,"
        " one matrix per study; rows: mesh approach and prism-stack height $h^+$, columns:"
        " nominal wall core cell size $L_\\mathrm{nom}/\\bar\\delta$.",
    "cf_l1_error_vs_lstar.png":
        "L1 deviation $\\varepsilon_{L1}$ (as cf_l1_error_map.png) of every run over the"
        " normalised core cell size $L^*$, one panel per study. Colours: mesh approach;"
        " markers: prism-stack height $h^+$; lines join the core sizes of one approach and"
        " stack. " + caps.LSTAR,
    "cd_error_scheme_effect.png":
        "$\\varepsilon_{C_D}$ with first-order upwind against base convection schemes, per"
        " model; colours: mesh approach; markers: prism-stack height $h^+$; marker size grows"
        " with core refinement. Dotted: identity. Symmetric-log axes, linear within"
        " $\\pm$0.3 %.",
    "cd_error_model_effect.png":
        "$\\varepsilon_{C_D}$ of SSG/LRR against SALSA, per scheme set; markers as"
        " cd_error_scheme_effect.png. Dotted: identity.",
    "cd_error_factor_effects.png":
        "Change $|\\Delta\\varepsilon_{C_D}|$ in percentage points when one factor changes"
        " (mesh approach, halving the core cell size, scheme set, model) over all matched run"
        " pairs, per prism-stack height $h^+$; bar: median. Shaded: scatter of the stop"
        " criterion ($\\le$ 0.05 pp).",
    "cd_error_refinement_order.png":
        "$|\\varepsilon_{C_D}|$ over the wall core cell size for the $h^+$ = 548 stack, per"
        " mesh approach. " + STYLE_NOTE + " Dotted: first- and second-order slopes; shaded:"
        " scatter of the stop criterion.",
    "cf_tmr_deviation.png":
        "Deviation of $C_f$ at x = 0.97 m from the TMR SST solution over the wall core cell"
        " size, per prism-stack height $h^+$. Markers: mesh approach; horizontal lines: the"
        " blockMesh run of each study. " + STYLE_NOTE,
    "cf_error_along_plate_finest.png":
        "$\\varepsilon_{C_f}$ along the plate at the finest core ($L_\\mathrm{nom}/\\bar\\delta$ ="
        f" {FINEST_DXN}). Rows: prism-stack height $h^+$; columns: mesh approach. " + STYLE_NOTE +
        " Dotted verticals: $x_\\mathrm{crit}$, where $\\delta_{99}$ of blockMesh reaches the"
        " stack height. SSG/LRR, base schemes, gmshHex at $h^+$ = 2637 had not settled at the"
        " stop ($C_f$ drift 0.3 % per 300 iterations).",
    "iterations.png":
        "Iterations to the stop criterion (8000 = cap) for blockMesh (bars) and over the wall"
        " core cell size per mesh approach. Markers: prism-stack height $h^+$. " + STYLE_NOTE,
}


def load_rows():
    rows = []
    with open(COMP / "comparison.csv") as f:
        for r in csv.DictReader(f):
            for k in ("cd", "cd_err", "cd_err_total", "err_rms", "err_l1", "err_station", "err_tmr",
                      "cf_station", "size", "h_plus", "dx_norm", "phi", "l_star"):
                r[k] = float(r[k]) if r[k] not in ("", "None") else None
            r["nPrisms"] = int(r["nPrisms"]) if r["nPrisms"] not in ("", "None") else None
            r["iters"] = int(r["iters"])
            r["converged"] = r["converged"] == "True"
            r["nCells"] = int(r["nCells"]) if r["nCells"] not in ("", "None") else None
            rows.append(r)
            if r["nPrisms"] is not None and r["h_plus"] is not None:
                HP.setdefault(r["nPrisms"], r["h_plus"])
            if r["size"] is not None and r["dx_norm"] is not None:
                DXN.setdefault(round(r["size"], 9), r["dx_norm"])
            if r["approach"] == "blockMesh" and not SCALES:
                SCALES.update(an.pp.case_scales(RESULTS / r["study"] / "_rundata" / r["name"]))
    return rows


def hp(n):
    return an.pp.fmt_hplus(HP[n]) if n in HP else f"nPrisms = {n}"


def dxn(size):
    return f"{DXN[round(size, 9)]:.2g}"


def get(rows, study, approach, n=None, size=None):
    for r in rows:
        if (r["study"] == study and r["approach"] == approach
                and (n is None or r["nPrisms"] == n)
                and (size is None or abs(r["size"] - size) < 1e-9)):
            return r
    return None


def symlog_y(ax, lin=0.1):
    ax.set_yscale("symlog", linthresh=lin)
    an.pp.plain_ticks(ax)
    ax.axhline(0, color=MUTED, linewidth=0.7)


def size_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([dxn(s) for s in SIZES], rotation=90)
    ax.minorticks_off()
    ax.set_xlim(0.0062, 0.0005)
    ax.set_xlabel(an.pp.DX_SYMBOL)


def fig_error_map(rows, path):
    fig, axes = plt.subplots(2, 2, figsize=(W, 7.4), sharey=True, sharex=True)
    norm = LogNorm(vmin=0.01, vmax=70)
    ylabels = [f"{a}, {hp(n)}" for n in NS for a in APPROACHES]
    for ax, (study, _, _) in zip(axes.flat, STUDIES):
        mat = np.full((len(ylabels), len(SIZES)), np.nan)
        for i, (n, a) in enumerate((n, a) for n in NS for a in APPROACHES):
            for j, s in enumerate(SIZES):
                r = get(rows, study, a, n, s)
                if r:
                    mat[i, j] = r["err_l1"]
        im = ax.imshow(mat, cmap="Blues", norm=norm, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                txt = f"{v:.1f}" if v >= 1 else f"{v:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                        color="white" if v > 3 else INK)
        for y in np.arange(1, len(NS)) * N_APP - 0.5:
            ax.axhline(y, color="white", linewidth=2.5)
        ax.set_xticks(range(len(SIZES)))
        ax.set_xticklabels([dxn(s) for s in SIZES])
        ax.set_title(LABEL[study])
        ax.grid(False)
        ax.tick_params(top=False, right=False)
    for ax in axes[1]:
        ax.set_xlabel(an.pp.DX_SYMBOL)
    axes[0, 0].set_yticks(range(len(ylabels)))
    axes[0, 0].set_yticklabels(ylabels)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label(r"$\varepsilon_{L1}$ [%]")
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    fig.savefig(path)
    plt.close(fig)


def fig_l1_vs_lstar(rows, path):
    fig, axes = plt.subplots(2, 2, figsize=(W, 5.4), sharex=True, sharey=True)
    for ax, (study, _, _) in zip(axes.flat, STUDIES):
        for a in APPROACHES:
            for n in NS:
                pts = [get(rows, study, a, n, s) for s in SIZES]
                pts = [r for r in pts if r and r["l_star"] and r["err_l1"]]
                if pts:
                    ax.plot([r["l_star"] for r in pts], [r["err_l1"] for r in pts],
                            color=APP_COLORS[a], marker=N_MARKERS[n], markersize=4,
                            markeredgecolor="white", markeredgewidth=0.4, linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        an.pp.plain_log_axis(ax)
        an.pp.plain_log_axis(ax, "x")
        ax.set_title(LABEL[study])
    for ax in axes[1]:
        ax.set_xlabel(an.pp.LSTAR_SYMBOL)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\varepsilon_{L1}$ [%]")
    handles = [an.pp.line_handle(a, APP_COLORS[a]) for a in APPROACHES]
    handles += [an.pp.line_handle(hp(n), MUTED, "none", N_MARKERS[n]) for n in NS]
    rect = an.pp.legend_below(fig, handles, ncol=N_APP)
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def paired_scatter(rows, pairs, xlab, ylab, path):
    fig, axes = plt.subplots(1, len(pairs), figsize=(W, 3.4))
    lim = 80
    sizes = {0.005: 2.5, 0.0025: 3.5, 0.00125: 4.5, 0.000625: 5.5}
    for ax, (sx, sy, sub) in zip(np.atleast_1d(axes), pairs):
        for a, n, s in itertools.product(APPROACHES, NS, SIZES):
            rx, ry = get(rows, sx, a, n, s), get(rows, sy, a, n, s)
            if not (rx and ry):
                continue
            ax.plot(rx["cd_err"], ry["cd_err"], marker=N_MARKERS[n], linestyle="none",
                    markersize=sizes[s], markeredgecolor=APP_COLORS[a], markeredgewidth=0.8,
                    markerfacecolor=APP_COLORS[a], alpha=0.85)
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, linewidth=0.8, linestyle=":")
        for ax_set in (ax.set_xscale, ax.set_yscale):
            ax_set("symlog", linthresh=0.3)
        ax.set_xticks([-10, -1, 0, 1, 10])
        ax.set_yticks([-10, -1, 0, 1, 10])
        an.pp.plain_ticks(ax, "x")
        an.pp.plain_ticks(ax, "y")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(sub)
    handles = [Line2D([], [], color=APP_COLORS[a], marker="o", linestyle="none", markersize=4,
                      label=a) for a in APPROACHES]
    handles += [Line2D([], [], color=MUTED, marker=N_MARKERS[n], linestyle="none", markersize=4,
                       label=hp(n)) for n in NS]
    handles += [Line2D([], [], color=MUTED, marker="o", linestyle="none", markersize=sizes[size],
                       label=f"{an.pp.DX_SYMBOL} = {dxn(size)}")
                for size in (0.005, 0.000625)]
    rect = an.pp.legend_below(fig, handles, ncol=5)
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def factor_effects(rows):
    by_key = {(r["study"], r["approach"], r["nPrisms"], round(r["size"], 7)): r for r in rows
              if r["approach"] in APPROACHES}
    eff = {f: {n: [] for n in NS} for f in ("topology", "core size (halving)", "schemes", "model")}
    for (st, a, n, s), r in by_key.items():
        for b in APPROACHES:
            if b > a:
                q = by_key.get((st, b, n, s))
                if q:
                    eff["topology"][n].append(abs(r["cd_err"] - q["cd_err"]))
        q = by_key.get((st, a, n, round(s / 2, 7)))
        if q:
            eff["core size (halving)"][n].append(abs(r["cd_err"] - q["cd_err"]))
    swap = {"SALSA": ("SALSA_upwind", "SSGLRR"),
            "SSGLRR": ("SSGLRR_upwind", None),
            "SALSA_upwind": (None, "SSGLRR_upwind")}
    for st, (sch, mod) in swap.items():
        for (s0, a, n, s), r in by_key.items():
            if s0 != st:
                continue
            for target, f in ((sch, "schemes"), (mod, "model")):
                q = by_key.get((target, a, n, s)) if target else None
                if q:
                    eff[f][n].append(abs(r["cd_err"] - q["cd_err"]))
    return eff


def fig_factor_effects(eff, path):
    fig, axes = plt.subplots(1, 3, figsize=(W, 3.0), sharey=True)
    colors = {"topology": "#2a78d6", "core size (halving)": "#1baf7a", "schemes": "#eb6834",
              "model": "#6250d6"}
    names = {"topology": "approach", "core size (halving)": r"$L_\mathrm{nom}/2$",
             "schemes": "schemes", "model": "model"}
    rng = np.random.default_rng(1)
    for ax, n in zip(axes, NS):
        for i, f in enumerate(eff):
            v = np.maximum(np.array(eff[f][n]), 1e-3)
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color=colors[f], alpha=0.55,
                       edgecolors="none")
            ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=INK, linewidth=1.6)
        ax.set_yscale("log")
        an.pp.plain_ticks(ax)
        ax.set_xticks(range(len(eff)))
        ax.set_xticklabels([names[f] for f in eff], rotation=30)
        ax.set_title(hp(n))
        ax.axhspan(1e-3, 0.05, color=GRID, alpha=0.6, zorder=0)
    axes[0].set_ylabel(r"$|\Delta\varepsilon_{C_D}|$ [pp]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_refinement_order(rows, path):
    fig, axes = plt.subplots(1, N_APP, figsize=(W, 2.6), sharey=True)
    for ax, a in zip(axes, APPROACHES):
        for st, _, k in STUDIES:
            pts = [(s, get(rows, st, a, 40, s)) for s in SIZES]
            pts = [(s, abs(r["cd_err"])) for s, r in pts if r]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=STUDY_COLORS[st],
                    linestyle=STUDY_LS[k], marker=STUDY_MARKER[st], markersize=3.5,
                    markeredgecolor="white", markeredgewidth=0.4, linewidth=1.0)
        for order, y0 in ((1, 8), (2, 8)):
            xs = np.array([0.005, 0.000625])
            ax.plot(xs, y0 * (xs / 0.005) ** order, color=MUTED, linewidth=0.8, linestyle=":")
            ax.text(0.00062, y0 * (0.000625 / 0.005) ** order, f" {order}", fontsize=7,
                    color=MUTED, va="center")
        ax.axhspan(0.01, 0.05, color=GRID, alpha=0.6, zorder=0)
        ax.set_yscale("log")
        ax.set_ylim(0.01, 30)
        an.pp.plain_ticks(ax)
        size_axis(ax)
        ax.set_title(a)
    axes[0].set_ylabel(r"$|\varepsilon_{C_D}|$ [%]")
    handles = [an.pp.line_handle(LABEL[st], STUDY_COLORS[st], STUDY_LS[k], STUDY_MARKER[st])
               for st, _, k in STUDIES]
    rect = an.pp.legend_below(fig, handles, ncol=4)
    fig.tight_layout(rect=rect, w_pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def fig_tmr(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), sharey=True)
    for ax, n in zip(axes, (40, 50)):
        for st, _, k in STUDIES:
            bm = get(rows, st, "blockMesh")
            ax.axhline(bm["err_tmr"], color=STUDY_COLORS[st], linestyle=STUDY_LS[k],
                       linewidth=0.9, alpha=0.8)
            for a, dx in zip(APPROACHES, np.linspace(0.94, 1.06, N_APP)):
                pts = [(s, get(rows, st, a, n, s)) for s in SIZES]
                pts = [(s, r) for s, r in pts if r]
                ax.plot([s * dx for s, _ in pts], [r["err_tmr"] for _, r in pts], linestyle="none",
                        marker=APP_MARKERS[a], color=STUDY_COLORS[st], markersize=3.5, alpha=0.9)
        ax.axhline(0, color=INK, linewidth=0.7)
        symlog_y(ax, 1)
        size_axis(ax)
        ax.set_title(hp(n))
    axes[0].set_ylabel(r"$(C_f - C_{f,\mathrm{TMR}})/C_{f,\mathrm{TMR}}$ [%]")
    handles = [an.pp.line_handle(LABEL[st], STUDY_COLORS[st], STUDY_LS[k]) for st, _, k in STUDIES]
    handles += [an.pp.line_handle(a, MUTED, "none", m) for a, m in APP_MARKERS.items()]
    rect = an.pp.legend_below(fig, handles, ncol=4)
    fig.tight_layout(rect=rect)
    fig.savefig(path)
    plt.close(fig)


def fig_along_plate(studies, path):
    fig, axes = an.pp.grid_figure(2, N_APP, sharex=True)
    dl = an.DELTA_MEAN
    for i, n in enumerate((40, 50)):
        for j, a in enumerate(APPROACHES):
            ax = axes[i][j]
            for sd in studies:
                r = next((r for r in sd["runs"] if r["approach"] == a and r["nPrisms"] == n
                          and abs(r["size"] - 0.000625) < 1e-9), None)
                if r is None:
                    continue
                grid, err = an.plate_error(r, sd["base"])
                k = an.pp.study_schemes(RESULTS / sd["name"])
                ax.plot(grid / dl, err, color=STUDY_COLORS[sd["name"]], linestyle=STUDY_LS[k],
                        linewidth=1.0)
                xc = r.get("x_crit")
                if xc:
                    ax.axvline(xc / dl, color=STUDY_COLORS[sd["name"]], linestyle=":",
                               linewidth=0.8)
            ax.axhline(0, color=MUTED, linewidth=0.7)
            ax.set_xlim(an.PLATE[0] / dl, an.PLATE[1] / dl)
    an.pp.label_grid(axes, [hp(n) for n in (40, 50)], APPROACHES, an.pp.error_label("cf"),
                     an.XI_LABEL)
    handles = [an.pp.line_handle(LABEL[sd["name"]], STUDY_COLORS[sd["name"]],
                                 STUDY_LS[an.pp.study_schemes(RESULTS / sd["name"])])
               for sd in studies]
    handles.append(an.pp.line_handle(an.XCRIT, MUTED, ":"))
    rect = an.pp.legend_below(fig, handles, ncol=5)
    fig.tight_layout(rect=rect, h_pad=0.6, w_pad=0.4)
    fig.savefig(path)
    plt.close(fig)


def fig_iterations(rows, path):
    fig, axes = plt.subplots(1, 1 + N_APP, figsize=(W, 2.8), sharey=True)
    for ax, a in zip(axes, ("blockMesh",) + APPROACHES):
        for st, _, k in STUDIES:
            if a == "blockMesh":
                r = get(rows, st, a)
                ax.bar([s for s, _, _ in STUDIES].index(st), r["iters"],
                       color=STUDY_COLORS[st], width=0.6,
                       hatch="//" if k == "upwind" else None, edgecolor="white")
                continue
            for n in NS:
                pts = [(s, get(rows, st, a, n, s)) for s in SIZES]
                pts = [(s, r) for s, r in pts if r]
                xs = [s * {"SALSA": 0.9, "SALSA_upwind": 0.97,
                           "SSGLRR": 1.03,
                           "SSGLRR_upwind": 1.1}[st] for s, _ in pts]
                ax.plot(xs, [r["iters"] for _, r in pts], color=STUDY_COLORS[st],
                        linestyle=STUDY_LS[k], marker=N_MARKERS[n], markersize=3,
                        linewidth=0.8, alpha=0.85)
        if a == "blockMesh":
            ax.set_xticks(range(4))
            ax.set_xticklabels([LABEL[s].replace(", ", "\n") for s, _, _ in STUDIES], fontsize=6,
                               rotation=90)
        else:
            size_axis(ax)
            ax.tick_params(axis="x", labelrotation=90)
        ax.set_title(a)
    axes[0].set_ylabel("iterations")
    handles = [an.pp.line_handle(LABEL[st], STUDY_COLORS[st], STUDY_LS[k]) for st, _, k in STUDIES]
    handles += [an.pp.line_handle(hp(n), MUTED, "none", N_MARKERS[n]) for n in NS]
    rect = an.pp.legend_below(fig, handles, ncol=4)
    fig.tight_layout(rect=rect, w_pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def write_captions(rows):
    lines = ["# Figure captions", "",
             "Caption drafts for the comparison figures (written by synthesis.py from the",
             "CAPTIONS of compare.py, synthesis.py and timing.py; the figures carry no titles).",
             "$h^+ = h u_\\tau/\\nu$ (h: prism-stack height) uses the analytical $u_\\tau$ of"
             " the composite boundary-layer profile at x = 2 m; x and the nominal wall core"
             " cell size $L_\\mathrm{nom}$ are divided by $\\bar\\delta$, the analytical"
             " $\\delta_{99}$ averaged over the plate." + caps.core_note(caps.core_factors(rows)) + " "
             + caps.LSTAR, ""]
    for folder, captions in (("", cmp.CAPTIONS), ("synthesis/", CAPTIONS),
                             ("timing/", tim.CAPTIONS)):
        for name, text in captions.items():
            lines += [f"## {folder}{name}", "", text, ""]
    (COMP / "CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")


def regime_table(rows, path):
    keys = ["study", "nPrisms", "h_plus", "approach", "err_5mm", "err_0.625mm", "rms_0.625mm",
            "capped"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for st, _, _ in STUDIES:
            for n in NS:
                for a in APPROACHES:
                    r5, r0 = get(rows, st, a, n, 0.005), get(rows, st, a, n, 0.000625)
                    if not (r5 and r0):
                        continue
                    capped = sum(not (get(rows, st, a, n, s) or {"converged": False})["converged"]
                                 for s in SIZES)
                    w.writerow([st, n, f"{HP.get(n, float('nan')):.0f}", a, f"{r5['cd_err']:.3f}", f"{r0['cd_err']:.3f}",
                                f"{r0['err_rms']:.3f}", capped])


def main():
    OUT.mkdir(exist_ok=True)
    for pattern in LEGACY:
        for stale in OUT.glob(pattern):
            stale.unlink()
    rows = load_rows()
    fig_error_map(rows, OUT / "cf_l1_error_map.png")
    fig_l1_vs_lstar(rows, OUT / "cf_l1_error_vs_lstar.png")
    paired_scatter(rows, [("SALSA", "SALSA_upwind", "SALSA"),
                          ("SSGLRR", "SSGLRR_upwind", "SSG/LRR")],
                   EPS_CD + ", base", EPS_CD + ", upwind",
                   OUT / "cd_error_scheme_effect.png")
    paired_scatter(rows, [("SALSA", "SSGLRR", "base schemes"),
                          ("SALSA_upwind", "SSGLRR_upwind", "upwind schemes")],
                   EPS_CD + ", SALSA", EPS_CD + ", SSG/LRR",
                   OUT / "cd_error_model_effect.png")
    eff = factor_effects(rows)
    fig_factor_effects(eff, OUT / "cd_error_factor_effects.png")
    with open(OUT / "factor_effects.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["factor", "nPrisms", "h_plus", "pairs", "median", "p90", "max"])
        for fac, d in eff.items():
            for n, v in d.items():
                v = np.array(v)
                w.writerow([fac, n, f"{HP.get(n, float('nan')):.0f}", len(v), f"{np.median(v):.3f}",
                            f"{np.percentile(v, 90):.3f}", f"{v.max():.3f}"])
    fig_refinement_order(rows, OUT / "cd_error_refinement_order.png")
    fig_tmr(rows, OUT / "cf_tmr_deviation.png")
    studies = [cmp.load_study(RESULTS / st) for st, _, _ in STUDIES]
    fig_along_plate([s for s in studies if s], OUT / "cf_error_along_plate_finest.png")
    fig_iterations(rows, OUT / "iterations.png")
    regime_table(rows, OUT / "regime_table.csv")
    write_captions(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
