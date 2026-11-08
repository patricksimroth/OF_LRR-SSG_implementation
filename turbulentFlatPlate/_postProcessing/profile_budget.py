#!/usr/bin/env python3
"""
profile_budget.py - term-by-term budgets of the mean momentum equation
and the turbulence-model transport equations at the profile_validation
stations, in the style of Pope (2000), Figs. 7.34-7.38:

    summary/profiles/x_<x>m/budget/<eq>.png          all terms, blockMesh run
    summary/profiles/x_<x>m/budget/<eq>_<term>_<ydelta|yplus>.png   one term,
                                                                     every mesh

(<eq>_<model>... when a study has several models).  Equations: Ux and Uy
momentum (every model); SSG/LRR k (half the trace of the R budgets), Rxx,
Ryy, Rzz, -Rxy and omega; SALSA nuTilda.  Terms are kinematic and signed as on the right-hand side of
0 = -D(.)/Dt + sum(terms): positive = gain.  The left-hand side enters as one
term, the mean convection -D(.)/Dt.  Terms that vanish identically (Rzz
production, Rxy dissipation, the trace of the pressure-strain) are left out.

<eq>.png, for the report (no title or notes - those go in the caption):
(a) terms normalised at each y so that the sum of their squares is one, over
y/delta = 1e-4 ... 1.1 (logarithmic, pp.Y_DELTA_RANGE); (b) terms in viscous units (U: nu/u_tau^3; R, k: nu/u_tau^4; omega:
nu^2/u_tau^4, symlog value axis; nuTilda: 1/u_tau^2) over y+ = 0 ... 50, as
Pope.  delta = delta99 and u_tau are the analytical (_bl_theory) values AT THE
STATION, the same for every run.  The imbalance (sum of all terms) is only
reported in the log, see closure().

<eq>_<term>_ydelta.png, <eq>_<term>_yplus.png: one term on every
mesh (rows h+, columns approach, lines L_nom/delta_mean, black dashed blockMesh,
dotted prism-stack top), normalised and ranged as panel (a) resp. (b).

Two samplings, the same figures for each:
  budget/        raw: one value per cell crossed by the line x = x_s (midPoint
                 set), the cell value at the midpoint of the line's chord.
  budget_strip/  strip average: every cell centroid in x_s +- strip_half_width()
                 (2 cm, at most 5 % of x_s so delta99 varies by < 4 %;
                 cellCentre set).  Cells that share a centroid height (rows:
                 prism stack, blockMesh, snappy, snappy45) are averaged along x;
                 the scattered centroids of the gmsh cores are grouped into
                 y-bins about one local cell size high (strip_bins()).  On
                 non-aligned cores single-cell terms carry an O(1) operator
                 error with zero mean (documentation/analysis_core_refinement/
                 budget_analysis/BUDGETS.md); the average shows the balance the
                 mean flow sees.

The terms come from the turbulenceBudget function object (src/turbulenceBudget),
which re-evaluates the model terms on the latest time with the case's
fvSchemes, run as `rhoSimpleFoam -postProcess` in a throw-away shadow case
(symlinked constant/ and time directory, system/ copied with the library
paths pointed at this repo's src/).  The sampled cell values are cached in
<run>/postProcessing/turbulenceBudget/ and reused while its .stamp is newer
than the latest time and the mesh and its key matches.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# pyrefly: ignore [missing-import]
import matplotlib

matplotlib.use("Agg")

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# pyrefly: ignore [missing-import]
import numpy as np

import _pp_common as pp
import profile_validation as pv

FOAM_BASHRC = os.environ.get("FOAM_BASHRC", "/usr/lib/openfoam/openfoam2512/etc/bashrc")
SOLVER = "rhoSimpleFoam"
CACHE_DIR = "turbulenceBudget"
BUDGET_DIR = "budget"
STRIP_DIR = "budget_strip"
STRIP_PREFIX = "stripBudget"
STRIP_HALF_WIDTH = 0.02
STRIP_MAX_FRACTION = 0.05
ROW_MIN = 3

# Sampled turbulenceBudget equations (field prefix budget<eq>:): the momentum
# equation for every model, the model's own equations where there are any.
MOMENTUM = "U"
MODEL_EQUATIONS = {
    "SSGLRROmega2019": ("R", "Omega"),
    "SpalartAllmarasSALSA": ("NuTilda",),
}

# component order of the csv columns _0 ... (vector, symmTensor)
VECTOR = ("x", "y", "z")
SYMM = ("xx", "xy", "xz", "yy", "yz", "zz")

# (stem, label, source equation, component(s), sign, viscous scale)
#   viscous scale: the term is multiplied by nu^a / u_tau^b, (a, b)
PLOT_EQUATIONS = (
    ("Ux", r"$U_x$", "U", ("x",), 1.0, (1, 3)),
    ("Uy", r"$U_y$", "U", ("y",), 1.0, (1, 3)),
    ("k", r"$k$", "R", ("xx", "yy", "zz"), 0.5, (1, 4)),
    ("Rxx", r"$\overline{u'u'}$", "R", ("xx",), 1.0, (1, 4)),
    ("Ryy", r"$\overline{v'v'}$", "R", ("yy",), 1.0, (1, 4)),
    ("Rzz", r"$\overline{w'w'}$", "R", ("zz",), 1.0, (1, 4)),
    ("Rxy", r"$-\overline{u'v'}$", "R", ("xy",), -1.0, (1, 4)),
    ("omega", r"$\omega$", "Omega", None, 1.0, (2, 4)),
    ("nuTilda", r"$\tilde\nu$", "NuTilda", None, 1.0, (0, 2)),
)

# (field name, legend label, colour, linestyle)
TERMS = (
    ("convection", "mean convection", "#0b0b0b", (0, (7, 2))),
    ("pressureGradient", "pressure gradient", "#9467bd", (0, (6, 1.5, 1, 1.5))),
    ("viscousStress", "viscous stress", "#2ca02c", "-."),
    ("reynoldsStress", "Reynolds stress", "#ff7f0e", ":"),
    ("production", "production", "#1f77b4", "-"),
    ("pressureStrain", "pressure-strain", "#9467bd", (0, (6, 1.5, 1, 1.5, 1, 1.5))),
    ("dissipation", "dissipation", "#d62728", "--"),
    ("destruction", "destruction", "#d62728", "--"),
    ("crossDiffusion", "cross diffusion", "#8c564b", (0, (3, 1, 1, 1))),
    ("viscousDiffusion", "viscous diffusion", "#2ca02c", "-."),
    ("turbulentDiffusion", "turbulent diffusion", "#ff7f0e", ":"),
)
IMBALANCE = "imbalance"

# Terms that never exceed this fraction of the local norm are identically
# zero in 2D (e.g. Rzz production ~1e-28 against 1e6) and are not drawn.
ZERO_FRACTION = 1e-6

YPLUS_MAX = 50.0
Y_DELTA_MIN, Y_DELTA_MAX = pp.Y_DELTA_RANGE

# <eq>.png are meant for the report: text width, report-sized fonts.
REPORT_SIZE = (6.3, 3.4)
REPORT_RC = {**pp.REPORT_RC, "axes.grid": False}


def strip_half_width(station):
    return min(STRIP_HALF_WIDTH, STRIP_MAX_FRACTION * station)


def station_tag(station):
    """`x0p2` for 0.2, `x1` for 1.0 - the sampleUyPlus set names."""
    return "x" + f"{station:g}".replace(".", "p")


def run_model(run_dir):
    text = (Path(run_dir) / "constant" / "turbulenceProperties").read_text()
    match = re.search(r"^\s*RASModel\s+(\w+)\s*;", text, re.M)
    return match.group(1) if match else None


def run_equations(run_dir):
    return (MOMENTUM,) + MODEL_EQUATIONS.get(run_model(run_dir), ())


def budget_dict(lib, equations, stations, write_fields=False):
    lines = "\n".join(
        f"    {station_tag(s)} {{ type midPoint; axis y; start ({s:g} 0 0.05); "
        f"end ({s:g} 0.1 0.05); }}" for s in stations)
    strips = "\n".join(
        f"    {station_tag(s)} {{ type cellCentre; axis xyz; bounds "
        f"({s - strip_half_width(s):g} 0 -1) ({s + strip_half_width(s):g} 0.1 1); }}"
        for s in stations)
    # One sets object per equation: the buffered writers name the file after
    # every sampled field and fail silently beyond 255 characters.
    samples = "\n".join(
        f'    budget{eq} {{ type sets; libs (sampling); interpolationScheme cell; '
        f'setFormat csv; fields ("budget{eq}:.*"); sets {{ $lines }} }}\n'
        f'    {STRIP_PREFIX}{eq} {{ type sets; libs (sampling); interpolationScheme cell; '
        f'setFormat csv; fields ("budget{eq}:.*"); sets {{ $strips }} }}'
        for eq in equations) if stations else ""
    if write_fields:
        samples += ('\n    budgetFields { type writeObjects; libs (utilityFunctionObjects); '
                    'objects ("budget.*"); writeOption anyWrite; }')
    return f"""FoamFile {{ version 2.0; format ascii; class dictionary; object budgetDict; }}
lines
{{
{lines}
}}
strips
{{
{strips}
}}
functions
{{
    budget {{ type turbulenceBudget; libs ("{lib}"); }}
{samples}
}}
"""


def cache_key(src_dir, equations, stations):
    fo_dir = Path(src_dir) / "turbulenceBudget"
    text = "".join((fo_dir / name).read_text() for name in ("turbulenceBudget.C",
                                                             "turbulenceBudget.H"))
    text += budget_dict("", equations, stations) + pp.source_key(budget_dict, shadow_case)
    return hashlib.sha1(text.encode()).hexdigest()


def shadow_case(run_dir, latest, shadow, src_dir):
    """Case in `shadow` that reads run_dir's mesh and latest time but writes
    nothing back: constant/ and the time-directory files are symlinks, its
    sub-directories (uniform/, rewritten by -postProcess) are copies, system/
    is a copy whose controlDict loads the libraries from src_dir."""
    shadow = Path(shadow)
    (shadow / "constant").symlink_to(Path(run_dir, "constant").resolve())
    time_dir = shadow / latest.name
    time_dir.mkdir()
    for entry in latest.iterdir():
        if entry.is_dir():
            shutil.copytree(entry, time_dir / entry.name, symlinks=False)
        else:
            (time_dir / entry.name).symlink_to(entry.resolve())
    shutil.copytree(Path(run_dir) / "system", shadow / "system", symlinks=False)
    control = shadow / "system" / "controlDict"
    control.write_text(re.sub(r'"[^"\n]*/src/([^/"\n]+)/lib/([^/"\n]+\.so)"',
                              lambda m: f'"{Path(src_dir) / m.group(1) / "lib" / m.group(2)}"',
                              control.read_text()))


def foam_command(case, dict_path):
    command = f"{SOLVER} -postProcess -latestTime -dict {dict_path} -case {case}"
    if os.environ.get("WM_PROJECT_DIR"):
        return ["bash", "-c", command]
    return ["bash", "-c", f"source {FOAM_BASHRC} > /dev/null 2>&1; {command}"]


def extract(run_dir, latest, src_dir, stations):
    """Cache dir with the sampled budget csv files of one run, or None."""
    equations = run_equations(run_dir)
    cache = Path(run_dir) / "postProcessing" / CACHE_DIR
    stamp = cache / ".stamp"
    key = cache_key(src_dir, equations, stations)
    outputs = sorted(cache.glob("budget*/*/*.csv")) + sorted(cache.glob(f"{STRIP_PREFIX}*/*/*.csv"))
    if pp.outputs_fresh(stamp, outputs, pp.run_data_inputs(run_dir, latest), key):
        return cache
    lib = Path(src_dir) / "turbulenceBudget" / "lib" / "libturbulenceBudget.so"
    if not lib.is_file():
        print(f"  [warn] {lib} missing - build it with `wmake libso` in its src dir")
        return None
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="budget_") as shadow:
        shadow_case(run_dir, latest, shadow, src_dir)
        dict_path = Path(shadow) / "system" / "budgetDict"
        dict_path.write_text(budget_dict(lib, equations, stations))
        result = subprocess.run(foam_command(shadow, dict_path), capture_output=True,
                                text=True)
        (cache / "log.budget").write_text(result.stdout + result.stderr)
        for eq in equations:
            for name in (f"budget{eq}", f"{STRIP_PREFIX}{eq}"):
                source = Path(shadow) / "postProcessing" / name
                if source.is_dir():
                    shutil.copytree(source, cache / name)
    if result.returncode != 0 or not list(cache.glob("budget*/*/*.csv")):
        print(f"  [warn] {Path(run_dir).name}: budget extraction failed "
              f"(see {cache / 'log.budget'})")
        return None
    pp.write_stamp(stamp, key)
    return cache


def read_budget(cache):
    """{station: {"y": y, "<eq>:<term>": (n,) or (n, 6)}} from the cache."""
    out = {}
    for path in sorted(Path(cache).glob("budget*/*/*.csv")):
        station = pv.station_from_tag(path.name.split("_budget", 1)[0])
        if station is None:
            continue
        header = path.open().readline().strip().split(",")
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        if data.shape[1] != len(header):
            print(f"  [warn] unreadable column layout: {path}")
            continue
        order = np.argsort(data[:, 0])
        data = data[order]
        prof = out.setdefault(station, {"y": data[:, 0]})
        if len(prof["y"]) != len(data):
            print(f"  [warn] {path.name}: point count differs from the other equations")
            continue
        fields = {}
        for i, name in enumerate(header[1:], start=1):
            base, _, comp = name.rpartition("_")
            if base.startswith("budget") and comp.isdigit():
                fields.setdefault(base, {})[int(comp)] = data[:, i]
            else:
                fields[name] = data[:, i]
        for name, value in fields.items():
            key = name[len("budget"):]
            prof[key] = (np.column_stack([value[c] for c in sorted(value)])
                         if isinstance(value, dict) else value)
    return out


def strip_bins(y, width):
    """Bin label per strip cell: rows of >= ROW_MIN cells with a common centroid
    height are one bin each; the remaining (scattered) centroids are grouped
    into y-bins about one local cell size high, the size sqrt(width * dy / n)
    estimated from the cell density of the 2*10 neighbouring centroids."""
    order = np.argsort(y, kind="stable")
    ys = y[order]
    new_row = np.r_[True, np.diff(ys) > 1e-9 + 1e-6 * np.abs(ys[1:])]
    row_id = np.cumsum(new_row) - 1
    in_row = np.bincount(row_id)[row_id] >= ROW_MIN
    label = np.where(in_row, row_id, -1)
    free = np.flatnonzero(~in_row)
    nxt, i, k = row_id.max() + 1, 0, 10
    while i < len(free):
        lo, hi = max(0, i - k), min(len(free) - 1, i + k)
        span = ys[free[hi]] - ys[free[lo]]
        size = np.sqrt(width * span / (hi - lo)) if hi > lo else 0.0
        j = i + 1
        while j < len(free) and ys[free[j]] - ys[free[i]] < size:
            j += 1
        label[free[i:j]] = nxt
        nxt, i = nxt + 1, j
    out = np.empty(len(y), dtype=int)
    out[order] = label
    return out


def read_strip_budget(cache):
    """As read_budget, from the strip samples: {station: {"y": bin mean y,
    "<eq>:<term>": bin means}}, one bin per cell row or y-bin (strip_bins)."""
    out = {}
    for path in sorted(Path(cache).glob(f"{STRIP_PREFIX}*/*/*.csv")):
        station = pv.station_from_tag(path.name.split("_budget", 1)[0])
        if station is None:
            continue
        header = path.open().readline().strip().split(",")
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        if data.shape[1] != len(header) or header[:3] != ["x", "y", "z"]:
            print(f"  [warn] unreadable column layout: {path}")
            continue
        width = 2 * strip_half_width(station)
        _, inv = np.unique(strip_bins(data[:, 1], width), return_inverse=True)
        count = np.bincount(inv)
        means = np.column_stack([np.bincount(inv, weights=data[:, i]) / count
                                 for i in range(data.shape[1])])
        means = means[np.argsort(means[:, 1])]
        prof = out.setdefault(station, {"y": means[:, 1]})
        if len(prof["y"]) != len(means):
            print(f"  [warn] {path.name}: bin count differs from the other equations")
            continue
        fields = {}
        for i, name in enumerate(header[3:], start=3):
            base, _, comp = name.rpartition("_")
            if base.startswith("budget") and comp.isdigit():
                fields.setdefault(base, {})[int(comp)] = means[:, i]
            else:
                fields[name] = means[:, i]
        for name, value in fields.items():
            prof[name[len("budget"):]] = (np.column_stack([value[c] for c in sorted(value)])
                                          if isinstance(value, dict) else value)
    return out


def equation_terms(profile, spec):
    """{term: (n,) array} of one plotted equation, incl. the imbalance."""
    stem, _label, source, comps, sign, _visc = spec
    terms = {}
    for term in [t[0] for t in TERMS] + [IMBALANCE]:
        value = profile.get(f"{source}:{term}")
        if value is None:
            continue
        if comps is not None:
            if stem == "k" and term == "pressureStrain":
                continue
            names = VECTOR if value.shape[1] == len(VECTOR) else SYMM
            value = sum(value[:, names.index(c)] for c in comps)
        terms[term] = sign * value
    return terms


def shown_terms(terms):
    """Terms to draw, in TERMS order: all but the identically zero ones."""
    names = [t for t, *_ in TERMS if t in terms]
    norm = local_norm(terms, names)
    return [t for t in names if float(np.max(np.abs(terms[t]) / norm)) > ZERO_FRACTION]


def closure(budget):
    """Median over all stations and points of |imbalance| / (sum terms^2)^1/2,
    per plotted equation - the check of the turbulenceBudget copy."""
    parts = []
    for spec in PLOT_EQUATIONS:
        rel = []
        for prof in budget.values():
            terms = equation_terms(prof, spec)
            if IMBALANCE not in terms:
                continue
            names = shown_terms(terms)
            rel.append(np.abs(terms[IMBALANCE]) / local_norm(terms, names))
        if rel:
            parts.append(f"{spec[0]} {np.median(np.concatenate(rel)):.1e}")
    return "closure " + ", ".join(parts)


def local_norm(terms, names):
    norm = np.sqrt(sum(terms[t] ** 2 for t in names))
    return np.where(norm > 0, norm, np.inf)


def plot_overview(budget, spec, out_path, nu, scales):
    """Report figure: (a) locally normalised terms over y/delta99, (b) terms in
    viscous units over y+, as Pope (2000), Fig. 7.34.  No title, no notes:
    they belong in the caption."""
    stem, _label, *_rest, (a, b) = spec
    terms = equation_terms(budget, spec)
    names = shown_terms(terms)
    if not names:
        return False
    y = budget["y"]
    norm = local_norm(terms, names)
    visc = nu ** a / scales["u_tau"] ** b
    y_delta = y / scales["delta_L"]
    y_plus = y * scales["u_tau"] / nu
    near = y_plus <= YPLUS_MAX

    with plt.rc_context(REPORT_RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=REPORT_SIZE)
        handles = []
        for name in names:
            lab, color, ls = next((lab, c, ls) for t, lab, c, ls in TERMS if t == name)
            ax_a.plot(terms[name] / norm, y_delta, color=color, linestyle=ls)
            keep = y_plus <= YPLUS_MAX * 1.05
            (line,) = ax_b.plot(terms[name][keep] * visc, y_plus[keep], color=color,
                                linestyle=ls, label=lab)
            handles.append(line)

        ax_a.set_xlim(-1.0, 1.0)
        pp.log_y_delta_axis(ax_a)
        ax_a.set_ylabel(r"$y/\delta$")
        ax_b.set_ylim(0.0, YPLUS_MAX)
        ax_b.set_ylabel(r"$y^+$")
        set_viscous_axis([ax_b], [(terms[t][near] * visc, y_plus[near]) for t in names],
                         stem)
        for ax, tag in ((ax_a, "(a)"), (ax_b, "(b)")):
            report_axes(ax)
            ax.set_title(tag, loc="left")
            gain_loss(ax)
        n_cols = len(handles) if len(handles) <= 4 else int(np.ceil(len(handles) / 2))
        fig.legend(handles=handles, loc="lower center", ncol=n_cols,
                   frameon=False, handlelength=2.6, columnspacing=1.4)
        n_rows = int(np.ceil(len(handles) / n_cols))
        fig.tight_layout(rect=(0, 0.055 * n_rows + 0.01, 1, 1), w_pad=2.0)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return True


def report_axes(ax):
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.tick_params(direction="in", top=True, right=True)


def gain_loss(ax):
    """Pope's value-axis label: `loss` at the left end, `gain` at the right."""
    ax.set_xlabel(" ")
    for x, text, ha in ((0.0, "loss", "left"), (1.0, "gain", "right")):
        ax.annotate(text, (x, 0.0), xycoords=("axes fraction", ax.xaxis.label),
                    ha=ha, va="bottom")


def set_viscous_axis(axes, pairs, stem, n_ticks=2):
    """Value axis of the y+ panels from [(values, y+)] (y+ <= YPLUS_MAX).
    omega ~ 1/y^2: every omega term grows like y^-4 towards the wall, so its
    axis is symlog; elsewhere linear."""
    if stem == "omega":
        outer = [np.abs(v[(yp >= 10.0) & np.isfinite(v)]) for v, yp in pairs]
        outer = np.concatenate([o for o in outer if o.size] + [np.ones(1)])
        linthresh = 0.1 * float(outer.max())
        lo, hi = symlog_limits([v[yp >= 1.0] for v, yp in pairs])
        for ax in axes:
            ax.set_xscale("symlog", linthresh=linthresh)
            ax.set_xlim(lo, hi)
            ax.set_xticks(symlog_ticks(linthresh, lo, hi, n_max=n_ticks))
    else:
        lims = viscous_limits(pairs)
        for ax in axes:
            ax.set_xlim(*lims)


def viscous_limits(pairs):
    """Value range of a y+ panel.  Terms can be singular at the wall; a range
    more than 3x wider than the one above y+ = 1 is cut to that."""
    def span(lower):
        sel = [v[(yp >= lower) & np.isfinite(v)] for v, yp in pairs]
        sel = np.concatenate([v for v in sel if v.size] + [np.zeros(1)])
        return float(sel.min()), float(sel.max())
    lo, hi = span(-np.inf)
    lo1, hi1 = span(1.0)
    if hi - lo > 3.0 * (hi1 - lo1) > 0:
        lo, hi = lo1, hi1
    pad = 0.05 * (hi - lo) or 1.0
    return lo - pad, hi + pad


def symlog_limits(values):
    finite = np.concatenate([v[np.isfinite(v)] for v in values] + [np.zeros(1)])
    return 1.5 * float(finite.min()), 1.5 * float(finite.max())


def symlog_ticks(linthresh, lo, hi, n_max=4):
    """0 and +-10^k from linthresh outwards, at most n_max per side."""
    ticks = [0.0]
    for sign, bound in ((-1.0, -lo), (1.0, hi)):
        if bound <= linthresh:
            continue
        k0, k1 = int(np.ceil(np.log10(linthresh))), int(np.floor(np.log10(bound)))
        step = max(1, int(np.ceil((k1 - k0 + 1) / n_max)))
        ticks += [sign * 10.0 ** k for k in range(k1, k0 - 1, -step)]
    return sorted(ticks)


VALUE_CAP = 3.0

GRID_MODES = {"ydelta": (r"$y/\delta$", Y_DELTA_MIN, Y_DELTA_MAX),
              "yplus": (r"$y^+$", 0.0, YPLUS_MAX)}


def term_curve(prof, spec, term, mode, nu, scales):
    """(value, y coordinate) of one term: (a) normalisation over y/delta or
    viscous units over y+, as the two overview panels; None if not drawn."""
    if prof is None:
        return None
    terms = equation_terms(prof, spec)
    names = shown_terms(terms)
    if term not in names:
        return None
    y = prof["y"]
    if mode == "ydelta":
        return terms[term] / local_norm(terms, names), y / scales["delta_L"]
    a, b = spec[5]
    return terms[term] * nu ** a / scales["u_tau"] ** b, y * scales["u_tau"] / nu


def linear_marks(y, y_max, slot, n_slots, n=4, y_min=0.0):
    """Marker indices spread evenly in y (in log y for y_min > 0), staggered per line."""
    frac = (np.arange(n) + (slot + 1) / (n_slots + 1)) / n
    if y_min > 0:
        targets = y_min * (y_max / y_min) ** frac
        with np.errstate(divide="ignore"):
            dist = lambda t: np.abs(np.log(np.maximum(y, 1e-300)) - np.log(t))
    else:
        targets = y_max * frac
        dist = lambda t: np.abs(y - t)
    return sorted({int(np.argmin(dist(t))) for t in targets if t <= y.max()})


def plot_term_grid(group, budgets, index, station, spec, term, mode, out_path, nu, scales):
    """Report figure of one term on every mesh: rows h+, columns approach,
    lines L_nom/delta_mean, black dashed blockMesh, dotted prism-stack top.
    No title, no notes: they belong in the caption."""
    rows, cols, cells = group["rows"], group["cols"], group["cells"]
    order = group["order"]
    style = {lab: (pv.LINE_COLORS[i % len(pv.LINE_COLORS)],
                   pv.LINE_MARKERS[i % len(pv.LINE_MARKERS)]) for i, lab in enumerate(order)}
    y_label, y_min, y_max = GRID_MODES[mode]
    reference = group["reference"]
    ref_curve = term_curve(budgets.get(reference, {}).get(station), spec, term, mode,
                           nu, scales) if reference else None
    curves = {n: term_curve(budgets.get(n, {}).get(station), spec, term, mode, nu, scales)
              for names in cells.values() for n in names}

    with plt.rc_context(REPORT_RC):
        fig, axes = plt.subplots(len(rows), len(cols),
                                 figsize=(REPORT_SIZE[0], 2.0 * len(rows) + 0.9),
                                 squeeze=False, sharex=True, sharey=True)
        plotted = set()
        for i, row in enumerate(rows):
            for j, approach in enumerate(cols):
                ax = axes[i][j]
                names = sorted(cells.get((row, approach), []),
                               key=lambda n: order.index(group["label"][n]))
                for name in names:
                    curve = curves.get(name)
                    if curve is None:
                        continue
                    lab = group["label"][name]
                    color, marker = style[lab]
                    keep = curve[1] <= 1.05 * y_max
                    ax.plot(curve[0][keep], curve[1][keep], color=color, marker=marker,
                            markersize=3.5, markeredgecolor="white", markeredgewidth=0.4,
                            linewidth=1.0, markevery=linear_marks(
                                curve[1][keep], y_max, order.index(lab), len(order),
                                y_min=y_min))
                    plotted.add(lab)
                if ref_curve is not None:
                    keep = ref_curve[1] <= 1.05 * y_max
                    ax.plot(ref_curve[0][keep], ref_curve[1][keep], color=pv.INK,
                            linestyle="--", linewidth=1.0, zorder=5)
                h = next(((index.get(n, {}).get("mesh") or {}).get("totalLayerThickness")
                          for n in names), None)
                if h:
                    y_h = h / scales["delta_L"] if mode == "ydelta" else h * scales["u_tau"] / nu
                    if y_min <= y_h <= y_max:
                        ax.axhline(y_h, color=pv.MUTED, linestyle=":", linewidth=0.9)
                if not names:
                    ax.text(0.5, 0.5, "no run", transform=ax.transAxes, ha="center",
                            va="center", color=pv.EDGE)
                report_axes(ax)
                if mode == "ydelta":
                    pp.log_y_delta_axis(ax)
                else:
                    ax.set_ylim(0.0, y_max)
                if i == 0 and approach is not None:
                    ax.set_title(approach)
                if j == 0:
                    row_label = f"{pp.fmt_hplus(row)}\n" if row is not None else ""
                    ax.set_ylabel(f"{row_label}{y_label}")
                if i == len(rows) - 1:
                    gain_loss(ax)
        if not plotted:
            plt.close(fig)
            return False

        flat = [ax for row_axes in axes for ax in row_axes]
        if mode == "ydelta":
            flat[0].set_xlim(-1.0, 1.0)
            flat[0].set_xticks([-0.5, 0.0, 0.5])
        else:
            pool = [c for c in curves.values() if c is not None]
            if ref_curve is not None:
                pool.append(ref_curve)
            set_viscous_axis(flat, [(v[yp <= y_max], yp[yp <= y_max]) for v, yp in pool],
                             spec[0], n_ticks=1)
            if spec[0] != "omega":
                if ref_curve is not None:
                    # One run far off must not flatten every other line.
                    v, yp = ref_curve
                    cap = VALUE_CAP * float(np.max(np.abs(v[(yp >= 1.0) & (yp <= y_max)]),
                                                   initial=0.0))
                    lo, hi = flat[0].get_xlim()
                    if cap > 0:
                        flat[0].set_xlim(max(lo, -cap), min(hi, cap))
                flat[0].xaxis.set_major_locator(
                    matplotlib.ticker.MaxNLocator(3, symmetric=False, prune="both"))

        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=style[lab][0], marker=style[lab][1], markersize=4,
                          linewidth=1.0, label=lab) for lab in order if lab in plotted]
        if ref_curve is not None:
            handles.append(Line2D([], [], color=pv.INK, linestyle="--", linewidth=1.0,
                                  label=pp.ERROR_BASELINE))
        handles.append(Line2D([], [], color=pv.MUTED, linestyle=":", linewidth=0.9,
                              label=pp.STACK_TOP_LABEL))
        n_cols = int(np.ceil(len(handles) / 2))
        fig.legend(handles=handles, loc="lower center", ncol=n_cols, frameon=False,
                   handlelength=2.4, columnspacing=1.2)
        fig.tight_layout(rect=(0, 0.5 / fig.get_size_inches()[1], 1, 1),
                         h_pad=0.6, w_pad=0.4)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return True


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path, case_name, run_case_dir, _source, results_root = \
        pp.resolve_case_paths(script_dir)
    print("=" * 64)
    print(f"{case_name} - turbulence-model budgets per station")
    print("=" * 64)
    src_dir = Path(base_path) / "src"

    solved = [(run, latest) for run, latest in pp.solved_run_dirs(run_case_dir)
              if run_model(run)]
    if not solved:
        print(f"[warn] no completed RAS runs under {run_case_dir}")
        return 0
    stations = pv.STATIONS

    def work(item):
        run, latest = item
        return run.name, extract(run, latest, src_dir, stations)

    budgets, strips = {}, {}
    for name, cache in pp.map_threads(work, solved, pp.pv_jobs()):
        if cache is None:
            continue
        budgets[name] = read_budget(cache)
        strips[name] = read_strip_budget(cache)
        print(f"  {name}: {closure(budgets[name])}")
    if not budgets:
        print("[warn] no budgets extracted")
        return 0
    print(f"Runs: {len(budgets)}")

    u_inf, nu, _rho = pp.detect_case_parameters(run_case_dir / sorted(budgets)[0])
    index = pp.combo_index(pp.load_manifest(run_case_dir))
    groups = pv.layout(sorted(budgets), index)
    out_root = Path(results_root) / pp.SUMMARY_DIR / pv.PROFILE_DIR

    tasks = []
    for station, (sampling, folder) in ((s, k) for s in stations
                                        for k in (("raw", BUDGET_DIR), ("strip", STRIP_DIR))):
        out_dir = out_root / f"x_{station:.2f}m" / folder
        shutil.rmtree(out_dir, ignore_errors=True)
        scales = pp.flow_scales(u_inf, nu, station)
        for group in groups:
            members = [n for names in group["cells"].values() for n in names]
            sources = {eq for n in members for eq in run_equations(run_case_dir / n)}
            equations = [s for s in PLOT_EQUATIONS if s[2] in sources]
            suffix = f"_{group['model']}" if len(groups) > 1 else ""
            overview_runs = [group["reference"]] if group["reference"] else members
            for spec in equations:
                for run in overview_runs:
                    name = spec[0] + suffix + ("" if group["reference"] else f"_{run}")
                    tasks.append(("overview", sampling, group, station, spec, None, run,
                                  out_dir / f"{name}.png", scales))
                for term, *_ in TERMS:
                    for mode in GRID_MODES:
                        tasks.append(("grid", sampling, group, station, spec, (term, mode),
                                      None, out_dir / f"{spec[0]}_{term}{suffix}_{mode}.png",
                                      scales))
        print(f"  x = {station:g} m -> {out_dir}")

    def draw(task):
        kind, sampling, group, station, spec, term, run, path, scales = task
        data = strips if sampling == "strip" else budgets
        path.parent.mkdir(parents=True, exist_ok=True)
        if kind == "overview":
            budget = data.get(run, {}).get(station)
            return budget is not None and plot_overview(budget, spec, path, nu, scales)
        term, mode = term
        return plot_term_grid(group, data, index, station, spec, term, mode, path,
                              nu, scales)

    written = sum(pp.map_processes(draw, tasks, pp.plot_jobs()))
    print(f"\nBudgets complete ({written} figure(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
