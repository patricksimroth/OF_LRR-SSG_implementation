"""
_captions.py - caption drafts for the report figures.

The figures carry no titles or notes (_pp_common report style): everything
that used to sit in a suptitle, an axis label or a legend note is here.
dashboard.py shows these under every image and writes summary/CAPTIONS.md.

caption(rel_path, context) matches a figure path relative to the study's
results folder (e.g. summary/along_plate/cf.png,
<run>/convergence_plots/residuals.png) against FIGURES, first match wins.
context: model, schemes, norm (see study_context()).
"""
from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

import _pp_common as pp
import field_images as fi

MODEL_NAMES = {"SSGLRROmega2019": "SSG/LRR-$\\omega$ (2019, simple diffusion)",
               "SpalartAllmarasSALSA": "SA-SALSA", "SpalartAllmaras": "SA",
               "kOmegaSST": "$k$-$\\omega$ SST"}
SCHEMES = {"base": "base schemes", "upwind": "first-order upwind convection schemes"}

QUANTITY = {"cf": "skin-friction coefficient $C_f$",
            "yplus": "wall-adjacent cell height in wall units $y^+$",
            "delta99": "boundary-layer thickness $\\delta_{99}$",
            "dstar": "displacement thickness $\\delta^*$",
            "theta": "momentum thickness $\\theta$",
            "H": "shape factor $H = \\delta^*/\\theta$"}
QUANTITY_NOTE = {
    "yplus": " The first cell is 1e-6 m high on every mesh.",
    "delta99": " $\\delta_{99}$: wall distance where $U_x$ reaches 99 % of the edge velocity."
               " Dotted: prism-stack top.",
    "dstar": " Integrated from the cell-centroid profiles with $U_e = U_x(y = 55$ mm$)$.",
    "theta": " Integrated from the cell-centroid profiles with $U_e = U_x(y = 55$ mm$)$.",
    "H": " Integrated from the cell-centroid profiles with $U_e = U_x(y = 55$ mm$)$.",
}
PROFILE = {
    "Ux": "Mean streamwise velocity $U_x/U_\\infty$",
    "nut": "Eddy viscosity $\\nu_t/\\nu$ (band: TMR SA, CFL3D, for the SA family)",
    "k": ("Turbulent kinetic energy $k$ (band: TMR SSG/LRR-RSM-w2012, CFL3D, for models that"
          " transport $R_{ij}$; for eddy-viscosity models $k$ is half the trace of the"
          " Boussinesq stresses, not transported)"),
    "Rxx": "Reynolds normal stress $\\overline{u'u'}$",
    "Ryy": "Reynolds normal stress $\\overline{v'v'}$",
    "Rzz": "Reynolds normal stress $\\overline{w'w'}$",
    "Rxy": "Reynolds shear stress $-\\overline{u'v'}$",
}
R_NOTE = (" Band: TMR SSG/LRR-RSM-w2012 (CFL3D) where the model transports $R_{ij}$; for an"
          " eddy-viscosity model these are the Boussinesq stresses (normal stresses isotropic"
          " by construction).")
EQUATION = {"Ux": "mean $x$-momentum ($U_x$)", "Uy": "mean $y$-momentum ($U_y$)",
            "Rxx": "$R_{xx}$", "Ryy": "$R_{yy}$", "Rzz": "$R_{zz}$", "Rxy": "$R_{xy}$",
            "k": "$k$", "omega": "$\\omega$", "nuTilda": "$\\tilde\\nu$"}
TERM = {"convection": "convection", "pressureGradient": "pressure gradient",
        "viscousStress": "viscous stress", "reynoldsStress": "Reynolds stress",
        "production": "production", "dissipation": "dissipation",
        "pressureStrain": "pressure-strain", "destruction": "destruction",
        "crossDiffusion": "cross diffusion", "viscousDiffusion": "viscous diffusion",
        "turbulentDiffusion": "turbulent diffusion"}

GRID = ("Rows: prism-stack height $h^+$; columns: mesh approach; lines: nominal wall core cell"
        " size $L_\\mathrm{nom}/\\bar\\delta$; dashed: the same model on the prism-free blockMesh mesh.")
LSTAR = ("$L^* = (L_\\mathrm{core}/\\bar\\delta)\\,\\langle(\\delta - h)^+\\rangle/\\bar\\delta$:"
         " core cell size times the plate-averaged fraction of the analytical boundary"
         " layer above the stack top, $\\langle(\\delta - h)^+\\rangle = A_\\mathrm{out}/L$ with"
         " $A_\\mathrm{out}$ the area between $\\delta(x)$ and the stack height h over the"
         " plate of length L. $L_\\mathrm{core}$ is measured on the mesh: square root of the"
         " median cell area of the core cells with centroids between $h + L_\\mathrm{nom}$ and"
         " $h + 3L_\\mathrm{nom}$ above the wall over 0.1 m $\\le x \\le$ 1.95 m"
         " ($L_\\mathrm{nom}$: nominal wall core cell size; the first core row, built on the"
         " prism top, is skipped).")


def core_factors(rows):
    """Median measured/nominal core cell size per approach, from summary/comparison CSV rows."""
    ratios = {}
    for r in rows:
        try:
            ratios.setdefault(r["approach"], []).append(float(r["l_core"]) / float(r["size"]))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
    return {a: statistics.median(v) for a, v in ratios.items() if v}


def core_note(factors):
    """Caption sentence: measured core cell size per approach as a multiple of L_nom."""
    if not factors:
        return ""
    order = [a for a in pp.APPROACHES if a in factors]
    parts = ", ".join(f"{factors[a]:.2f} ({a})" for a in order)
    return (" The core cell size measured next to the stack (see $L^*$) is"
            f" $L_\\mathrm{{core}} = c\\,L_\\mathrm{{nom}}$ with c = {parts}.")


ERROR = "$\\varepsilon_q = (q - q_\\mathrm{blockMesh})/q_\\mathrm{blockMesh}$"
FIELD_IMAGE = {"domain_p": "Pressure $p$, whole domain",
               "domain_U": "Velocity magnitude $|U|$, whole domain",
               "bl_U": "Velocity magnitude $|U|$", "bl_nut": "Eddy viscosity $\\nu_t$",
               "bl_k": "Turbulent kinetic energy $k$",
               "bl_omega": "Specific dissipation rate $\\omega$ (logarithmic colour scale)",
               "bl_nuTilda": "SA working variable $\\tilde\\nu$"}


def study_context(results_root):
    """{model, schemes, norm} of a study folder (manifest + overlay)."""
    root = Path(results_root)
    models = []
    for name in ("study_manifest.json", "_rundata/study_manifest.json"):
        path = root / name
        if path.is_file():
            combos = json.loads(path.read_text()).get("combos", [])
            models = sorted({c.get("model") for c in combos if c.get("model")})
            break
    scales = pp.case_scales()
    for run in sorted((root / "_rundata").glob("*__blockMesh*")):
        scales = pp.case_scales(run)
        break
    rows = []
    table = root / pp.SUMMARY_DIR / "tables" / "summary.csv"
    if table.is_file():
        with open(table, newline="") as f:
            rows = list(csv.DictReader(f))
    norm = (f"$h^+ = h u_\\tau/\\nu$ (h: prism-stack height) uses the analytical $u_\\tau$ ="
            f" {scales['u_tau']:.3f} m/s of the composite boundary-layer profile at"
            f" x = {scales['station']:g} m; x and the nominal wall core cell size"
            f" $L_\\mathrm{{nom}}$ are divided by $\\bar\\delta$ = {1e3 * scales['delta_mean']:.2f}"
            f" mm, the analytical $\\delta_{{99}}$ averaged over the plate."
            f"{core_note(core_factors(rows))}")
    return {"model": ", ".join(MODEL_NAMES.get(m, m) for m in models) or "all runs",
            "schemes": SCHEMES[pp.study_schemes(root)], "norm": norm}


def _study(c):
    return f"{c['model']}, {c['schemes']}"


def _along_plate(m, c):
    q = m["q"]
    return (f"{QUANTITY[q][0].upper()}{QUANTITY[q][1:]} along the plate ({_study(c)})."
            f"{QUANTITY_NOTE.get(q, '')} {GRID} {c['norm']}")


def _error_along_plate(m, c):
    return (f"Relative deviation {ERROR} of the {QUANTITY[m['q']]} from blockMesh along the"
            f" plate ({_study(c)}). Dotted: $x_\\mathrm{{crit}}$, where the analytical"
            f" $\\delta_{{99}}$ reaches the stack height h. {GRID} {c['norm']}")


def _error_vs_core(m, c):
    return (f"Relative deviation {ERROR} of the {QUANTITY[m['q']]} from blockMesh over the"
            f" normalised core cell size $L^*$ ({_study(c)}): (a) at the TMR station"
            f" $x_\\mathrm{{ref}}$ = 0.97 m (symmetric-log axis, linear within $\\pm$0.1 %),"
            f" (b) RMS over 0.1 m $\\le x \\le$ 1.95 m. Colours: mesh approach; markers:"
            f" prism-stack height $h^+$; lines join the core sizes of one approach and stack."
            f" {LSTAR} {c['norm']}")


def _cost(m, c):
    return ("RMS of $\\varepsilon_{C_f}$ over 0.1 m $\\le x \\le$ 1.95 m against the number of"
            f" cells ({_study(c)}). Each line runs from the coarsest to the finest core;"
            f" colours: mesh approach; markers: prism-stack height; dashed: blockMesh."
            f" {c['norm']}")


def _wall_law(m, c):
    return (f"Mean velocity in wall units at x = 1 m ({_study(c)}), $u_\\tau$ from each run's"
            " own wall shear stress. Grey: viscous sublayer $u^+ = y^+$, log law"
            " ($\\kappa$ = 0.41, B = 5.0) and Coles' profile at $Re_\\theta$ = 10 000. Dotted:"
            f" prism-stack top. {GRID} {c['norm']}")


def _prism_top(m, c):
    return ("Height of the prism-stack top in local wall units $h u_\\tau(x)/\\nu$, with"
            f" $u_\\tau(x)$ of the blockMesh run ({_study(c)}), over the boundary-layer"
            " regions: viscous sublayer ($y^+ < 5$), buffer layer ($y^+ < 30$), log layer"
            " ($y^+ < 0.2\\,\\delta_{99}^+$), wake region ($y^+ < \\delta_{99}^+$) and free"
            " stream. Line labels: $h^+$ with the analytical $u_\\tau$ at x = 2 m."
            f" {c['norm']}")


def _profile(m, c):
    note = R_NOTE if m["q"] in ("Rxx", "Ryy", "Rzz", "Rxy") else ""
    return (f"{PROFILE[m['q']]} over the wall distance $y/\\delta$ (logarithmic, $\\delta$ the"
            f" analytical $\\delta_{{99}}$ at the station) at x = {float(m['x']):g} m"
            f" ({_study(c)}).{note} Dotted: prism-stack top; dash-dotted: $\\delta_{{99}}$ of"
            f" blockMesh. {GRID} {c['norm']}")


SAMPLING = {
    None: " Cell values along the line x = const, one per cell.",
    "_strip": (" Strip average: cell values in x $\\pm$ 2 cm (at most 5 % of x) averaged along x,"
               " per cell row where the cells share a centroid height, else in y-bins about one"
               " cell size high."),
}


def _budget(m, c):
    return (f"Budget of the {EQUATION.get(m['eq'], m['eq'])} equation on blockMesh at"
            f" x = {float(m['x']):g} m ({_study(c)}): (a) every term divided by the local"
            " magnitude of all terms over $y/\\delta$ (logarithmic), (b) in viscous units over $y^+$, as"
            " Pope (2000), Fig. 7.34. Kinematic terms, gain positive; convection is"
            " $-D/Dt$. $u_\\tau$ and $\\delta_{99}$ analytical at the station."
            f"{SAMPLING[m.get('strip')]}")


def _budget_term(m, c):
    axis = ("divided by the local magnitude of all terms, over $y/\\delta$ (logarithmic)"
            if m["mode"] == "ydelta" else "in viscous units, over $y^+$")
    return (f"{TERM.get(m['term'], m['term']).capitalize()} in the"
            f" {EQUATION.get(m['eq'], m['eq'])} equation at x = {float(m['x']):g} m"
            f" ({_study(c)}), {axis}. Dotted: prism-stack top.{SAMPLING[m.get('strip')]}"
            f" {GRID} {c['norm']}")


def _field_budget(m, c):
    return (f"{TERM.get(m['term'], m['term']).capitalize()} in the"
            f" {EQUATION.get(m['eq'], m['eq'])} equation in the boundary layer"
            f" {fi.window_text(m.get('win') or '')}, mesh overlaid, at the last iteration."
            " Kinematic, gain positive (red), loss negative (blue); convection is $-D/Dt$."
            " Symmetric logarithmic colour scale $\\mathrm{sign}(q)\\log_{10}(1 + |q|/q_0)$,"
            " $q_0$ the area-weighted median of $|q|$ in the window (at least $10^{-4}$ of the"
            " upper limit), upper limit the area-weighted 99.5th percentile; larger values"
            " take the end colours.")


def _field(m, c):
    name = m["name"]
    if name.startswith("bl_R_"):
        what = f"Reynolds stress $R_{{{name[5:].lower()}}}$"
    else:
        what = FIELD_IMAGE.get(name, name)
    where = "" if name.startswith("domain") else (
        f" in the boundary layer {fi.window_text(m.get('win') or '')}")
    return f"{what}{where}, mesh overlaid, at the last iteration."


FIGURES = (
    (r"summary/along_plate/(?P<q>\w+)\.png", _along_plate),
    (r"summary/error_along_plate/(?P<q>\w+)\.png", _error_along_plate),
    (r"summary/error_vs_core/(?P<q>\w+)\.png", _error_vs_core),
    (r"summary/cost/cf_error_vs_cells\.png", _cost),
    (r"summary/wall_law/uplus_x1m\w*\.png", _wall_law),
    (r"summary/prism_coverage/hplus_regions\.png", _prism_top),
    (r"summary/profiles/x_(?P<x>[\d.]+)m/budget(?P<strip>_strip)?/(?P<eq>[A-Za-z]+)_(?P<term>[A-Za-z]+?)"
     r"(?:_\w+)?_(?P<mode>ydelta|yplus)\.png", _budget_term),
    (r"summary/profiles/x_(?P<x>[\d.]+)m/budget(?P<strip>_strip)?/(?P<eq>[A-Za-z]+)(?:_\w+)?\.png",
     _budget),
    (r"summary/profiles/x_(?P<x>[\d.]+)m/(?P<q>Ux|nut|k|Rxx|Ryy|Rzz|Rxy)(?:_\w+)?\.png",
     _profile),
    (r"[^/]+/convergence_plots/residuals\.png",
     lambda m, c: "Initial (solid) and final (dotted) residuals per SIMPLE iteration."),
    (r"[^/]+/convergence_plots/minmax_(?P<f>\w+)\.png",
     lambda m, c: f"Minimum and maximum of {m['f']} in the domain per iteration."),
    (r"[^/]+/convergence_plots/Ux_profile_history\.png",
     lambda m, c: "$U_x(y)$ at x = 1 m at 20, 40, 60, 80 and 100 % of the iterations."),
    (r"[^/]+/field_plots/(?:(?P<win>front|back)/)?budget/(?P<eq>[A-Za-z]+)_(?P<term>[A-Za-z]+)"
     r"\.png", _field_budget),
    (r"[^/]+/field_plots/(?:(?P<win>front|back)/)?\d+_(?P<name>\w+)\.png", _field),
)


def caption(rel_path, context):
    rel = str(rel_path).replace("\\", "/")
    for pattern, build in FIGURES:
        match = re.fullmatch(pattern, rel)
        if match:
            try:
                return build(match.groupdict(), context)
            except KeyError:
                return ""
    return ""
