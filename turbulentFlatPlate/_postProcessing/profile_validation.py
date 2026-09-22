#!/usr/bin/env python3
"""
profile_validation.py - wall-normal turbulence profiles of every run against
the NASA TMR reference, at the ten `sampleUyPlus` stations.

Two independent comparisons:

* **Reynolds stresses and omega** vs TMR SSG/LRR-RSM-w2012 (CFL3D), one figure
  per component (Rxx, Ryy, Rzz, Rxy, omega), each a 2 x 5 grid of stations with
  every run overlaid.  Runs that transport R (the RSM variants) are plotted
  from the solved `R`; the others fall back to `turbulenceProperties:R`, i.e.
  the stresses their eddy-viscosity model implies, which is a fair thing to
  hold against an RSM reference as long as the legend says so.
* **Eddy viscosity of the SA family** vs TMR SA (CFL3D), one figure.  Only
  SA-family runs appear: nu_t of an RSM or SST run is not comparable with an
  SA reference.

The reference profiles come from `_validation_data/tmr_*.dat`, extracted from
the TMR field dumps by `_tmr_extract_profiles.py` (which also documents the
unit and axis conversions).  Wall distance is plotted on a log axis because
the near-wall stress peak sits at y+ ~ 15, i.e. inside the first 1 % of the
boundary-layer height.
"""
from __future__ import annotations

import glob
import re
import os
import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import matplotlib

matplotlib.use("Agg")

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# pyrefly: ignore [missing-import]
from matplotlib.ticker import MaxNLocator

# pyrefly: ignore [missing-import]
import numpy as np

import _pp_common as pp
from cf_validation import DISPLAY_NAME, assign_plot_styles

RSTRESS_REFERENCE = "tmr_rstress_ssglrrrsm.dat"
NUT_REFERENCE = "tmr_nut_sa.dat"

# Reference label per figure set - these are near-neighbour models, not the
# same closure, and the plots have to say so.
RSTRESS_LABEL = "TMR SSG/LRR-RSM-w2012 (CFL3D)"
NUT_LABEL = "TMR SA (CFL3D)"

# (column in the reference, candidate columns in a sampled profile, axis label)
RSTRESS_QUANTITIES = (
    ("Rxx", ("R_xx", "R_tp_xx"), r"$\overline{u'u'}$  [m$^2$/s$^2$]"),
    ("Ryy", ("R_yy", "R_tp_yy"), r"$\overline{v'v'}$  [m$^2$/s$^2$]"),
    ("Rzz", ("R_zz", "R_tp_zz"), r"$\overline{w'w'}$  [m$^2$/s$^2$]"),
    ("Rxy", ("R_xy", "R_tp_xy"), r"$\overline{u'v'}$  [m$^2$/s$^2$]"),
    ("omega", ("omega",), r"$\omega$  [1/s]"),
)
NUT_QUANTITY = ("nut", ("nut",), r"$\nu_t$  [m$^2$/s]")

# Models whose nu_t is comparable with the TMR SA reference.
SA_FAMILY = ("SpalartAllmarasSALSA", "SpalartAllmaras")

LOG_SCALE_VALUES = ("omega",)

# An eddy-viscosity model carries no normal-stress anisotropy, so these three
# are identically 2k/3 when they come from turbulenceProperties:R.
NORMAL_STRESSES = ("Rxx", "Ryy", "Rzz")

# Law-of-the-wall constants for the u+/y+ panels (same as postpro.py).
KAPPA, B_LOG = 0.41, 5.0

# Extra figure width reserved for the right-hand legend, in inches.
LEGEND_WIDTH_IN = 3.0

# Mesh approach every other approach is compared against in the
# velocity-error figure (structured, prism-free, finest wall-normal grid).
ERROR_BASELINE = "blockMesh"

# Samples below this y+ are the wall face itself, not a cell value.
WALL_POINT_YPLUS = 1e-6


def station_from_tag(tag):
    """0.2 from `x0p2`, 1.0 from `x1`, 1.95 from `x1p95`."""
    try:
        return float(tag.lstrip("x").replace("p", "."))
    except ValueError:
        return None


def load_reference(val_dir, name):
    """{station: {column: ndarray}} from a `_validation_data` profile file."""
    path = os.path.join(val_dir, name)
    if not os.path.exists(path):
        print(f"  [warn] reference not found: {path}")
        return None, None
    header = [line for line in open(path) if line.lower().startswith("variables")]
    if not header:
        print(f"  [warn] no variables line in {path}")
        return None, None
    columns = [c.strip().strip('"') for c in header[0].split("=", 1)[1].split(",")]
    zones = pp.parse_tecplot_zones(path)
    out = {}
    for zone, arr in zones.items():
        station = station_from_tag(zone.split("=")[-1].strip())
        if station is None or arr.shape[1] != len(columns):
            continue
        out[station] = dict(zip(columns, arr.T))
    print(f"  reference {name}: {len(out)} stations, columns {columns[1:]}")
    return out, columns


def read_run_profiles(case_dir):
    """{station: {column: ndarray}} for one run, from sampleUyPlus."""
    sample_dir = pp.latest_time_directory(Path(case_dir) / "postProcessing" / "sampleUyPlus")
    if sample_dir is None:
        return {}
    out = {}
    for path in sorted(glob.glob(os.path.join(sample_dir, "*_profile_*.xy"))):
        station = station_from_tag(os.path.basename(path).split("_profile_", 1)[0])
        if station is None:
            continue
        cols, arr = pp.read_profile_set(path)
        if cols is None:
            print(f"  [warn] unreadable column layout: {os.path.basename(path)}")
            continue
        out[station] = dict(zip(cols, arr.T))
    return out


def pick_column(profile, candidates):
    """First candidate column present in a profile, with its name."""
    for name in candidates:
        if name in profile:
            return name, profile[name]
    return None, None


def delta99(x, u_inf, nu):
    """Analytical turbulent flat-plate thickness, only used to scale the axes."""
    return 0.37 * x / (u_inf * x / nu) ** 0.2


def plot_quantity(quantity, candidates, axis_label, runs, reference,
                  ref_label, output_path, u_inf, nu, colors, markers, sections):
    """One figure: a 2 x 5 grid of stations, every run plus the reference."""
    stations = sorted(reference) if reference else sorted(
        {s for profiles in runs.values() for s in profiles})
    if not stations:
        print(f"  [warn] no stations for {quantity}")
        return False
    n_cols = 5
    n_rows = int(np.ceil(len(stations) / n_cols))
    # +LEGEND_WIDTH_IN so moving the legend out of the axes does not squeeze
    # the panels: the plotting area keeps its full 4.0 in per column.
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.0 * n_cols + LEGEND_WIDTH_IN, 3.6 * n_rows),
                             squeeze=False)
    plotted_any = False
    sources = {}

    for index, station in enumerate(stations):
        ax = axes[index // n_cols][index % n_cols]
        for name, profiles in runs.items():
            profile = profiles.get(station)
            if profile is None:
                continue
            column, values = pick_column(profile, candidates)
            if column is None:
                continue
            sources.setdefault(name, column)
            y = profile["y"]
            mask = y > 0
            line, = ax.plot(values[mask], y[mask] * 1e3,
                            color=colors.get(name, "k"),
                            marker=markers.get(name, "o"), markersize=3,
                            markevery=max(1, int(mask.sum() // 12)),
                            linewidth=1.2)
            plotted_any = True

        if reference and station in reference and quantity in reference[station]:
            ref = reference[station]
            mask = ref["y"] > 0
            ax.plot(ref[quantity][mask], ref["y"][mask] * 1e3,
                    color="black", linestyle="--", linewidth=1.3, alpha=0.8)
            plotted_any = True

        ax.set_yscale("log")
        # From the first cell of the finest TMR grid to well outside the
        # boundary layer, so every station is framed the same way.
        ax.set_ylim(1e-4, 1.6 * delta99(station, u_inf, nu) * 1e3)
        if quantity in LOG_SCALE_VALUES:
            ax.set_xscale("log")
        else:
            # Four ticks in scientific notation: the stresses and nu_t span
            # 1e-4..1e-2, whose default tick labels run into each other.
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
            ax.xaxis.get_offset_text().set_fontsize(8)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_title(f"x = {station:g} m", fontsize=10)
        if index % n_cols == 0:
            ax.set_ylabel(r"$y$  [mm]")
        ax.set_xlabel(axis_label, fontsize=9)

    for index in range(len(stations), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    if not plotted_any:
        plt.close(fig)
        print(f"  [warn] nothing to plot for {quantity}")
        return False

    fallback = sorted({name for name, column in sources.items()
                       if column.startswith("R_tp_")})
    note = ""
    if fallback:
        # Run names are long (<model>__<approach>); listing four of them
        # overflows the figure width, so collapse the common case.
        who = ("all runs" if len(fallback) == len(sources)
               else ", ".join(DISPLAY_NAME.get(n, n) for n in fallback))
        note = f"\n{who}: modelled stresses (turbulenceProperties:R)"
        if quantity in NORMAL_STRESSES:
            # Boussinesq gives R = (2/3)k I - nu_t dev(twoSymm(grad U)); in a
            # thin shear layer the deviatoric normal parts are ~0, so all three
            # normal stresses collapse onto 2k/3.  Saying so on the figure
            # stops the flat Rxx = Ryy = Rzz curves reading as a bug.
            note += ", isotropic by construction (= 2k/3)"
        else:
            note += ", not transported"
    fig.suptitle(f"{axis_label.split('  ')[0]} profiles vs {ref_label}{note}",
                 fontsize=11)
    rect = pp.legend_right(fig, sections + [("Reference", [(ref_label, "black", None)])],
                           width_in=LEGEND_WIDTH_IN)
    fig.tight_layout(rect=(rect[0], 0, rect[2], 0.94))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  {quantity:<6} -> {output_path}")
    return True


def friction_velocity(profile):
    """u_tau = max over the line of sqrt(|tau_x| / rho), or None.

    `wallShearStress` is a volume field that is only non-zero at the wall, so
    the maximum along a wall-normal line is the wall value.  Same convention as
    postpro.compute_uplus_yplus.
    """
    tau = profile.get("tau_x")
    if tau is None:
        return None
    rho = profile.get("rho")
    u_tau = float(np.sqrt(np.abs(tau / (rho if rho is not None else 1.0))).max())
    return u_tau if np.isfinite(u_tau) and u_tau > 0.0 else None


def uplus_yplus(profile, nu):
    """(y+, u+) for one sampled station, or (None, None).

    Same convention as postpro.compute_uplus_yplus so the two figures agree:
    u_tau = max over the line of sqrt(|tau_x| / rho) - `wallShearStress` is a
    volume field that is only non-zero at the wall, so the maximum along a
    wall-normal line is the wall value - and y+ = y u_tau / nu with the
    freestream nu = mu/rho from the case's thermophysicalProperties.
    """
    y, ux = profile.get("y"), profile.get("Ux")
    u_tau = friction_velocity(profile)
    if y is None or ux is None or u_tau is None:
        return None, None
    return y * u_tau / nu, ux / u_tau


def plot_uplus_yplus(runs, output_path, nu, colors, markers, sections):
    """u+ over y+ (semi-log) at every station, 2 x 5 grid, one line per run.

    The references are the two universal ones - the viscous sublayer u+ = y+
    and the log law - deliberately NOT the Coles profile that the single
    station figure in postpro.py uses: that one is tabulated at Re_theta =
    10000, which this plate only reaches near x = 1.2 m, so it would be wrong
    on nine of the ten panels.
    """
    stations = sorted({s for profiles in runs.values() for s in profiles})
    if not stations:
        print("  [warn] no stations for u+/y+")
        return False
    n_cols = 5
    n_rows = int(np.ceil(len(stations) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.0 * n_cols + LEGEND_WIDTH_IN, 3.6 * n_rows),
                             squeeze=False)
    plotted_any = False

    yp_visc = np.logspace(-1, np.log10(12.0), 40)
    yp_log = np.logspace(np.log10(8.0), 4.2, 60)

    for index, station in enumerate(stations):
        ax = axes[index // n_cols][index % n_cols]
        for name, profiles in runs.items():
            profile = profiles.get(station)
            if profile is None:
                continue
            yp, up = uplus_yplus(profile, nu)
            if yp is None:
                continue
            mask = yp > 0
            line, = ax.plot(yp[mask], up[mask],
                            color=colors.get(name, "k"),
                            marker=markers.get(name, "o"), markersize=3,
                            markevery=max(1, int(mask.sum() // 12)),
                            linewidth=1.2)
            plotted_any = True

        for xs, ys, style, lab in (
                (yp_visc, yp_visc, dict(color="0.4", linestyle=":", linewidth=1.2),
                 r"$u^+ = y^+$"),
                (yp_log, np.log(yp_log) / KAPPA + B_LOG,
                 dict(color="black", linestyle="--", linewidth=1.2),
                 rf"log law ($\kappa$={KAPPA}, B={B_LOG})")):
            ax.plot(xs, ys, **style)

        ax.set_xscale("log")
        ax.set_xlim(0.1, 2e4)
        ax.set_ylim(0, 32)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_title(f"x = {station:g} m", fontsize=10)
        ax.set_xlabel(r"$y^+$", fontsize=9)
        if index % n_cols == 0:
            ax.set_ylabel(r"$u^+$")

    for index in range(len(stations), n_rows * n_cols):
        axes[index // n_cols][index % n_cols].axis("off")

    if not plotted_any:
        plt.close(fig)
        print("  [warn] nothing to plot for u+/y+")
        return False

    fig.suptitle(r"$u^+$ over $y^+$ at the sampling stations", fontsize=12)
    rect = pp.legend_right(fig, sections + [("Law of the wall", [
        (r"$u^+ = y^+$", "0.4", None),
        (rf"log law ($\kappa$={KAPPA}, B={B_LOG})", "black", None)])],
        width_in=LEGEND_WIDTH_IN)
    fig.tight_layout(rect=(rect[0], 0, rect[2], 0.94))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  {'u+/y+':<6} -> {output_path}")
    return True


def sampled_delta99(profile):
    """(delta99, U_e) from one sampled station, same rule as bl_thickness.py.

    Distinct from delta99() above, which is the analytical correlation used
    only to frame the profile axes.

    U_e = max Ux on the line, delta99 = first wall distance where Ux reaches
    0.99 U_e, linearly interpolated between the bracketing samples.
    """
    y, ux = profile.get("y"), profile.get("Ux")
    if y is None or ux is None or len(y) < 2:
        return None, None
    order = np.argsort(y)
    y, ux = y[order], ux[order]
    u_e = float(ux.max())
    if u_e <= 0:
        return None, None
    target = 0.99 * u_e
    above = np.flatnonzero(ux >= target)
    if above.size == 0:
        return None, None
    i = int(above[0])
    if i == 0:
        return float(y[0]), u_e
    y0, y1, u0, u1 = y[i - 1], y[i], ux[i - 1], ux[i]
    frac = 0.0 if u1 == u0 else (target - u0) / (u1 - u0)
    return float(y0 + frac * (y1 - y0)), u_e


def _column(profile, *names):
    """First of `names` present in a sampled profile, else None."""
    for name in names:
        if name in profile:
            return profile[name]
    return None


def _k(profile):
    """Turbulent kinetic energy: the sampled k, else half the trace of R."""
    k = _column(profile, "k")
    if k is not None:
        return k
    for prefix in ("R_", "R_tp_"):
        parts = [profile.get(f"{prefix}{c}") for c in ("xx", "yy", "zz")]
        if all(p is not None for p in parts):
            return 0.5 * (parts[0] + parts[1] + parts[2])
    return None


def _stress(component):
    return lambda p, u_tau, nu: (
        None if _column(p, f"R_{component}", f"R_tp_{component}") is None
        else _column(p, f"R_{component}", f"R_tp_{component}") / u_tau**2)


# Quantities compared against the baseline, all in WALL UNITS so every line is
# compared at matching y+ (same basis as u+/y+):
#   (key, symbol, value(profile, u_tau, nu) -> array | None, output file)
ERROR_QUANTITIES = (
    ("U", r"u^+", lambda p, u_tau, nu: _column(p, "Ux") / u_tau
     if _column(p, "Ux") is not None else None, "profile_velocity_error.png"),
    ("nut", r"\nu_t^+ = \nu_t/\nu",
     lambda p, u_tau, nu: None if _column(p, "nut") is None else _column(p, "nut") / nu,
     "profile_error_nut.png"),
    ("k", r"k^+ = k/u_\tau^2",
     lambda p, u_tau, nu: None if _k(p) is None else _k(p) / u_tau**2,
     "profile_error_k.png"),
    ("omega", r"\omega^+ = \omega\nu/u_\tau^2",
     lambda p, u_tau, nu: None if _column(p, "omega") is None
     else _column(p, "omega") * nu / u_tau**2, "profile_error_omega.png"),
    ("Rxx", r"\overline{u'u'}^+", _stress("xx"), "profile_error_Rxx.png"),
    ("Ryy", r"\overline{v'v'}^+", _stress("yy"), "profile_error_Ryy.png"),
    ("Rzz", r"\overline{w'w'}^+", _stress("zz"), "profile_error_Rzz.png"),
    ("Rxy", r"\overline{u'v'}^+", _stress("xy"), "profile_error_Rxy.png"),
)


def variant_colors(names, index):
    """(run -> colour, run -> variant label, ordered labels) for the error plots.

    Each error panel holds ONE mesh approach, so colouring by approach (as the
    other figures do) would make every line in a panel the same colour.  Here
    colour encodes the swept parameter instead, on a sequential colormap so the
    order of e.g. nPrisms 5 < 20 < 40 is visible at a glance.
    """
    keys = pp.swept_keys([index[n] for n in names if n in index])
    label = {n: pp._variant_label(index.get(n, {}).get("params") or {}, keys) or n
             for n in names}
    ordered = sorted(set(label.values()),
                     key=lambda v: ([float(x) for x in re.findall(r"[\d.]+", v)], v))
    cmap = plt.get_cmap("viridis")
    shade = {v: cmap(0.9 * i / max(1, len(ordered) - 1)) for i, v in enumerate(ordered)}
    return {n: shade[label[n]] for n in names}, label, ordered, shade


def plot_profile_error(runs, station, index, markers, nu, out_dir, quantity,
                       symbol, value_fn, file_name):
    """|relative error| of one wall-unit quantity against the baseline, over y+.

    One panel per mesh approach, one line per run (coloured by the swept
    parameter), at x = station, with an open symbol at the top of each run's
    prism stack and a dotted line at its delta99.

    CAVEAT: every quantity is scaled with the run's OWN u_tau (u+, k+, R+,
    omega+, y+), so an error in the wall shear stress largely cancels.  In the
    viscous sublayer Ux = tau_w y / mu exactly, hence u+ = y+ whatever tau_w is
    - snappy nPrisms=5 has u_tau 3.5x too low and raw Ux 91.6 % low at
    y = 1e-6 m yet shows near-zero u+ error there.  Read Cf for that error.
    """
    base = sorted(n for n in runs if index.get(n, {}).get("approach") == ERROR_BASELINE)
    if not base:
        print(f"  [info] no {ERROR_BASELINE} run - skipping the {quantity} error figure")
        return False
    base_profile = runs[base[0]].get(station)
    base_ut = friction_velocity(base_profile) if base_profile else None
    base_val = value_fn(base_profile, base_ut, nu) if base_ut else None
    if base_val is None:
        print(f"  [info] {quantity} not sampled on {base[0]} - skipping its error figure")
        return False
    base_yp = base_profile["y"] * base_ut / nu
    order = np.argsort(base_yp)
    base_yp, base_val = base_yp[order], base_val[order]

    others = [n for n in sorted(runs) if index.get(n, {}).get("approach") != ERROR_BASELINE]
    approaches = sorted({index.get(n, {}).get("approach", "?") for n in others})
    if not approaches:
        return False
    colors, label, ordered, shade = variant_colors(others, index)

    n_cols = len(approaches)
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(4.6 * n_cols + LEGEND_WIDTH_IN, 5.0),
                             squeeze=False, sharex=True, sharey=True)
    plotted = set()

    for col, approach in enumerate(approaches):
        ax = axes[0][col]
        for name in others:
            if index.get(name, {}).get("approach") != approach:
                continue
            profile = runs[name].get(station)
            u_tau = friction_velocity(profile) if profile else None
            val = value_fn(profile, u_tau, nu) if u_tau else None
            if val is None:
                continue
            yp = profile["y"] * u_tau / nu
            order = np.argsort(yp)
            yp, val = yp[order], val[order]
            ref = np.interp(yp, base_yp, base_val)
            with np.errstate(divide="ignore", invalid="ignore"):
                err = np.abs((val - ref) / ref) * 100.0
            # `type face` sampling includes the wall face itself (y ~ 1e-19 m,
            # sometimes slightly negative), where the value and y+ are ~0 and
            # the relative error is 0/0 - it used to be drawn as a streak
            # across the panel from below the frame.
            mask = (yp > WALL_POINT_YPLUS) & (ref != 0) & np.isfinite(err)
            if not mask.any():
                continue
            ax.plot(err[mask], yp[mask], color=colors[name], linewidth=1.4)
            plotted.add(label[name])

            h = (index.get(name, {}).get("mesh") or {}).get("totalLayerThickness")
            if h:
                h_plus = h * u_tau / nu
                ax.plot([np.interp(h_plus, yp[mask], err[mask])], [h_plus],
                        marker=markers.get(name, "o"), markersize=9,
                        color=colors[name], markerfacecolor="none",
                        markeredgewidth=1.8, linestyle="none")
            d99, _u_e = sampled_delta99(profile)
            if d99:
                ax.axhline(d99 * u_tau / nu, color=colors[name],
                           linestyle=":", linewidth=0.8, alpha=0.6)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(left=0.001, right=1000.0)
        ax.set_ylim(0.03, 3e4)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_title(approach, fontsize=11)
        # Braces around the symbol: `\nu_t^+_{blockMesh}` is a double
        # subscript to mathtext, `{\nu_t^+}_{blockMesh}` is not.
        sym = "{" + symbol.split("=")[0].strip() + "}"
        ax.set_xlabel(rf"$|{sym} - {sym}_{{\mathrm{{{ERROR_BASELINE}}}}}|"
                      rf"\,/\,{sym}_{{\mathrm{{{ERROR_BASELINE}}}}}$  [%]",
                      fontsize=9)
        if col == 0:
            ax.set_ylabel(r"$y^+$")

    if not plotted:
        plt.close(fig)
        print(f"  [warn] nothing to plot for the {quantity} error")
        return False

    fig.suptitle(f"${symbol}$ error vs {ERROR_BASELINE} at x = {station:g} m  -  open "
                 f"symbol = top of the prism stack, dotted line = $\\delta_{{99}}$   "
                 f"(wall units cancel the $u_\\tau$ error - see Cf for that)",
                 fontsize=10)
    title = ", ".join(pp.swept_keys([index[n] for n in others if n in index])) or "run"
    entries = [(v, shade[v], markers.get(next(n for n in others if label[n] == v), "o"), "-")
               for v in ordered if v in plotted]
    rect = pp.legend_right(fig, [(title, entries)], width_in=LEGEND_WIDTH_IN)
    fig.tight_layout(rect=(rect[0], 0, rect[2], 0.90))
    path = os.path.join(out_dir, file_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {quantity + 'err':<6} -> {path}")
    return True


def plot_uplus_yplus_by_mesh(runs, station, index, markers, nu, out_dir):
    """Classical semi-log u+ over y+ at x = station, one panel per mesh approach.

    Every layer count of an approach in its own panel (coloured by the swept
    parameter, as in the error figures), with the baseline as a black dashed
    line for reference, the viscous sublayer and the log law.  An open symbol
    marks the top of each run's prism stack.
    """
    names = sorted(runs)
    approaches = sorted({index.get(n, {}).get("approach", "?") for n in names})
    if not approaches:
        return False
    colors, label, ordered, shade = variant_colors(names, index)

    base = sorted(n for n in names if index.get(n, {}).get("approach") == ERROR_BASELINE)
    base_curve = None
    if base and runs[base[0]].get(station) is not None:
        byp, bup = uplus_yplus(runs[base[0]][station], nu)
        if byp is not None:
            keep = byp > WALL_POINT_YPLUS
            base_curve = (byp[keep], bup[keep])

    yp_visc = np.logspace(-1, np.log10(12.0), 40)
    yp_log = np.logspace(np.log10(8.0), 4.2, 60)
    n_cols = len(approaches)
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(4.6 * n_cols + LEGEND_WIDTH_IN, 5.0),
                             squeeze=False, sharex=True, sharey=True)
    plotted = set()

    for col, approach in enumerate(approaches):
        ax = axes[0][col]
        ax.plot(yp_visc, yp_visc, color="0.4", linestyle=":", linewidth=1.2)
        ax.plot(yp_log, np.log(yp_log) / KAPPA + B_LOG, color="0.4",
                linestyle="-.", linewidth=1.2)
        if base_curve is not None and approach != ERROR_BASELINE:
            # On top (zorder): the well-resolved stacks lie right on it and
            # would otherwise hide it completely.
            ax.plot(*base_curve, color="black", linestyle="--", linewidth=1.3,
                    zorder=5)
        for name in names:
            if index.get(name, {}).get("approach") != approach:
                continue
            profile = runs[name].get(station)
            yp, up = uplus_yplus(profile, nu) if profile is not None else (None, None)
            if yp is None:
                continue
            order = np.argsort(yp)
            yp, up = yp[order], up[order]
            keep = yp > WALL_POINT_YPLUS
            colour = "black" if approach == ERROR_BASELINE else colors[name]
            ax.plot(yp[keep], up[keep], color=colour, linewidth=1.4)
            plotted.add(label[name])
            h = (index.get(name, {}).get("mesh") or {}).get("totalLayerThickness")
            u_tau = friction_velocity(profile)
            if h and u_tau:
                h_plus = h * u_tau / nu
                ax.plot([h_plus], [np.interp(h_plus, yp[keep], up[keep])],
                        marker=markers.get(name, "o"), markersize=9, color=colour,
                        markerfacecolor="none", markeredgewidth=1.8, linestyle="none")
        ax.set_xscale("log")
        ax.set_xlim(0.1, 2e4)
        ax.set_ylim(0, 35)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_title(approach, fontsize=11)
        ax.set_xlabel(r"$y^+$")
        if col == 0:
            ax.set_ylabel(r"$u^+$")

    if not plotted:
        plt.close(fig)
        print("  [warn] nothing to plot for u+/y+ by mesh")
        return False

    fig.suptitle(f"$u^+$ over $y^+$ at x = {station:g} m, per mesh approach  -  "
                 f"open symbol = top of the prism stack", fontsize=11)
    title = ", ".join(pp.swept_keys([index[n] for n in names if n in index])) or "run"
    entries = [(v, shade[v], markers.get(next(n for n in names if label[n] == v), "o"), "-")
               for v in ordered if v in plotted and v != "(none)"
               and any(label[n] == v and index.get(n, {}).get("approach") != ERROR_BASELINE
                       for n in names)]
    refs = [(f"{ERROR_BASELINE}", "black", None, "--"),
            (r"$u^+ = y^+$", "0.4", None, ":"),
            (rf"log law ($\kappa$={KAPPA}, B={B_LOG})", "0.4", None, "-.")]
    rect = pp.legend_right(fig, [(title, entries), ("Reference", refs)],
                           width_in=LEGEND_WIDTH_IN)
    fig.tight_layout(rect=(rect[0], 0, rect[2], 0.92))
    path = os.path.join(out_dir, "profile_uplus_yplus_by_mesh.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {'u+mesh':<6} -> {path}")
    return True


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    _, case_name, run_case_dir, source_case_dir, results_root = \
        pp.resolve_case_paths(script_dir)
    print("=" * 64)
    print(f"{case_name} - turbulence profile validation")
    print("=" * 64)

    val_dir = os.path.join(source_case_dir, "_validation_data")
    rstress_ref, _ = load_reference(val_dir, RSTRESS_REFERENCE)
    nut_ref, _ = load_reference(val_dir, NUT_REFERENCE)

    runs = {}
    for run, _latest in pp.solved_run_dirs(run_case_dir):
        profiles = read_run_profiles(run)
        if profiles:
            runs[run.name] = profiles
        else:
            print(f"  [warn] {run.name}: no sampleUyPlus profiles")
    if not runs:
        print(f"[warn] no runs with sampled profiles under {run_case_dir}")
        return 0
    print(f"Runs: {', '.join(sorted(runs))}")

    first = os.path.join(run_case_dir, sorted(runs)[0])
    u_inf, nu, _rho = pp.detect_case_parameters(first)
    # Colour = mesh approach, marker = swept parameter, both from the manifest.
    manifest = pp.load_manifest(run_case_dir)
    index = pp.combo_index(manifest)
    colors, markers, sections = pp.study_styles(runs, manifest)
    print("Styles: " + "; ".join(
        f"{title or 'runs'}: {', '.join(lbl for lbl, _c, _m in entries)}"
        for title, entries in sections))

    out_dir = os.path.join(results_root, "validation")
    os.makedirs(out_dir, exist_ok=True)
    written = 0

    for quantity, candidates, axis_label in RSTRESS_QUANTITIES:
        written += plot_quantity(
            quantity, candidates, axis_label, runs, rstress_ref, RSTRESS_LABEL,
            os.path.join(out_dir, f"profile_{quantity}.png"),
            u_inf, nu, colors, markers, sections)

    written += plot_uplus_yplus(
        runs, os.path.join(out_dir, "profile_uplus_yplus.png"), nu, colors,
        markers, sections)

    written += plot_uplus_yplus_by_mesh(runs, 1.0, index, markers, nu, out_dir)

    # Error against the baseline for u+ and every turbulence quantity sampled.
    # For an eddy-viscosity study Rxx = Ryy = Rzz = 2k/3 exactly (see the
    # normal-stress note above), so their error figures would repeat k's.
    transported_R = any("R_xx" in prof for profiles in runs.values()
                        for prof in profiles.values())
    for quantity, symbol, value_fn, file_name in ERROR_QUANTITIES:
        if quantity in NORMAL_STRESSES and not transported_R:
            print(f"  [info] {quantity} error skipped - no transported R, so it "
                  f"equals the k error (2k/3)")
            continue
        written += plot_profile_error(runs, 1.0, index, markers, nu, out_dir,
                                      quantity, symbol, value_fn, file_name)

    # Eddy viscosity of the SA family only - see the module docstring.
    quantity, candidates, axis_label = NUT_QUANTITY
    sa_runs = {name: profiles for name, profiles in runs.items()
               if name.split("__")[0].startswith(SA_FAMILY)}
    if sa_runs:
        written += plot_quantity(
            quantity, candidates, axis_label, sa_runs, nut_ref, NUT_LABEL,
            os.path.join(out_dir, "profile_nut_sa.png"),
            u_inf, nu, colors, markers, sections)
    else:
        print("  [info] no SA-family run - skipping the nu_t comparison "
              f"(models present: {', '.join(sorted({n.split('__')[0] for n in runs}))})")

    print(f"\nProfile validation complete ({written} figure(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
