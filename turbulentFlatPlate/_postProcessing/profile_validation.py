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
                  ref_label, output_path, u_inf, nu, colors, markers):
    """One figure: a 2 x 5 grid of stations, every run plus the reference."""
    stations = sorted(reference) if reference else sorted(
        {s for profiles in runs.values() for s in profiles})
    if not stations:
        print(f"  [warn] no stations for {quantity}")
        return False
    n_cols = 5
    n_rows = int(np.ceil(len(stations) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.6 * n_rows),
                             squeeze=False)
    handles, labels = [], []
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
                            linewidth=1.2,
                            label=DISPLAY_NAME.get(name, name))
            plotted_any = True
            if line.get_label() not in labels:
                handles.append(line)
                labels.append(line.get_label())

        if reference and station in reference and quantity in reference[station]:
            ref = reference[station]
            mask = ref["y"] > 0
            line, = ax.plot(ref[quantity][mask], ref["y"][mask] * 1e3,
                            color="black", linestyle="--", linewidth=1.3,
                            alpha=0.8, label=ref_label)
            plotted_any = True
            if ref_label not in labels:
                handles.append(line)
                labels.append(ref_label)

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
        note = ("  -  " + ", ".join(DISPLAY_NAME.get(n, n) for n in fallback)
                + ": modelled stresses (turbulenceProperties:R), not transported")
    fig.suptitle(f"{axis_label.split('  ')[0]} profiles vs {ref_label}{note}",
                 fontsize=12)
    fig.legend(handles, labels, loc="lower center",
               ncol=min(4, len(labels)), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  {quantity:<6} -> {output_path}")
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
    colors, markers = assign_plot_styles(runs)

    out_dir = os.path.join(results_root, "validation")
    os.makedirs(out_dir, exist_ok=True)
    written = 0

    for quantity, candidates, axis_label in RSTRESS_QUANTITIES:
        written += plot_quantity(
            quantity, candidates, axis_label, runs, rstress_ref, RSTRESS_LABEL,
            os.path.join(out_dir, f"profile_{quantity}.png"),
            u_inf, nu, colors, markers)

    # Eddy viscosity of the SA family only - see the module docstring.
    quantity, candidates, axis_label = NUT_QUANTITY
    sa_runs = {name: profiles for name, profiles in runs.items()
               if name.split("__")[0].startswith(SA_FAMILY)}
    if sa_runs:
        written += plot_quantity(
            quantity, candidates, axis_label, sa_runs, nut_ref, NUT_LABEL,
            os.path.join(out_dir, "profile_nut_sa.png"),
            u_inf, nu, colors, markers)
    else:
        print("  [info] no SA-family run - skipping the nu_t comparison "
              f"(models present: {', '.join(sorted({n.split('__')[0] for n in runs}))})")

    print(f"\nProfile validation complete ({written} figure(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
