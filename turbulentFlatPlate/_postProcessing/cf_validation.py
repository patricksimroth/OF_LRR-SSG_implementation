#!/usr/bin/env python3
"""
cf_validation.py – Cf validation against the NASA TMR reference for the
turbulentFlatPlate case.

This script mirrors the methodology used in
``../flat_plate_andre_v2/scripts/validate_matrix.py``:

* Sampling – the bottomWall patch is sampled with a ``surfaces`` function
  object (``yPlus_reference_2``) and one wall-normal profile at ``x=0.97008``
  is sampled with a ``sets`` function object (``sampleProfile``).  The raw
  ``wallShearStress`` and ``yPlus`` fields are written as ASCII ``.raw``
  tables in the same ``postProcessing/<func>/<latestTime>`` layout used by
  the reference.
* Reference data – ONE curve for every run: the TMR CFL3D/FUN3D compressible
  flat-plate ``Cf(x)`` for the SST model, ``_validation_data/tmr_cf_plate_sst.dat``
  (see TMR_REFERENCE_FILE).  It is applied to every run regardless of that run's
  turbulence model, so ``ref_error_pct`` is the distance to the SST reference,
  not a like-for-like model error.
* Calculation – ``Cf = 2 |tau_w| / (rho U_inf^2)`` is computed from the
  sampled ``wallShearStress``.  ``U_inf``, ``nu`` and ``rho`` are detected
  per model from the case files (see ``_pp_common.detect_case_parameters``).
  The friction velocity ``u_tau`` is computed from ``Cf`` at the reference
  station.
* Validation – the local error against the TMR reference is reported at the
  reference station ``x = 0.97008`` and the log-law RMS error of the
  ``u+``/``y+`` profile is computed in the range ``y+ in [30, 300]``.

The figure layout differs from the reference: instead of one subplot per
model (the reference produces a row of subplots because it also spans
multiple grids), this script produces **a single ``Cf`` versus ``x`` plot
containing every available model** plus the TMR reference.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys

# pyrefly: ignore [missing-import]
import matplotlib

matplotlib.use("Agg")

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# pyrefly: ignore [missing-import]
import numpy as np

import _pp_common as pp

# ---------------------------------------------------------------------------
# Default constants for the test_cases/turbulentFlatPlate setup
# ---------------------------------------------------------------------------
# Fallbacks only; U_inf, nu and rho are detected per model by
# pp.detect_case_parameters() (0/U, plus thermophysicalProperties + 0/p + 0/T
# for the compressible cases here: nu = mu/rho).  These defaults mirror the
# TMR flat plate, M = 0.2 and Re/m = 5e6 -> U = 69.4 m/s, nu = 1.388e-05.
U_INF_DEFAULT = 69.4
NU_DEFAULT = 1.388e-05
# Reference station used by the flat_plate_andre_v2 validation.
REF_STATION_X = 0.97008

# The single Cf reference every run is compared against.  TMR publishes SA,
# SST and SSG/LRR-RSM-w2012 curves for this case; only the SST one is used
# here, as the most trusted of the three.  Its CFL3D and FUN3D zones differ by
# 0.01 % at the reference station, so they are averaged into one curve.
# _validation_data/tmr_cf_plate.dat (the SA curve) stays in the repo unused.
TMR_REFERENCE_FILE = "tmr_cf_plate_sst.dat"
TMR_REFERENCE_LABEL = "TMR SST (CFL3D/FUN3D)"
# log-law constants
KAPPA = 0.41
B = 5.0
LOG_YPLUS_RANGE = (30.0, 300.0)

# Pretty-print model names (the on-disk folders are LRR, SSG, kEpsilon, ...).
DISPLAY_NAME = {
    "LRR": "LRR",
    "SSG": "SSG",
    "kEpsilon": "k-epsilon",
    "kOmegaSST": "k-omega SST",
    "kOmegaSST_andre": "k-omega SST (andre)",
    "SSGLRROmega2019": "SSG/LRR-omega 2019 (SD)",
    "SpalartAllmarasSALSA": "SA SALSA",
    "SpalartAllmaras": "SA",
}

# Markers per model for the combined plot (chosen to be readable in print).
MODEL_MARKERS = {
    "LRR": "s",
    "SSG": "X",
    "kEpsilon": "D",
    "kOmegaSST": "^",
    "kOmegaSST_andre": "v",
    "SSGLRROmega2019": "P",
    "SpalartAllmarasSALSA": "*",
    "SpalartAllmaras": "o",
}

# Colours per model for the combined plot.
MODEL_COLORS = {
    "LRR": "tab:blue",
    "SSG": "tab:orange",
    "kEpsilon": "tab:green",
    "kOmegaSST": "tab:red",
    "kOmegaSST_andre": "tab:purple",
    "SSGLRROmega2019": "tab:brown",
    "SpalartAllmarasSALSA": "tab:cyan",
    "SpalartAllmaras": "tab:olive",
}


# Extra figure width reserved for the right-hand legend, in inches.
LEGEND_WIDTH_IN = 3.0

# Set by main() so the plot helpers can colour by mesh approach and mark by the
# swept parameter instead of giving each of N runs its own colour.
STUDY_MANIFEST = None


def assign_plot_styles(models):
    """(color, marker, legend_sections) per run.

    With a study manifest: colour = mesh approach, marker = swept parameter
    (pp.study_styles), so a 4 x 5 sweep needs 9 legend entries, not 20.
    Without one: the static per-model registry, then a tab10 cycle for names it
    does not know (a plain run_simulation.sh variant).
    """
    models = sorted(set(models))
    if STUDY_MANIFEST:
        return pp.study_styles(models, STUDY_MANIFEST)
    colors = dict(MODEL_COLORS)
    markers = dict(MODEL_MARKERS)
    cmap = plt.get_cmap("tab10")
    marker_cycle = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">")
    for i, name in enumerate([m for m in models if m not in colors]):
        colors[name] = cmap(i % cmap.N)
        markers[name] = marker_cycle[i % len(marker_cycle)]
    return colors, markers, [(None, [(DISPLAY_NAME.get(m, m), colors[m], markers[m])
                                     for m in models])]


# ---------------------------------------------------------------------------
# Reference data parsing (Techplot .dat format used by the TMR repository)
# ---------------------------------------------------------------------------
# parse_tecplot_zones moved to _pp_common (shared with profile_validation.py).
parse_tecplot_zones = pp.parse_tecplot_zones


def load_reference(val_dir):
    """The single TMR reference curve as (x, cf), or None if the file is absent.

    The per-code zones (CFL3D, FUN3D) are averaged onto the union of their x
    stations: they agree to 0.01 % here, so one curve carries the reference
    without drawing every zone separately.
    """
    path = os.path.join(val_dir, TMR_REFERENCE_FILE)
    if not os.path.exists(path):
        print(f"  [warn] TMR reference not found: {path}")
        return None
    zones = parse_tecplot_zones(path)
    if not zones:
        print(f"  [warn] no zones parsed from {path}")
        return None
    x = np.unique(np.concatenate([c[:, 0] for c in zones.values()]))
    cf = np.mean([np.interp(x, c[:, 0], c[:, 1]) for c in zones.values()], axis=0)
    print(f"  reference   : {TMR_REFERENCE_FILE} "
          f"({', '.join(zones)}; {len(x)} stations)")
    return np.column_stack((x, cf))


def compute_cf_from_shear(tau_components, rho, u_inf=U_INF_DEFAULT):
    """Compute Cf = 2 |tau| / (rho U_inf^2).

    ``tau_components`` is an (N, 3) or (N, >=3) array of (tau_x, tau_y, tau_z).
    Only the first three columns are used.  This matches the reference's
    ``2*np.linalg.norm(shear[order, 3:5], axis=1)`` formulation, generalised
    to the in-plane (x, y) wall-shear magnitude.
    """
    tau_in_plane = np.linalg.norm(tau_components[:, :2], axis=1)
    return 2.0 * tau_in_plane / (rho * u_inf * u_inf)


# ---------------------------------------------------------------------------
# Per-model processing
# ---------------------------------------------------------------------------
def discover_models(run_case_dir):
    """Return a sorted list of model directory names whose ``postProcessing``
    contains a wallShearStress sample on the bottomWall patch.

    Prefers the dedicated ``yPlus_reference_2``/``samplePlate`` directories
    and falls back to anything that has a usable ``wallShearStress_*.raw``
    file.  Runs whose solver crashed are excluded: they leave partly written
    samples behind that would enter the validation plot as if they were
    converged results (see ``_pp_common.run_completed``).
    """
    if not os.path.isdir(run_case_dir):
        return []
    found = []
    crashed = []
    for entry in sorted(os.listdir(run_case_dir)):
        case_dir = os.path.join(run_case_dir, entry)
        post = os.path.join(case_dir, "postProcessing")
        if not os.path.isdir(post):
            continue
        has_data = False
        for func in ("yPlus_reference_2", "samplePlate"):
            tdir = pp.latest_time_directory(os.path.join(post, func))
            if tdir is None:
                continue
            for fname in os.listdir(tdir):
                if fname.startswith("wallShearStress") and fname.endswith(".raw"):
                    has_data = True
                    break
            if has_data:
                break
        if not has_data:
            continue
        if pp.run_completed(case_dir):
            found.append(entry)
        else:
            crashed.append(entry)
    pp.warn_failed(crashed, "the Cf validation")
    return found


def find_wall_shear_file(case_dir):
    """Locate the most recent wallShearStress_<surface>.raw file."""
    for func in ("yPlus_reference_2", "samplePlate"):
        tdir = pp.latest_time_directory(os.path.join(case_dir, "postProcessing", func))
        if tdir is None:
            continue
        for fname in sorted(os.listdir(tdir)):
            if fname.startswith("wallShearStress") and fname.endswith(".raw"):
                return os.path.join(tdir, fname), fname
    return None, None


def find_yplus_file(case_dir):
    """Locate the most recent yPlus_<surface>.raw file."""
    for func in ("yPlus_reference_2", "samplePlate"):
        tdir = pp.latest_time_directory(os.path.join(case_dir, "postProcessing", func))
        if tdir is None:
            continue
        for fname in sorted(os.listdir(tdir)):
            if fname.startswith("yPlus") and fname.endswith(".raw"):
                return os.path.join(tdir, fname), fname
    return None, None


def find_profile_file(case_dir):
    """Locate the x_0p97008_U.xy file produced by sampleProfile."""
    tdir = pp.latest_time_directory(os.path.join(case_dir, "postProcessing", "sampleProfile"))
    if tdir is None:
        return None
    for fname in sorted(os.listdir(tdir)):
        if fname.endswith(".xy"):
            return os.path.join(tdir, fname)
    return None


def process_model(model_name, case_dir, reference, ref_station_x=REF_STATION_X):
    """Build the per-model record used for the plots and the summary table."""
    record = {"model": model_name, "display": DISPLAY_NAME.get(model_name, model_name)}

    # Detect the freestream conditions for this case.
    u_inf, nu, rho = pp.detect_case_parameters(case_dir,
                                                fallback_u_inf=U_INF_DEFAULT,
                                                fallback_nu=NU_DEFAULT)
    record["u_inf"] = u_inf
    record["nu"] = nu
    record["rho"] = rho

    shear_path, shear_fname = find_wall_shear_file(case_dir)
    if shear_path is None:
        print(f"  [warn] {model_name}: no wallShearStress .raw found")
        return None
    shear = pp.read_numeric_rows(shear_path)
    if shear is None:
        print(f"  [warn] {model_name}: empty wallShearStress file")
        return None

    # Sort by x so that downstream plots are monotonic.
    order = np.argsort(shear[:, 0])
    x = shear[order, 0]
    cf = compute_cf_from_shear(shear[order, 3:6], rho=rho, u_inf=u_inf)
    record["shear_source"] = os.path.relpath(shear_path, case_dir)
    record["x"] = x
    record["cf"] = cf

    # Cf at the reference station (closest sampled x to REF_STATION_X).
    station = int(np.argmin(np.abs(x - ref_station_x)))
    record["station_x"] = float(x[station])
    record["cf_station"] = float(cf[station])

    # Wall y+ on the bottomWall patch.
    yp_path, yp_fname = find_yplus_file(case_dir)
    if yp_path is not None:
        yp = pp.read_numeric_rows(yp_path)
        if yp is not None:
            order_y = np.argsort(yp[:, 0])
            record["yp_x"] = yp[order_y, 0]
            record["yp"] = yp[order_y, 3]
            record["yp_min"] = float(record["yp"].min())
            record["yp_max"] = float(record["yp"].max())
            record["yp_source"] = os.path.relpath(yp_path, case_dir)

    # Distance to the TMR SST reference at the station.  Applied to every run:
    # the run dirs of a study are named <model>__<approach>, so keying the
    # reference by model name (as this used to do) matched nothing and left
    # cf_tmr empty for every study run.
    if reference is None:
        record["ref_error_pct"] = float("nan")
    else:
        cf_tmr = float(np.interp(record["station_x"],
                                 reference[:, 0], reference[:, 1]))
        record["cf_tmr"] = cf_tmr
        record["ref_error_pct"] = 100.0 * (record["cf_station"] - cf_tmr) / cf_tmr

    # Wall-normal profile at x=0.97008 -> log-law RMS.
    profile_path = find_profile_file(case_dir)
    if profile_path is not None:
        prof = pp.read_numeric_rows(profile_path)
        if prof is not None and prof.shape[0] > 1:
            y = prof[:, 0]
            u = prof[:, 1]
            # u_tau = sqrt(|tau_w|/rho) = sqrt(Cf/2) * U_inf
            utau = math.sqrt(record["cf_station"] / 2.0) * u_inf
            yplus = y * utau / nu
            uplus = u / utau
            mask = (yplus >= LOG_YPLUS_RANGE[0]) & (yplus <= LOG_YPLUS_RANGE[1])
            if mask.any():
                err = uplus[mask] - (np.log(yplus[mask]) / KAPPA + B)
                record["log_rms"] = float(np.sqrt(np.mean(err ** 2)))
                record["log_max"] = float(np.max(np.abs(err)))
            else:
                record["log_rms"] = float("nan")
                record["log_max"] = float("nan")
            record["profile_source"] = os.path.relpath(profile_path, case_dir)

    return record


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_cf_combined(records, reference, output_path, ref_station_x=REF_STATION_X):
    """Single Cf-vs-x plot with every model and the TMR reference overlaid.

    This is the test_cases equivalent of
    flat_plate_andre_v2's ``cf_plate_matrix.png`` (one subplot per model),
    here collapsing the row of subplots into a single comparison plot.
    """
    fig, ax = plt.subplots(figsize=(9.5 + LEGEND_WIDTH_IN, 5.5))
    colors, markers, sections = assign_plot_styles(
        r["model"] for r in records if r is not None)
    plotted_any = False
    for record in records:
        if record is None or "x" not in record:
            continue
        marker = markers.get(record["model"], "o")
        color = colors.get(record["model"], "k")
        # Keep only the physically meaningful domain (skip leading edge spike).
        mask = (record["x"] >= 0.05) & (record["x"] <= 1.95)
        ax.plot(record["x"][mask], record["cf"][mask],
                color=color, marker=marker, markersize=3.5,
                markevery=max(1, mask.sum() // 25),
                linewidth=1.2, label=record["display"])
        plotted_any = True

    # The one TMR reference, drawn once and labelled.
    if reference is not None:
        mask = (reference[:, 0] >= 0.05) & (reference[:, 0] <= 1.95)
        ax.plot(reference[mask, 0], reference[mask, 1],
                color="black", linestyle="--", linewidth=1.2, alpha=0.7,
                label=TMR_REFERENCE_LABEL)
        plotted_any = True

    if plotted_any:
        # Mark the reference station used for the table.
        for record in records:
            if record is None or "station_x" not in record:
                continue
            color = colors.get(record["model"], "k")
            ax.axvline(record["station_x"], color=color, alpha=0.10, linewidth=0.6)
        ax.axvline(ref_station_x, color="grey", alpha=0.25,
                   linewidth=0.8, linestyle=":",
                   label=f"reference station x = {ref_station_x:g}")

    ax.set_xlabel(r"$x$  [m]")
    ax.set_ylabel(r"$C_f$")
    ax.set_title("Local skin-friction coefficient – all runs vs. TMR SST")
    ax.grid(True, alpha=0.3, which="both")
    rect = pp.legend_right(fig, sections + [("Reference", [
        (TMR_REFERENCE_LABEL, "black", None)])], width_in=LEGEND_WIDTH_IN)
    fig.tight_layout(rect=rect)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"  Cf plot     -> {output_path}")


def plot_yplus_combined(records, output_path):
    """Single wall-y+-versus-x plot with every model overlaid.

    Uses the same sampled ``yPlus_bottomWall.raw`` data as the summary
    table (``yPlus_reference_2``, falling back to ``samplePlate``).  The
    y-axis is logarithmic because the model matrix spans wall-resolved
    (y+ < 1) and wall-function (y+ > 30) meshes simultaneously.
    """
    fig, ax = plt.subplots(figsize=(9.5 + LEGEND_WIDTH_IN, 5.5))
    colors, markers, sections = assign_plot_styles(
        r["model"] for r in records if r is not None)
    plotted_any = False
    yp_all = []
    for record in records:
        if record is None or "yp_x" not in record:
            continue
        marker = markers.get(record["model"], "o")
        color = colors.get(record["model"], "k")
        # Keep only the physically meaningful domain (skip leading edge spike).
        mask = (record["yp_x"] >= 0.05) & (record["yp_x"] <= 1.95)
        if not mask.any():
            continue
        yp_all.append(record["yp"][mask])
        ax.plot(record["yp_x"][mask], record["yp"][mask],
                color=color, marker=marker, markersize=3.5,
                markevery=max(1, mask.sum() // 25),
                linewidth=1.2, label=record["display"])
        plotted_any = True

    if plotted_any:
        ax.set_yscale("log")
        # Automatic y range from the model data only, so the reference lines
        # cannot stretch the axis into empty regions.
        yp_cat = np.concatenate(yp_all)
        yp_cat = yp_cat[yp_cat > 0]
        if yp_cat.size:
            ax.set_ylim(yp_cat.min() / 1.5, yp_cat.max() * 1.5)
            # Mesh-regime reference lines: y+ = 1 (viscous sublayer
            # resolution) and y+ = 30 (log-law wall-function range), drawn
            # only when they fall inside the data range.
            for y_ref, style, label in ((1.0, "--", r"$y^+ = 1$"),
                                        (30.0, ":", r"$y^+ = 30$")):
                if yp_cat.min() <= y_ref <= yp_cat.max():
                    ax.axhline(y_ref, color="grey", linestyle=style,
                               linewidth=0.8, alpha=0.55, label=label)

    ax.set_xlabel(r"$x$  [m]")
    ax.set_ylabel(r"$y^+$")
    ax.set_title("Wall y+ along the plate – all models")
    ax.grid(True, alpha=0.3, which="both")
    rect = pp.legend_right(fig, sections, width_in=LEGEND_WIDTH_IN)
    fig.tight_layout(rect=rect)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"  y+ plot     -> {output_path}")


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------
def write_summary_csv(records, output_path):
    fields = ("model", "station_x", "cf_station", "cf_tmr", "ref_error_pct",
              "yp_min", "yp_max", "log_rms", "log_max",
              "shear_source", "profile_source")
    with open(output_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for record in records:
            if record is None:
                continue
            row = []
            for key in fields:
                value = record.get(key)
                if value is None:
                    row.append("")
                elif isinstance(value, float) and math.isnan(value):
                    row.append("n/a")
                else:
                    row.append(value)
            writer.writerow(row)
    print(f"  CSV summary -> {output_path}")


def write_summary_txt(records, output_path):
    lines = [f"reference: {TMR_REFERENCE_LABEL} - applied to every run, "
             f"regardless of its turbulence model",
             "model, station_x, Cf(station), TMR Cf, error %, "
             "wall y+ min/max, log-law RMS, log-law max"]
    for record in records:
        if record is None:
            continue

        def fmt(value, spec=".8g"):
            if value is None:
                return "n/a"
            if isinstance(value, float) and math.isnan(value):
                return "n/a"
            return format(value, spec)

        yp_min = record.get("yp_min")
        yp_max = record.get("yp_max")
        yp_range = "n/a"
        if yp_min is not None and yp_max is not None:
            yp_range = f"{yp_min:.2f}/{yp_max:.2f}"
        lines.append(
            f"{record['display']}, {fmt(record.get('station_x'), '.5f')}, "
            f"{fmt(record.get('cf_station'))}, "
            f"{fmt(record.get('cf_tmr'))}, "
            f"{fmt(record.get('ref_error_pct'), '.3f')}, "
            f"{yp_range}, "
            f"{fmt(record.get('log_rms'), '.4f')}, "
            f"{fmt(record.get('log_max'), '.4f')}"
        )
    with open(output_path, "w") as stream:
        stream.write("\n".join(lines) + "\n")
    print(f"  TXT summary -> {output_path}")
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path, case_name, run_case_dir, source_case_dir, results_root = \
        pp.resolve_case_paths(script_dir)
    print("=" * 64)
    print(f"{case_name} – Cf validation (TMR reference)")
    print("=" * 64)
    print(f"run_case_dir   : {run_case_dir}")
    print(f"source_case_dir: {source_case_dir}")
    print(f"results_root   : {results_root}")

    # Reference data: the TMR SST flat-plate curve, used for every run.
    val_dir = os.path.join(source_case_dir, "_validation_data")
    reference = load_reference(val_dir)

    # Discover and process models.
    models = discover_models(run_case_dir)
    if not models:
        print(f"[warn] no models with sampled wallShearStress found under "
              f"{run_case_dir}")
        sys.exit(1)
    print(f"Discovered {len(models)} model(s): {', '.join(models)}")

    records = []
    for model_name in models:
        print(f"\n--- {model_name} ---")
        case_dir = os.path.join(run_case_dir, model_name)
        record = process_model(model_name, case_dir, reference)
        if record is None:
            continue
        records.append(record)
        print(f"  Cf(x = {record['station_x']:.4f}) = {record['cf_station']:.6f}")
        if not math.isnan(record.get("ref_error_pct", float("nan"))):
            print(f"  TMR error   : {record['ref_error_pct']:+.3f}%  "
                  f"(TMR Cf = {record['cf_tmr']:.6f})")
        if not math.isnan(record.get("log_rms", float("nan"))):
            print(f"  log-law RMS : {record['log_rms']:.4f}  "
                  f"(max |err| = {record['log_max']:.4f})")

    if not records:
        print("[warn] no records to plot")
        sys.exit(1)

    # Output directory (always the same one the rest of the pipeline writes to).
    out_dir = os.path.join(results_root, "validation")
    os.makedirs(out_dir, exist_ok=True)

    global STUDY_MANIFEST
    STUDY_MANIFEST = pp.load_manifest(run_case_dir)
    plot_cf_combined(records, reference,
                     os.path.join(out_dir, "cf_validation.png"))
    plot_yplus_combined(records, os.path.join(out_dir, "yplus_validation.png"))
    write_summary_csv(records, os.path.join(out_dir, "cf_validation.csv"))
    summary_lines = write_summary_txt(records, os.path.join(out_dir, "cf_validation.txt"))
    print("\n" + "\n".join(summary_lines))
    print("\nCf validation complete.")


if __name__ == "__main__":
    main()
