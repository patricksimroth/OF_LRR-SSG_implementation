"""
postpro.py – Post-processing for the turbulentFlatPlate case.

Per run, into results/<case>/<study>/<run>/convergence_plots/:
    residuals.png            initial and final residuals per iteration
    minmax_<field>.png       field minimum and maximum per iteration
    Ux_profile_history.png   Ux(y) at x = 1 m at 20/40/60/80/100 % of the run

Per study, into summary/wall_law/:
    uplus_x1m.png            u+ over y+ at x = 1 m against the viscous sublayer,
                             the log law (kappa 0.41, B 5.0) and Coles' profile;
                             rows h+, columns approach, one line per core size,
                             the same model's blockMesh run dashed (as
                             profile_validation.py).  u_tau is each run's own.
Can be called either from the source _postProcessing/ directory or from the
run/turbulentFlatPlate/ directory.
"""
import glob
import io
import itertools
import os
import re

# pyrefly: ignore [missing-import]
import matplotlib

matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# pyrefly: ignore [missing-import]
import numpy as np

# pyrefly: ignore [missing-import]
import pandas as pd

import _pp_common as pp
import profile_validation as pv

pp.report_style()

# (label, unit) of the minMax fields; unknown ones keep their OpenFOAM name.
FIELD_LABELS = {
    "p": (r"$p$", "Pa"), "mag(U)": (r"$|U|$", "m/s"), "T": (r"$T$", "K"),
    "e": (r"$e$", "J/kg"), "k": (r"$k$", "m$^2$/s$^2$"), "omega": (r"$\omega$", "1/s"),
    "epsilon": (r"$\varepsilon$", "m$^2$/s$^3$"), "nut": (r"$\nu_t$", "m$^2$/s"),
    "nuTilda": (r"$\tilde\nu$", "m$^2$/s"), "rho": (r"$\rho$", "kg/m$^3$"),
}
RESIDUAL_LABELS = {"Ux": r"$U_x$", "Uy": r"$U_y$", "Uz": r"$U_z$", "p": r"$p$", "e": r"$e$",
                   "h": r"$h$", "k": r"$k$", "omega": r"$\omega$", "epsilon": r"$\varepsilon$",
                   "nuTilda": r"$\tilde\nu$", "Rxx": r"$R_{xx}$", "Rxy": r"$R_{xy}$",
                   "Rxz": r"$R_{xz}$", "Ryy": r"$R_{yy}$", "Ryz": r"$R_{yz}$", "Rzz": r"$R_{zz}$"}
SINGLE_SIZE = (pp.REPORT_WIDTH_IN, 3.0)
# ---------------------------------------------------------------------------
# Convergence plots  (identical helpers to backwards_step_tut)
# ---------------------------------------------------------------------------
def plot_minMax_data(file_path, output_path):
    line_regex = re.compile(
        r"^\s*(?P<time>\d+\.?\d*)\s+"
        r"(?P<field>[^\s\t]+)\s+"
        r"(?P<min>[e\d\.\-\+]+)\s+\(.*\)\s+(?:\d+\s+)?"
        r"(?P<max>[e\d\.\-\+]+)\s+\(.*\)"
    )
    field_data = {}
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = line_regex.match(line)
            if m:
                field = m.group("field")
                field_data.setdefault(field, []).append(
                    {"Time": float(m.group("time")),
                     "Min":  float(m.group("min")),
                     "Max":  float(m.group("max"))}
                )
    output_dir = os.path.join(output_path, "convergence_plots")
    os.makedirs(output_dir, exist_ok=True)
    for field, entries in field_data.items():
        df = pd.DataFrame(entries)
        symbol, unit = FIELD_LABELS.get(field, (field, None))
        fig, ax = plt.subplots(figsize=SINGLE_SIZE)
        ax.plot(df["Time"], df["Max"], color=pp.CORE_COLORS[1], label="maximum")
        ax.plot(df["Time"], df["Min"], color=pp.CORE_COLORS[2], label="minimum")
        ax.set_xlabel("iteration")
        ax.set_ylabel(f"{symbol} [{unit}]" if unit else symbol)
        rect = pp.legend_below(fig, ax.get_legend_handles_labels()[0])
        fig.tight_layout(rect=rect)
        safe = re.sub(r"[^A-Za-z0-9]+", "", field.replace("mag(", "mag"))
        fig.savefig(os.path.join(output_dir, f"minmax_{safe}.png"))
        plt.close(fig)


def plot_residual_data(file_path, output_path):
    if not os.path.exists(file_path):
        print(f"  [warn] residual file not found: {file_path}")
        return
    with open(file_path) as f:
        lines = f.readlines()
    header_parts = []
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("#"):
            parts = line.strip("#").strip().split()
            if "Time" in parts or any("_initial" in p for p in parts):
                header_parts.extend(parts)
        else:
            data_start = i
            break
    df_full = pd.read_csv(
        io.StringIO("".join(lines[data_start:])),
        sep=r"\s+", names=header_parts, engine="python",
    )
    field_prefixes = sorted({col[:-8] for col in df_full.columns if col.endswith("_initial")})
    colors = itertools.cycle(pp.CORE_COLORS + ("#2a78d6", "#1baf7a", "#9a3fb5", pp.INK))
    fig, ax = plt.subplots(figsize=SINGLE_SIZE)
    handles = []
    for field in field_prefixes:
        init_col, final_col = f"{field}_initial", f"{field}_final"
        if init_col in df_full.columns and final_col in df_full.columns:
            color = next(colors)
            ax.plot(df_full["Time"], df_full[init_col], color=color, linewidth=1.0)
            ax.plot(df_full["Time"], df_full[final_col], color=color, linewidth=0.6,
                    linestyle=":")
            handles.append(pp.line_handle(RESIDUAL_LABELS.get(field, field), color))
    handles += [pp.line_handle("initial", pp.MUTED, "-"), pp.line_handle("final", pp.MUTED, ":")]
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual")
    rect = pp.legend_below(fig, handles, ncol=min(len(handles), 8))
    fig.tight_layout(rect=rect)
    output_dir = os.path.join(output_path, "convergence_plots")
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "residuals.png"))
    plt.close(fig)
    print(f"  Residual plot → {output_dir}/residuals.png")
# ---------------------------------------------------------------------------
# Validation data parser
# ---------------------------------------------------------------------------
def parse_theory_data(validation_file):
    """
    Parse u+/y+ theory file.
    Columns: u+, log10(y+)_inner, log10(y+)_log, log10(y+)_spal1, log10(y+)_spal2
    Returns a dict with arrays 'u_plus', 'log_y_inner', 'log_y_log'.
    """
    rows = []
    with open(validation_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "variables")):
                continue
            try:
                rows.append([float(v) for v in line.split()])
            except ValueError:
                pass
    if not rows:
        return None
    arr = np.array(rows)
    return {
        "u_plus":     arr[:, 0],
        "log_y_inner": arr[:, 1],   # inner law (viscous sublayer)
        "log_y_log":   arr[:, 2],   # log law (κ=0.41, B=5.0)
    }
# ---------------------------------------------------------------------------
# Sample-file parser
# ---------------------------------------------------------------------------
# Columns per field token in a sets file name.  OpenFOAM orders the fields by
# type (scalars, vectors, symmTensors), alphabetically within a type, and
# skips fields the model does not have (e.g. k/omega for SA), so the column
# layout differs per model and must come from the file name.
# The field -> column mapping and the filename parser live in _pp_common,
# shared with profile_validation.py.
_PROFILE_FIELD_COLUMNS = pp.PROFILE_FIELD_COLUMNS
_profile_columns_from_filename = pp.profile_columns_from_filename


def parse_sample_profile(sample_dir):
    """
    Find the main-line x1_profile_*.xy file in *sample_dir* and return a DataFrame.
    Columns are `y` followed by the sampled fields in the order OpenFOAM writes
    them (by type, alphabetical within a type; fields the model lacks are
    absent), e.g. SA: nut, rho, yPlus, U, wallShearStress, turbulenceProperties:R.
    Column names come from the file name when it is in the
    `<set>_profile_<fields>.xy` pattern and matches the column count, otherwise
    from the column count directly (9 = with rho, 8 = without; the old
    sampleUyPlusConv layout).
    """
    if not sample_dir or not os.path.isdir(sample_dir):
        return None
    # Find any xy file produced by the x1_profile set
    candidates = sorted(glob.glob(os.path.join(sample_dir, "x1_profile*.xy")))
    if not candidates:
        print(f"  [warn] no x1_profile*.xy in {sample_dir}")
        return None
    file_path = candidates[0]
    arr = pp.read_numeric_rows(file_path)
    if arr is None:
        print(f"  [warn] empty sample file: {file_path}")
        return None

    cols = _profile_columns_from_filename(file_path)
    if cols is not None and len(cols) != arr.shape[1]:
        cols = None
    if cols is None:
        if arr.shape[1] == 9:
            cols = ["y", "rho", "yPlus", "Ux", "Uy", "Uz", "tau_x", "tau_y", "tau_z"]
        elif arr.shape[1] == 8:
            cols = ["y", "yPlus", "Ux", "Uy", "Uz", "tau_x", "tau_y", "tau_z"]
        else:
            print(f"  [warn] unexpected column count ({arr.shape[1]}) in {file_path}")
            return None

    # Old method (kept for reference): tried the 9-column mapping and fell
    # back to the 8-column one via IndexError.
    # try:
    #     df = pd.DataFrame({
    #         "y": arr[:, 0], "rho": arr[:, 1], "yPlus": arr[:, 2],
    #         "Ux": arr[:, 3], "Uy": arr[:, 4], "Uz": arr[:, 5],
    #         "tau_x": arr[:, 6], "tau_y": arr[:, 7], "tau_z": arr[:, 8],
    #     })
    # except IndexError:
    #     df = pd.DataFrame({
    #         "y": arr[:, 0], "yPlus": arr[:, 1], "Ux": arr[:, 2],
    #         "Uy": arr[:, 3], "Uz": arr[:, 4],
    #         "tau_x": arr[:, 5], "tau_y": arr[:, 6], "tau_z": arr[:, 7],
    #     })

    df = pd.DataFrame(arr, columns=cols)
    df.sort_values("y", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Loaded {len(df)} sample points from {os.path.basename(file_path)}")
    print(df[["y", "Ux", "yPlus"]].head(5))
    print(df[["y", "Ux", "yPlus"]].tail(5))
    return df

# ---------------------------------------------------------------------------
# u+ / y+ computation
# ---------------------------------------------------------------------------
def compute_uplus_yplus(df, nu, rho=None):
    """
    Compute u+ and y+ from the sampled profile DataFrame.

    * u_tau(cell) = sqrt(|tau_x(cell)| / rho(cell)); rho is taken from the
      sampled `rho` column when present (compressible case), otherwise from
      the scalar fallback `rho` from pp.detect_case_parameters().
    * u_tau = max(u_tau_local) over the profile.
    * u+  = Ux / u_tau
    * y+  = u_tau * y / nu, with nu = mu/rho_inf from the case's own
      thermophysicalProperties (pp.detect_case_parameters).  mu is constant
      here (`transport const`), so the only approximation is using the
      freestream rho rather than the local one.
    """
    df = df.copy()
    if "rho" in df.columns:
        rho_field = df["rho"].astype(float)
    else:
        rho_field = float(rho if rho is not None else 1.0)
    df["u_tau_local"] = np.sqrt(np.abs(df["tau_x"] / rho_field))
    u_tau = max(df["u_tau_local"])
    print(f"  u_tau = {u_tau:.5f} m/s")
    df["u_plus"] = df["Ux"] / u_tau
    df["y_plus"] = df["y"] * u_tau / nu
    print(df[["y", "Ux", "yPlus", "u_plus", "y_plus", "u_tau_local"]].head(5))
    print(df[["y", "Ux", "yPlus", "u_plus", "y_plus", "u_tau_local"]].tail(5))
    return df, u_tau
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_uplus_yplus(theory, models_data, output_dir, coles=None, manifest=None):
    """u+ over y+ at x = 1 m: rows h+, columns approach, one line per core size
    and the same model's blockMesh run dashed (profile_validation.layout).
    Without a manifest every run is a line in one panel."""
    os.makedirs(output_dir, exist_ok=True)
    index = pp.combo_index(manifest)
    groups = pv.layout(sorted(models_data), index)
    refs = [(10 ** theory["log_y_inner"], theory["u_plus"], "-", r"$u^+ = y^+$"),
            (10 ** theory["log_y_log"], theory["u_plus"], "--", "log law")]
    if coles is not None:
        refs.append((coles[:, 0], coles[:, 1], "-.", "Coles"))
    written = []
    for group in groups:
        rows, cols, cells = group["rows"], group["cols"], group["cells"]
        style = {lab: (pp.CORE_COLORS[i % len(pp.CORE_COLORS)],
                       pp.CORE_MARKERS[i % len(pp.CORE_MARKERS)])
                 for i, lab in enumerate(group["order"])}
        reference = group["reference"]
        fig, axes = pp.grid_figure(len(rows), len(cols), row_height=1.9,
                                   sharex=True, sharey=True)
        plotted = set()
        for i, row in enumerate(rows):
            for j, approach in enumerate(cols):
                ax = axes[i][j]
                for x, u, ls, _ in refs:
                    ax.plot(x, u, color=pp.EDGE, linestyle=ls, linewidth=0.8)
                names = sorted(cells.get((row, approach), []),
                               key=lambda n: group["order"].index(group["label"][n]))
                for slot, name in enumerate(names):
                    df = models_data[name]
                    valid = df[df["y_plus"] > 0]
                    lab = group["label"][name]
                    color, marker = style[lab]
                    ax.plot(valid["y_plus"], valid["u_plus"], color=color, marker=marker,
                            markersize=3.5, markeredgecolor="white", markeredgewidth=0.4,
                            linewidth=1.0, markevery=max(1, len(valid) // 6))
                    plotted.add(lab)
                if reference:
                    df = models_data[reference]
                    valid = df[df["y_plus"] > 0]
                    ax.plot(valid["y_plus"], valid["u_plus"], **pp.BASELINE_STYLE)
                h = next(((index.get(n, {}).get("mesh") or {}).get("totalLayerThickness")
                          for n in names), None)
                if h and reference:
                    df = models_data[reference]
                    ax.axvline(h * df.attrs["u_tau"] / df.attrs["nu"], **pp.STACK_TOP_STYLE)
                ax.set_xscale("log")
                ax.set_xlim(0.1, 1e5)
                ax.set_ylim(0, 30)
        pp.label_grid(axes, [pp.fmt_hplus(r) if r is not None else "" for r in rows],
                      cols, r"$u^+$", r"$y^+$")
        handles = [pp.line_handle(lab, style[lab][0], "-", style[lab][1])
                   for lab in group["order"] if lab in plotted]
        if reference:
            handles.append(pp.line_handle(pp.ERROR_BASELINE, pp.INK, "--"))
            handles.append(pp.line_handle(pp.STACK_TOP_LABEL, pp.MUTED, ":"))
        handles += [pp.line_handle(label, pp.EDGE, ls) for _, _, ls, label in refs]
        rect = pp.legend_below(fig, handles)
        fig.tight_layout(rect=rect, h_pad=0.6, w_pad=0.4)
        suffix = f"_{group['model']}" if len(groups) > 1 else ""
        out_path = os.path.join(output_dir, f"uplus_x1m{suffix}.png")
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  u+/y+ plot → {out_path}")
        written.append(out_path)
    return written


def plot_velocity_convergence(model_name, case_folder, output_path):
    """
    Per-model convergence figure: one Ux-vs-y profile for each of the
    20/40/60/80/100 % checkpoints of the run (closest available
    sampleUyPlusConv snapshot), light -> dark in time order.
    Saved to <output_path>/convergence_plots/Ux_profile_history.png.
    """
    conv_base = os.path.join(case_folder, "postProcessing", "sampleUyPlusConv")
    if not os.path.isdir(conv_base):
        print(f"  [warn] no sampleUyPlusConv output for {model_name} – skipping convergence plot")
        return None
    times = []
    for item in os.listdir(conv_base):
        try:
            times.append(float(item))
        except ValueError:
            pass
    if not times:
        print(f"  [warn] no sampleUyPlusConv snapshots for {model_name} – skipping convergence plot")
        return None

    end_time = max(times)

    profiles = []
    for pct in (20, 40, 60, 80, 100):
        target = end_time * pct / 100.0
        t = min(times, key=lambda t: abs(t - target))
        if t in [p[0] for p in profiles]:
            continue
        t_name = str(int(t)) if t == int(t) else str(t)
        df = parse_sample_profile(os.path.join(conv_base, t_name))
        if df is None or df.empty:
            print(f"  [warn] could not parse sampleUyPlusConv snapshot t={t:g} for {model_name}")
            continue
        label = f"t = {t:g}" + (f" ({pct:g}%)" if t == target else "")
        profiles.append((t, df, label))

    if not profiles:
        print(f"  [warn] no parseable sampleUyPlusConv snapshots for {model_name}")
        return None

    profiles.sort(key=lambda p: p[0])

    # Data-driven limits: full U range and the 99 %-U boundary-layer edge
    u_max = max(df["Ux"].max() for _, df, _ in profiles)
    y_edges = []
    for _, df, _ in profiles:
        d = df.sort_values("y")
        above = d[d["Ux"] >= 0.99 * d["Ux"].max()]
        if not above.empty:
            y_edges.append(float(above["y"].min()))
    y_max = 1.05 * max(y_edges) if y_edges else max(df["y"].max() for _, df, _ in profiles)

    fig, ax = plt.subplots(figsize=SINGLE_SIZE)
    cmap = plt.get_cmap("Blues")
    n = len(profiles)
    for i, (t, df, label) in enumerate(profiles):
        frac = 0.35 + 0.65 * (i / max(n - 1, 1))
        ax.plot(df["Ux"], 1e3 * df["y"], color=cmap(frac),
                linewidth=1.3 if i == n - 1 else 1.0, label=f"iteration {t:g}")
    ax.set_xlabel(r"$U_x$ [m/s]")
    ax.set_ylabel(r"$y$ [mm]")
    ax.set_xlim(0, 1.05 * u_max)
    ax.set_ylim(0, 1e3 * y_max)
    rect = pp.legend_below(fig, ax.get_legend_handles_labels()[0])
    fig.tight_layout(rect=rect)

    conv_dir = os.path.join(output_path, "convergence_plots")
    os.makedirs(conv_dir, exist_ok=True)
    out_path = os.path.join(conv_dir, "Ux_profile_history.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Velocity convergence plot → {out_path}")
    return out_path
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("turbulentFlatPlate – post-processing")
    print("=" * 60)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path, case_name, run_case_dir, source_case_dir, results_root = \
        pp.resolve_case_paths(script_dir)
    print(f"run_case_dir   : {run_case_dir}")
    print(f"source_case_dir: {source_case_dir}")
    print(f"results_root   : {results_root}")
    # Validation data
    validation_file = os.path.join(
        source_case_dir, "_validation_data", "u+y+theory.dat"
    )
    theory = None
    if os.path.exists(validation_file):
        theory = parse_theory_data(validation_file)
        print(f"Loaded theory data: {len(theory['u_plus'])} points")
    else:
        print(f"[warn] validation file not found: {validation_file}")
    coles_file = os.path.join(source_case_dir, "_validation_data", "tmr_coles_upyp.dat")
    coles = pp.read_numeric_rows(coles_file) if os.path.exists(coles_file) else None
    if coles is not None:
        print(f"Loaded Coles profile: {len(coles)} points")
    else:
        print(f"[warn] Coles profile not found: {coles_file}")
    # Discover model folders.  Crashed runs keep a postProcessing/ tree, so
    # they are dropped here rather than silently plotted (see
    # _pp_common.run_completed).
    model_folders = []
    if os.path.isdir(run_case_dir):
        model_folders = [
            item for item in os.listdir(run_case_dir)
            if os.path.isdir(os.path.join(run_case_dir, item, "postProcessing"))
        ]
    model_folders, crashed = pp.split_completed(run_case_dir, model_folders)
    pp.warn_failed(crashed, "the u+/y+ validation")
    if not model_folders:
        print(f"[warn] no model folders found under {run_case_dir}")
    models_uplus = {}   # {model_name: DataFrame(u_plus, y_plus)}
    for model_name in sorted(model_folders):
        print(f"\n--- Model: {model_name} ---")
        case_folder  = os.path.join(run_case_dir, model_name)
        output_path  = os.path.join(results_root, model_name)
        os.makedirs(output_path, exist_ok=True)
        # --- Convergence plots (reused while the run and the plotting code are unchanged) ---
        min_max_path = os.path.join(case_folder, "postProcessing/minMax/0/fieldMinMax.dat")
        solver_path = os.path.join(case_folder, "postProcessing/solverInfo/0/solverInfo.dat")
        conv_base = os.path.join(case_folder, "postProcessing", "sampleUyPlusConv")
        conv_dir = os.path.join(output_path, "convergence_plots")
        conv_stamp = os.path.join(conv_dir, ".stamp")
        conv_key = pp.source_key(plot_minMax_data, plot_residual_data,
                                 plot_velocity_convergence, parse_sample_profile)
        conv_fresh = pp.outputs_fresh(
            conv_stamp, glob.glob(os.path.join(conv_dir, "*.png")),
            [p for p in (min_max_path, solver_path, conv_base) if os.path.exists(p)], conv_key)
        if conv_fresh:
            print("  Convergence plots unchanged, reused")
        else:
            if os.path.exists(conv_stamp):
                os.remove(conv_stamp)
            for stale in glob.glob(os.path.join(conv_dir, "*.png")):
                os.remove(stale)
            if os.path.exists(min_max_path):
                print("  Plotting minMax convergence…")
                plot_minMax_data(min_max_path, output_path)
            if os.path.exists(solver_path):
                print("  Plotting residuals…")
                plot_residual_data(solver_path, output_path)
        # --- u+/y+ profile ---
        sample_base = os.path.join(case_folder, "postProcessing", "sampleUyPlus")
        sample_dir  = pp.latest_time_directory(sample_base)
        if sample_dir is None:
            print(f"  [warn] no sampleUyPlus output found – skipping profile for {model_name}")
            continue
        df_raw = parse_sample_profile(sample_dir)
        if df_raw is None or df_raw.empty:
            continue
        u_inf, nu, rho = pp.detect_case_parameters(case_folder)
        print(f"  detected U_inf = {u_inf:g} m/s, nu = {nu:g} m²/s, rho = {rho:.6f} kg/m³")
        df_prof, u_tau = compute_uplus_yplus(df_raw, nu, rho)
        df_prof.attrs.update(u_tau=u_tau, nu=nu)
        models_uplus[model_name] = df_prof

        # --- Per-model velocity convergence plot (sampleUyPlusConv snapshots) ---
        if not conv_fresh:
            plot_velocity_convergence(model_name, case_folder, output_path)
            if os.path.isdir(conv_dir):
                pp.write_stamp(conv_stamp, conv_key)

    # --- u+/y+ per approach and h+ ---
    if theory and models_uplus:
        out_dir = os.path.join(results_root, pp.SUMMARY_DIR, "wall_law")
        print(f"\nGenerating u+/y+ plot in {out_dir} …")
        plot_uplus_yplus(theory, models_uplus, out_dir, coles=coles,
                         manifest=pp.load_manifest(run_case_dir))
    else:
        if not theory:
            print("[warn] skipping u+/y+ plot – no theory data")
        if not models_uplus:
            print("[warn] skipping u+/y+ plot – no simulation profiles found")
            print("       (run the simulation first so that postProcessing/sampleUyPlus/ exists)")
    print("\nPost-processing complete.")