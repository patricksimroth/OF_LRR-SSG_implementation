"""
postpro.py – Post-processing for the turbulentFlatPlate case.
Reads sampled wall-normal profiles at x = 1 m (from the sampleUyPlus
function object) and computes u+ / y+ for comparison against the
theoretical inner-law and log-law curves provided in _validation_data/.
Can be called either from the source _postProcessing/ directory or from the
run/turbulentFlatPlate/ directory (same path-resolution logic as the
backwards-step case).
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
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MU_LAM = 1.831e-05   # Pa·s  – from thermophysicalProperties (const transport)
RHO = 1.2747          # kg/m³ – from thermophysicalProperties (const transport)
RHO=1
# ---------------------------------------------------------------------------
# Path helpers  (mirrored from backwards_step_tut)
# ---------------------------------------------------------------------------
def resolve_case_paths(script_dir):
    """Return (base_path, case_name, run_case_dir, source_case_dir, results_root)."""
    if os.path.basename(script_dir) == "_postProcessing":
        source_case_dir = os.path.dirname(script_dir)
        case_name = os.path.basename(source_case_dir)
        base_path = os.path.dirname(source_case_dir)
        run_case_dir = os.path.join(base_path, "run", case_name)
    else:
        case_name = os.path.basename(script_dir)
        parent_dir = os.path.dirname(script_dir)
        if os.path.basename(parent_dir) == "run":
            base_path = os.path.dirname(parent_dir)
            run_case_dir = script_dir
            source_case_dir = os.path.join(base_path, case_name)
        else:
            base_path = parent_dir
            run_case_dir = os.path.join(base_path, "run", case_name)
            source_case_dir = script_dir
    results_root = os.path.join(base_path, "results", case_name)
    return base_path, case_name, run_case_dir, source_case_dir, results_root
def latest_time_directory(parent):
    """Return the path to the highest-numbered time-step subdirectory."""
    if not parent or not os.path.isdir(parent):
        return None
    candidates = []
    for item in os.listdir(parent):
        try:
            candidates.append((float(item), os.path.join(parent, item)))
        except ValueError:
            pass
    return max(candidates, key=lambda x: x[0])[1] if candidates else None
# ---------------------------------------------------------------------------
# Convergence plots  (identical helpers to backwards_step_tut)
# ---------------------------------------------------------------------------
def plot_minMax_data(file_path, output_path):
    line_regex = re.compile(
        r"^\s*(?P<time>\d+\.?\d*)\s+"
        r"(?P<field>[^\s\t]+)\s+"
        r"(?P<min>[e\d\.\-\+]+)\s+\(.*\)\s+"
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
        plt.figure(figsize=(10, 6))
        plt.plot(df["Time"], df["Min"], label="Min Value")
        plt.plot(df["Time"], df["Max"], label="Max Value")
        plt.title(f"Field Variation Over Time: {field}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        safe = "plot_" + field.replace("(", "_").replace(")", "_") + ".png"
        plt.savefig(os.path.join(output_dir, safe))
        plt.close()

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
    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    plt.figure(figsize=(12, 7))
    for field in field_prefixes:
        init_col, final_col = f"{field}_initial", f"{field}_final"
        if init_col in df_full.columns and final_col in df_full.columns:
            color = next(colors)
            plt.plot(df_full["Time"], df_full[init_col], label=f"{field} Initial",
                     color=color, linestyle=":")
            plt.plot(df_full["Time"], df_full[final_col], label=f"{field} Final",
                     color=color, linestyle="-")
    plt.yscale("log")
    plt.title("Solver Residuals")
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    output_dir = os.path.join(output_path, "convergence_plots")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "plot_residuals.png"))
    plt.close()
    print(f"  Residual plot → {output_dir}/plot_residuals.png")
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
            if not line or line.startswith("#") or line.startswith("variables"):
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
def _read_numeric_rows(file_path):
    rows = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append([float(v) for v in line.split()])
            except ValueError:
                pass
    return rows
def parse_sample_profile(sample_dir):
    """
    Find the x1_profile_*.xy file in *sample_dir* and return a DataFrame
    with columns [y, Ux, Uy, Uz, yPlus, rho].
    The sampleUyPlus function object writes fields (U yPlus rho), so the
    raw columns are:  y  Ux Uy Uz  yPlus  rho
    """
    if not sample_dir or not os.path.isdir(sample_dir):
        return None
    # Find any xy file produced by the x1_profile set
    candidates = sorted(glob.glob(os.path.join(sample_dir, "x1_profile*.xy")))
    if not candidates:
        print(f"  [warn] no x1_profile*.xy in {sample_dir}")
        return None
    file_path = candidates[0]
    rows = _read_numeric_rows(file_path)
    if not rows:
        print(f"  [warn] empty sample file: {file_path}")
        return None

    arr = np.array(rows)

    if arr.shape[1] < 6:
        print(f"  [warn] unexpected column count ({arr.shape[1]}) in {file_path}")
        return None
    try:
        df = pd.DataFrame({
            "y":     arr[:, 0],
            "rho":   arr[:, 1],
            "yPlus": arr[:, 2],
            "Ux":    arr[:, 3],
            "Uy":    arr[:, 4],
            "Uz":    arr[:, 5],
            "tau_x": arr[:, 6],
            "tau_y": arr[:, 7],
            "tau_z": arr[:, 8],
        })
    except IndexError:
        df = pd.DataFrame({
            "y":     arr[:, 0],
            "yPlus": arr[:, 1],
            "Ux":    arr[:, 2],
            "Uy":    arr[:, 3],
            "Uz":    arr[:, 4],
            "tau_x": arr[:, 5],
            "tau_y": arr[:, 6],
            "tau_z": arr[:, 7],
        })
    # Keep only the boundary layer (positive Ux, positive y)
    # df = df[df["y"] > 0].copy()
    df.sort_values("y", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Loaded {len(df)} sample points from {os.path.basename(file_path)}")
    print(df[["y", "Ux", "yPlus"]].head(5))
    print(df[["y", "Ux", "yPlus"]].tail(5))
    return df

def parse_probes_data(probes_dir):
    """
    Parse probes data from the probes function object.
    Returns a DataFrame with time and velocity components at probe locations.

    The probe file format is:
        # Probe 0 (x y z)
        # Probe 1 (x y z)
        # Time   0   1   2
        time   (Ux0 Uy0 Uz0) (Ux1 Uy1 Uz1) (Ux2 Uy2 Uz2)
        ...
    """
    if not probes_dir or not os.path.isdir(probes_dir):
        return None

    # Find the probes file (named "U" in the probes directory)
    candidates = sorted(glob.glob(os.path.join(probes_dir, "U")))
    if not candidates:
        print(f"  [warn] no probes U file in {probes_dir}")
        return None

    file_path = candidates[0]

    parsed_data = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Extract time (first number on the line)
            time_match = re.match(r'^\s*(\d+\.?\d*)', line)
            if not time_match:
                continue
            time_val = float(time_match.group(1))

            # Extract all parenthesized groups (probe data) - handles cases
            # where parentheses are split across whitespace tokens
            probe_matches = re.findall(r'\(([^)]+)\)', line)

            for probe_idx, probe_str in enumerate(probe_matches):
                try:
                    u_vals = [float(x) for x in probe_str.split()]
                    if len(u_vals) >= 3:
                        parsed_data.append({
                            'time': time_val,
                            'probe_idx': probe_idx,
                            'Ux': u_vals[0],
                            'Uy': u_vals[1],
                            'Uz': u_vals[2]
                        })
                except ValueError:
                    continue

    if not parsed_data:
        print(f"  [warn] could not parse valid probe data from {file_path}")
        return None

    # Convert to DataFrame and pivot so each probe has its own columns
    df = pd.DataFrame(parsed_data)
    if 'probe_idx' in df.columns and len(df) > 0:
        pivot_df = df.pivot_table(
            index='time', columns='probe_idx', values=['Ux', 'Uy', 'Uz']
        )
        pivot_df.columns = [
            f'probe_{idx}_{field}' for field, idx in pivot_df.columns
        ]
        df = pivot_df.reset_index()

    print(f"  Loaded {len(df)} time steps with probe data from "
          f"{os.path.basename(file_path)}")
    return df
# ---------------------------------------------------------------------------
# u+ / y+ computation
# ---------------------------------------------------------------------------
def compute_uplus_yplus(df, mu=MU_LAM, rho=RHO):
    """
    Compute u+ and y+ from the sampled profile DataFrame.
    Strategy
    --------
    * nu = mu / rho  (cell-local, so it varies with rho along the profile)
    * u_tau is computed from the yPlus field and the physical y distance:
          u_tau(cell) = yPlus(cell) * nu(cell) / y(cell)
      We use the mean of the near-wall cells (y+ < 200) for a single u_tau
      estimate per profile, which is robust to mesh non-uniformity.
    * u+  = Ux / u_tau
    * y+  = u_tau * y / nu   (equivalent to the original yPlus field)
    """
    df = df.copy()
    df["nu"] = mu / rho
    # u_tau from yPlus definition: y+ = u_tau * y / nu  → u_tau = y+ * nu / y
    df["u_tau_local"] = np.sqrt(np.abs(df["tau_x"] / rho))
    # Average u_tau over near-wall region (yPlus < 200) to get a single value
    # near_wall = df.iloc[1]   # first point
    u_tau = max(df["u_tau_local"])
    print(f"  u_tau = {u_tau:.5f} m/s")
    df["u_plus"] = df["Ux"] / u_tau
    df["y_plus"] = df["y"] * u_tau / df["nu"]
    print(df[["y", "Ux", "yPlus", "u_plus", "y_plus", "u_tau_local"]].head(5))
    print(df[["y", "Ux", "yPlus", "u_plus", "y_plus", "u_tau_local"]].tail(5))
    return df, u_tau
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_uplus_yplus(theory, models_data, output_dir):
    """
    Plot u+ vs y+ (on log scale) for all models against the theoretical curves.
    Parameters
    ----------
    theory      : dict from parse_theory_data()
    models_data : {model_name: DataFrame with u_plus, y_plus columns}
    output_dir  : directory where the PNG is saved
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    # Theory curves (convert from log10(y+) to y+)
    u_plus_th = theory["u_plus"]
    ax.plot(10**theory["log_y_inner"], u_plus_th,
            color="green", linewidth=1.2, linestyle="-", label="viscous sublayer (u⁺ = y⁺)")
    ax.plot(10**theory["log_y_log"],   u_plus_th,
            color="green", linewidth=1.2, linestyle="--", label="log law (κ=0.41, B=5.0)")
    # Simulation profiles
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    sim_colors = [c for c in prop_cycle if c.lower() not in ("#00ff00", "#008000")]
    for idx, (model_name, df) in enumerate(sorted(models_data.items())):
        color = sim_colors[idx % len(sim_colors)]
        valid = df[df["y_plus"] > 0].copy()
        ax.plot(valid["y_plus"], valid["u_plus"],
                color=color, linewidth=1.8, label=f"{model_name}")
    ax.set_xscale("log")
    ax.set_xlabel("y+", fontsize=13)
    ax.set_ylabel("u+", fontsize=13)
    ax.set_title("Law of the Wall – u+ vs y+ (log scale)  [x = 1 m]", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5, which="both")
    ax.set_xlim(0.1, 1e5)
    ax.set_ylim(0, 30)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "plot_uplus_yplus.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  u+/y+ plot → {out_path}")
    return out_path

def plot_Ux_vs_y(models_data, output_dir):
    """
    Plot Ux vs y (linear scale) for all models.
    Parameters
    ----------
    models_data : {model_name: DataFrame with Ux, y columns}
    output_dir  : directory where the PNG is saved
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    # Q1: both axes linear; y-axis focused on the near-wall region
    ax.set_xscale("linear")
    ax.set_yscale("linear")

    # Simulation profiles
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    sim_colors = [c for c in prop_cycle if c.lower() not in ("#00ff00", "#008000")]
    csv_frames = []
    for idx, (model_name, df) in enumerate(sorted(models_data.items())):
        color = sim_colors[idx % len(sim_colors)]
        # Q2: keep the wall point at y = 0 (do not drop it)
        ax.plot(df["Ux"], df["y"],
                color=color, linewidth=1.8, label=f"{model_name}")
        # Q4 (self-chosen): also export the plotted data to CSV for reuse
        csv_frames.append(pd.DataFrame({
            "model": model_name,
            "y": df["y"].values,
            "Ux": df["Ux"].values,
        }))

    # Q3: no theoretical/reference profile is plotted in this figure yet
    ax.set_xlabel("Ux (m/s)", fontsize=13)
    ax.set_ylabel("y (m)", fontsize=13)
    ax.set_title("Velocity Profile – Ux vs y  [x = 1 m]", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5, which="both")
    ax.set_ylim(0, 0.02)
    ax.set_xlim(0, 70)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "plot_Ux_vs_y.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    if csv_frames:
        csv_path = os.path.join(output_dir, "Ux_vs_y_profiles.csv")
        pd.concat(csv_frames, ignore_index=True).to_csv(csv_path, index=False)
        print(f"  Ux vs y CSV  → {csv_path}")

    print(f"  Ux vs y plot → {out_path}")
    return out_path

def plot_velocity_evolution_at_points(models_data, output_dir):
    """
    Plot velocity evolution at the probe points over time.
    Creates one subplot per probe location.
    """
    # Determine number of probes from the data
    n_probes = 0
    for df in models_data.values():
        if df is not None:
            probe_columns = [
                col for col in df.columns
                if col.startswith('probe_') and col.endswith('_Ux')
            ]
            if probe_columns:
                n_probes = max(n_probes, len(probe_columns))

    if n_probes == 0:
        print(f"  [warn] no probe data to plot")
        return None

    # Y-positions matching the probes (from controlDict)
    y_positions = [0.005, 0.01, 0.015]

    # Create subplots - one per probe
    fig, axes = plt.subplots(1, n_probes, figsize=(5 * n_probes, 5), sharey=True)
    if n_probes == 1:
        axes = [axes]

    fig.suptitle('Velocity Evolution at Probe Points (x=1m)', fontsize=14)

    # Color cycle for different models
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = [c for c in prop_cycle if c.lower() not in ("#00ff00", "#008000")]

    # Plot for each model
    for idx, (model_name, df) in enumerate(models_data.items()):
        if df is None or 'time' not in df.columns:
            continue
        color = colors[idx % len(colors)]

        for probe_idx in range(n_probes):
            ux_col = f'probe_{probe_idx}_Ux'
            if ux_col in df.columns:
                y_label = y_positions[probe_idx] if probe_idx < len(y_positions) else probe_idx
                axes[probe_idx].plot(
                    df['time'], df[ux_col],
                    color=color, label=model_name, linewidth=1.5
                )
                axes[probe_idx].set_title(f'Probe {probe_idx + 1} (y={y_label} m)')
                axes[probe_idx].set_xlabel('Time')
                axes[probe_idx].grid(True, linestyle='--', alpha=0.5)

    axes[0].set_ylabel('Ux Velocity')
    axes[0].legend(loc='upper right')

    # Save the plot
    output_path = os.path.join(output_dir, "velocity_evolution_at_points.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Velocity evolution plot → {output_path}")
    return output_path
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("turbulentFlatPlate – post-processing")
    print("=" * 60)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path, case_name, run_case_dir, source_case_dir, results_root = \
        resolve_case_paths(script_dir)
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
    # Discover model folders
    model_folders = []
    if os.path.isdir(run_case_dir):
        model_folders = [
            item for item in os.listdir(run_case_dir)
            if os.path.isdir(os.path.join(run_case_dir, item, "postProcessing"))
        ]
    if not model_folders:
        print(f"[warn] no model folders found under {run_case_dir}")
    models_uplus = {}   # {model_name: DataFrame(u_plus, y_plus)}
    models_probes = {}  # {model_name: DataFrame(time, Ux, Uy, Uz)}
    for model_name in sorted(model_folders):
        print(f"\n--- Model: {model_name} ---")
        case_folder  = os.path.join(run_case_dir, model_name)
        output_path  = os.path.join(results_root, model_name)
        os.makedirs(output_path, exist_ok=True)
        # --- Convergence plots ---
        min_max_path = os.path.join(case_folder, "postProcessing/minMax/0/fieldMinMax.dat")
        if os.path.exists(min_max_path):
            print("  Plotting minMax convergence…")
            plot_minMax_data(min_max_path, output_path)
        solver_path = os.path.join(case_folder, "postProcessing/solverInfo/0/solverInfo.dat")
        if os.path.exists(solver_path):
            print("  Plotting residuals…")
            plot_residual_data(solver_path, output_path)
        # --- u+/y+ profile ---
        sample_base = os.path.join(case_folder, "postProcessing", "sampleUyPlus")
        sample_dir  = latest_time_directory(sample_base)
        if sample_dir is None:
            print(f"  [warn] no sampleUyPlus output found – skipping profile for {model_name}")
            continue
        df_raw = parse_sample_profile(sample_dir)
        if df_raw is None or df_raw.empty:
            continue
        df_prof, u_tau = compute_uplus_yplus(df_raw, mu=MU_LAM)
        models_uplus[model_name] = df_prof
        
        # --- Probes data ---
        probes_base = os.path.join(case_folder, "postProcessing", "probes")
        probes_dir = latest_time_directory(probes_base)
        if probes_dir is not None:
            df_probes = parse_probes_data(probes_dir)
            if df_probes is not None:
                models_probes[model_name] = df_probes
            else:
                print(f"  [warn] failed to parse probes data for {model_name}")
        else:
            print(f"  [warn] no probes output found for {model_name}")

    # --- Combined u+/y+ validation plot ---
    if theory and models_uplus:
        val_dir = os.path.join(results_root, "validation")
        print(f"\nGenerating u+/y+ validation plot in {val_dir} …")
        plot_uplus_yplus(theory, models_uplus, val_dir)
        
        # --- Ux vs y profile plot ---
        print(f"Generating Ux vs y profile plot in {val_dir} …")
        plot_Ux_vs_y(models_uplus, val_dir)
    
    # --- Velocity evolution plot ---
    if models_probes:
        val_dir = os.path.join(results_root, "validation")
        print(f"\nGenerating velocity evolution plot in {val_dir} …")
        plot_velocity_evolution_at_points(models_probes, val_dir)
    else:
        if not theory:
            print("[warn] skipping u+/y+ plot – no theory data")
        if not models_uplus:
            print("[warn] skipping u+/y+ plot – no simulation profiles found")
            print("       (run the simulation first so that postProcessing/sampleUyPlus/ exists)")
    print("\nPost-processing complete.")