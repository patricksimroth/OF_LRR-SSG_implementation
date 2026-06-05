import itertools
import glob
import io
import os
import re

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Convergence plots
# ---------------------------------------------------------------------------

def plot_minMax_data(file_path, output_path):
    line_regex = re.compile(
        r'^\s*(?P<time>\d+\.?\d*)\s+'
        r'(?P<field>[^\s\t]+)\s+'
        r'(?P<min>[e\d\.\-\+]+)\s+\(.*\)\s+'
        r'(?P<max>[e\d\.\-\+]+)\s+\(.*\)'
    )

    field_data = {}
    print(f"Reading file: {file_path}")
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = line_regex.match(line)
            if m:
                field = m.group('field')
                field_data.setdefault(field, []).append({
                    'Time': float(m.group('time')),
                    'Min':  float(m.group('min')),
                    'Max':  float(m.group('max')),
                })

    output_dir = os.path.join(output_path, 'convergence_plots')
    os.makedirs(output_dir, exist_ok=True)

    for field, entries in field_data.items():
        df = pd.DataFrame(entries)
        print(f"Processing field: {field} ({len(df)} entries)")

        plt.figure(figsize=(10, 6))
        plt.plot(df['Time'], df['Min'], label='Min Value')
        plt.plot(df['Time'], df['Max'], label='Max Value')
        plt.title(f'Field Variation Over Time: {field}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)

        safe_name = 'plot_' + field.replace('(', '_').replace(')', '_') + '.png'
        plt.savefig(os.path.join(output_dir, safe_name))
        plt.close()


def plot_residual_data(file_path, output_path):
    print(f"Reading residual file: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    with open(file_path, 'r') as f:
        lines = f.readlines()

    header_parts = []
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('#'):
            parts = line.strip('#').strip().split()
            if 'Time' in parts or any('_initial' in p for p in parts):
                header_parts.extend(parts)
        else:
            data_start = i
            break

    df_full = pd.read_csv(
        io.StringIO("".join(lines[data_start:])),
        sep=r'\s+', names=header_parts, engine='python',
    )

    field_prefixes = sorted({col[:-8] for col in df_full.columns if col.endswith('_initial')})
    colors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    plt.figure(figsize=(12, 7))
    for field in field_prefixes:
        init_col, final_col = f"{field}_initial", f"{field}_final"
        if init_col in df_full.columns and final_col in df_full.columns:
            color = next(colors)
            plt.plot(df_full['Time'], df_full[init_col], label=f'{field} Initial', color=color, linestyle=':')
            plt.plot(df_full['Time'], df_full[final_col], label=f'{field} Final',   color=color, linestyle='-')

    plt.yscale('log')
    plt.title('Solver Residuals Trend')
    plt.xlabel('Time / Iteration')
    plt.ylabel('Residual')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    output_dir = os.path.join(output_path, 'convergence_plots')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'plot_residuals.png'))
    plt.close()
    print(f"Residual plot saved to {output_dir}/plot_residuals.png")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def get_x_coordinate_from_filename(file_name):
    m = re.search(r'x_by_h_([mM]?\d+)(?:_p|_)', file_name)
    if not m:
        return None

    token = m.group(1)
    sign = -1 if token[0].lower() == 'm' else 1
    num_str = token[1:] if sign == -1 else token

    if not num_str.isdigit():
        return None

    if len(num_str) == 1:
        x_val = float(num_str)
    elif len(num_str) == 2 and num_str[0] == '0':
        x_val = float(num_str[1])
    elif len(num_str) >= 3 and num_str[0] == '0':
        x_val = float(num_str[1:-1] + '.' + num_str[-1])
    else:
        x_val = float(num_str)

    return sign * x_val


def _read_numeric_rows(file_path):
    rows = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                rows.append([float(p) for p in line.split()])
            except ValueError:
                pass
    return rows


def read_Uref_and_rho_from_file(file_path, default_rho=1.225, default_Uref=44.2):
    """Read rho and velocity magnitude from the Uref sample file."""
    if not os.path.exists(file_path):
        return default_rho, default_Uref

    rows = _read_numeric_rows(file_path)
    if rows:
        values = rows[0]
        file_name = os.path.basename(file_path)
        try:
            if '_p_rho_U' in file_name and len(values) >= 6:
                rho = values[2]
                Uref = float(np.linalg.norm(values[3:6]))
            elif '_p_U_rho' in file_name and len(values) >= 6:
                rho = values[5]
                Uref = float(np.linalg.norm(values[2:5]))
            elif len(values) >= 6:
                rho = values[2]
                Uref = float(np.linalg.norm(values[-3:]))
            elif 'rho' not in file_name:
                Uref = float(np.linalg.norm(values[-3:]))
                rho = default_rho
            else:
                rho, Uref = default_rho, default_Uref

            if 0.5 <= rho <= 2.5 and Uref > 10:
                print(f"Extracted from {file_path}: rho={rho}, Uref={Uref}")
                return rho, Uref
        except (TypeError, ValueError):
            pass

    print(f"Using fallback values: rho={default_rho}, Uref={default_Uref}")
    return default_rho, default_Uref


def parse_x_by_h_file(file_path, H=0.0127, Uref=44.2):
    """Parse profile samples. File name fields define the column expansion."""
    rows = _read_numeric_rows(file_path)
    if not rows: 
        return pd.DataFrame()

    num_cols_in_file = len(rows[0])
    file_name = os.path.basename(file_path)

    if 'U_turbulenceProperties' in file_name or 'U_R' in file_name:
        cols = ['y', 'U', 'V', 'W', 'uu', 'uv', 'uw', 'vv', 'vw', 'ww']
    elif 'rho' in file_name:
        cols = ['y', 'p', 'rho', 'cp', 'U', 'V', 'W', 'Cfx', 'Cfy', 'Cfz', 'uu', 'uv', 'uw', 'vv', 'vw', 'ww']
    else:
        cols = ['y', 'p', 'cp', 'U', 'V', 'W', 'Cfx', 'Cfy', 'Cfz', 'uu', 'uv', 'uw', 'vv', 'vw', 'ww']

    df = pd.DataFrame(rows, columns=cols[:num_cols_in_file])
    if not {'y', 'U', 'V'}.issubset(df.columns):
        return pd.DataFrame()

    Uref2 = Uref ** 2
    df['Y/H']  = df['y'] / H
    df['U/Ur'] = df['U'] / Uref
    df['V/Ur'] = df['V'] / Uref
    
    # Normalize if columns exist
    if 'uu' in df.columns: 
        df['uu'] = df['uu'] / Uref2 * 1000
    if 'vv' in df.columns: 
        df['vv'] = df['vv'] / Uref2 * 1000
    if 'uv' in df.columns: 
        df['uv'] = df['uv'] / Uref2 * 1000

    out_cols = ['Y/H', 'U/Ur', 'V/Ur']
    for c in ['uu', 'vv', 'uv']:
        if c in df.columns: 
            out_cols.append(c)
            
    return df[out_cols].sort_values('Y/H')


def parse_x_by_h_directory(directory, H=0.0127, Uref=44.2):
    results = {}
    for path in sorted(glob.glob(os.path.join(directory, '*x_by_h*'))):
        x = get_x_coordinate_from_filename(os.path.basename(path))
        if x is None:
            continue
        try:
            results[x] = parse_x_by_h_file(path, H=H, Uref=Uref)
        except Exception as exc:
            print(f"Warning: failed to parse {path}: {exc}")
    return results


def parse_bakstp1_file(file_path, alpha_deg=0.0):
    data = {}
    current_x = None
    alpha_ok = False

    with open(file_path, 'r') as f:
        for line in f:
            x_match = re.search(r'X/H\s*=\s*([+-]?\d*\.?\d+)', line)
            if x_match:
                current_x = float(x_match.group(1))
                alpha_ok = False

            if current_x is None:
                continue

            alpha_match = re.search(r'alpha\s*=\s*([+-]?\d*\.\d+|[+-]?\d+)\s*deg', line, re.I)
            if alpha_match:
                alpha_ok = abs(float(alpha_match.group(1)) - alpha_deg) < 1e-6
                if alpha_ok:
                    data[current_x] = []
                continue

            if not alpha_ok:
                continue

            tokens = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', line)
            if len(tokens) < 7 or not re.match(r'^\d+$', tokens[0]):
                continue

            row = [float(t) for t in tokens[:7]]
            data[current_x].append({
                'Y/H':  row[1], 'U/Ur': row[2], 'V/Ur': row[3],
                'uu':   row[4], 'vv':   row[5], 'uv':   row[6],
            })

    return {x: pd.DataFrame(rows).sort_values('Y/H') for x, rows in data.items() if rows}


def parse_bakstp2_file(file_path):
    exp_data = {
        'cp_stepside': [],
        'cp_opposing': [],
        'cf_stepside': []
    }
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    mode = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        if "Cp for Stepside wall" in line:
            mode = 'cp_stepside'
            continue
        elif "Cp  Opposing wall" in line:
            mode = 'cp_opposing'
            continue
        elif "Alpha = 0 degrees" in line:
            mode = 'cf_stepside'
            continue
        elif "Alpha = 6 degrees" in line:
            mode = None
            continue
            
        # Parse Fixed Width Tables for Cp
        if mode in ['cp_stepside', 'cp_opposing']:
            if line.startswith(' ---') or 'X/H' in line or '___' in line:
                continue
            try:
                xh_str = line[0:6].strip()
                # Target column for Alpha=0 is approximately characters 14 to 24
                a0_str = line[14:24].strip()
                if xh_str and a0_str:
                    exp_data[mode].append({'X/H': float(xh_str), 'Cp': float(a0_str)})
            except ValueError:
                pass
                
        # Parse Standard Space-Delimited Table for Cf
        elif mode == 'cf_stepside':
            if 'X/H' in line or 'Uncertainty' in line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    exp_data[mode].append({'X/H': float(parts[0]), 'Cf': float(parts[1])})
                except ValueError:
                    pass

    for cp_key in ('cp_stepside', 'cp_opposing'):
        cp_data = exp_data.get(cp_key, [])
        if not cp_data:
            continue

        reference_point = max(cp_data, key=lambda row: row['X/H'])
        cp_offset = reference_point['Cp']
        for row in cp_data:
            row['Cp'] -= cp_offset
        reference_point['Cp'] = 0.0

    return exp_data


def parse_wall_file(file_path):
    rows = _read_numeric_rows(file_path)
    if not rows:
        return pd.DataFrame()

    num_cols_in_file = len(rows[0])
    file_name = os.path.basename(file_path)

    if 'static(p)_coeff_wallShearStress' in file_name:
        cols = ['x', 'cp', 'wallShearStress_x', 'wallShearStress_y', 'wallShearStress_z']
    elif 'wallShearStress_static(p)_coeff' in file_name:
        cols = ['x', 'wallShearStress_x', 'wallShearStress_y', 'wallShearStress_z', 'cp']
    elif 'rho' in file_name:
        cols = ['x', 'p', 'rho', 'cp', 'U', 'V', 'W', 'Cfx', 'Cfy', 'Cfz', 'uu', 'uv', 'uw', 'vv', 'vw', 'ww']
    else:
        cols = ['x', 'p', 'cp', 'U', 'V', 'W', 'Cfx', 'Cfy', 'Cfz', 'uu', 'uv', 'uw', 'vv', 'vw', 'ww']

    df = pd.DataFrame(rows, columns=cols[:num_cols_in_file])
    return df


def get_wall_data(folder, wall_prefix, H=0.0127, rho=1.225, Uref=44.2, nu=1.56e-05):
    # This automatically combines upstream and downstream if wall_prefix='lowerWall'
    if not folder or not os.path.isdir(folder):
        return None

    files = glob.glob(os.path.join(folder, f"{wall_prefix}*.xy"))
    if not files:
        return None
        
    df_list = []
    for f in files:
        df = parse_wall_file(f)
        if not df.empty:
            df_list.append(df)
            
    if not df_list:
        return None
        
    # Merge correctly, order by coordinate x
    combined = pd.concat(df_list).drop_duplicates('x').sort_values('x')
    
    # Calculate non-dimensional properties
    combined['X/H'] = combined['x'] / H
    if 'wallShearStress_x' in combined.columns:
        combined['Cf'] = -1 * combined['wallShearStress_x'] / (0.5 * rho * Uref**2)
    elif 'Cfx' in combined.columns:
        combined['Cf'] = -1 * combined['Cfx'] / (0.5 * rho * Uref**2)
    elif 'Cfx_kinematic' in combined.columns:
        combined['Cf'] = -1 * combined['Cfx_kinematic'] * nu / (0.5 * Uref**2)
    
    return combined


# ---------------------------------------------------------------------------
# Metrics & Combined validation profiles
# ---------------------------------------------------------------------------

def _compute_nrmse(sim_df, val_df, variable, x, model_name):
    if sim_df is None or val_df is None:
        return np.nan
    if variable not in sim_df.columns or variable not in val_df.columns:
        return np.nan

    sim_y = sim_df['Y/H'].values
    val_y = val_df['Y/H'].values

    if len(sim_y) < 2 or len(val_y) == 0:
        return np.nan

    valid_mask = (val_y >= sim_y.min()) & (val_y <= sim_y.max())
    if not valid_mask.any():
        return np.nan

    sim_interp = np.interp(val_y[valid_mask], sim_y, sim_df[variable].values)
    val_values = val_df.loc[valid_mask, variable].astype(float).values

    rmse = np.sqrt(np.mean((sim_interp - val_values) ** 2))
    val_range = val_values.max() - val_values.min()
    return rmse / val_range if val_range > 1e-8 else np.nan


def _compute_pearson_r(sim_df, val_df, variable):
    if sim_df is None or val_df is None:
        return np.nan
    if variable not in sim_df.columns or variable not in val_df.columns:
        return np.nan

    sim_y = sim_df['Y/H'].values
    val_y = val_df['Y/H'].values

    if len(sim_y) < 2 or len(val_y) < 2:
        return np.nan

    valid_mask = (val_y >= sim_y.min()) & (val_y <= sim_y.max())
    if valid_mask.sum() < 2:
        return np.nan

    sim_interp = np.interp(val_y[valid_mask], sim_y, sim_df[variable].values)
    val_values = val_df.loc[valid_mask, variable].astype(float).values

    r = np.corrcoef(sim_interp, val_values)[0, 1]
    return float(r)


def _format_x_label(x_coord):
    label = f"{abs(x_coord):g}".replace('.', 'p')
    return f"xm{label}" if x_coord < 0 else f"x{label}"


def _plot_nrmse_overview(nrmse_records, output_root):
    if not nrmse_records:
        return
    df_nrmse = pd.DataFrame(nrmse_records)
    overview_dir = os.path.join(output_root, 'overview')
    os.makedirs(overview_dir, exist_ok=True)

    for var in df_nrmse['variable'].unique():
        df_var = df_nrmse[df_nrmse['variable'] == var]
        plt.figure(figsize=(10, 6))
        for model_name in sorted(df_var['model'].unique()):
            df_model = df_var[df_var['model'] == model_name].sort_values('x')
            plt.plot(df_model['x'], df_model['nrmse'], label=model_name, marker='o')
        plt.xlabel('X/H')
        plt.ylabel('Normalized RMSE (NRMSE)')
        plt.title(f'Model Normalized Error Trend over X/H for Quantity: {var}')
        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(overview_dir, f"rmse_{var.replace('/', '_')}.png"))
        plt.close()


def _plot_absolute_nrmse_bar_chart(nrmse_records, output_root):
    if not nrmse_records:
        return
    df = pd.DataFrame(nrmse_records)
    df_avg = df.groupby(['variable', 'model'])['nrmse'].mean().unstack(level='model')
    if df_avg.empty:
        return

    models = sorted(df_avg.columns)
    x_indices = np.arange(len(df_avg.index))
    bar_width = 0.8 / len(models)

    plt.figure(figsize=(11, 6))
    for i, model_name in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * bar_width
        plt.bar(x_indices + offset, df_avg[model_name], width=bar_width, label=model_name)

    plt.xticks(x_indices, df_avg.index)
    plt.ylim(0, 0.65)
    plt.xlabel('Quantity / Field Variable')
    plt.ylabel('Average Range-Normalized RMSE (NRMSE)')
    plt.title('Overall Summary: Profile Normalized Error Comparison (Lower is Better)')
    plt.grid(True, which='both', linestyle='--', alpha=0.3, axis='y')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, 'relative_rmse_comparison.png'))
    plt.close()


def _plot_correlation_overview(correlation_records, output_dir):
    if not correlation_records:
        return
    df_corr = pd.DataFrame(correlation_records)
    plt.figure(figsize=(10, 6))
    for (model, variable), group in df_corr.groupby(['model', 'variable']):
        grp = group.sort_values('x')
        plt.plot(grp['x'], grp['correlation'], label=f"{variable} ({model})", marker='o')

    plt.xlabel('X/H')
    plt.ylabel('Correlation (r)')
    plt.title('Validation vs Simulation Correlation by X/H')
    plt.ylim(-1.05, 1.05)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_correlation_overview.png'))
    plt.close()


def plot_validation_profiles_across_models(validation_data, models_x_by_h, output_path, variables=None):
    if not validation_data or not models_x_by_h:
        return

    if variables is None:
        variables = ['U/Ur', 'V/Ur', 'uu', 'vv', 'uv']

    combined_root = os.path.join(output_path, 'validation')
    os.makedirs(combined_root, exist_ok=True)

    x_coords = sorted(set(validation_data.keys()).union(*(set(d.keys()) for d in models_x_by_h.values())))
    nrmse_records = []
    correlation_records = []

    for x in x_coords:
        val_df = validation_data.get(x)
        for var in variables:
            var_dir = os.path.join(combined_root, var.replace('/', '_'))
            os.makedirs(var_dir, exist_ok=True)
            plt.figure(figsize=(10, 7))
            plotted = False

            if val_df is not None and var in val_df.columns:
                plt.plot(val_df[var], val_df['Y/H'],
                         label='Experiment', color='black', marker='x', linewidth=0)
                plotted = True

            for model_name in sorted(models_x_by_h):
                model_df = models_x_by_h[model_name].get(x)
                if model_df is None or var not in model_df.columns:
                    continue
                plt.plot(model_df[var], model_df['Y/H'], label=model_name, linewidth=2)
                plotted = True

                if val_df is not None and var in val_df.columns:
                    nrmse_val = _compute_nrmse(model_df, val_df, var, x, model_name)
                    if not np.isnan(nrmse_val):
                        nrmse_records.append({'model': model_name, 'variable': var, 'x': x, 'nrmse': nrmse_val})
                    r = _compute_pearson_r(model_df, val_df, var)
                    if not np.isnan(r):
                        correlation_records.append({'model': model_name, 'variable': var, 'x': x, 'correlation': r})

            if not plotted:
                plt.close()
                continue

            plt.xlabel(var)
            plt.ylabel('Y/H')
            plt.ylim(0, 3)
            plt.title(f'{var} profiles at X/H={x} across all models and experiment')
            plt.grid(True, which='both', linestyle='--', alpha=0.3)
            plt.legend(loc='best', fontsize='small')
            plt.tight_layout()
            plt.savefig(os.path.join(var_dir, f'{_format_x_label(x)}.png'))
            plt.close()

    _plot_nrmse_overview(nrmse_records, combined_root)
    _plot_absolute_nrmse_bar_chart(nrmse_records, combined_root)
    _plot_correlation_overview(correlation_records, combined_root)


# --- NEW: Combined Wall Data Plotting ---
def plot_combined_wall_comparisons(exp_data, models_lower_wall, models_upper_wall, output_root):
    wall_dir = os.path.join(output_root, 'validation_wall')
    os.makedirs(wall_dir, exist_ok=True)

    # 1. Lower Wall Cp
    if models_lower_wall and exp_data.get('cp_stepside'):
        exp_df = pd.DataFrame(exp_data['cp_stepside'])
        plt.figure(figsize=(10, 6))
        plt.plot(exp_df['X/H'], exp_df['Cp'], 'kx', label='Experiment (Table 2, a=0)', markersize=6)
        
        for model_name, lower_wall in sorted(models_lower_wall.items()):
            plt.plot(lower_wall['X/H'], lower_wall['cp'], label=f'Simulation ({model_name})', linewidth=1.5)
            
        plt.xlabel('X/H')
        plt.ylabel('Cp')
        plt.title('Lower Wall (Stepside) Pressure Coefficient')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(wall_dir, 'plot_lowerWall_Cp.png'))
        plt.close()

    # 2. Upper Wall Cp
    if models_upper_wall and exp_data.get('cp_opposing'):
        exp_df = pd.DataFrame(exp_data['cp_opposing'])
        plt.figure(figsize=(10, 6))
        plt.plot(exp_df['X/H'], exp_df['Cp'], 'kx', label='Experiment (Table 2, a=0)', markersize=6)
        
        for model_name, upper_wall in sorted(models_upper_wall.items()):
            plt.plot(upper_wall['X/H'], upper_wall['cp'], label=f'Simulation ({model_name})', linewidth=1.5)
            
        plt.xlabel('X/H')
        plt.ylabel('Cp')
        plt.title('Upper Wall (Opposing) Pressure Coefficient')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(wall_dir, 'plot_upperWall_Cp.png'))
        plt.close()

    # 3. Lower Wall Cf
    if models_lower_wall and exp_data.get('cf_stepside'):
        exp_df = pd.DataFrame(exp_data['cf_stepside'])
        plt.figure(figsize=(10, 6))
        plt.plot(exp_df['X/H'], exp_df['Cf'], 'kx', label='Experiment (Table 3, a=0)', markersize=6)
        
        for model_name, lower_wall in sorted(models_lower_wall.items()):
            plt.plot(lower_wall['X/H'], lower_wall['Cf'], label=f'Simulation ({model_name})', linewidth=1.5)
            
        plt.xlabel('X/H')
        plt.ylabel('Cf')
        plt.title('Lower Wall (Stepside) Skin Friction Coefficient')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize='small')
        plt.tight_layout()
        plt.savefig(os.path.join(wall_dir, 'plot_lowerWall_Cf.png'))
        plt.close()


def latest_time_directory(parent):
    if not os.path.isdir(parent):
        return None

    candidates = []
    for item in os.listdir(parent):
        path = os.path.join(parent, item)
        if not os.path.isdir(path):
            continue
        try:
            candidates.append((float(item), path))
        except ValueError:
            continue

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def find_first_xy(directory, pattern='*.xy'):
    if not directory or not os.path.isdir(directory):
        return None
    matches = sorted(glob.glob(os.path.join(directory, pattern)))
    return matches[0] if matches else None


def resolve_case_paths(script_dir):
    if os.path.basename(script_dir) == '_postProcessing':
        source_case_dir = os.path.dirname(script_dir)
        case_name = os.path.basename(source_case_dir)
        base_path = os.path.dirname(source_case_dir)
        run_case_dir = os.path.join(base_path, 'run', case_name)
        return base_path, case_name, run_case_dir, source_case_dir

    case_name = os.path.basename(script_dir)
    parent_dir = os.path.dirname(script_dir)

    if os.path.basename(parent_dir) == 'run':
        base_path = os.path.dirname(parent_dir)
        run_case_dir = script_dir
        source_case_dir = os.path.join(base_path, case_name)
    else:
        base_path = parent_dir
        run_case_dir = os.path.join(base_path, 'run', case_name)
        source_case_dir = script_dir

    return base_path, case_name, run_case_dir, source_case_dir

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting post-processing...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path, case_name, run_case_dir, source_case_dir = resolve_case_paths(script_dir)
    results_root = os.path.join(base_path, 'results', case_name)

    model_folders = []
    if os.path.isdir(run_case_dir):
        model_folders = [
            item for item in os.listdir(run_case_dir)
            if os.path.isdir(os.path.join(run_case_dir, item, 'postProcessing'))
        ]
    if not model_folders:
        print(f"Warning: no model folders found under {run_case_dir}")

    validation_data = None
    models_x_by_h = {}
    
    # New wall data dictionaries for combined plotting
    exp_wall_data = None
    models_lower_wall = {}
    models_upper_wall = {}

    for model_name in model_folders:
        print(f"Processing folder: {model_name}")
        output_path = os.path.join(base_path, 'results', case_name, model_name)
        case_folder = os.path.join(run_case_dir, model_name)

        if not os.path.isdir(case_folder):
            continue

        min_max_path = os.path.join(case_folder, 'postProcessing/minMax/0/fieldMinMax.dat')
        if os.path.exists(min_max_path):
            plot_minMax_data(min_max_path, output_path)

        solver_path = os.path.join(case_folder, 'postProcessing/solverInfo/0/solverInfo.dat')
        if os.path.exists(solver_path):
            plot_residual_data(solver_path, output_path)

        validation_file = os.path.join(source_case_dir, '_validation_data/experimental/bakstp1.dat')
        validation_file2 = os.path.join(source_case_dir, '_validation_data/experimental/bakstp2.dat')

        post_processing_dir = os.path.join(case_folder, 'postProcessing')
        x_by_h_dir = latest_time_directory(os.path.join(post_processing_dir, 'sample'))
        uref_dir = latest_time_directory(os.path.join(post_processing_dir, 'Uref'))
        sample_wall_dir = latest_time_directory(os.path.join(post_processing_dir, 'sample_wall'))

        if os.path.exists(validation_file) and x_by_h_dir and os.path.isdir(x_by_h_dir):
            uref_file = find_first_xy(uref_dir, 'Uref*.xy')
            rho, Uref = read_Uref_and_rho_from_file(uref_file) if uref_file else (1.225, 44.2)
            print(f"Using Uref={Uref} m/s and rho={rho} kg/m^3 for normalization in {model_name}")
            if validation_data is None:
                validation_data = parse_bakstp1_file(validation_file, alpha_deg=0.0)

            x_by_h_data = parse_x_by_h_directory(x_by_h_dir, H=0.0127, Uref=Uref)
            if x_by_h_data:
                models_x_by_h[model_name] = x_by_h_data
                
            # Process and stash wall comparison data
            if os.path.exists(validation_file2):
                if exp_wall_data is None:
                    exp_wall_data = parse_bakstp2_file(validation_file2)
                    
                lower_wall = get_wall_data(sample_wall_dir, 'lowerWall', H=0.0127, rho=rho, Uref=Uref)
                upper_wall = get_wall_data(sample_wall_dir, 'upperWall', H=0.0127, rho=rho, Uref=Uref)
                
                if lower_wall is not None and not lower_wall.empty:
                    models_lower_wall[model_name] = lower_wall
                if upper_wall is not None and not upper_wall.empty:
                    models_upper_wall[model_name] = upper_wall

    # Overlay profiles across all available models
    if validation_data and models_x_by_h:
        plot_validation_profiles_across_models(validation_data, models_x_by_h, results_root)
        
    # Overlay wall profiles across all models
    if exp_wall_data:
        print("Generating consolidated wall comparisons...")
        plot_combined_wall_comparisons(exp_wall_data, models_lower_wall, models_upper_wall, results_root)
        
