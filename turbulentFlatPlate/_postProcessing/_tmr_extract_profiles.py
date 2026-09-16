#!/usr/bin/env python3
"""
_tmr_extract_profiles.py - turn the NASA TMR flat-plate field dumps into the
small wall-normal profile files kept in `_validation_data/`.

Not part of the pipeline (leading underscore = the drivers copy it but never
run it).  Re-run it only when the sampling stations change.

    python3 _tmr_extract_profiles.py turb_contours_cfl3d_ssglrrrsm.dat \
                                     mut_contours_cfl3d.dat

Inputs (zipped Tecplot dumps, CFL3D, finest 545 x 385 grid; both are ~10-24 MB
so they are NOT committed - only the extracted profiles are):

  Reynolds stresses + omega, SSG/LRR-RSM-w2012
    https://tmbwg.github.io/turbmodels/flatplate_ssglrrrsm.html
    -> turb-contours-cfl3d-ssglrrrsm-dat.zip     (structured, at cell centres)
  Eddy viscosity, SA
    https://tmbwg.github.io/turbmodels/flatplate_sa.html
    -> mut-contours-cfl3d-dat.zip                (structured, at cell centres)

CFL3D was chosen over the TAU / FUN3D dumps of the same cases because it ran
the freestream this case now uses (Tu = 0.039 %, mu_t/mu = 0.009; TAU used
0.1 % / 0.1), because an ordered 544 x 384 BLOCK zone makes a wall-normal
profile one array column, and because it is half the size.

Two conversions happen here, and both are verified against values TMR states
independently (see the asserts below):

  * Non-dimensionalisation.  CFL3D writes R/a_inf^2, omega*mu_inf/(rho*a^2)
    and mu_t/mu_inf; everything is converted to SI using this case's own
    freestream, which is the same M = 0.2, Re/m = 5e6 state.
  * Axis convention.  In the TMR dumps **z is the wall-normal direction**, so
    R33 is the wall-normal stress and R13 the shear - not R22/R12.  The
    components are relabelled to this case's y-up convention on the way out.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np

import _pp_common as pp

# Wall-normal sampling stations of base_case/system/sampleUyPlus.
STATIONS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.95)

# Ordered-zone shape of both dumps (cells of the finest 545 x 385 node grid).
NI, NJ = 544, 384

RSTRESS_OUT = "tmr_rstress_ssglrrrsm.dat"
NUT_OUT = "tmr_nut_sa.dat"


def read_ordered_block(path, n_vars):
    """Read an ordered BLOCK Tecplot dump as a list of (NJ, NI) arrays.

    Every numeric token in the file belongs to the data: the header lines
    (`VARIABLES`, `ZONE`, `I=...`, `DT=(...)`) contain no bare numbers that
    parse as floats on their own, so a flat read then a reshape is enough.
    """
    values = []
    with open(path) as stream:
        for line in stream:
            parts = line.split()
            if not parts:
                continue
            try:
                values.extend(float(v) for v in parts)
            except ValueError:
                continue
    values = np.asarray(values)
    n = NI * NJ
    if values.size != n_vars * n:
        raise SystemExit(f"[error] {path}: got {values.size} numbers, "
                         f"expected {n_vars} x {n} - not the expected "
                         f"{NI} x {NJ} ordered dump")
    return [values[i * n:(i + 1) * n].reshape(NJ, NI) for i in range(n_vars)]


def freestream(case_dir):
    """(rho, mu, nu, a) of the case, so the conversion cannot drift from it."""
    thermo = (Path(case_dir) / "constant" / "thermophysicalProperties").read_text()
    mol_w = float(re.search(r"molWeight\s+([-\d.eE+]+)", thermo).group(1))
    cp = float(re.search(r"^\s*Cp\s+([-\d.eE+]+)\s*;", thermo, re.M).group(1))
    _, nu, rho = pp.detect_case_parameters(case_dir)
    r_s = 8314.462618 / mol_w
    t_inf = pp._read_internal_field(Path(case_dir) / "0.orig" / "T")
    a_inf = math.sqrt((cp / (cp - r_s)) * r_s * t_inf)
    return rho, nu * rho, nu, a_inf


def profiles_at_stations(x_grid, fields):
    """{station: (NJ, n_fields) array} linearly interpolated in x.

    The grid is a tensor product - x is constant down a column and the
    wall-normal coordinate is identical in every column - so interpolating
    each j-row in x is exact for the station, with no wall-normal
    interpolation at all.
    """
    x_line = x_grid[0]
    out = {}
    for station in STATIONS:
        cols = np.column_stack([
            [np.interp(station, x_line, field[j]) for j in range(NJ)]
            for field in fields
        ])
        out[station] = cols
    return out


def write_zones(path, variables, profiles, header):
    with open(path, "w") as stream:
        stream.write(header)
        stream.write('variables=' + ','.join(f'"{v}"' for v in variables) + "\n")
        for station, block in profiles.items():
            stream.write(f'zone, t="x={station:g}"\n')
            for row in block:
                stream.write("".join(f" {v: .9E}" for v in row) + "\n")
    print(f"  wrote {path}  ({len(profiles)} zones x {NJ} points)")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        raise SystemExit(__doc__)
    rstress_src, nut_src = (Path(a) for a in argv)

    script_dir = Path(__file__).resolve().parent
    _, _, _, source_case_dir, _ = pp.resolve_case_paths(script_dir)
    val_dir = source_case_dir / "_validation_data"
    base_case = source_case_dir / "base_case"

    rho, mu, nu, a_inf = freestream(base_case)
    print(f"case freestream: rho={rho:.6f} mu={mu:.4e} nu={nu:.6e} a={a_inf:.3f}")

    # --- Reynolds stresses + omega -----------------------------------------
    # Variables in file order: x, z, R11, R22, R33, R13, omega.
    x, z, r11, r22, r33, r13, omega = read_ordered_block(rstress_src, 7)
    a2 = a_inf * a_inf
    # z up -> y up:  R11->Rxx, R33->Ryy (wall normal), R22->Rzz, R13->Rxy.
    fields = [z, r11 * a2, r33 * a2, r22 * a2, r13 * a2, omega * rho * a2 / mu]
    profiles = profiles_at_stations(x, fields)

    # TMR states the inflow BCs as R_ii = 6e-9 a^2 and omega = 1e-6 rho a^2/mu;
    # the first cell inside the inlet must be just below both (the freestream
    # decays downstream, so only the inlet is a meaningful check).
    inlet = np.column_stack([f[NJ - 1, 0] for f in fields[1:]])[0]
    assert 0.8 < inlet[0] / (6e-9 * a2) < 1.0, inlet
    assert 0.8 < inlet[4] / (1e-6 * rho * a2 / mu) < 1.0, inlet
    print(f"  inlet check: R_ii = {inlet[0]:.4e} (BC {6e-9 * a2:.4e}), "
          f"omega = {inlet[4]:.4e} (BC {1e-6 * rho * a2 / mu:.4e})")

    write_zones(
        val_dir / RSTRESS_OUT,
        ("y", "Rxx", "Ryy", "Rzz", "Rxy", "omega"),
        profiles,
        "# NASA TMR 2D zero-pressure-gradient flat plate, SSG/LRR-RSM-w2012,\n"
        "# CFL3D, finest 545x385 grid.  Extracted from\n"
        "# turb_contours_cfl3d_ssglrrrsm.dat by _tmr_extract_profiles.py.\n"
        "# SI units (R in m2/s2, omega in 1/s, y in m), and the TMR z-up\n"
        "# components are relabelled to this case's y-up convention:\n"
        "#   TMR R11->Rxx, R33->Ryy (wall normal), R22->Rzz, R13->Rxy.\n"
        "# TMR R12 and R23 are identically zero in 2D (= Rxz, Ryz here).\n",
    )

    # --- SA eddy viscosity --------------------------------------------------
    # Variables in file order: x, y, mut.  mut is mu_t/mu_inf: TMR's SA page
    # gives nut_farfield = 0.210438 nu_inf and the dump holds 0.21049 in the
    # freestream, which pins the normalisation.
    x, y, mut = read_ordered_block(nut_src, 3)
    assert 0.2 < mut[NJ - 1, 0] < 0.22, mut[NJ - 1, 0]
    print(f"  freestream mu_t/mu_inf = {mut[NJ - 1, 0]:.5f} "
          f"(TMR SA: nu_t = 0.210438 nu)")
    profiles = profiles_at_stations(x, [y, mut * nu])
    write_zones(
        val_dir / NUT_OUT,
        ("y", "nut"),
        profiles,
        "# NASA TMR 2D zero-pressure-gradient flat plate, SA, CFL3D, finest\n"
        "# 545x385 grid.  Extracted from mut_contours_cfl3d.dat by\n"
        "# _tmr_extract_profiles.py.  nut = (mu_t/mu_inf)*nu_inf in m2/s,\n"
        "# y in m.\n",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
