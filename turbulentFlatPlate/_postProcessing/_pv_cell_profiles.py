"""Cell-centre wall-normal profiles of U, nut (and k, omega, epsilon, nuTilda
when present) at every x station (pvbatch).

    pvbatch --force-offscreen-rendering _pv_cell_profiles.py <case.foam> <out.npz> <x0> <x1> <nx>

Why not the reader's cell-to-point data: ParaView averages the neighbouring
cell values UNWEIGHTED at each vertex, and blockMesh (growth 1.0585) and the
prism stacks (1.17) are stretched differently, so the vertex values carry a
mesh-dependent bias - +2.8 % at y = 1 um between two meshes whose wall shear
agrees to 0.04 %.  Instead the raw (piecewise-constant) cell values are sampled
densely along each vertical line; every run of equal values is one cell, whose
wall-normal centre is the middle of the run.  Profiles are those centres plus
the wall point (y = 0, U = 0, nut = 0), to be interpolated linearly.

The centre of each run is then replaced by the EXACT centroid of that cell
(CellCenters, matched by its (U, nut) value): the middle of a run of samples is
only as good as the sample spacing, and a 0.6 % position error in the linear
sublayer is a 0.3 % velocity error.
"""
import sys

import numpy as np
from paraview import servermanager
from paraview.simple import CellCenters, MergeBlocks, OpenFOAMReader, ResampleToImage
from vtk.numpy_interface import dataset_adapter as dsa

foam, out = sys.argv[1], sys.argv[2]
x0, x1, nx = float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
# Sample spacing well below the local cell size everywhere: 0.025 um near the
# 1 um wall cell, 0.25 um to 2 mm, 5 um to the top of the boundary layer.
GRIDS = ((0.0, 25e-6, 1001), (25e-6, 2e-3, 7901), (2e-3, 0.06, 11601))

reader = OpenFOAMReader(FileName=foam)
reader.CaseType = "Reconstructed Case"
reader.MeshRegions = ["internalMesh"]
reader.UpdatePipelineInformation()
# Turbulence scalars sampled alongside, whichever of them this model writes.
EXTRAS = [f for f in ("k", "omega", "epsilon", "nuTilda") if f in reader.CellArrays.Available]
reader.CellArrays = ["U", "nut"] + EXTRAS
reader.Createcelltopointfiltereddata = 0
reader.UpdatePipelineInformation()
times = reader.TimestepValues
times = list(times) if hasattr(times, "__len__") else [times]
latest = max(times)
reader.UpdatePipeline(latest)
_, _, _, _, z0, z1 = reader.GetDataInformation().GetBounds()
zm = 0.5 * (z0 + z1)

# Exact centroids of every cell, keyed by the cell's own (Ux, nut) value.
cc = CellCenters(Input=MergeBlocks(Input=reader))
cc.VertexCells = 0
cc.UpdatePipeline(latest)
cd = dsa.WrapDataObject(servermanager.Fetch(cc))
cpts = np.asarray(cd.Points)
cu = np.asarray(cd.PointData["U"])[:, 0]
cn = np.asarray(cd.PointData["nut"])
centroid = {}
for (px, py, _pz), uu, nn in zip(cpts, cu, cn):
    centroid.setdefault((float(uu), float(nn)), []).append((px, py))

ys, us, nuts = [], [], []
extras = {f: [] for f in EXTRAS}
for ylo, yhi, ny in GRIDS:
    rs = ResampleToImage(Input=reader)
    rs.UseInputBounds = 0
    rs.SamplingBounds = [x0, x1, ylo, yhi, zm, zm]
    rs.SamplingDimensions = [nx, ny, 1]
    rs.UpdatePipeline(latest)
    d = dsa.WrapDataObject(servermanager.Fetch(rs))
    valid = np.asarray(d.PointData["vtkValidPointMask"]).reshape(ny, nx) > 0
    u = np.asarray(d.PointData["U"])[:, 0].reshape(ny, nx)
    nut = np.asarray(d.PointData["nut"]).reshape(ny, nx)
    y = np.linspace(ylo, yhi, ny)
    keep = slice(1, None) if ys else slice(None)     # drop the duplicated grid seam
    ys.append(y[keep]); us.append(np.where(valid, u, np.nan)[keep]); nuts.append(nut[keep])
    for f in EXTRAS:
        extras[f].append(np.asarray(d.PointData[f]).reshape(ny, nx)[keep])
y = np.concatenate(ys)
U = np.concatenate(us)
NUT = np.concatenate(nuts)
EXTRA = {f: np.concatenate(v) for f, v in extras.items()}
# Value at the wall point (y = 0): known from the wall BC for these, unknown
# (NaN) for omega/epsilon, whose wall values are singular.
WALL = {"k": 0.0, "nuTilda": 0.0}

offsets, yc_all, uc_all, nc_all = [0], [], [], []
ec_all = {f: [] for f in EXTRAS}
for i in range(nx):
    u, nut = U[:, i], NUT[:, i]
    ok = np.isfinite(u)
    yy, uu, nn = y[ok], u[ok], nut[ok]
    change = np.flatnonzero((np.diff(uu) != 0) | (np.diff(nn) != 0)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(uu)]]) - 1
    yc = 0.5 * (yy[starts] + yy[ends])
    xs = x0 + (x1 - x0) * i / (nx - 1)
    for k, s in enumerate(starts):
        cand = centroid.get((float(uu[s]), float(nn[s])))
        if cand:
            px, py = min(cand, key=lambda c: abs(c[0] - xs) + abs(c[1] - yc[k]))
            if abs(py - yc[k]) < 0.5 * (yy[ends[k]] - yy[s]) + 1e-6:
                yc[k] = py
    # the first run starts at the wall face (y = 0): its centre is half its top
    yc_all.append(np.concatenate([[0.0], yc]))
    uc_all.append(np.concatenate([[0.0], uu[starts]]))
    nc_all.append(np.concatenate([[0.0], nn[starts]]))
    for f in EXTRAS:
        ec_all[f].append(np.concatenate([[WALL.get(f, np.nan)], EXTRA[f][:, i][ok][starts]]))
    offsets.append(offsets[-1] + len(yc) + 1)
np.savez_compressed(out, x=np.linspace(x0, x1, nx), offsets=np.array(offsets),
                    y=np.concatenate(yc_all), u=np.concatenate(uc_all),
                    nut=np.concatenate(nc_all),
                    **{f: np.concatenate(v) for f, v in ec_all.items()})
