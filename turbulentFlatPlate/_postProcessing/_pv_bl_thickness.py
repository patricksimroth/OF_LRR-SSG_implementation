"""
_pv_bl_thickness.py – boundary-layer thickness delta99(x) of one run.

Runs inside ParaView's pvbatch, not the project venv:
    pvbatch --force-offscreen-rendering _pv_bl_thickness.py \
        <case.foam> <out.csv> <x0> <x1> <nx> <y_top> <ny>
Resamples the latest U onto a regular x-y grid in the mid-z plane (point data,
cell-to-point interpolated by the reader).  Per x station:
    U_e     = maximum Ux of the station
    delta99 = first wall distance where Ux >= 0.99 U_e (linear interpolation)
Called per run by bl_thickness.py; the leading underscore keeps the drivers
from executing it directly.
"""
import sys

import numpy as np
from paraview import servermanager
from paraview.simple import OpenFOAMReader, ResampleToImage
from vtk.numpy_interface import dataset_adapter as dsa

foam_path, out_path = sys.argv[1], sys.argv[2]
x0, x1, nx = float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
y_top, ny = float(sys.argv[6]), int(sys.argv[7])

reader = OpenFOAMReader(FileName=foam_path)
reader.CaseType = "Reconstructed Case"
reader.MeshRegions = ["internalMesh"]
reader.CellArrays = ["U"]
# Must be set explicitly: without the reader's cell-to-point data the resample
# probes piecewise-constant cell values, so delta99 snaps to cell faces and
# plots as a staircase (e.g. stuck at the 15 mm prism-stack top on snappy).
reader.Createcelltopointfiltereddata = 1
reader.UpdatePipelineInformation()
times = reader.TimestepValues
times = list(times) if hasattr(times, "__len__") else [times]
latest = max(times) if times else 0.0
reader.UpdatePipeline(latest)
_, _, _, _, z0, z1 = reader.GetDataInformation().GetBounds()
z_mid = 0.5 * (z0 + z1)

resample = ResampleToImage(Input=reader)
resample.UseInputBounds = 0
resample.SamplingBounds = [x0, x1, 0.0, y_top, z_mid, z_mid]
resample.SamplingDimensions = [nx, ny, 1]
resample.UpdatePipeline(latest)

data = dsa.WrapDataObject(servermanager.Fetch(resample))
ux = np.asarray(data.PointData["U"])[:, 0].reshape(ny, nx)
valid = np.asarray(data.PointData["vtkValidPointMask"]).reshape(ny, nx) > 0
x = np.linspace(x0, x1, nx)
y = np.linspace(0.0, y_top, ny)

rows = []
for i in range(nx):
    inside = valid[:, i]
    if inside.sum() < 2:
        continue
    u, wall_dist = ux[inside, i], y[inside]
    u_edge = u.max()
    target = 0.99 * u_edge
    j = int(np.argmax(u >= target))
    if j == 0:
        delta = wall_dist[0]
    else:
        delta = wall_dist[j - 1] + ((target - u[j - 1]) * (wall_dist[j] - wall_dist[j - 1])
                                    / (u[j] - u[j - 1]))
    rows.append((x[i], delta, u_edge))

np.savetxt(out_path, np.array(rows), delimiter=",", header="x,delta99,U_e",
           comments="")
