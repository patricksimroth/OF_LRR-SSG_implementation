"""Centroid and 2D area of every cell near the plate (pvbatch).

    pvbatch --force-offscreen-rendering _pv_cell_sizes.py <case.foam> <out.npz> <x0> <x1> <y1>

Writes x, y (cell centroids) and area (cell volume / domain depth) of the cells
whose centroid lies in x0 <= x <= x1, y <= y1; _core_size.py reduces them to
the measured core cell size.
"""
import sys

import numpy as np
from paraview import servermanager
from paraview.simple import CellCenters, CellSize, MergeBlocks, OpenFOAMReader
from vtk.numpy_interface import dataset_adapter as dsa

foam, out = sys.argv[1], sys.argv[2]
x0, x1, y1 = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])

reader = OpenFOAMReader(FileName=foam)
reader.CaseType = "Reconstructed Case"
reader.MeshRegions = ["internalMesh"]
reader.CellArrays = []
reader.Createcelltopointfiltereddata = 0
reader.UpdatePipelineInformation()
times = reader.TimestepValues
times = list(times) if hasattr(times, "__len__") else [times]
latest = max(times)
reader.UpdatePipeline(latest)
_, _, _, _, z0, z1 = reader.GetDataInformation().GetBounds()

size = CellSize(Input=MergeBlocks(Input=reader))
size.ComputeVertexCount = 0
size.ComputeLength = 0
size.ComputeArea = 0
size.ComputeVolume = 1
cc = CellCenters(Input=size)
cc.VertexCells = 0
cc.UpdatePipeline(latest)
d = dsa.WrapDataObject(servermanager.Fetch(cc))
pts = np.asarray(d.Points)
vol = np.asarray(d.PointData["Volume"])
keep = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] <= y1)
np.savez_compressed(out, x=pts[keep, 0], y=pts[keep, 1], area=vol[keep] / (z1 - z0))
