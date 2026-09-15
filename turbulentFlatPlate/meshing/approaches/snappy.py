"""
snappy approach (stub) – blockMesh base + snappyHexMesh prism layers.

Intended recipe (v2512 has no changeDimension):
  1. blockMesh base: 1 cell in z, frontAndBack as symmetryPlane (snappy does
     not accept empty patches).
  2. snappyHexMesh with addLayers -> prisms on bottomWall.
  3. Flip frontAndBack patch type to empty in constant/polyMesh/boundary
     (1-cell z mesh, no z-cell removal -> boundary-file edit is sufficient).
  4. renumberMesh + checkMesh.
Open point: verify snappy tolerates the 1-cell z base; fallback would be a
few z-cells plus keeping a single layer (mesh surgery).
"""
from .base import Approach, ApproachError


class SnappyApproach(Approach):
    name = "snappy"
    defaults = {
        "nPrisms": 8,
        "firstLayerThickness": 5e-06,
        "expansionRatio": 1.2,
        "prismCoreRatio": 0.5,
    }

    def generate(self, target_case, params):
        raise ApproachError("snappy approach not implemented yet (structure stub)")
