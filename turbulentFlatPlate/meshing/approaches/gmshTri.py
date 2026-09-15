"""
gmshTri approach (stub) – gmsh 2D triangle core, prisms added by snappy.

Intended recipe:
  1. Write flat-plate .geo (rectangle, distance field controls core size).
  2. gmsh -2 -> mesh.msh (gmsh via apt, not installed yet).
  3. gmshToFoam -> constant/polyMesh, rename patches, frontAndBack 1 cell z.
  4. Prisms via snappyHexMesh addLayers (as in the snappy approach).
  5. renumberMesh + checkMesh.
"""
from .base import Approach, ApproachError


class GmshTriApproach(Approach):
    name = "gmshTri"
    defaults = {
        "nPrisms": 8,
        "firstLayerThickness": 5e-06,
        "expansionRatio": 1.2,
        "prismCoreRatio": 0.5,
        "coreCellSize": 0.05,
    }

    def generate(self, target_case, params):
        raise ApproachError("gmshTri approach not implemented yet (structure stub)")
