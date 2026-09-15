"""
gmshHex approach (stub) – gmsh 2D triangles recombined to quads (hex in 2D),
prisms added by snappy.

Intended recipe:
  1. Write flat-plate .geo (rectangle, distance field controls core size).
  2. gmsh -2 with 2D recombination (triangles -> quads; riskiest step, may
     leave mixed cells on unstructured regions).
  3. gmshToFoam -> constant/polyMesh, patch naming as in gmshTri.
  4. Prisms via snappyHexMesh addLayers.
  5. renumberMesh + checkMesh.
"""
from .base import Approach, ApproachError


class GmshHexApproach(Approach):
    name = "gmshHex"
    defaults = {
        "nPrisms": 8,
        "firstLayerThickness": 5e-06,
        "expansionRatio": 1.2,
        "prismCoreRatio": 0.5,
        "coreCellSize": 0.05,
    }

    def generate(self, target_case, params):
        raise ApproachError("gmshHex approach not implemented yet (structure stub)")
