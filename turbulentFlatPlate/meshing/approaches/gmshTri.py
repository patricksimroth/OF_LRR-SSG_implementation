"""
gmshTri approach – unstructured triangular gmsh core, prism layers by snappy.

  1. system/plate.geo: rectangle, cell size graded away from the plate,
     extruded one cell in z (Layers{1}; Recombine -> triangular prisms).
  2. gmsh -3 -format msh2 -> gmshToFoam -> constant/polyMesh.
  3. patch types fixed up in constant/polyMesh/boundary.
  4. snappyHexMesh addLayers -> prism stack on bottomWall.
  5. renumberMesh + checkMesh.

The shared steps live in _gmsh.py; gmshHex is the same pipeline with
polyDualMesh inserted after step 3.
"""
from . import _gmsh
from .base import Approach


class GmshTriApproach(Approach):
    name = "gmshTri"
    defaults = {
        # geometry (matches base_case/system/blockMeshDict after its scale)
        "xInlet": -0.3333333333333333,
        "xOutlet": 2.0,
        "yTop": 1.0,
        "depth": 0.1,
        # core sizing
        "wallCellSize": 0.004,
        "farCellSize": 0.06,
        "distMin": 0.01,
        "distMax": 0.4,
        # prism layers (absolute sizes).  nPrisms is set so the outermost layer
        # lands within ~3x of wallCellSize: the stack has to hand over to the
        # isotropic core smoothly, and an abrupt jump produces negative-k
        # overshoots in the log layer that eventually blow the run up
        # (nPrisms 30 gives a 20x jump and diverges around iteration 750).
        # _gmsh.add_prism_layers warns when the ratio gets large.
        "nPrisms": 40,
        "firstLayerThickness": 1.0e-06,
        "totalLayerThickness": 0.015,   # overall stack height; ratio is derived
        "minThickness": 1.0e-09,
    }

    def generate(self, target_case, params):
        p = self.validate(params)
        _gmsh.write_geo(target_case, p)
        _gmsh.run_gmsh(target_case)
        _gmsh.gmsh_to_foam(target_case)
        _gmsh.set_patch_types(target_case)
        _gmsh.add_prism_layers(target_case, p)
        return _gmsh.finalize(
            target_case, {"tool": "gmsh+snappy", "core": "tri", "params": p}
        )
