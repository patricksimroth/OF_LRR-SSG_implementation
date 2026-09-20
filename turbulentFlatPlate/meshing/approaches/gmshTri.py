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
        # 5 mm matches snappy (backgroundCellSize/2**wallLevel) so all three
        # prism approaches share one streamwise wall spacing.
        "wallCellSize": 0.005,
        "farCellSize": 0.06,
        "distMin": 0.01,
        "distMax": 0.4,
        # Prism stack: first layer + expansion ratio + count, handed to snappy
        # verbatim.  The overall height is derived (40 x 1e-6 at 1.2252 =
        # 0.01498 m) and reported in the manifest.
        # nPrisms is set so the outermost layer lands within ~3x of
        # wallCellSize: the stack has to hand over to the isotropic core
        # smoothly, and an abrupt jump produces negative-k overshoots in the
        # log layer that eventually blow the run up (nPrisms 30 gives a 20x
        # jump and diverges around iteration 750).  _gmsh.add_prism_layers
        # warns when the ratio gets large.
        "nPrisms": 50,
        "firstLayerThickness": 1.0e-06,
        "expansionRatio": 1.17,
        "minThickness": 1.0e-09,
    }

    def generate(self, target_case, params):
        p = self.validate(params)
        _gmsh.write_geo(target_case, p)
        _gmsh.run_gmsh(target_case)
        _gmsh.gmsh_to_foam(target_case)
        _gmsh.set_patch_types(target_case)
        stack = _gmsh.add_prism_layers(target_case, p)
        return _gmsh.finalize(
            target_case, {"tool": "gmsh+snappy", "core": "tri", "params": p,
                          **stack}
        )
