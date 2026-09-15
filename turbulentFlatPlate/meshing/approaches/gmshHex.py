"""
gmshHex approach – gmsh triangles turned into their hexagon-dominant polygonal
dual, prism layers by snappy.

Same core as gmshTri, then polyDualMesh: the dual cells sit on the vertices of
the triangulation, so a vertex surrounded by six triangles becomes a hexagon
(mixed in with pentagons and heptagons where the triangulation is irregular).

  1.-3. as gmshTri (triangular core -> constant/polyMesh), but the two z faces
        stay in separate front/back groups - the dual repair needs to tell them
        apart.
  4. polyDualMesh, then the OpenFOAMWiki 2-D repair (setSet -> subsetMesh ->
     combinePatchFaces -> extrudeMesh plane) and a translate back onto the
     baseline z range.
  5. createPatch merges front+back into the frontAndBack patch the case fields
     expect.
  6. snappyHexMesh addLayers -> prism stack on bottomWall.
  7. renumberMesh + checkMesh.
"""
from . import _gmsh
from .base import Approach
from .gmshTri import GmshTriApproach


class GmshHexApproach(Approach):
    name = "gmshHex"
    defaults = dict(GmshTriApproach.defaults, dualFeatureAngle=60)

    def generate(self, target_case, params):
        p = self.validate(params)
        _gmsh.write_geo(target_case, p, split_front_back=True)
        _gmsh.run_gmsh(target_case)
        _gmsh.gmsh_to_foam(target_case)
        _gmsh.set_patch_types(target_case)
        _gmsh.dualize(target_case, p)
        _gmsh.merge_front_back(target_case)
        _gmsh.set_patch_types(target_case)
        _gmsh.add_prism_layers(target_case, p)
        return _gmsh.finalize(
            target_case, {"tool": "gmsh+polyDualMesh+snappy", "core": "polyDual",
                          "params": p}
        )
