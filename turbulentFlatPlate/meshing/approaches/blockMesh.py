"""
blockMesh approach – structured baseline mesh (wall-resolved base mesh).

Params map to the variables at the top of base_case/system/blockMeshDict
(n_y, n_x, n_x_symmetry) plus the shared firstLayerThickness.  The wall-normal
GRADING is derived from it, so the wall cell is exactly as thick as the first
prism layer of snappy/gmshTri/gmshHex and all approaches see the same wall y+.
The approach writes system/meshParams with the resulting values; blockMeshDict
picks them up via #includeIfPresent "meshParams" (defaults stay inline in the
dict itself).
"""
import re

from ..common import run_foam, stack_expansion_ratio
from .base import Approach, ApproachError

# Wall-normal extent of both blocks: y 0..3 in blockMeshDict at scale 1/3.
DOMAIN_HEIGHT = 1.0
# Streamwise lengths of the plate block (x 0..6) and the upstream block (x -1..0).
PLATE_LENGTH = 2.0
SYMMETRY_LENGTH = 1.0 / 3.0


class BlockMeshApproach(Approach):
    name = "blockMesh"
    # gradingX is the plate block's streamwise last/first cell ratio; with
    # n_x 248 it puts dx at x = 1 m within ~2x of the gmsh meshes instead of
    # ~11x (see the GRADING_X comment in base_case/system/blockMeshDict).
    # dxSymmetryLE > 0 replaces the upstream block's fixed two-section grading
    # (4.8 mm at x = 0) by one ratio ending in a cell of that size at x = 0.
    # Matching it to a fine plate cell (0.56 mm on both sides) diverges within
    # ~150 iterations for every model; refining only one side is stable.
    defaults = {"firstLayerThickness": 1.0e-06, "n_y": 193, "n_x": 248,
                "n_x_symmetry": 24, "gradingX": 3.0, "dxSymmetryLE": 0.0}

    def generate(self, target_case, params):
        params = self.validate(params)
        # n_y cells growing by r fill the height like an n_y-layer prism stack;
        # simpleGrading is last/first cell = r**(n_y - 1).
        n_y = params["n_y"]
        ratio = stack_expansion_ratio(params["firstLayerThickness"],
                                      DOMAIN_HEIGHT, n_y)
        grading = ratio ** (n_y - 1)
        dict_params = {"GRADING": f"{grading:.10g}", "n_y": n_y,
                       "n_x": params["n_x"], "n_x_symmetry": params["n_x_symmetry"],
                       "GRADING_X": f'{params["gradingX"]:.10g}'}
        n_x, g_x = params["n_x"], params["gradingX"]
        r_x = g_x ** (1.0 / (n_x - 1)) if g_x != 1 else 1.0
        leading_edge = {"dxLeadingEdge": (PLATE_LENGTH * (r_x - 1.0) / (r_x ** n_x - 1.0)
                                          if g_x != 1 else PLATE_LENGTH / n_x)}
        if params["dxSymmetryLE"] > 0:
            n_sym = params["n_x_symmetry"]
            # Growth away from x = 0, then last/first cell toward the plate.
            r_sym = stack_expansion_ratio(params["dxSymmetryLE"], SYMMETRY_LENGTH, n_sym)
            dict_params["SYM_GRADING_X"] = f"{r_sym ** -(n_sym - 1):.10g}"
            leading_edge["symmetryRatio"] = round(r_sym, 5)
        system_dir = target_case / "system"
        system_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"{key} {dict_params[key]};" for key in sorted(dict_params)]
        (system_dir / "meshParams").write_text("\n".join(lines) + "\n")
        rc = run_foam(["blockMesh"], target_case, "log.blockMesh")
        if rc != 0:
            raise ApproachError(f"blockMesh failed (rc={rc}), see log.blockMesh")
        # nCells in the manifest so mesh_study.py can tabulate every approach
        # on the same footing (snappy and the gmsh approaches report it too).
        match = re.search(r"^\s*nCells:\s+(\d+)",
                          (target_case / "log.blockMesh").read_text(), re.M)
        return {"tool": "blockMesh", "params": params,
                "GRADING": round(grading, 4), **leading_edge,
                "nCells": int(match.group(1)) if match else None}
