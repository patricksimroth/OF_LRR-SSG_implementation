"""
blockMesh approach – structured baseline mesh (wall-resolved base mesh).

Params map 1:1 to the variables at the top of base_case/system/blockMeshDict
(GRADING, n_y, n_x, n_x_symmetry).  The approach writes system/meshParams with
the requested values; blockMeshDict picks them up via
#includeIfPresent "meshParams" (defaults stay inline in the dict itself).
"""
from ..common import run_foam
from .base import Approach, ApproachError


class BlockMeshApproach(Approach):
    name = "blockMesh"
    defaults = {"GRADING": 50000, "n_y": 193, "n_x": 113, "n_x_symmetry": 24}

    def generate(self, target_case, params):
        params = self.validate(params)
        system_dir = target_case / "system"
        system_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"{key} {params[key]};" for key in sorted(params)]
        (system_dir / "meshParams").write_text("\n".join(lines) + "\n")
        rc = run_foam(["blockMesh"], target_case, "log.blockMesh")
        if rc != 0:
            raise ApproachError(f"blockMesh failed (rc={rc}), see log.blockMesh")
        return {"tool": "blockMesh", "params": params}
