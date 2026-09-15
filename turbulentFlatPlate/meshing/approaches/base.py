"""
base.py – Approach interface for the mesh-generation pipeline.

An approach turns a parameter dict into a ready constant/polyMesh inside the
target run directory: it writes its own system/ dicts (blockMeshDict,
snappyHexMeshDict, gmsh .geo, ...) and runs the OpenFOAM tools it needs
(blockMesh / snappyHexMesh / gmsh / gmshToFoam / renumberMesh / ...),
returning mesh metadata for the study manifest.

Params: scalars fixed, lists swept by the runner; `defaults` also feed the
auto-generated combo tag (only deviations appear in the tag).
"""
from pathlib import Path


class ApproachError(RuntimeError):
    pass


class Approach:
    name = ""
    defaults = {}

    def validate(self, params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ApproachError(
                f"[{self.name}] unknown parameters {sorted(unknown)}, "
                f"available: {sorted(self.defaults)}"
            )
        return {**self.defaults, **params}

    def generate(self, target_case: Path, params: dict) -> dict:
        raise NotImplementedError
