from .base import Approach, ApproachError
from .blockMesh import BlockMeshApproach
from .gmshHex import GmshHexApproach
from .gmshTri import GmshTriApproach
from .snappy import SnappyApproach
from .snappy45 import Snappy45Approach

REGISTRY = {
    approach.name: approach
    for approach in (
        BlockMeshApproach(),
        SnappyApproach(),
        Snappy45Approach(),
        GmshTriApproach(),
        GmshHexApproach(),
    )
}


def get_approach(name):
    if name not in REGISTRY:
        raise ApproachError(f"unknown approach '{name}', available: {sorted(REGISTRY)}")
    return REGISTRY[name]
