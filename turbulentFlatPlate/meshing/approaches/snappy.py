"""
snappy approach – full snappyHexMesh workflow over the whole domain.

  1. blockMesh builds only a coarse *background* box of near-cubic cells.  It
     reaches a few rows below the plate and has grid lines exactly at y = 0
     (the castellated cut then lands on a face, so snapping does not distort
     the wall cells) and at x = 0 (the plate/upstream split at the leading edge
     then falls on a face edge).
  2. snappyHexMesh, all three stages in one pass, against an STL plane at y = 0
     spanning the whole bottom boundary:
       castellate - octree refinement toward the plate (wallLevel within
                    refineDistance, one level coarser per 3x that distance);
                    cells below the plane are removed;
       snap       - the cut boundary is moved onto the plane;
       addLayers  - the prism stack on the plane.  The plane is ONE patch
                    (bottomAll) from inlet to outlet, so the stack never has to
                    taper to nothing at the leading edge.
  3. Collapse to 2-D: octree refinement splits cells in z too, so the `back`
     patch is re-extruded into a single cell of the case depth (extrudeMesh)
     and moved onto z = 0 .. depth (the sampling planes sit at depth/2).
  4. topoSet + createPatch: bottomAll split by face centre into symmetry
     (x < 0, slip `patch`) and bottomWall (x > 0, wall); front + back ->
     frontAndBack (empty).
  5. renumberMesh + checkMesh + patch-set check via _gmsh.finalize.

Two v2512 traps this relies on:
  - The stack is passed as firstLayerThickness + expansionRatio + nPrisms,
    which is also how it is requested (the overall height is derived).  With
    snappy's own `thickness` keyword the layers are silently rolled back
    (0% extrusion) - that was the entire "full-domain snappy cannot add
    layers" problem.
  - snappy writes a 0/ directory; a stale 0/polyMesh shadows constant/polyMesh
    for every later tool (and crashes a re-run), so it is removed after each
    mesh-writing step.
"""
import re
import shutil
from pathlib import Path

from ..common import stack_total_thickness
from . import _gmsh
from .base import Approach, ApproachError

X_INLET = -1.0 / 3.0
X_OUTLET = 2.0
Y_TOP = 1.0
N_BELOW = 4  # background rows below the plate; castellation removes them

BLOCK_MESH_DICT = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}

// Background box for snappyHexMesh: near-cubic cells, grid lines exactly at
// x = 0 (leading edge) and y = 0 (plate).  The rows below y = 0 are cut away.
scale 1;

vertices
(
    (@XI@ @YMIN@ 0)    (0 @YMIN@ 0)    (@XO@ @YMIN@ 0)
    (@XO@ @YTOP@ 0)    (0 @YTOP@ 0)    (@XI@ @YTOP@ 0)
    (@XI@ @YMIN@ @DZ@) (0 @YMIN@ @DZ@) (@XO@ @YMIN@ @DZ@)
    (@XO@ @YTOP@ @DZ@) (0 @YTOP@ @DZ@) (@XI@ @YTOP@ @DZ@)
);

blocks
(
    hex (0 1 4 5 6 7 10 11) (@NUP@ @NY@ 1) simpleGrading (1 1 1)
    hex (1 2 3 4 7 8 9 10) (@NPLATE@ @NY@ 1) simpleGrading (1 1 1)
);

edges ();

boundary
(
    inlet   { type patch;    faces ((0 6 11 5)); }
    outlet  { type patch;    faces ((2 3 9 8)); }
    topWall { type wall;     faces ((5 11 10 4) (4 10 9 3)); }
    yMin    { type patch;    faces ((0 1 7 6) (1 2 8 7)); }
    front   { type symmetry; faces ((6 7 10 11) (7 8 9 10)); }
    back    { type symmetry; faces ((0 5 4 1) (1 4 3 2)); }
);

mergePatchPairs ();
"""

SNAPPY_DICT = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}

castellatedMesh true;
snap            true;
addLayers       true;

geometry
{
    bottomAll { type triSurfaceMesh; file "bottomAll.stl"; name bottomAll; }
}

castellatedMeshControls
{
    maxLocalCells       2000000;
    maxGlobalCells      8000000;
    minRefinementCells  0;
    maxLoadUnbalance    0.1;
    nCellsBetweenLevels @NBETWEEN@;
    features            ();
    refinementSurfaces
    {
        bottomAll { level (@LEVEL@ @LEVEL@); patchInfo { type wall; } }
    }
    resolveFeatureAngle 30;
    refinementRegions
    {
        @REGIONS@
    }
    locationInMesh      @LOCATION@;
    allowFreeStandingZoneFaces true;
}

snapControls
{
    nSmoothPatch        3;
    tolerance           2.0;
    nSolveIter          50;
    nRelaxIter          5;
    nFeatureSnapIter    10;
    implicitFeatureSnap false;
    explicitFeatureSnap false;
}

addLayersControls
{
    relativeSizes       false;
    layers
    {
        bottomAll { nSurfaceLayers @NPRISMS@; }
    }
    // NOT `thickness`: v2512 rolls the whole stack back with it.
    firstLayerThickness @FIRST@;
    expansionRatio      @RATIO@;
    minThickness        @MINTHICK@;
    nGrow               0;
    featureAngle        130;
    slipFeatureAngle    30;
    nRelaxIter          5;
    nSmoothSurfaceNormals 3;
    nSmoothNormals      5;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle  90;
    nBufferCellsNoExtrude 0;
    nLayerIter          50;
    nRelaxedIter        20;
}

meshQualityControls
{
    maxNonOrtho         80;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          180;
    minVol              -1;
    minTetQuality       -1;
    minArea             -1;
    minTwist            -1;
    minDeterminant      -1;
    minFaceWeight       -1;
    minVolRatio         -1;
    minTriangleTwist    -1;
    nSmoothScale        4;
    errorReduction      0.75;
    relaxed { maxNonOrtho 90; }
}

mergeTolerance 1e-6;
"""

TOPO_SET_DICT = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      topoSetDict;
}

// bottomAll split at the leading edge by face centre, as two DISJOINT sets.
// Building bottomWall from the bottomAll patch instead re-takes the upstream
// faces, and createPatch then silently duplicates them (open cells).
actions
(
    { name upstreamFaces; type faceSet; action new;    source patchToFace; patch bottomAll; }
    { name upstreamFaces; type faceSet; action subset; source boxToFace; box (-1e3 -1e3 -1e3) (0 1e3 1e3); }
    { name plateFaces;    type faceSet; action new;    source patchToFace; patch bottomAll; }
    { name plateFaces;    type faceSet; action subset; source boxToFace; box (0 -1e3 -1e3) (1e3 1e3 1e3); }
);
"""

CREATE_PATCH_DICT = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      createPatchDict;
}

pointSync false;

patches
(
    { name symmetry;     patchInfo { type patch; } constructFrom set;     set upstreamFaces; }
    { name bottomWall;   patchInfo { type wall;  } constructFrom set;     set plateFaces; }
    { name frontAndBack; patchInfo { type empty; } constructFrom patches; patches (front back); }
);
"""


def _fill(template, substitutions):
    for key, value in substitutions.items():
        template = template.replace(key, str(value))
    return template


def _plane_stl(x0, x1, z0, z1):
    corners = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
    lines = ["solid bottomAll"]
    for tri in ((0, 1, 2), (0, 2, 3)):
        lines += [" facet normal 0 1 0", "  outer loop"]
        lines += [f"   vertex {corners[i][0]:.9g} 0 {corners[i][1]:.9g}" for i in tri]
        lines += ["  endloop", " endfacet"]
    lines.append("endsolid bottomAll")
    return "\n".join(lines) + "\n"


class SnappyApproach(Approach):
    name = "snappy"
    defaults = {
        # background box (near-cubic cells, face lines at x = 0 and y = 0).
        # The wall cell is backgroundCellSize / 2**wallLevel = 5 mm, matching
        # gmshTri/gmshHex wallCellSize so the three prism approaches differ in
        # topology only.  h is snapped to Y_TOP / round(Y_TOP / h), and 0.02
        # divides 1.0 exactly, so the requested size is what you get.
        "backgroundCellSize": 0.02,
        # octree refinement toward the plate: wallLevel within refineDistance,
        # one level coarser per 3x that distance
        "wallLevel": 2,
        "refineDistance": 0.1,
        "nCellsBetweenLevels": 3,
        "depth": 0.1,
        # prism stack, shared with gmshTri / gmshHex: first layer + expansion
        # ratio + count, handed to snappy verbatim.  The overall height is
        # derived (40 x 1e-6 at 1.2252 = 0.01498 m) and goes in the manifest.
        "nPrisms": 50,
        "firstLayerThickness": 1.0e-06,
        "expansionRatio": 1.17,
    }

    def generate(self, target_case, params):
        p = self.validate(params)
        target = Path(target_case)
        system = target / "system"
        system.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(target / "constant" / "polyMesh", ignore_errors=True)

        # 1. background box.  h divides Y_TOP exactly, so a row boundary sits
        # at y = 0; the upstream / plate blocks meet at x = 0.
        h = Y_TOP / round(Y_TOP / p["backgroundCellSize"])
        (system / "blockMeshDict").write_text(_fill(BLOCK_MESH_DICT, {
            "@XI@": f"{X_INLET:.16g}", "@XO@": f"{X_OUTLET:.16g}",
            "@YMIN@": f"{-N_BELOW * h:.16g}", "@YTOP@": f"{Y_TOP:.16g}",
            "@DZ@": f"{h:.16g}",
            "@NUP@": max(1, round(-X_INLET / h)),
            "@NPLATE@": max(1, round(X_OUTLET / h)),
            "@NY@": round(Y_TOP / h) + N_BELOW,
        }))
        surfaces = target / "constant" / "triSurface"
        surfaces.mkdir(parents=True, exist_ok=True)
        (surfaces / "bottomAll.stl").write_text(
            _plane_stl(X_INLET - 0.5, X_OUTLET + 0.5, -h, 2.0 * h))
        _gmsh._foam(["blockMesh"], target, "log.blockMesh")

        # 2. castellate + snap + layers
        level = int(p["wallLevel"])
        bands = " ".join(f"({p['refineDistance'] * 3 ** i:.6g} {level - i})"
                         for i in range(level))
        regions = (f"bottomAll {{ mode distance; levels ({bands}); }}"
                   if level > 0 else "")
        ratio = float(p["expansionRatio"])
        total = stack_total_thickness(p["firstLayerThickness"], ratio,
                                      int(p["nPrisms"]))
        print(f"    prisms: {int(p['nPrisms'])} layers, ratio {ratio:.4f}, "
              f"stack {total:.3e} m, outermost "
              f"{p['firstLayerThickness'] * ratio ** (int(p['nPrisms']) - 1):.3e} m")
        (system / "snappyHexMeshDict").write_text(_fill(SNAPPY_DICT, {
            "@NBETWEEN@": int(p["nCellsBetweenLevels"]),
            "@LEVEL@": level,
            "@REGIONS@": regions,
            # off every face at any refinement level up to 3
            "@LOCATION@": f"(1.0123 0.5037 {0.3123 * h:.9g})",
            "@NPRISMS@": int(p["nPrisms"]),
            "@FIRST@": f"{p['firstLayerThickness']:.16g}",
            "@RATIO@": f"{ratio:.9f}",
            "@MINTHICK@": f"{p['firstLayerThickness'] / 10.0:.16g}",
        }))
        _gmsh._foam(["snappyHexMesh", "-overwrite"], target, "log.snappyHexMesh")
        shutil.rmtree(target / "0", ignore_errors=True)

        log = (target / "log.snappyHexMesh").read_text()
        extruded = re.findall(r"Extruding \d+ out of \d+ faces \(([\d.]+)%\)", log)
        coverage = float(extruded[-1]) if extruded else 0.0
        if coverage < 50.0:
            raise ApproachError(
                f"prism layers placed on only {coverage:.1f}% of the wall faces - "
                f"snappy rejected the stack, see log.snappyHexMesh")

        # 3. one cell in z at the case depth
        self._collapse_to_2d(target, p["depth"])

        # 4. leading-edge split + frontAndBack
        (system / "topoSetDict").write_text(TOPO_SET_DICT)
        _gmsh._foam(["topoSet", "-constant"], target, "log.topoSet")
        (system / "createPatchDict").write_text(CREATE_PATCH_DICT)
        _gmsh._foam(["createPatch", "-overwrite"], target, "log.createPatch")
        if "Creating duplicate face" in (target / "log.createPatch").read_text():
            raise ApproachError("createPatch duplicated boundary faces (open cells) "
                                "- see log.createPatch")
        shutil.rmtree(target / "0", ignore_errors=True)
        _gmsh.set_patch_types(target)

        # 5. renumber, check, patch-set assertion, manifest metadata
        meta = _gmsh.finalize(target, {
            "tool": "snappyHexMesh (castellate+snap+layers)",
            "params": p,
            "layer_coverage_pct": coverage,
            "expansionRatio": round(ratio, 6),
            "totalLayerThickness": round(total, 9),
            "outermostLayer": round(
                p["firstLayerThickness"] * ratio ** (int(p["nPrisms"]) - 1), 9),
        })
        # finalize reads snappy's "Added" count, which here is the 3-D stage
        # (before the z collapse) - label it as such
        if "layerCells" in meta:
            meta["layerCells3D"] = meta.pop("layerCells")
        return meta

    @staticmethod
    def _collapse_to_2d(target, depth):
        """Re-extrude the `back` patch of the refined 3-D mesh into one cell."""
        work = target / "_snapped3d"
        shutil.rmtree(work, ignore_errors=True)
        (work / "system").mkdir(parents=True)
        for name in ("controlDict", "fvSchemes", "fvSolution"):
            shutil.copy2(target / "system" / name, work / "system" / name)
        (work / "constant").mkdir()
        shutil.move(str(target / "constant" / "polyMesh"),
                    str(work / "constant" / "polyMesh"))
        _gmsh._render("extrudeMeshDict", target, "extrudeMeshDict", {
            "@SOURCE_CASE@": work.resolve(), "@DEPTH@": f"{depth:.16g}"})
        _gmsh._foam(["extrudeMesh"], target, "log.extrudeMesh")
        shutil.rmtree(target / "0", ignore_errors=True)

        _gmsh._run(["checkMesh", "-constant"], target, "log.checkMesh.extruded")
        box = re.search(r"Overall domain bounding box \(\S+ \S+ (\S+)\) \(\S+ \S+ (\S+)\)",
                        (target / "log.checkMesh.extruded").read_text())
        if not box:
            raise ApproachError("could not read the extruded mesh bounding box, "
                                "see log.checkMesh.extruded")
        z_min = float(box.group(1))
        if abs(z_min) > 1e-12:
            _gmsh._foam(["transformPoints", "-translate", f"(0 0 {-z_min:.16g})"],
                        target, "log.transformPoints")
