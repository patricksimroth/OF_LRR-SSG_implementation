"""
snappy45 approach – full-domain snappyHexMesh like `snappy`, but with the core
grid tilted by `angle` (45 deg) against the flow.

Everything is built in a computational frame C in which the background box is
axis-aligned and the domain is rotated by -angle about z; the finished mesh is
rotated back by +angle.  In C the snappy dict is the same as snappy's:

  1. Background box: squares of backgroundCellSize covering the rotated domain,
     one cell in z of depth backgroundCellSize / 2**wallLevel.  Its corner is
     shifted so the plate plane runs midway between two rows of the finest
     cells' centres and vertices (a plane through cell centres leaves the
     castellation undecided).
  2. snappyHexMesh against the four domain sides as separate STL planes
     (bottomAll from inlet to outlet, inlet, outlet, topWall), overhanging in
     z.  None of them is grid-aligned, so all four are castellated and
     snapped; the domain corners and the leading edge are explicit feature
     edges, so the corners stay sharp and a wall vertex lands on x = 0.
     Refinement is the same directional x-y band refinement as snappy
     (wallLevel within refineDistance of the plate, one level coarser per 3x
     that distance), with the bands as searchableRotatedBox parallel to the
     plate.  Directional x-y refinement in C is isotropic in the plane, so
     the refined cells stay square at the tilt angle.  Prism stack on
     bottomAll, identical to snappy's.
  3. Collapse to 2-D (snappy's _collapse_to_2d), rotate by +angle about z.
  4. topoSet + createPatch: bottomAll split at x = 0 into symmetry / bottomWall,
     front + back -> frontAndBack; the background box sides have no faces left
     and createPatch drops them.
  5. renumberMesh + checkMesh + patch-set check via _gmsh.finalize.

wallLevel / refineDistance mean the same as in snappy.  The snapped staircase
of the tilted grid leaves wall faces h*sqrt(2) long (snappy merges each pair of
staircase faces), so backgroundCellSize defaults to 0.02/sqrt(2): the
streamwise wall spacing then matches the other approaches (5 mm at wallLevel 2)
and the core squares are 3.54 mm.  The measured spacing goes in the manifest as
wallFaces / wallFaceSpacing.
"""
import math
import re
import shutil
from pathlib import Path

from ..common import stack_total_thickness
from . import _gmsh
from .base import ApproachError
from .snappy import (CREATE_PATCH_DICT, SNAPPY_DICT, TOPO_SET_DICT, X_INLET,
                     X_OUTLET, Y_TOP, SnappyApproach, _fill)

BLOCK_MESH_DICT = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}

// Background box for snappyHexMesh in the rotated frame: squares, one cell in
// z.  Every side lies outside the domain and is cut away.
scale 1;

vertices
(
    (@X0@ @Y0@ 0)    (@X1@ @Y0@ 0)    (@X1@ @Y1@ 0)    (@X0@ @Y1@ 0)
    (@X0@ @Y0@ @DZ@) (@X1@ @Y0@ @DZ@) (@X1@ @Y1@ @DZ@) (@X0@ @Y1@ @DZ@)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (@NX@ @NY@ 1) simpleGrading (1 1 1)
);

edges ();

boundary
(
    outerBox { type patch;    faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2)); }
    front    { type symmetry; faces ((4 5 6 7)); }
    back     { type symmetry; faces ((0 3 2 1)); }
);

mergePatchPairs ();
"""

# the four domain sides in the physical frame: (name, patch type, start, end)
SIDES = (
    ("bottomAll", "wall",  (X_INLET, 0.0), (X_OUTLET, 0.0)),
    ("outlet",    "patch", (X_OUTLET, 0.0), (X_OUTLET, Y_TOP)),
    ("topWall",   "wall",  (X_OUTLET, Y_TOP), (X_INLET, Y_TOP)),
    ("inlet",     "patch", (X_INLET, Y_TOP), (X_INLET, 0.0)),
)
FEATURE_POINTS = ((X_INLET, 0.0), (0.0, 0.0), (X_OUTLET, 0.0),
                  (X_OUTLET, Y_TOP), (X_INLET, Y_TOP))


def _rotate(xy, angle):
    c, s = math.cos(angle), math.sin(angle)
    return (c * xy[0] - s * xy[1], s * xy[0] + c * xy[1])


def _strip_stl(name, a, b, z0, z1):
    """Planar strip from a to b (x-y), z0..z1, as two triangles."""
    corners = [(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)]
    lines = [f"solid {name}"]
    for tri in ((0, 1, 2), (0, 2, 3)):
        lines += [" facet normal 0 0 0", "  outer loop"]
        lines += ["   vertex " + " ".join(f"{v:.16g}" for v in corners[i]) for i in tri]
        lines += ["  endloop", " endfacet"]
    lines.append(f"endsolid {name}")
    return "\n".join(lines) + "\n"


def _feature_edges(points, z0, z1):
    """featureEdgeMesh: one z-parallel line through every x-y point."""
    verts = []
    for x, y in points:
        verts += [f"({x:.16g} {y:.16g} {z0:.16g})", f"({x:.16g} {y:.16g} {z1:.16g})"]
    edges = [f"({2 * i} {2 * i + 1})" for i in range(len(points))]
    return ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
            "    class       featureEdgeMesh;\n    object      domain.eMesh;\n}\n\n"
            f"{len(verts)}\n(\n" + "\n".join(verts) + "\n)\n\n"
            f"{len(edges)}\n(\n" + "\n".join(edges) + "\n)\n")


class Snappy45Approach(SnappyApproach):
    name = "snappy45"
    defaults = {
        **SnappyApproach.defaults,
        # tilted squares of h snap to wall faces of h*sqrt(2) (pairs of
        # staircase faces merged), so h = 0.02/sqrt(2) gives the same 5 mm
        # streamwise wall spacing as snappy / gmshTri / gmshHex at wallLevel 2
        "backgroundCellSize": 0.02 / math.sqrt(2.0),
        "angle": 45.0,
        # the refinement bands stop at this height: a level transition that
        # reaches the snapped outlet/top corner (wallLevel 4: 0.675 m,
        # wallLevel 5: 1.01 m) blows the solver up within ~100 iterations
        "maxRefineHeight": 0.45,
        # bands above the finest one grow from this distance instead of
        # 3*refineDistance (0 = the snappy layout).  wallLevel 5
        # with refineDistance's 3x bands never converges the energy equation
        # (crash ~iteration 45); 0.025 reuses wallLevel 4's working bands
        "farRefineDistance": 0.0,
        # above outerWallLevel the bands of levels <= outerWallLevel are those
        # of the outerWallLevel mesh (outerRefineDistance * 3**i: 0.05 / 0.15 /
        # 0.45 m), and every finer level k sits in a strip of
        # outerRefineDistance / 2**(k - outerWallLevel) at the wall (25 mm for
        # level 4, 12.5 mm for level 5).  With snappy's bands (level
        # transitions at 0.075 / 0.225 / 0.675 m for wallLevel 4) the steady
        # state is unstable: a pressure mode trapped between the transitions
        # grows ~2.7x per 10 iterations even from a converged solution
        "outerWallLevel": 3,
        "outerRefineDistance": 0.05,
    }

    def _band_tops(self, p):
        level = int(p["wallLevel"])
        outer = int(p["outerWallLevel"])
        if outer and level > outer:
            d = p["outerRefineDistance"]
            tops = [d / 2 ** (level - i - outer) if level - i > outer
                    else d * 3 ** (outer - (level - i)) for i in range(level)]
        else:
            far = p["farRefineDistance"] or 3.0 * p["refineDistance"]
            tops = [p["refineDistance"] if i == 0 else far * 3 ** (i - 1)
                    for i in range(level)]
        return [min(top, p["maxRefineHeight"]) for top in tops]

    def generate(self, target_case, params):
        p = self.validate(params)
        target = Path(target_case)
        system = target / "system"
        system.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(target / "constant" / "polyMesh", ignore_errors=True)

        theta = math.radians(float(p["angle"]))
        h = float(p["backgroundCellSize"])
        level = int(p["wallLevel"])
        h_fine = h / 2 ** level
        dz = h_fine

        def to_c(xy):
            return _rotate(xy, -theta)

        # 1. background box in C.  The plate in C is the line through the
        # origin along to_c((1, 0)); the physical y of a C point is
        # (-sin, cos).(x, y), so offsetting the corner along that normal by
        # h_fine/2 (modulo h_fine) puts the plane between finest rows.
        corners = [to_c(xy) for xy in ((X_INLET, 0.0), (X_OUTLET, 0.0),
                                       (X_OUTLET, Y_TOP), (X_INLET, Y_TOP))]
        pad = 2.0 * h
        x0 = math.floor((min(c[0] for c in corners) - pad) / h) * h
        y0 = math.floor((min(c[1] for c in corners) - pad) / h) * h
        nx = math.ceil((max(c[0] for c in corners) + pad - x0) / h)
        ny = math.ceil((max(c[1] for c in corners) + pad - y0) / h)
        normal = (math.sin(theta), math.cos(theta))  # physical y-axis in C
        y_phys0 = normal[0] * x0 + normal[1] * y0
        shift = (0.5 * h_fine - y_phys0) % h_fine
        if abs(normal[1]) > 1e-9:
            y0 += shift / normal[1]
        else:
            x0 += shift / normal[0]
        (system / "blockMeshDict").write_text(_fill(BLOCK_MESH_DICT, {
            "@X0@": f"{x0:.16g}", "@X1@": f"{x0 + nx * h:.16g}",
            "@Y0@": f"{y0:.16g}", "@Y1@": f"{y0 + ny * h:.16g}",
            "@DZ@": f"{dz:.16g}", "@NX@": nx, "@NY@": ny,
        }))

        surfaces = target / "constant" / "triSurface"
        surfaces.mkdir(parents=True, exist_ok=True)
        overhang = 2.0 * h
        for name, _, a, b in SIDES:
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            t = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
            a_ext = (a[0] - overhang * t[0], a[1] - overhang * t[1])
            b_ext = (b[0] + overhang * t[0], b[1] + overhang * t[1])
            (surfaces / f"{name}.stl").write_text(
                _strip_stl(name, to_c(a_ext), to_c(b_ext), -h, 2.0 * h))
        (surfaces / "domain.eMesh").write_text(
            _feature_edges([to_c(xy) for xy in FEATURE_POINTS], -h, 2.0 * h))
        _gmsh._foam(["blockMesh"], target, "log.blockMesh")

        # 2. castellate + snap + layers
        e1 = to_c((1.0, 0.0))
        boxes, regions = [], []
        for i, top in enumerate(self._band_tops(p)):
            origin = to_c((X_INLET - 1.0, -h))
            boxes.append(
                f"refineBox{i} {{ type searchableRotatedBox; "
                f"span ({X_OUTLET - X_INLET + 2.0:.16g} {top + h:.16g} 3); "
                f"origin ({origin[0]:.16g} {origin[1]:.16g} -1); "
                f"e1 ({e1[0]:.16g} {e1[1]:.16g} 0); e3 (0 0 1); }}")
            regions.append(f"refineBox{i} {{ mode inside; levels ((1e15 0)); "
                           f"levelIncrement (0 0 ({level - i} {level - i} 0)); }}")
        side_geometry = [f'{name} {{ type triSurfaceMesh; file "{name}.stl"; name {name}; }}'
                         for name, *_ in SIDES]
        side_surfaces = "\n        ".join(
            f"{name} {{ level (0 0); patchInfo {{ type {kind}; }} }}"
            for name, kind, *_ in SIDES)
        ratio = float(p["expansionRatio"])
        n = int(p["nPrisms"])
        total = stack_total_thickness(p["firstLayerThickness"], ratio, n)
        print(f"    prisms: {n} layers, ratio {ratio:.4f}, stack {total:.3e} m, "
              f"outermost {p['firstLayerThickness'] * ratio ** (n - 1):.3e} m")
        inside = to_c((1.0123, 0.5037))
        template = SNAPPY_DICT.replace(
            '    bottomAll { type triSurfaceMesh; file "bottomAll.stl"; name bottomAll; }\n', "")
        template = template.replace(
            "bottomAll { level (0 0); patchInfo { type wall; } }", side_surfaces)
        template = template.replace(
            "    features            ();",
            '    features            ({ file "domain.eMesh"; level 0; });')
        template = template.replace(
            "    explicitFeatureSnap false;", "    explicitFeatureSnap true;\n"
            "    multiRegionFeatureSnap true;")
        snappy_dict = _fill(template, {
            "@BOXES@": "\n    ".join(side_geometry + boxes),
            "@REGIONS@": "\n        ".join(regions),
            "@LOCATION@": f"({inside[0]:.9g} {inside[1]:.9g} {0.3123 * dz:.9g})",
            "@NPRISMS@": n,
            "@FIRST@": f"{p['firstLayerThickness']:.16g}",
            "@RATIO@": f"{ratio:.9f}",
            "@MINTHICK@": f"{p['firstLayerThickness'] / 10.0:.16g}",
        })
        (system / "snappyHexMeshDict").write_text(snappy_dict)
        _gmsh._foam(["snappyHexMesh", "-overwrite"], target, "log.snappyHexMesh")
        shutil.rmtree(target / "0", ignore_errors=True)

        log = (target / "log.snappyHexMesh").read_text()
        extruded = re.findall(r"Extruding \d+ out of \d+ faces \(([\d.]+)%\)", log)
        coverage = float(extruded[-1]) if extruded else 0.0
        if coverage < 50.0:
            raise ApproachError(
                f"prism layers placed on only {coverage:.1f}% of the wall faces - "
                f"snappy rejected the stack, see log.snappyHexMesh")

        # 3. one cell in z, back to the physical frame
        self._collapse_to_2d(target, p["depth"])
        _gmsh._foam(["transformPoints", "-rotate-z", f"{float(p['angle']):.16g}"],
                    target, "log.transformPoints.rotate")

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

        # a capped refinement would leave the plate with background-sized
        # faces; the snapped faces of the finest level are h_fine/sqrt(2) ..
        # h_fine*sqrt(2) long
        boundary = (target / "constant" / "polyMesh" / "boundary").read_text()
        wall = re.search(r"bottomWall\s*\{[^}]*?nFaces\s+(\d+);", boundary)
        n_wall = int(wall.group(1)) if wall else 0
        minimum = X_OUTLET / (h_fine * math.sqrt(2.0)) * 0.9
        if n_wall < minimum:
            raise ApproachError(
                f"bottomWall has {n_wall} faces, expected at least {minimum:.0f} "
                f"for wallLevel {level} - refinement was capped, see "
                f"log.snappyHexMesh")

        # 5. renumber, check, patch-set assertion, manifest metadata
        meta = _gmsh.finalize(target, {
            "tool": f"snappyHexMesh (castellate+snap+layers), core tilted "
                    f"{float(p['angle']):g} deg",
            "params": p,
            "layer_coverage_pct": coverage,
            "expansionRatio": round(ratio, 6),
            "totalLayerThickness": round(total, 9),
            "outermostLayer": round(p["firstLayerThickness"] * ratio ** (n - 1), 9),
            "wallFaces": n_wall,
            "wallFaceSpacing": round(X_OUTLET / n_wall, 9) if n_wall else None,
        })
        if "layerCells" in meta:
            meta["layerCells3D"] = meta.pop("layerCells")
        return meta
