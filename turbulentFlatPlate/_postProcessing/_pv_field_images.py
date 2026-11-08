"""
_pv_field_images.py – field images with the mesh on top, one run, one pvbatch session.

Runs inside ParaView's pvbatch, not the project venv:
    pvbatch --force-offscreen-rendering _pv_field_images.py <case.foam> <label> <specs-json>
Each spec: {"out", "field", "component" (None / "Magnitude" / "XX" ...), "title",
"caption", "window" (None = whole domain, else [x0, x1, y0, y1]), "preset", "log"},
optionally "weights" ([[component index or None, weight], ...]: colour by
sum(weight * field[:, index]) instead of field/component, VTK component order)
and "symlog" ({"top": q1, "ref": q2, "decades": n}: colour by
sign(v) log10(1 + |v|/v0), v0 = the area-weighted q2-quantile of |v| in the
window but at least 10^-n of the upper limit, the area-weighted q1-quantile;
symmetric about 0, values beyond it in the end colours, labels at +-powers of
ten).
Called per run by field_images.py; the leading underscore keeps the drivers
from executing it directly.
"""
import json
import sys

import numpy as np
from paraview import servermanager
from paraview.simple import (CellSize, Clip, ColorBy, CreateView, Delete, GetColorTransferFunction,
                             GetScalarBar, Hide, HideScalarBarIfNotNeeded, MergeBlocks,
                             OpenFOAMReader, PythonCalculator, SaveScreenshot, Show)
from vtkmodules.numpy_interface import dataset_adapter as dsa

IMAGE_SIZE = [2800, 1300]
DOMAIN_WIDTH_FRACTION = 0.76   # rest of the width is kept free for the colour bar

foam_path, label, specs = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])

reader = OpenFOAMReader(FileName=foam_path)
reader.CaseType = "Reconstructed Case"
reader.MeshRegions = ["internalMesh"]
reader.CellArrays = sorted({spec["field"] for spec in specs})
reader.UpdatePipelineInformation()
times = reader.TimestepValues
times = list(times) if hasattr(times, "__len__") else [times]
latest = max(times) if times else 0.0
reader.UpdatePipeline(latest)
_, _, _, _, z0, z1 = reader.GetDataInformation().GetBounds()

view = CreateView("RenderView")
view.ViewSize = IMAGE_SIZE
view.ViewTime = latest
view.OrientationAxesVisibility = 0
for name, value in (("UseColorPaletteForBackground", 0),
                    ("BackgroundColorMode", "Single Color")):
    if hasattr(view, name):
        setattr(view, name, value)
view.Background = [1.0, 1.0, 1.0]
view.CameraParallelProjection = 1

clips = {}
temporary = []


def source_for(window):
    """Whole mesh, or the mesh cut exactly at the zoom window: the colour range
    comes from the visible region only, and nothing is drawn outside the window,
    so the colour bar and caption margins stay white (a crinkle clip keeps whole
    cells that stick out under them)."""
    if window is None:
        return reader
    key = tuple(window)
    if key not in clips:
        x0, x1, y0, y1 = window
        clip = Clip(Input=reader)
        clip.ClipType = "Box"
        if hasattr(clip.ClipType, "UseReferenceBounds"):
            clip.ClipType.UseReferenceBounds = 0
        clip.ClipType.Position = [x0, y0, z0 - 1.0]
        clip.ClipType.Length = [x1 - x0, y1 - y0, z1 - z0 + 2.0]
        clip.Crinkleclip = 0
        clip.Exact = 1
        clip.Invert = 1
        clip.UpdatePipeline(latest)
        clips[key] = clip
    return clips[key]


def weighted_source(source, spec):
    """Cell array `budgetTerm` = sum(weight * field[:, index]) on source."""
    field = spec["field"]
    parts = []
    for index, weight in spec["weights"]:
        column = f"inputs[0].CellData['{field}']"
        if index is not None:
            column += f"[:, {int(index)}]"
        parts.append(f"({float(weight)!r})*{column}")
    calc = PythonCalculator(Input=source)
    calc.ArrayAssociation = "Cell Data"
    calc.ArrayName = "budgetTerm"
    calc.Expression = " + ".join(parts)
    calc.UpdatePipeline(latest)
    temporary.append(calc)
    return calc


def abs_quantiles(source, array, qs):
    """Area-weighted quantiles of |array| over the cells of source."""
    sizes = CellSize(Input=MergeBlocks(Input=source))
    sizes.ComputeVertexCount = 0
    sizes.ComputeLength = 0
    sizes.ComputeArea = 0
    sizes.ComputeVolume = 1
    data = dsa.WrapDataObject(servermanager.Fetch(sizes))
    value = np.abs(np.asarray(data.CellData[array]))
    weight = np.abs(np.asarray(data.CellData["Volume"]))
    merged = sizes.Input
    Delete(sizes)
    Delete(merged)
    keep = np.isfinite(value)
    value, weight = value[keep], weight[keep]
    if not value.size:
        return [0.0 for _ in qs]
    order = np.argsort(value)
    cumulative = np.cumsum(weight[order])
    index = [min(int(np.searchsorted(cumulative, q * cumulative[-1])), value.size - 1)
             for q in qs]
    return [float(value[order][i]) for i in index]


def symlog_source(source, array, v0):
    """Cell array `<array>Log` = sign(v) log10(1 + |v|/v0)."""
    calc = PythonCalculator(Input=source)
    calc.ArrayAssociation = "Cell Data"
    calc.ArrayName = array + "Log"
    calc.Expression = (f"numpy.sign(inputs[0].CellData['{array}'])"
                       f"*log10(1.0 + abs(inputs[0].CellData['{array}'])/{v0!r})")
    calc.UpdatePipeline(latest)
    temporary.append(calc)
    return calc


def symlog_labels(v0, top, n_max=3, gap=0.12):
    """[(position, text)]: 0 and +-10^k up to top, at most n_max per side,
    the largest always included, none within gap of the range from 0."""
    level = np.log10(1.0 + top / v0)
    k1 = int(np.floor(np.log10(top)))
    exps = [k for k in range(k1, int(np.floor(np.log10(v0))) - 2, -1)
            if np.log10(1.0 + 10.0 ** k / v0) >= gap * level]
    step = max(1, int(np.ceil(len(exps) / n_max)))
    labels = [(0.0, "0")]
    for k in exps[::step]:
        s = float(np.log10(1.0 + 10.0 ** k / v0))
        labels += [(s, f"$10^{{{k}}}$"), (-s, f"$-10^{{{k}}}$")]
    return sorted(labels)


def fit_camera(x0, x1, y0, y1):
    """Parallel camera along -z; the region fills the left DOMAIN_WIDTH_FRACTION
    of the image, the colour bar sits on the right."""
    aspect = IMAGE_SIZE[0] / IMAGE_SIZE[1]
    # extra vertical headroom so the caption never overlaps a height-limited region
    scale = max(1.12 * 0.5 * (y1 - y0),
                1.04 * 0.5 * (x1 - x0) / (DOMAIN_WIDTH_FRACTION * aspect))
    image_width = 2.0 * scale * aspect
    cx = 0.5 * (x0 + x1) + (0.5 - 0.5 * DOMAIN_WIDTH_FRACTION) * image_width
    cy = 0.5 * (y0 + y1)
    view.CameraFocalPoint = [cx, cy, 0.5 * (z0 + z1)]
    view.CameraPosition = [cx, cy, z1 + 10.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelScale = scale


for spec in specs:
    source = source_for(spec["window"])
    if spec.get("weights"):
        source = weighted_source(source, spec)
        array = "budgetTerm"
    else:
        array = spec["field"]
    symlog = spec.get("symlog")
    if symlog:
        top, ref = abs_quantiles(source, array, (symlog["top"], symlog["ref"]))
        top = top or 1.0
        v0 = max(ref, top * 10.0 ** -symlog["decades"])
        source = symlog_source(source, array, v0)
        array += "Log"
        level = float(np.log10(1.0 + top / v0))
    display = Show(source, view)
    display.SetRepresentationType("Surface With Edges")
    display.EdgeColor = [0.25, 0.25, 0.25]
    if spec["component"]:
        ColorBy(display, ("CELLS", array, spec["component"]))
    else:
        ColorBy(display, ("CELLS", array))
    lut = GetColorTransferFunction(array)
    lut.ApplyPreset(spec["preset"], True)
    lut.UseLogScale = 0
    if symlog:
        lut.AutomaticRescaleRangeMode = "Never"
        lut.RescaleTransferFunction(-level, level)
    else:
        lut.AutomaticRescaleRangeMode = "Grow and update on 'Apply'"
        display.RescaleTransferFunctionToDataRange(False, True)
    if spec["log"]:
        lo, hi = lut.RGBPoints[0], lut.RGBPoints[-4]
        if lo > 0.0 and hi > lo:
            lut.UseLogScale = 1
    display.SetScalarBarVisibility(view, True)
    bar = GetScalarBar(lut, view)
    if symlog:
        labels = symlog_labels(v0, top)
        lut.Annotations = [x for s, text in labels for x in (repr(s), text)]
        bar.DrawAnnotations = 1
        bar.UseCustomLabels = 1
        bar.CustomLabels = [s for s, _ in labels]
        bar.DrawTickLabels = 0
        bar.AddRangeLabels = 0
        bar.TextPosition = "Ticks left/bottom, annotations right/top"
        bar.ScalarBarLength = 0.70
    else:
        lut.Annotations = []
        bar.DrawAnnotations = 0
        bar.UseCustomLabels = 0
        bar.DrawTickLabels = 1
        bar.AddRangeLabels = 1
        bar.TextPosition = "Ticks right/top, annotations left/bottom"
        bar.ScalarBarLength = 0.74
    bar.Title = spec["title"]
    bar.ComponentTitle = ""
    bar.TitleColor = [0.0, 0.0, 0.0]
    bar.LabelColor = [0.0, 0.0, 0.0]
    # %g-style range labels print p (~1.1e5, varying by ~100 Pa) as "1.1e+05"
    # and collide with the tick labels
    bar.RangeLabelFormat = "{:.6g}"   # std::format; printf style is deprecated in 6.1
    bar.TitleFontSize = 44
    bar.LabelFontSize = 36
    bar.TitleBold = 1
    bar.ScalarBarThickness = 48
    # title above the bar: rotated beside it, it collides with wide labels (p ~ 110077)
    bar.HorizontalTitle = 1
    bar.WindowLocation = "Any Location"
    bar.Orientation = "Vertical"
    bar.Position = [0.81, 0.08]

    if spec["window"] is None:
        x0, x1, y0, y1, _, _ = reader.GetDataInformation().GetBounds()
    else:
        x0, x1, y0, y1 = spec["window"]
    fit_camera(x0, x1, y0, y1)

    SaveScreenshot(spec["out"], view, ImageResolution=IMAGE_SIZE,
                   TransparentBackground=0)
    Hide(source, view)
    HideScalarBarIfNotNeeded(lut, view)
    while temporary:
        Delete(temporary.pop())
