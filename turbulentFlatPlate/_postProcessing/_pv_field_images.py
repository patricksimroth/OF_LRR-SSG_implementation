"""
_pv_field_images.py – field images with the mesh on top, one run, one pvbatch session.

Runs inside ParaView's pvbatch, not the project venv:
    pvbatch --force-offscreen-rendering _pv_field_images.py <case.foam> <label> <specs-json>
Each spec: {"out", "field", "component" (None / "Magnitude" / "XX" ...), "title",
"caption", "window" (None = whole domain, else [x0, x1, y0, y1]), "preset", "log"}.
Called per run by field_images.py; the leading underscore keeps the drivers
from executing it directly.
"""
import json
import sys

from paraview.simple import (Clip, ColorBy, CreateView, GetColorTransferFunction,
                             GetScalarBar, Hide, HideScalarBarIfNotNeeded,
                             OpenFOAMReader, SaveScreenshot, Show, Text)

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

caption_source = Text()
caption = Show(caption_source, view)
caption.Color = [0.0, 0.0, 0.0]
caption.WindowLocation = "Upper Left Corner"

clips = {}


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


previous = None
for spec in specs:
    if previous is not None:
        prev_source, prev_lut = previous
        Hide(prev_source, view)
        HideScalarBarIfNotNeeded(prev_lut, view)

    source = source_for(spec["window"])
    display = Show(source, view)
    display.SetRepresentationType("Surface With Edges")
    display.EdgeColor = [0.25, 0.25, 0.25]
    if spec["component"]:
        ColorBy(display, ("CELLS", spec["field"], spec["component"]))
    else:
        ColorBy(display, ("CELLS", spec["field"]))
    lut = GetColorTransferFunction(spec["field"])
    lut.ApplyPreset(spec["preset"], True)
    lut.UseLogScale = 0
    display.RescaleTransferFunctionToDataRange(False, True)
    if spec["log"]:
        lo, hi = lut.RGBPoints[0], lut.RGBPoints[-4]
        if lo > 0.0 and hi > lo:
            lut.UseLogScale = 1
    display.SetScalarBarVisibility(view, True)
    bar = GetScalarBar(lut, view)
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
    bar.ScalarBarLength = 0.74

    caption_source.Text = f"{label}   {spec['caption']}, time {latest:g}"
    if spec["window"] is None:
        x0, x1, y0, y1, _, _ = reader.GetDataInformation().GetBounds()
    else:
        x0, x1, y0, y1 = spec["window"]
    fit_camera(x0, x1, y0, y1)

    SaveScreenshot(spec["out"], view, ImageResolution=IMAGE_SIZE,
                   TransparentBackground=0)
    previous = (source, lut)
