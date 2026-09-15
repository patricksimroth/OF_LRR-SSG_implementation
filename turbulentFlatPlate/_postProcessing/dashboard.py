"""
dashboard.py – Static HTML results dashboard for the turbulentFlatPlate case.
Adapted from the backwards_step_tut dashboard with the same lightbox, gallery,
and log-rendering logic. The validation section shows the u+/y+ plot produced
by postpro.py and a dedicated Cf Validation section from cf_validation.py.
"""
import csv
import html
# import os
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import _pp_common as pp

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
# ---------------------------------------------------------------------------
# Lightbox  (identical to backwards_step_tut)
# ---------------------------------------------------------------------------
LIGHTBOX_CSS = """
body.lightbox-open { overflow: hidden; }
.lightbox[hidden] { display: none; }
.lightbox {
  position: fixed; inset: 0; z-index: 1000;
  display: grid; grid-template-rows: auto minmax(0, 1fr) auto;
  gap: 12px; padding: 16px;
  background: rgba(10, 15, 20, 0.94); color: #fff;
}
.lightbox-bar { display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 12px; }
.lightbox-title { margin: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #f4f7fa; }
.lightbox-counter { color: #c8d0d7; font-size: 13px; }
.lightbox-stage { position: relative; min-height: 0; display: grid; place-items: center; }
.lightbox-stage img { max-width: 100%; max-height: 100%; object-fit: contain; background: #fff; border-radius: 4px; }
.lightbox-button {
  border: 1px solid rgba(255,255,255,0.28); background: rgba(255,255,255,0.12);
  color: #fff; border-radius: 6px; min-width: 38px; min-height: 38px;
  padding: 7px 11px; font: inherit; cursor: pointer;
}
.lightbox-button:hover { background: rgba(255,255,255,0.2); }
.lightbox-nav { position: absolute; top: 50%; transform: translateY(-50%); width: 44px; height: 64px; font-size: 30px; }
.lightbox-prev { left: 0; }
.lightbox-next { right: 0; }
.lightbox-help { margin: 0; text-align: center; color: #c8d0d7; font-size: 13px; }
@media (max-width: 720px) {
  .lightbox { padding: 10px; }
  .lightbox-nav { width: 38px; height: 54px; }
}
"""
LIGHTBOX_JS = """
<script>
(function () {
  const links = Array.from(document.querySelectorAll('.plot'));
  if (!links.length) return;
  const overlay = document.createElement('div');
  overlay.className = 'lightbox';
  overlay.hidden = true;
  overlay.innerHTML = `
    <div class="lightbox-bar">
      <div>
        <p class="lightbox-title"></p>
        <span class="lightbox-counter"></span>
      </div>
      <button class="lightbox-button lightbox-close" type="button">Close</button>
    </div>
    <div class="lightbox-stage">
      <button class="lightbox-button lightbox-nav lightbox-prev" type="button">&#8249;</button>
      <img alt="">
      <button class="lightbox-button lightbox-nav lightbox-next" type="button">&#8250;</button>
    </div>
    <p class="lightbox-help">ESC closes &middot; Arrow keys navigate</p>
  `;
  document.body.appendChild(overlay);
  const titleEl   = overlay.querySelector('.lightbox-title');
  const counterEl = overlay.querySelector('.lightbox-counter');
  const imageEl   = overlay.querySelector('img');
  const closeBtn  = overlay.querySelector('.lightbox-close');
  const prevBtn   = overlay.querySelector('.lightbox-prev');
  const nextBtn   = overlay.querySelector('.lightbox-next');
  let gallery = [], current = 0;
  function galleryFromLink(link) {
    const grid = link.closest('.plot-grid');
    const items = grid ? Array.from(grid.querySelectorAll('.plot')) : [link];
    return items.map(item => ({
      href:  item.href,
      label: item.querySelector('span')?.textContent.trim() || item.querySelector('img')?.alt || item.href
    }));
  }
  function show(idx) {
    if (!gallery.length) return;
    current = (idx + gallery.length) % gallery.length;
    const item = gallery[current];
    imageEl.src = item.href; imageEl.alt = item.label;
    titleEl.textContent = item.label;
    counterEl.textContent = `${current + 1} / ${gallery.length}`;
    prevBtn.hidden = nextBtn.hidden = gallery.length < 2;
  }
  function open(link) {
    gallery = galleryFromLink(link);
    current = gallery.findIndex(i => i.href === link.href);
    if (current < 0) current = 0;
    show(current);
    overlay.hidden = false;
    document.body.classList.add('lightbox-open');
    closeBtn.focus();
  }
  function close() {
    overlay.hidden = true;
    document.body.classList.remove('lightbox-open');
    imageEl.removeAttribute('src');
  }
  links.forEach(link => link.addEventListener('click', e => { e.preventDefault(); open(link); }));
  closeBtn.addEventListener('click', close);
  prevBtn.addEventListener('click',  () => show(current - 1));
  nextBtn.addEventListener('click',  () => show(current + 1));
  overlay.addEventListener('click',  e => { if (e.target === overlay) close(); });
  document.addEventListener('keydown', e => {
    if (overlay.hidden) return;
    if (e.key === 'Escape')      close();
    if (e.key === 'ArrowLeft')   show(current - 1);
    if (e.key === 'ArrowRight')  show(current + 1);
  });
})();
</script>
"""
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]
try:
    import re
except ImportError:
    pass
def image_files(directory: Path):
    if not directory or not directory.is_dir():
        return []
    return sorted(
        [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )
def log_files(directory: Path):
    if not directory or not directory.is_dir():
        return []
    return sorted(
        [p for p in directory.iterdir() if p.suffix == ".log" or p.name.startswith("log.")],
        key=lambda p: p.name.lower(),
    )
def model_names(run_case_dir: Path, results_root: Path):
    names = set()
    for src in [run_case_dir, results_root]:
        if src.is_dir():
            for item in src.iterdir():
                if item.is_dir() and not item.name.startswith("_") and item.name not in ("validation",):
                    names.add(item.name)
    return sorted(names)
def img_src(image_path: Path, results_root: Path):
    try:
        rel = image_path.relative_to(results_root)
        return quote(str(rel).replace("\\", "/"))
    except ValueError:
        return quote(str(image_path).replace("\\", "/"))
def render_gallery(images, results_root: Path, empty_msg="No plots found."):
    if not images:
        return f'<p class="empty">{html.escape(empty_msg)}</p>'
    items = []
    for p in images:
        src   = img_src(p, results_root)
        label = html.escape(p.name)
        items.append(
            f'<a class="plot" href="{src}">'
            f'<img src="{src}" alt="{label}" loading="lazy">'
            f'<span>{label}</span>'
            f"</a>"
        )
    return f'<div class="plot-grid">{"".join(items)}</div>'
def render_logs(log_list, results_root: Path):
    if not log_list:
        return '<p class="empty">No log files found.</p>'
    items = []
    for p in log_list:
        try:
            rel = p.relative_to(results_root)
            href = quote(str(rel).replace("\\", "/"))
        except ValueError:
            href = quote(str(p).replace("\\", "/"))
        items.append(f'<li><a href="{href}">{html.escape(p.name)}</a></li>')
    return f'<ul class="links">{"".join(items)}</ul>'
def render_model_summary(models, run_case_dir: Path, results_root: Path):
    if not models:
        return '<p class="empty">No models found.</p>'
    rows = []
    for m in models:
        conv_dir = results_root / m / "convergence_plots"
        n_plots  = len(image_files(conv_dir))
        n_logs   = len(log_files(run_case_dir / m))
        rows.append(
            f"<tr><td>{html.escape(m)}</td><td>{n_plots}</td><td>{n_logs}</td></tr>"
        )
    header = "<tr><th>Model</th><th>Convergence plots</th><th>Log files</th></tr>"
    return f"<table>{header}{''.join(rows)}</table>"
# ---------------------------------------------------------------------------
# Dashboard builder
# ---------------------------------------------------------------------------
def render_cf_section(cf_plot, val_dir: Path, results_root: Path):
    if cf_plot is None:
        return ('<p class="empty">No Cf validation plot found – '
                "run cf_validation.py first.</p>")
    parts = [
        f'<p><img src="{img_src(cf_plot, results_root)}" alt="Cf validation" '
        'style="max-width:100%;"></p>'
    ]
    csv_path = val_dir / "cf_validation.csv"
    if csv_path.is_file():
        columns = ("model", "station_x", "cf_station", "cf_tmr",
                   "ref_error_pct", "yp_min", "yp_max", "log_rms", "log_max")
        labels = {"model": "Model", "station_x": "x [m]", "cf_station": "Cf",
                  "cf_tmr": "Cf (TMR)", "ref_error_pct": "err [%]",
                  "yp_min": "y⁺ min", "yp_max": "y⁺ max",
                  "log_rms": "log-law RMS", "log_max": "log-law max"}
        with open(csv_path, newline="") as stream:
            rows = list(csv.reader(stream))
        if len(rows) >= 2:
            header = rows[0]
            idx = {name: header.index(name) for name in columns if name in header}
            if idx:
                th = "".join(f"<th>{labels[c]}</th>" for c in columns if c in idx)
                trs = []
                for row in rows[1:]:
                    tds = "".join(
                        f"<td>{html.escape(row[i])}</td>"
                        for c, i in idx.items() if i < len(row)
                    )
                    trs.append(f"<tr>{tds}</tr>")
                parts.append(f"<table><tr>{th}</tr>{''.join(trs)}</table>")
    links = []
    for name in ("cf_validation.csv", "cf_validation.txt"):
        p = val_dir / name
        if p.is_file():
            rel = quote(str(p.relative_to(results_root)).replace("\\", "/"))
            links.append(f'<li><a href="{rel}">{html.escape(name)}</a></li>')
    if links:
        parts.append(f'<ul class="links">{"".join(links)}</ul>')
    return "".join(parts)

def build_dashboard(case_name, run_case_dir: Path, results_root: Path):
    models       = model_names(run_case_dir, results_root)
    val_dir      = results_root / "validation"
    all_logs_run = log_files(run_case_dir)
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    # Collect validation images (u+/y+ plot)
    val_images = image_files(val_dir)
    # Separate the individual validation figures from the gallery
    ux_y_plot = None
    cf_plot = None
    yplus_plot = None
    bl_plot = None
    remaining_images = []
    for img in val_images:
        if "plot_Ux_vs_y" in img.name:
            ux_y_plot = img
        elif "yplus_validation" in img.name:
            yplus_plot = img
        elif "cf_validation" in img.name:
            cf_plot = img
        elif "delta99_comparison" in img.name:
            bl_plot = img
        else:
            remaining_images.append(img)
    val_images = remaining_images
    per_model_plots = sum(
        len(image_files(results_root / m / "convergence_plots"))
        + len(image_files(results_root / m / "field_plots")) for m in models
    )
    n_plots = (len(val_images) + per_model_plots
               + (1 if ux_y_plot else 0)
               + (1 if cf_plot else 0)
               + (1 if yplus_plot else 0)
               + (1 if bl_plot else 0))
    # Per-model sections
    model_sections = []
    for model in models:
        conv_images  = image_files(results_root / model / "convergence_plots")
        field_images = image_files(results_root / model / "field_plots")
        m_logs       = log_files(run_case_dir / model)
        field_section = (
            f"<h3>Field images (mesh overlaid)</h3>"
            f"{render_gallery(field_images, results_root)}"
        ) if field_images else ""
        model_sections.append(
            f'<section id="model-{html.escape(model)}">'
            f"<h2>{html.escape(model)}</h2>"
            f"{field_section}"
            f"<h3>Convergence</h3>"
            f"{render_gallery(conv_images, results_root, 'No convergence plots found.')}"
            f"<h3>Logs</h3>"
            f"{render_logs(m_logs, results_root)}"
            f"</section>"
        )
    model_nav = "".join(
        f'<a href="#model-{html.escape(m)}">{html.escape(m)}</a>' for m in models
    ) or '<span class="empty">No models found.</span>'
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(case_name)} results</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f5f7f8; --panel: #ffffff; --text: #172026;
  --muted: #66727b; --line: #d9e0e4; --accent: #1f6feb; --accent-soft: #e8f1ff;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font: 15px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background: var(--bg); color: var(--text); }}
header {{ background: var(--panel); border-bottom: 1px solid var(--line); padding: 28px clamp(18px,4vw,52px); }}
main {{ width: min(1400px, calc(100% - 32px)); margin: 24px auto 56px; }}
h1, h2, h3 {{ margin: 0; }}
h1 {{ font-size: clamp(28px,4vw,44px); }}
h2 {{ font-size: 24px; margin-bottom: 16px; }}
h3 {{ font-size: 16px; margin: 20px 0 10px; color: var(--muted); }}
p {{ margin: 8px 0; }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
nav {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }}
nav a, .pill {{
  display: inline-flex; align-items: center; min-height: 32px;
  padding: 5px 11px; border: 1px solid var(--line); border-radius: 6px;
  background: var(--panel); color: var(--text);
}}
section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 20px; margin: 18px 0; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; color: var(--muted); }}
table {{ width: 100%; border-collapse: collapse; overflow: hidden; }}
th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
th {{ color: var(--muted); font-weight: 650; background: #fbfcfd; }}
.plot-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px,1fr)); gap: 14px; }}
.plot {{ display: grid; grid-template-rows: 200px auto; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: #fbfcfd; color: var(--text); }}
.plot:hover {{ text-decoration: none; border-color: #a8bdd3; }}
.plot img {{ width: 100%; height: 200px; object-fit: contain; background: #fff; border-bottom: 1px solid var(--line); }}
.plot span {{ padding: 9px 10px; overflow-wrap: anywhere; color: var(--muted); font-size: 13px; }}
.links {{ columns: 2 320px; padding-left: 20px; }}
.links li {{ break-inside: avoid; margin: 4px 0; overflow-wrap: anywhere; }}
.empty {{ color: var(--muted); background: var(--accent-soft); border-radius: 6px; padding: 10px 12px; }}
{LIGHTBOX_CSS}
@media (max-width: 720px) {{
  main {{ width: calc(100% - 20px); }}
  section {{ padding: 14px; }}
  .plot-grid {{ grid-template-columns: 1fr; }}
  th, td {{ padding: 8px; }}
}}
</style>
</head>
<body>
<header>
  <h1>{html.escape(case_name)}</h1>
  <p>Static results dashboard – turbulent flat plate u<sup>+</sup>/y<sup>+</sup> validation.</p>
  <div class="meta">
    <span class="pill">Generated: {html.escape(generated_at)}</span>
    <span class="pill">Models: {len(models)}</span>
    <span class="pill">Plots: {n_plots}</span>
    <span class="pill">Logs: {len(all_logs_run)}</span>
  </div>
  <nav>
    <a href="#summary">Summary</a>
    <a href="#validation">u⁺/y⁺ Validation</a>
    <a href="#cf">Cf Validation</a>
    <a href="#yplus">Wall y⁺</a>
    <a href="#delta99">Boundary-layer thickness</a>
    <a href="#models">Models</a>
    <a href="#logs">Logs</a>
  </nav>
</header>
<main>
  <section id="summary">
    <h2>Summary</h2>
    {render_model_summary(models, run_case_dir, results_root)}
  </section>
  <section id="validation">
    <h2>u⁺/y⁺ Validation  <small style="color:var(--muted);font-size:14px;">x = 1 m</small></h2>
    <p style="color:var(--muted);font-size:13px;">
      Simulation profiles compared to theoretical viscous sublayer (u⁺&nbsp;=&nbsp;y⁺) and
      log-law (κ&nbsp;=&nbsp;0.41, B&nbsp;=&nbsp;5.0) from White (1974).
    </p>
    {render_gallery(val_images, results_root, "No validation plot found – run postpro.py first.")}
    {f'<h2>Velocity Profile – Ux vs y</h2>' if ux_y_plot else ''}
    {f'<p><img src="{img_src(ux_y_plot, results_root)}" alt="Ux vs y Profile" style="max-width:100%;">' if ux_y_plot else ''}
  </section>
  <section id="cf">
    <h2>Cf Validation  <small style="color:var(--muted);font-size:14px;">station x = 0.97008 vs. TMR reference</small></h2>
    {render_cf_section(cf_plot, val_dir, results_root)}
  </section>
  <section id="yplus">
    <h2>Wall y⁺ along the plate  <small style="color:var(--muted);font-size:14px;">all models, log scale</small></h2>
    <p style="color:var(--muted);font-size:13px;">
      Sampled wall y⁺ on the bottomWall patch (y⁺_reference_2 / samplePlate) versus x for every model.
      Reference lines: y⁺&nbsp;=&nbsp;1 (wall-resolved) and y⁺&nbsp;=&nbsp;30 (wall-function range).
    </p>
    {f'<p><img src="{img_src(yplus_plot, results_root)}" alt="Wall y+ along the plate" style="max-width:100%;"></p>' if yplus_plot else '<p class="empty">No wall y+ plot found – run cf_validation.py first.</p>'}
  </section>
  <section id="delta99">
    <h2>Boundary-layer thickness  <small style="color:var(--muted);font-size:14px;">δ₉₉ over x, all runs</small></h2>
    <p style="color:var(--muted);font-size:13px;">
      δ₉₉ = wall distance where Ux reaches 99&nbsp;% of the local edge velocity, compared to the
      turbulent flat-plate correlation δ&nbsp;=&nbsp;0.37&nbsp;x&nbsp;/&nbsp;Re<sub>x</sub><sup>1/5</sup>;
      dotted: prism-layer height of the meshes that have one.
    </p>
    {f'<p><img src="{img_src(bl_plot, results_root)}" alt="Boundary-layer thickness comparison" style="max-width:100%;"></p>' if bl_plot else '<p class="empty">No boundary-layer plot found – run bl_thickness.py first.</p>'}
  </section>
  <section id="models">
    <h2>Models</h2>
    <nav>{model_nav}</nav>
  </section>
  {"".join(model_sections)}
  <section id="logs">
    <h2>All Logs</h2>
    {render_logs(all_logs_run, results_root)}
  </section>
</main>
{LIGHTBOX_JS}
</body>
</html>
"""
    return html_doc
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    script_dir = Path(__file__).resolve().parent
    _, case_name, run_case_dir, _, results_root = pp.resolve_case_paths(script_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    html_content = build_dashboard(case_name, run_case_dir, results_root)
    out_file = results_root / "index.html"
    out_file.write_text(html_content, encoding="utf-8")
    print(f"Dashboard written → {out_file}")
if __name__ == "__main__":
    main()
