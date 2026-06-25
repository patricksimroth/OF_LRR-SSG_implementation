"""
dashboard.py – Static HTML results dashboard for the turbulentFlatPlate case.
Adapted from the backwards_step_tut dashboard with the same lightbox, gallery,
and log-rendering logic. The validation section shows the u+/y+ plot produced
by postpro.py.
"""
import html
# import os
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
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
# Path resolution  (shared with postpro.py)
# ---------------------------------------------------------------------------
def resolve_case_paths(script_dir: Path):
    if script_dir.name == "_postProcessing":
        source_case_dir = script_dir.parent
        case_name       = source_case_dir.name
        base_path       = source_case_dir.parent
        run_case_dir    = base_path / "run" / case_name
    else:
        case_name    = script_dir.name
        parent       = script_dir.parent
        if parent.name == "run":
            base_path       = parent.parent
            run_case_dir    = script_dir
            source_case_dir = base_path / case_name
        else:
            base_path       = parent
            run_case_dir    = base_path / "run" / case_name
            source_case_dir = script_dir
    results_root = base_path / "results" / case_name
    return base_path, case_name, run_case_dir, source_case_dir, results_root
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
def recursive_image_files(directory: Path):
    if not directory.is_dir():
        return []
    return sorted(
        [p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: str(p).lower(),
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
def build_dashboard(case_name, run_case_dir: Path, results_root: Path):
    models       = model_names(run_case_dir, results_root)
    val_dir      = results_root / "validation"
    all_images   = recursive_image_files(results_root)
    all_logs_run = log_files(run_case_dir)
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    # Collect validation images (u+/y+ plot)
    val_images = image_files(val_dir)
    # Per-model sections
    model_sections = []
    for model in models:
        conv_images = image_files(results_root / model / "convergence_plots")
        m_logs      = log_files(run_case_dir / model)
        model_sections.append(
            f'<section id="model-{html.escape(model)}">'
            f"<h2>{html.escape(model)}</h2>"
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
    <span class="pill">Plots: {len(all_images)}</span>
    <span class="pill">Logs: {len(all_logs_run)}</span>
  </div>
  <nav>
    <a href="#summary">Summary</a>
    <a href="#validation">u⁺/y⁺ Validation</a>
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
    # import re  # needed for natural_key inside this scope
    script_dir = Path(__file__).resolve().parent
    _, case_name, run_case_dir, _, results_root = resolve_case_paths(script_dir)
    results_root.mkdir(parents=True, exist_ok=True)
    html_content = build_dashboard(case_name, run_case_dir, results_root)
    out_file = results_root / "index.html"
    out_file.write_text(html_content, encoding="utf-8")
    print(f"Dashboard written → {out_file}")
if __name__ == "__main__":
    main()
