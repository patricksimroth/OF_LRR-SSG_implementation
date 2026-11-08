"""
dashboard.py – Static HTML results dashboard for the turbulentFlatPlate case.
Adapted from the backwards_step_tut dashboard with the same lightbox, gallery,
and log-rendering logic.  One gallery per summary/ subfolder (SUMMARY_SECTIONS
order, profiles per station), then the per-run plots, each image with its
caption draft from _captions.py; also writes summary/CAPTIONS.md.
"""
import html
import json
# import os
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import _captions
import _pp_common as pp

# (summary subfolder, heading, short description)
SUMMARY_SECTIONS = (
    ("along_plate", "Along the plate", "q(x) per approach and h⁺, one line per core size"),
    ("error_along_plate", "Error along the plate", "relative deviation from blockMesh"),
    ("error_vs_core", "Error over the core cell size", "at x = 0.97 m and RMS over the plate"),
    ("cost", "Cost", "plate-RMS C_f error over the cell count"),
    ("wall_law", "Law of the wall", "u⁺ over y⁺ at x = 1 m"),
    ("prism_coverage", "Prism coverage", "stack top over the boundary-layer regions"),
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
FIELD_WINDOWS = {"front": "leading edge", "": "whole domain and x = 1 m", "back": "outlet"}
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
                if (item.is_dir() and not item.name.startswith("_")
                        and item.name not in ("validation", pp.SUMMARY_DIR)):
                    names.add(item.name)
    return sorted(names)
def img_src(image_path: Path, results_root: Path):
    try:
        rel = image_path.relative_to(results_root)
        return quote(str(rel).replace("\\", "/"))
    except ValueError:
        return quote(str(image_path).replace("\\", "/"))
def render_gallery(images, results_root: Path, empty_msg="No plots found.", context=None):
    if not images:
        return f'<p class="empty">{html.escape(empty_msg)}</p>'
    items = []
    for p in images:
        src   = img_src(p, results_root)
        label = html.escape(p.name)
        text = _captions.caption(p.relative_to(results_root), context) if context else ""
        note = f'<small>{html.escape(text)}</small>' if text else ""
        items.append(
            f'<a class="plot" href="{src}" title="{html.escape(text)}">'
            f'<img src="{src}" alt="{label}" loading="lazy">'
            f'<span>{label}{note}</span>'
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
def provenance_pills(results_root: Path):
    """Study and commit pills from the provenance.json run_study.py wrote.

    The dashboard is what gets looked at months later, so it has to say which
    code produced the data - and say it loudly when the tree was dirty, since
    the commit alone then describes code that never ran."""
    path = results_root / "provenance.json"
    if not path.is_file():
        return ""
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    pills = []
    if record.get("study"):
        pills.append(f'<span class="pill">Study: {html.escape(record["study"])}</span>')
    git = record.get("git") or {}
    if git.get("commit"):
        label = f"Commit: {git['commit'][:7]}"
        style = ""
        if git.get("dirty"):
            label += f" + {len(git.get('changed', []))} uncommitted change(s)"
            style = ' style="color:#c0392b;border-color:#c0392b;"'
        pills.append(f'<span class="pill"{style}>{html.escape(label)}</span>')
    return "".join(pills)
def run_status(run_case_dir: Path, model: str):
    """(label, html colour) for the Summary table's status column.  Runs the
    solver did not finish are excluded from every validation plot, so the
    dashboard has to say so instead of just showing an empty section."""
    case_dir = run_case_dir / model
    if not case_dir.is_dir():
        return "no run dir", "var(--muted)"
    if pp.run_completed(case_dir):
        return "completed", "inherit"
    return "FAILED (solver did not reach End)", "#c0392b"
def render_model_summary(models, run_case_dir: Path, results_root: Path):
    if not models:
        return '<p class="empty">No models found.</p>'
    rows = []
    for m in models:
        conv_dir = results_root / m / "convergence_plots"
        n_plots  = len(image_files(conv_dir))
        n_logs   = len(log_files(run_case_dir / m))
        status, colour = run_status(run_case_dir, m)
        rows.append(
            f"<tr><td>{html.escape(m)}</td>"
            f'<td style="color:{colour};">{html.escape(status)}</td>'
            f"<td>{n_plots}</td><td>{n_logs}</td></tr>"
        )
    header = ("<tr><th>Model</th><th>Status</th><th>Convergence plots</th>"
              "<th>Log files</th></tr>")
    return f"<table>{header}{''.join(rows)}</table>"
# ---------------------------------------------------------------------------
# Dashboard builder
# ---------------------------------------------------------------------------
def summary_images(summary: Path):
    """Every summary figure in dashboard order."""
    images = []
    for name, _h, _d in SUMMARY_SECTIONS:
        images += image_files(summary / name)
    for station in sorted((d for d in (summary / "profiles").glob("x_*m") if d.is_dir()),
                          key=lambda d: natural_key(d.name)):
        images += (image_files(station) + image_files(station / "budget")
                   + image_files(station / "budget_strip"))
    return images


def write_captions(summary: Path, results_root: Path, context):
    lines = ["# Figure captions", "",
             "Caption drafts for the figures in this folder (generated by dashboard.py from",
             "_postProcessing/_captions.py; the figures themselves carry no titles or notes).", ""]
    for img in summary_images(summary):
        rel = img.relative_to(results_root)
        lines += [f"## {img.relative_to(summary)}", "", _captions.caption(rel, context), ""]
    (summary / "CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")


def build_dashboard(case_name, run_case_dir: Path, results_root: Path):
    models       = model_names(run_case_dir, results_root)
    summary      = results_root / pp.SUMMARY_DIR
    all_logs_run = log_files(run_case_dir)
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    context = _captions.study_context(results_root)
    if summary.is_dir():
        write_captions(summary, results_root, context)
    sections, nav = [], []
    for name, heading, description in SUMMARY_SECTIONS:
        images = image_files(summary / name)
        if not images:
            continue
        nav.append(f'<a href="#{name}">{html.escape(heading)}</a>')
        sections.append(
            f'<section id="{name}"><h2>{html.escape(heading)}  <small style="color:var(--muted);'
            f'font-size:14px;">{html.escape(description)}</small></h2>'
            f"{render_gallery(images, results_root, context=context)}</section>")
    stations = sorted((d for d in (summary / "profiles").glob("x_*m") if d.is_dir()),
                      key=lambda d: natural_key(d.name)) if summary.is_dir() else []
    if stations:
        nav.append('<a href="#profiles">Profiles</a>')
        body = "".join(
            f"<h3>{html.escape(d.name.replace('x_', 'x = ').replace('m', ' m'))}</h3>"
            f"{render_gallery(image_files(d), results_root, context=context)}"
            + "".join(
                f"<h3>{html.escape(d.name.replace('x_', 'x = ').replace('m', ' m'))}: "
                f"transport-equation budgets, {label}</h3>"
                f"{render_gallery(image_files(d / sub), results_root, context=context)}"
                for sub, label in (("budget", "cell values along the line"),
                                   ("budget_strip", "strip-averaged"))
                if image_files(d / sub))
            for d in stations)
        sections.append(f'<section id="profiles"><h2>Wall-normal profiles</h2>{body}</section>')
    tables = sorted((summary / "tables").glob("*")) if summary.is_dir() else []
    table_links = "".join(
        f'<li><a href="{img_src(t, results_root)}">{html.escape(t.name)}</a></li>'
        for t in tables)
    if (summary / "CAPTIONS.md").is_file():
        table_links += (f'<li><a href="{img_src(summary / "CAPTIONS.md", results_root)}">'
                        "CAPTIONS.md</a></li>")
    n_plots = len(summary_images(summary)) + sum(
        len(image_files(results_root / m / "convergence_plots"))
        + sum(len(image_files(results_root / m / "field_plots" / w))
              + len(image_files(results_root / m / "field_plots" / w / "budget"))
              for w in FIELD_WINDOWS) for m in models)
    model_sections = []
    for model in models:
        conv_images  = image_files(results_root / model / "convergence_plots")
        m_logs       = log_files(run_case_dir / model)
        field_section = ""
        for window, where in FIELD_WINDOWS.items():
            field_dir = results_root / model / "field_plots" / window
            field_images = image_files(field_dir)
            budget_images = image_files(field_dir / "budget")
            if field_images:
                field_section += (
                    f"<h3>Field images, {where} (mesh overlaid)</h3>"
                    f"{render_gallery(field_images, results_root, context=context)}")
            if budget_images:
                field_section += (
                    f"<h3>Budget-term fields, {where} (mesh overlaid)</h3>"
                    f"{render_gallery(budget_images, results_root, context=context)}")
        model_sections.append(
            f'<section id="model-{html.escape(model)}">'
            f"<h2>{html.escape(model)}</h2>"
            f"{field_section}"
            f"<h3>Convergence</h3>"
            f"{render_gallery(conv_images, results_root, 'No convergence plots found.', context)}"
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
.plot small {{ display: block; margin-top: 4px; font-size: 11px; line-height: 1.3; }}
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
  <p>Static results dashboard – turbulent flat plate, mesh-topology study.</p>
  <div class="meta">
    <span class="pill">Generated: {html.escape(generated_at)}</span>
    {provenance_pills(results_root)}
    <span class="pill">Models: {len(models)}</span>
    <span class="pill">Plots: {n_plots}</span>
    <span class="pill">Logs: {len(all_logs_run)}</span>
  </div>
  <nav>
    <a href="#summary">Summary</a>
    {"".join(nav)}
    <a href="#models">Models</a>
    <a href="#logs">Logs</a>
  </nav>
</header>
<main>
  <section id="summary">
    <h2>Summary</h2>
    {render_model_summary(models, run_case_dir, results_root)}
  </section>
  {"".join(sections)}
  <section id="tables">
    <h2>Tables</h2>
    {f'<ul class="links">{table_links}</ul>' if table_links else '<p class="empty">No tables found.</p>'}
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
