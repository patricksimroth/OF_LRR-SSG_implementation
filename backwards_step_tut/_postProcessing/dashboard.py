import html
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import quote


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.svg', '.webp'}


LIGHTBOX_CSS = """
body.lightbox-open {
  overflow: hidden;
}
.lightbox[hidden] {
  display: none;
}
.lightbox {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto;
  gap: 12px;
  padding: 16px;
  background: rgba(10, 15, 20, 0.94);
  color: #fff;
}
.lightbox-bar {
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: center;
  gap: 12px;
}
.lightbox-title {
  margin: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #f4f7fa;
}
.lightbox-counter {
  color: #c8d0d7;
  font-size: 13px;
}
.lightbox-stage {
  position: relative;
  min-height: 0;
  display: grid;
  place-items: center;
}
.lightbox-stage img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  background: #fff;
  border-radius: 4px;
}
.lightbox-button {
  border: 1px solid rgba(255, 255, 255, 0.28);
  background: rgba(255, 255, 255, 0.12);
  color: #fff;
  border-radius: 6px;
  min-width: 38px;
  min-height: 38px;
  padding: 7px 11px;
  font: inherit;
  cursor: pointer;
}
.lightbox-button:hover {
  background: rgba(255, 255, 255, 0.2);
}
.lightbox-nav {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 44px;
  height: 64px;
  font-size: 30px;
}
.lightbox-prev {
  left: 0;
}
.lightbox-next {
  right: 0;
}
.lightbox-help {
  margin: 0;
  text-align: center;
  color: #c8d0d7;
  font-size: 13px;
}
@media (max-width: 720px) {
  .lightbox {
    padding: 10px;
  }
  .lightbox-nav {
    width: 38px;
    height: 54px;
  }
}
"""


LIGHTBOX_JS = """
<script>
(function () {
  const links = Array.from(document.querySelectorAll('.plot'));
  if (!links.length) {
    return;
  }

  const overlay = document.createElement('div');
  overlay.className = 'lightbox';
  overlay.hidden = true;
  overlay.innerHTML = `
    <div class="lightbox-bar">
      <div>
        <p class="lightbox-title"></p>
        <span class="lightbox-counter"></span>
      </div>
      <button class="lightbox-button lightbox-close" type="button" aria-label="Close">Close</button>
    </div>
    <div class="lightbox-stage">
      <button class="lightbox-button lightbox-nav lightbox-prev" type="button" aria-label="Previous image">&#8249;</button>
      <img alt="">
      <button class="lightbox-button lightbox-nav lightbox-next" type="button" aria-label="Next image">&#8250;</button>
    </div>
    <p class="lightbox-help">ESC closes. Arrow keys move within this category.</p>
  `;
  document.body.appendChild(overlay);

  const titleEl = overlay.querySelector('.lightbox-title');
  const counterEl = overlay.querySelector('.lightbox-counter');
  const imageEl = overlay.querySelector('img');
  const closeButton = overlay.querySelector('.lightbox-close');
  const prevButton = overlay.querySelector('.lightbox-prev');
  const nextButton = overlay.querySelector('.lightbox-next');

  let gallery = [];
  let currentIndex = 0;

  function galleryFromLink(link) {
    const grid = link.closest('.plot-grid');
    const items = grid ? Array.from(grid.querySelectorAll('.plot')) : [link];
    return items.map((item) => {
      const label = item.querySelector('span')?.textContent.trim() || item.querySelector('img')?.alt || item.href;
      return {
        href: item.href,
        label: label
      };
    });
  }

  function show(index) {
    if (!gallery.length) {
      return;
    }
    currentIndex = (index + gallery.length) % gallery.length;
    const item = gallery[currentIndex];
    imageEl.src = item.href;
    imageEl.alt = item.label;
    titleEl.textContent = item.label;
    counterEl.textContent = `${currentIndex + 1} / ${gallery.length}`;
    prevButton.hidden = gallery.length < 2;
    nextButton.hidden = gallery.length < 2;
  }

  function open(link) {
    gallery = galleryFromLink(link);
    currentIndex = gallery.findIndex((item) => item.href === link.href);
    if (currentIndex < 0) {
      currentIndex = 0;
    }
    show(currentIndex);
    overlay.hidden = false;
    document.body.classList.add('lightbox-open');
    closeButton.focus();
  }

  function close() {
    overlay.hidden = true;
    document.body.classList.remove('lightbox-open');
    imageEl.removeAttribute('src');
  }

  function next() {
    show(currentIndex + 1);
  }

  function previous() {
    show(currentIndex - 1);
  }

  links.forEach((link) => {
    link.addEventListener('click', (event) => {
      event.preventDefault();
      open(link);
    });
  });

  closeButton.addEventListener('click', close);
  nextButton.addEventListener('click', next);
  prevButton.addEventListener('click', previous);
  overlay.addEventListener('click', (event) => {
    if (event.target === overlay) {
      close();
    }
  });

  document.addEventListener('keydown', (event) => {
    if (overlay.hidden) {
      return;
    }
    if (event.key === 'Escape') {
      close();
    } else if (event.key === 'ArrowRight') {
      next();
    } else if (event.key === 'ArrowLeft') {
      previous();
    }
  });
}());
</script>
"""


def natural_key(value):
    parts = []
    current = ''
    is_digit = None
    for char in str(value):
        char_is_digit = char.isdigit()
        if is_digit is None or char_is_digit == is_digit:
            current += char
        else:
            parts.append(int(current) if is_digit else current.lower())
            current = char
        is_digit = char_is_digit
    if current:
        parts.append(int(current) if is_digit else current.lower())
    return parts


def resolve_case_paths(script_dir):
    script_dir = Path(script_dir).resolve()

    if script_dir.name == '_postProcessing':
        source_case_dir = script_dir.parent
        case_name = source_case_dir.name
        base_path = source_case_dir.parent
        run_case_dir = base_path / 'run' / case_name
    elif script_dir.parent.name == 'run':
        run_case_dir = script_dir
        case_name = script_dir.name
        base_path = script_dir.parent.parent
        source_case_dir = base_path / case_name
    else:
        source_case_dir = script_dir
        case_name = script_dir.name
        base_path = script_dir.parent
        run_case_dir = base_path / 'run' / case_name

    results_root = base_path / 'results' / case_name
    return base_path, case_name, run_case_dir, source_case_dir, results_root


def url_for(path, relative_to):
    rel = os.path.relpath(path, relative_to)
    return quote(rel.replace(os.sep, '/'), safe='/._-()')


def image_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: natural_key(p.name),
    )


def recursive_image_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        [p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: natural_key(str(p.relative_to(directory))),
    )


def log_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        [p for p in directory.rglob('*') if p.is_file() and (p.name.endswith('.log') or p.name.startswith('log.'))],
        key=lambda p: natural_key(str(p.relative_to(directory))),
    )


def latest_time_dir(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return None
    candidates = []
    for item in directory.iterdir():
        if not item.is_dir():
            continue
        try:
            candidates.append((float(item.name), item.name))
        except ValueError:
            continue
    return max(candidates, default=(None, None))[1]


def model_names(run_case_dir, results_root):
    names = set()
    if run_case_dir.is_dir():
        for item in run_case_dir.iterdir():
            if (item / 'postProcessing').is_dir():
                names.add(item.name)
    if results_root.is_dir():
        for item in results_root.iterdir():
            if item.is_dir() and item.name not in {'validation', 'validation_wall', 'overview'}:
                names.add(item.name)
    return sorted(names, key=natural_key)


def image_card(path, results_root, title=None):
    title = title or path.stem.replace('_', ' ')
    rel_url = url_for(path, results_root)
    text = html.escape(title)
    return (
        '<a class="plot" href="{href}">'
        '<img src="{href}" alt="{alt}" loading="lazy">'
        '<span>{label}</span>'
        '</a>'
    ).format(href=rel_url, alt=text, label=text)


def file_link(path, results_root, label=None):
    rel_url = url_for(path, results_root)
    label = label or str(path)
    return '<a href="{href}" target="_blank" rel="noopener">{label}</a>'.format(
        href=rel_url,
        label=html.escape(label),
    )


def render_gallery(paths, results_root, empty_text='No files found.'):
    if not paths:
        return '<p class="empty">{}</p>'.format(html.escape(empty_text))
    cards = '\n'.join(image_card(path, results_root) for path in paths)
    return '<div class="plot-grid">{}</div>'.format(cards)


def render_model_summary(models, run_case_dir, results_root):
    rows = []
    for model in models:
        pp_dir = run_case_dir / model / 'postProcessing'
        result_dir = results_root / model
        sample_time = latest_time_dir(pp_dir / 'sample') or '-'
        uref_time = latest_time_dir(pp_dir / 'Uref') or '-'
        wall_time = latest_time_dir(pp_dir / 'sample_wall') or '-'
        convergence_count = len(image_files(result_dir / 'convergence_plots'))
        logs_count = len(log_files(run_case_dir / model))
        rows.append(
            '<tr>'
            '<td><a href="#model-{anchor}">{model}</a></td>'
            '<td>{sample}</td><td>{uref}</td><td>{wall}</td>'
            '<td>{plots}</td><td>{logs}</td>'
            '</tr>'.format(
                anchor=html.escape(model),
                model=html.escape(model),
                sample=html.escape(sample_time),
                uref=html.escape(uref_time),
                wall=html.escape(wall_time),
                plots=convergence_count,
                logs=logs_count,
            )
        )

    if not rows:
        return '<p class="empty">No model folders with postProcessing output were found.</p>'

    return (
        '<table>'
        '<thead><tr><th>Model</th><th>sample</th><th>Uref</th><th>sample_wall</th>'
        '<th>Convergence plots</th><th>Logs</th></tr></thead>'
        '<tbody>{}</tbody></table>'
    ).format('\n'.join(rows))


def render_logs(logs, results_root):
    if not logs:
        return '<p class="empty">No logs found.</p>'
    items = []
    for path in logs:
        label = str(path)
        items.append('<li>{}</li>'.format(file_link(path, results_root, label)))
    return '<ul class="links">{}</ul>'.format('\n'.join(items))


def render_validation_profiles(validation_dir, results_root):
    if not validation_dir.is_dir():
        return '<p class="empty">No validation profile directory found.</p>'

    sections = []
    for variable_dir in sorted(validation_dir.iterdir(), key=lambda p: natural_key(p.name)):
        if not variable_dir.is_dir() or variable_dir.name == 'overview':
            continue
        images = image_files(variable_dir)
        if not images:
            continue
        sections.append(
            '<details><summary>{name} <span>{count} plots</span></summary>{gallery}</details>'.format(
                name=html.escape(variable_dir.name),
                count=len(images),
                gallery=render_gallery(images, results_root),
            )
        )

    if not sections:
        return '<p class="empty">No validation profile plots found.</p>'
    return '\n'.join(sections)


def build_dashboard(case_name, run_case_dir, results_root):
    models = model_names(run_case_dir, results_root)
    validation_dir = results_root / 'validation'
    wall_dir = results_root / 'validation_wall'

    validation_overview = []
    for name in ['relative_rmse_comparison.png', 'validation_correlation_overview.png']:
        path = validation_dir / name
        if path.is_file():
            validation_overview.append(path)
    validation_overview.extend(image_files(validation_dir / 'overview'))

    all_result_images = recursive_image_files(results_root)
    all_logs = log_files(run_case_dir)
    generated_at = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')

    model_sections = []
    for model in models:
        convergence = image_files(results_root / model / 'convergence_plots')
        model_logs = log_files(run_case_dir / model)
        model_sections.append(
            '<section id="model-{anchor}">'
            '<h2>{model}</h2>'
            '<h3>Convergence</h3>{convergence}'
            '<h3>Logs</h3>{logs}'
            '</section>'.format(
                anchor=html.escape(model),
                model=html.escape(model),
                convergence=render_gallery(convergence, results_root, 'No convergence plots found.'),
                logs=render_logs(model_logs, results_root),
            )
        )

    html_doc = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{case_name} results</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f5f7f8;
  --panel: #ffffff;
  --text: #172026;
  --muted: #66727b;
  --line: #d9e0e4;
  --accent: #1f6feb;
  --accent-soft: #e8f1ff;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font: 15px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--text);
}}
header {{
  background: var(--panel);
  border-bottom: 1px solid var(--line);
  padding: 28px clamp(18px, 4vw, 52px);
}}
main {{
  width: min(1400px, calc(100% - 32px));
  margin: 24px auto 56px;
}}
h1, h2, h3 {{ margin: 0; }}
h1 {{ font-size: clamp(28px, 4vw, 44px); }}
h2 {{ font-size: 24px; margin-bottom: 16px; }}
h3 {{ font-size: 16px; margin: 20px 0 10px; color: var(--muted); }}
p {{ margin: 8px 0; }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
nav {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 18px;
}}
nav a, .pill {{
  display: inline-flex;
  align-items: center;
  min-height: 32px;
  padding: 5px 11px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
  color: var(--text);
}}
section {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 20px;
  margin: 18px 0;
}}
.meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 16px;
  color: var(--muted);
}}
table {{
  width: 100%;
  border-collapse: collapse;
  overflow: hidden;
}}
th, td {{
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}}
th {{
  color: var(--muted);
  font-weight: 650;
  background: #fbfcfd;
}}
.plot-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 14px;
}}
.plot {{
  display: grid;
  grid-template-rows: 160px auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  background: #fbfcfd;
  color: var(--text);
}}
.plot:hover {{ text-decoration: none; border-color: #a8bdd3; }}
.plot img {{
  width: 100%;
  height: 160px;
  object-fit: contain;
  background: #fff;
  border-bottom: 1px solid var(--line);
}}
.plot span {{
  padding: 9px 10px;
  overflow-wrap: anywhere;
  color: var(--muted);
  font-size: 13px;
}}
details {{
  border: 1px solid var(--line);
  border-radius: 8px;
  margin: 12px 0;
  padding: 0;
  background: #fbfcfd;
}}
summary {{
  cursor: pointer;
  padding: 12px 14px;
  font-weight: 650;
}}
summary span {{ color: var(--muted); font-weight: 500; margin-left: 8px; }}
details .plot-grid {{ padding: 0 14px 14px; }}
.links {{
  columns: 2 320px;
  padding-left: 20px;
}}
.links li {{ break-inside: avoid; margin: 4px 0; overflow-wrap: anywhere; }}
.empty {{
  color: var(--muted);
  background: var(--accent-soft);
  border-radius: 6px;
  padding: 10px 12px;
}}
{lightbox_css}
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
  <h1>{case_name}</h1>
  <p>Static results dashboard generated from the OpenFOAM run and results folders.</p>
  <div class="meta">
    <span class="pill">Generated: {generated_at}</span>
    <span class="pill">Models: {model_count}</span>
    <span class="pill">Plots: {plot_count}</span>
    <span class="pill">Logs: {log_count}</span>
  </div>
  <nav>
    <a href="#summary">Summary</a>
    <a href="#validation">Validation</a>
    <a href="#wall">Wall data</a>
    <a href="#models">Models</a>
    <a href="#logs">Logs</a>
  </nav>
</header>
<main>
  <section id="summary">
    <h2>Summary</h2>
    {summary_table}
  </section>

  <section id="validation">
    <h2>Validation Overview</h2>
    {validation_overview}
    <h3>Profiles</h3>
    {validation_profiles}
  </section>

  <section id="wall">
    <h2>Wall Data</h2>
    {wall_gallery}
  </section>

  <section id="models">
    <h2>Models</h2>
    <nav>{model_nav}</nav>
  </section>
  {model_sections}

  <section id="logs">
    <h2>All Logs</h2>
    {all_logs}
  </section>
</main>
{lightbox_js}
</body>
</html>
'''.format(
        case_name=html.escape(case_name),
        generated_at=html.escape(generated_at),
        model_count=len(models),
        plot_count=len(all_result_images),
        log_count=len(all_logs),
        summary_table=render_model_summary(models, run_case_dir, results_root),
        validation_overview=render_gallery(validation_overview, results_root, 'No validation overview plots found.'),
        validation_profiles=render_validation_profiles(validation_dir, results_root),
        wall_gallery=render_gallery(image_files(wall_dir), results_root, 'No wall plots found.'),
        model_nav=''.join(
            '<a href="#model-{anchor}">{model}</a>'.format(
                anchor=html.escape(model),
                model=html.escape(model),
            )
            for model in models
        ) or '<span class="empty">No models found.</span>',
        model_sections='\n'.join(model_sections),
        all_logs=render_logs(all_logs, results_root),
        lightbox_css=LIGHTBOX_CSS,
        lightbox_js=LIGHTBOX_JS,
    )
    return html_doc


def main():
    script_dir = Path(__file__).resolve().parent
    _, case_name, run_case_dir, _, results_root = resolve_case_paths(script_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    html_doc = build_dashboard(case_name, run_case_dir, results_root)
    output_file = results_root / 'index.html'
    output_file.write_text(html_doc, encoding='utf-8')
    print(f"Dashboard written to {output_file}")


if __name__ == '__main__':
    main()
