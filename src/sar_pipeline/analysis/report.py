"""Turn a Markdown findings document into one self-contained HTML file.

Why self-contained
------------------
Findings get read on a phone, forwarded, or opened long after the working folders have moved. A
single HTML file with every figure embedded (as base64) still renders anywhere, with no broken
image links and nothing to host. It is written locally only; publishing it anywhere is a separate,
deliberate decision.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

_STYLE = """
:root { --ink:#0b0b0b; --ink2:#52514e; --line:#e4e2dc; --surface:#fcfcfb; --accent:#2a78d6; }
@media (prefers-color-scheme: dark) {
  :root { --ink:#f4f4f2; --ink2:#c3c2b7; --line:#3a3a37; --surface:#1a1a19; --accent:#3987e5; }
  img { background:#fcfcfb; border-radius:6px; }
}
body { background:var(--surface); color:var(--ink); font:16px/1.55 system-ui, -apple-system, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 24px 16px 80px; }
h1 { font-size: 1.7rem; margin-top: 0.4rem; } h2 { margin-top: 2.4rem; border-bottom: 1px solid var(--line);
       padding-bottom: 4px; } h3 { margin-top: 1.6rem; }
p, li { color: var(--ink); } small, em { color: var(--ink2); }
table { border-collapse: collapse; width: 100%; display: block; overflow-x: auto; font-size: 0.9rem; margin: 12px 0; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 10px; text-align: left; vertical-align: top; }
th { color: var(--ink2); font-weight: 600; }
img { max-width: 100%; height: auto; margin: 10px 0; }
code { font-size: 0.88em; } a { color: var(--accent); }
blockquote { border-left: 3px solid var(--accent); margin: 12px 0; padding: 4px 14px; color: var(--ink2); }
"""


def _embed_images(html: str, base_dir: Path) -> str:
    def swap(match):
        src = match.group(1)
        path = (base_dir / src) if not Path(src).is_absolute() else Path(src)
        if not path.exists():
            return match.group(0)
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode()
        return f'src="data:{mime};base64,{data}"'
    return re.sub(r'src="([^"]+)"', swap, html)


def markdown_to_html(md_path, out_path, title):
    """Render ``md_path`` to ``out_path`` with images embedded; relative image paths resolve from the md."""
    import markdown

    md_path = Path(md_path)
    body = markdown.markdown(md_path.read_text(), extensions=["tables", "fenced_code"])
    body = _embed_images(body, md_path.parent)
    html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{title}</title><style>{_STYLE}</style></head><body>{body}</body></html>")
    Path(out_path).write_text(html)
    return out_path
