"""Tests for the self-contained HTML report builder."""
from __future__ import annotations

import base64

from sar_pipeline.analysis import report

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def test_images_are_embedded_and_tables_render(tmp_path):
    (tmp_path / "fig.png").write_bytes(PNG)
    (tmp_path / "doc.md").write_text("# Title\n\n![x](fig.png)\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
    out = report.markdown_to_html(tmp_path / "doc.md", tmp_path / "doc.html", "T")
    html = out.read_text()
    assert "data:image/png;base64," in html and 'src="fig.png"' not in html
    assert "<table>" in html and "<title>T</title>" in html


def test_missing_image_is_left_as_is(tmp_path):
    (tmp_path / "doc.md").write_text("![x](nope.png)\n")
    html = report.markdown_to_html(tmp_path / "doc.md", tmp_path / "doc.html", "T").read_text()
    assert 'src="nope.png"' in html
