"""Tests for the portfolio image export (no project data)."""
from sar_pipeline import outreach_portfolio as op


def test_save_jpg_size_and_limit(tmp_path):
    plt = op._style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([0, 1], [0, 1], color=op.GREEN)
    p = op.save_jpg(fig, tmp_path / "x.jpg")
    from PIL import Image

    im = Image.open(p)
    assert max(im.size) == 1600 and p.stat().st_size <= 400 * 1024
