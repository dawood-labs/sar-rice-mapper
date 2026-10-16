"""Tests for the post video helpers (a tiny render, no project data)."""
import pytest

from sar_pipeline import outreach_video as ov


def test_scene_at_and_count_text():
    assert ov.scene_at(0.0) == (0, 0.0)
    assert ov.scene_at(6.0)[0] == 1
    assert ov.scene_at(99.0)[0] == len(ov.SCENES) - 2
    assert ov.count_text("199,655", 1.0) == "199,655"
    assert ov.count_text("233", 0.0) == "0"
    assert ov.ease(-1) == 0 and ov.ease(2) == 1


def test_render_writes_a_short_video(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    ex = {"clear": [5, 40, 120], "radar": list(range(10, 210, 12)), "vh": [(d, -18.0 + d / 50) for d in range(10, 210, 12)],
          "fit": [(d, 0.2 + d / 400) for d in range(0, 215, 5)], "clear_vals": []}
    out = ov.render(str(tmp_path / "v.mp4"), ex, fps=5, size_px=320, seconds=0.6)
    assert out.exists() and out.stat().st_size > 1000
