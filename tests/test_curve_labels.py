"""Second fresh start: the user's curve labels are stored, and a repeated pixel takes the latest label."""
from sar_pipeline.analysis import curve_labels as cl


def test_add_stores_and_a_repeated_pixel_takes_the_latest_label(tmp_path):
    store = tmp_path / "labels.csv"
    cl.add([(39, 24703, "rice"), (39, 7636, "not rice", "water")], store)
    t = cl.add([(39, 24703, "late rice")], store)
    assert len(t) == 2 and t.set_index("pixel").loc[24703, "label"] == "late rice"
    assert cl.load(store).set_index("pixel").loc[7636, "note"] == "water"


def test_sowing_and_establishment_are_kept_when_a_label_is_repeated_without_them(tmp_path):
    store = tmp_path / "labels.csv"
    cl.add([(160, 78199, "standing rice")], store, sowing="2026-06-15", establishment="transplanted (water)")
    t = cl.add([(160, 78199, "standing rice", "new note")], store)
    row = t.set_index("pixel").loc[78199]
    assert row["sowing"] == "2026-06-15" and row["establishment"] == "transplanted (water)" and row["note"] == "new note"
