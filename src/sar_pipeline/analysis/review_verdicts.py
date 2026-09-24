"""The reviewed fields as a regression test: one table of every field a reviewer judged by eye.

Why
---
The review of 20 AOIs (docs outside the repository, ``review_2026/aoi<N>.md``) ended with a
"Fields checked" table per AOI: field id, current label, the reviewer's verdict (right / wrong /
uncertain) and what the field really is. Those ~430 judgements are the closest thing to ground
truth outside the surveyed plots, so every fix is measured against them: a fix must turn the
"wrong" fields right and leave the "right" fields alone. This module reads the markdown tables
once into a CSV (``report/review_verdicts.csv``) and scores a set of field labels against it.

The markdown tables have the columns ``field_id | acres | label | verdict | true class | issue``.
The verdict and true-class cells are free text; they are reduced to

* ``verdict``: ``right`` / ``wrong`` / ``uncertain`` (the leading word; "acceptable", "plausible"
  and "likely right" count as right, "likely wrong" as wrong, anything else as uncertain);
* ``expected_rice``: 1 if the reviewer's true class is delivered rice (1 or 6), 0 if it is a
  class that is not delivered (0, 2, 3, 4, 5, trees, roads, dry-land crops), blank when the
  reviewer was undecided between a delivered and a non-delivered class ("1 or 3").

Use::

    python -m sar_pipeline.analysis.review_verdicts --docs ../docs/review_2026   # writes the CSV
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

OUT = "processed/_batch/s2_2026/report/review_verdicts.csv"
DELIVERED = {1, 6}
NOT_DELIVERED = {0, 2, 3, 4, 5}
_RICE_WORDS = ("rice", "paddy")
_NOT_RICE_WORDS = ("not rice", "not paddy", "non-paddy", "dry", "tree", "road", "village", "garden",
                   "sand", "stream", "bank", "orchard", "homestead", "water all", "weeds", "regrowth",
                   "not a field", "never flooded", "not standing", "fallow", "open water", "marsh", "pond",
                   "mixed")


def parse_table(path) -> pd.DataFrame:
    """The 'Fields checked' rows of one review file (rows starting with ``| aoi``)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith("| aoi"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        fid = cells[0].strip("`")
        rows.append({"field_id": fid, "aoi": fid.split("_")[0], "acres_text": cells[1],
                     "label_text": cells[2], "verdict_text": cells[3], "true_class_text": cells[4],
                     "issue": cells[5] if len(cells) > 5 else ""})
    return pd.DataFrame(rows)


def reduce_verdict(text: str) -> str:
    t = text.lower()
    if t.startswith(("right", "acceptable", "plausible", "likely right", "probably right", "rice right",
                     "name wrong", "wrong name", "class name wrong", "wrong label, no",
                     "wrong label (not rice anyway)")):
        return "right"
    if t.startswith(("wrong", "likely wrong", "not rice")):
        return "wrong"
    return "uncertain"


def expected_rice(text: str):
    """1 / 0 / None from the reviewer's 'true class' cell."""
    t = text.lower().strip()
    codes = [int(c) for c in re.findall(r"(?<![\d.-])([0-6])(?![\d.])", t.split("(")[0][:12])]
    if codes:
        if all(c in DELIVERED for c in codes):
            return 1
        if all(c in NOT_DELIVERED for c in codes):
            return 0
        return None
    if any(w in t for w in _NOT_RICE_WORDS):
        return 0
    if any(w in t for w in _RICE_WORDS):
        return 1
    return None


def build(docs_dir, out=OUT) -> pd.DataFrame:
    """All review files -> one CSV with the reduced columns."""
    frames = [parse_table(p) for p in sorted(Path(docs_dir).glob("aoi*.md"))]
    t = pd.concat(frames, ignore_index=True)
    t["label"] = pd.to_numeric(t["label_text"].str.extract(r"(\d)")[0], errors="coerce").astype("Int64")
    t["acres"] = pd.to_numeric(t["acres_text"].str.extract(r"([\d.]+)")[0], errors="coerce")
    t["verdict"] = t["verdict_text"].map(reduce_verdict)
    t["expected_rice"] = t["true_class_text"].map(expected_rice).astype("Int64")
    t = t.drop_duplicates("field_id", keep="first")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    return t


def score(labels: pd.DataFrame, verdicts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compare new field labels (``field_id``, ``label``) with the reviewers' expectations.

    Returns one row per verdict group with: fields, how many now agree with the reviewer's
    expected delivery status (rice vs not), how many disagree, how many are undecidable.
    """
    v = pd.read_csv(OUT) if verdicts is None else verdicts
    m = v.merge(labels[["field_id", "label"]].rename(columns={"label": "new_label"}), on="field_id", how="left")
    m["new_rice"] = m["new_label"].isin(DELIVERED).astype("Int64").where(m["new_label"].notna())
    m["old_rice"] = m["label"].isin(DELIVERED).astype(int)
    rows = []
    for verdict, g in m.groupby("verdict"):
        decidable = g[g["expected_rice"].notna() & g["new_rice"].notna()]
        rows.append({"verdict": verdict, "fields": len(g),
                     "decidable": len(decidable),
                     "agree_now": int((decidable["new_rice"] == decidable["expected_rice"]).sum()),
                     "disagree_now": int((decidable["new_rice"] != decidable["expected_rice"]).sum()),
                     "agreed_before": int((decidable["old_rice"] == decidable["expected_rice"]).sum()),
                     "label_changed": int((g["new_label"].notna() & (g["new_label"] != g["label"])).sum())})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.review_verdicts", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs", required=True, help="folder with the review files aoi<N>.md")
    p.add_argument("--out", default=OUT)
    args = p.parse_args(argv)
    t = build(args.docs, args.out)
    print(f"{len(t)} fields from {t['aoi'].nunique()} AOIs -> {args.out}")
    print(t.groupby(["verdict", "expected_rice"], dropna=False).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
