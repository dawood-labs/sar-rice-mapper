"""Rice vs non-rice agreement between the rule's delivered map and the human QC, as a confusion matrix (26 Oct 2026).

Why: there is no field survey with both rice and non-rice for these AOIs, but every AOI was checked field by field in
QGIS by a reviewer (the QC, second round 15 Oct 2026). The reviewer's decision is used as the reference:

* a field the rule called rice and the reviewer kept as rice: true positive (TP);
* rule rice, reviewer deleted it or set it to non-rice: false positive (FP);
* rule non-rice, reviewer set it to rice, or a rice polygon the reviewer drew where the map had none: false negative (FN);
* rule non-rice, reviewer kept it non-rice (or deleted it): true negative (TN).

Counted in acres (each field's delivered area; a drawn polygon's own area) and in fields. Limit: the reviewer worked on
the rule's map, so this is agreement with a human check, not an independent field accuracy; it tends to read high.

Inputs: the delivered layers the QC started from (``qc_compare.DELIVERED_DIR``, layer named in
``../data/qc/qc_2026-10-15_summary.csv`` column ``base``) and the QC files (``qc_compare.QC2_DIR``).
Outputs (``processed/_batch/s2_2026/accuracy/``): ``qc_confusion.csv`` (per AOI), ``qc_metrics.json`` (totals),
``confusion_matrix.png`` and ``metrics.png``.

Run::

    python -m sar_pipeline.analysis.qc_accuracy
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .aoi_stats import ACRE_M2

SUMMARY = "../data/qc/qc_2026-10-15_summary.csv"
OUT = Path("processed/_batch/s2_2026/accuracy")
METRIC_CRS = 32646


def confusion_one(base, qc) -> dict:
    """TP / FP / FN / TN in acres and fields for one AOI from its delivered layer and its QC layer (both with
    ``field_id`` and ``major_class``; a QC polygon without ``field_id`` was drawn by the reviewer)."""
    base = base.to_crs(METRIC_CRS)
    qc = qc.to_crs(METRIC_CRS)
    b = base.assign(ac=base.area / ACRE_M2).groupby("field_id").agg(rule=("major_class", "first"), ac=("ac", "sum"))
    drawn = qc["field_id"].isna() | (qc["field_id"].astype(str) == "") | (qc["field_id"].astype(str) == "None")
    q = qc[~drawn].groupby(qc.loc[~drawn, "field_id"].astype(str))["major_class"].agg(lambda s: "rice" if (s == "rice").any()
                                                                                   else "non-rice")
    b.index = b.index.astype(str)
    b["qc"] = q.reindex(b.index).fillna("non-rice")            # deleted by the reviewer = not rice
    rr, qr = b["rule"] == "rice", b["qc"] == "rice"
    d = qc[drawn & (qc["major_class"].fillna("rice") == "rice")]
    out = {}
    for k, m in (("tp", rr & qr), ("fp", rr & ~qr), ("fn", ~rr & qr), ("tn", ~rr & ~qr)):
        out[f"{k}_ac"] = float(b.loc[m, "ac"].sum())
        out[f"{k}_fields"] = int(m.sum())
    out["fn_ac"] += float(d.area.sum() / ACRE_M2)
    out["fn_fields"] += int(len(d))
    return out


def metrics(tp: float, fp: float, fn: float, tn: float) -> dict:
    """Precision, recall, F1 (rice as the positive class) and overall accuracy, in percent."""
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * p * r / (p + r) if p + r else np.nan
    acc = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else np.nan
    return {k: round(100 * v, 1) for k, v in (("precision", p), ("recall", r), ("f1", f1), ("accuracy", acc))}


def _one(args):
    import geopandas as gpd

    from .qc_compare import DELIVERED_DIR
    from .rice_map_delivery import bucket_name

    aoi, base_layer, qc_file = args
    name = bucket_name(aoi)
    base = gpd.read_file(Path(DELIVERED_DIR) / name / f"{name}_{base_layer}.gpkg")
    return {"aoi": name, "set": "second" if aoi >= 1000 else "first", **confusion_one(base, gpd.read_file(qc_file))}


def run(jobs: int = 0) -> tuple[pd.DataFrame, dict]:
    from concurrent.futures import ProcessPoolExecutor

    from .. import resources
    from .aoi_batch2 import OFFSET
    from .qc_compare import qc2_files

    s = pd.read_csv(SUMMARY)
    files = qc2_files()
    todo = []
    for a, base in zip(s["aoi"], s["base"]):
        n = OFFSET + int(a.split("aoi")[1]) if a.startswith("b2_") else int(a[3:])
        if n in files and isinstance(base, str):
            todo.append((n, base, files[n]))
    with ProcessPoolExecutor(jobs or resources.detect_resources().cpus) as ex:
        t = pd.DataFrame(list(ex.map(_one, todo)))
    tot = {"aois": len(t)}
    for unit in ("ac", "fields"):
        c = {k: float(t[f"{k}_{unit}"].sum()) for k in ("tp", "fp", "fn", "tn")}
        tot[unit] = {**{k: round(v, 3) for k, v in c.items()}, **metrics(**c)}
        for st in ("first", "second"):
            cs = {k: float(t.loc[t["set"] == st, f"{k}_{unit}"].sum()) for k in ("tp", "fp", "fn", "tn")}
            tot[f"{unit}_{st}_set"] = metrics(**cs)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "qc_confusion.csv", index=False)
    (OUT / "qc_metrics.json").write_text(json.dumps(tot, indent=1))
    return t, tot


def figures(tot: dict, out: Path = OUT) -> list[Path]:
    """Confusion matrix (acres, with field counts) and a bar chart of the four metrics (by acres and by fields)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..outreach_figures import HAIRLINE, SLOT_1, SLOT_2, SURFACE, TEXT_1, TEXT_2

    a, f = tot["ac"], tot["fields"]
    fig, ax = plt.subplots(figsize=(8, 6.6), facecolor=SURFACE)
    cells = [[("tp", "Rice", "Rice"), ("fn", "Rice", "Non-rice")], [("fp", "Non-rice", "Rice"), ("tn", "Non-rice", "Non-rice")]]
    total = sum(a[k] for k in ("tp", "fp", "fn", "tn"))
    for i, row in enumerate(cells):
        for j, (k, _, _) in enumerate(row):
            good = k in ("tp", "tn")
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, color="#d6e6f7" if good else "#fbe2d5", ec=SURFACE, lw=4))
            ax.text(j + 0.5, 1.5 - i + 0.12, f"{a[k]:,.0f} ac", ha="center", va="center", fontsize=20, weight="bold",
                    color=TEXT_1)
            ax.text(j + 0.5, 1.5 - i - 0.08, f"{100 * a[k] / total:.1f} % of area", ha="center", fontsize=12, color=TEXT_2)
            ax.text(j + 0.5, 1.5 - i - 0.22, f"{int(f[k]):,} fields  ({k.upper()})", ha="center", fontsize=11, color=TEXT_2)
    ax.set_xlim(0, 2), ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], ["Rice", "Non-rice"], fontsize=13)
    ax.set_yticks([1.5, 0.5], ["Rice", "Non-rice"], fontsize=13)
    ax.set_xlabel("Map (rule) says", fontsize=13, color=TEXT_2)
    ax.set_ylabel("Human QC says", fontsize=13, color=TEXT_2)
    ax.xaxis.set_label_position("top"), ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"Rice vs non-rice: map vs human QC ({tot['aois']} areas)", fontsize=15, weight="bold", color=TEXT_1,
                 pad=48)
    p1 = out / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    names = ["Accuracy", "Precision", "Recall", "F1"]
    keys = ["accuracy", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
    x = np.arange(4)
    w = 0.36
    for off, d, col, lab in ((-w / 2, a, SLOT_1, "by area (acres)"), (w / 2, f, SLOT_2, "by fields")):
        bars = ax.bar(x + off, [d[k] for k in keys], w, color=col, label=lab, zorder=2)
        for bb, v in zip(bars, [d[k] for k in keys]):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=11, color=TEXT_1)
    ax.set_ylim(0, 112)                                    # bars start at zero (a cut axis would exaggerate gaps)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks(x, names, fontsize=13)
    ax.set_ylabel("%", color=TEXT_2)
    ax.grid(axis="y", color=HAIRLINE, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Rice mapping: agreement with human QC (rice = positive class)", fontsize=14, weight="bold",
                 color=TEXT_1, pad=14)
    p2 = out / "metrics.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return [p1, p2]


def main(argv=None) -> int:
    t, tot = run()
    print(json.dumps(tot, indent=1))
    for p in figures(tot):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
