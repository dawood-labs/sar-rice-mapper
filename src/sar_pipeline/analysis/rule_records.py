"""One record per AOI of the rules it was mapped with and WHY: ``docs/rules/aoi<N>.md``.

Why
---
User (5 Oct 2026): "har aoi pr jo jo rule bany ya change ho uski sari reasons details ko save zrur krna ta k bad me
kaam aaye". The switches live in ``curve_rules.AOI_OVERRIDES`` (code comments give the pixel or field behind each),
but the story of an AOI - which rule sets were compared, how the reviewers' verdicts scored them, which set was chosen
and why, every later change with its reason - is spread over chat, logs and tables. This module writes that story as
one readable file per AOI, appended to whenever something changes, so a junior can see later why an AOI looks the way
it does. Tracked in the repo: only AOI numbers, switch names, scores and reasons (no place names).

Use (from code)::

    from sar_pipeline.analysis import rule_records as rr
    rr.record(125, "chosen", "defaults + ANY_WATER_TRANSPLANTED + YOUNG_WHILE_RADAR_LOW",
              reason="best of 9 sets on 200 reviewed fields (54 %)", scores=table)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "docs" / "rules"


def path(aoi: int, root: Path = ROOT) -> Path:
    return Path(root) / f"aoi{aoi}.md"


def _switches(aoi: int) -> str:
    from . import curve_rules as cr

    rules = cr.AOI_OVERRIDES.get(aoi, {})
    if not rules:
        return "- (none: the default rules)"
    return "\n".join(f"- `{k}` = `{v}`" for k, v in sorted(rules.items()))


def record(aoi: int, event: str, what: str, reason: str = "", scores: pd.DataFrame | None = None,
           when: str | None = None, root: Path = ROOT) -> Path:
    """Appends one dated entry (``event``: e.g. "rule sets compared", "chosen", "changed", "locked", "delivered") and
    rewrites the file's header with the AOI's switches as they are now."""
    p = path(aoi, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = p.read_text().split("## History", 1)[1] if p.exists() and "## History" in p.read_text() else "\n"
    when = when or str(pd.Timestamp.now().floor("min"))
    entry = f"\n### {when} - {event}\n\n{what}\n"
    if reason:
        entry += f"\nWhy: {reason}\n"
    if scores is not None and len(scores):
        cols = [str(c) for c in scores.columns]       # a plain markdown table (no extra library needed)
        rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in scores.itertuples(index=False)]
        entry += "\n" + "\n".join(rows) + "\n"
    head = (f"# aoi{aoi}: rules and the reasons behind them\n\n"
            f"Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[{aoi}]`"
            f" (code comments name the pixel or field behind each).\n\n## Switches now\n\n{_switches(aoi)}\n\n")
    p.write_text(head + "## History\n" + body.rstrip("\n") + "\n" + entry)
    return p
