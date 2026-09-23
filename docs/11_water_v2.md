# 11 — Refining the water test: why rice-like fields lost their water, and the fix (v2)

**Read this after docs/09 (the rule) and docs/10 (its validation).** Code: `analysis/radar_water.py`
(`water_evidence`), `analysis/monsoon_rule.py` (`WATER_DEFAULT`, class 5),
`analysis/water_investigation.py` (the investigation tools), `analysis/validation.py`
(`compare_versions`).

## The question

After the first four batches, over a third of the standing rice-like area (the rule's class 3,
"standing, water unconfirmed") had no radar water. In AOIs where almost everything looks like rice,
that points at the detection rather than the crop: rice is always transplanted into water. So every
step of the water test was re-examined, starting from what the pixels actually show.

## How it was investigated

1. **Radar curves aligned on the optical dates** (`water_investigation.aligned_curves`,
   `plot_aligned`): the median VV/VH of class 1 and class 3 pixels, per track, day by day around the
   trough and around the climb.
2. **Blocks of pixels from the middle of fields** (`group_centers`, `group_sheet`): a 5 x 5 block whose
   9 x 9 surroundings all carry the same class, with its median NDVI (clear and hazy observations
   marked), median VV/VH per track, and every clear 5-3-2 chip. Single pixels and edge pixels were
   dropped after one "field" turned out to be the rim of a fish pond: at an edge the radar 5 x 5 mean
   mixes bunds, roads and neighbours, and the chips are hard to read.
3. **The same evidence as numbers for every pixel sample** (`evidence_features`,
   `aoi_evidence_sample`), for class 1 and class 3, field interior and field edge, in every
   processed AOI, and for the field plots and the negative sets of docs/10.

## What was found

| Finding | Evidence |
|---|---|
| Class 3 was mostly **edges and fragments** | 15 % of class 3 sat inside a field of its own class, against 42 % of class 1; 25 % sat on edges, against 10 % |
| **The optical trough can be weeks after the water** | a delta field flooded from early June to mid-July (VV -20, VH -30 dB) got its trough at the end of July, because cloud hid the field from May to September; the rule's "dry" reference fell inside the flood, so the dip came out negative |
| **Haze fakes troughs on trees and houses** | blocks on tree lines and village strips: NDVI 0.6-0.85 in the dry season, a trough built from hazy observations across a two-month cloud gap, radar steady (VV -5 to -8, VH -12 to -15) all season. The 40-day criterion does not stop this when the cloud lasts two months |
| **Some rice-like crops really had no water** | large fields bare and dry until May (VV -11, VH -19 to -25), then brighter after the rains (VV -5, VH -14) and never dark again: dry-land crops |
| **Fish ponds** | smooth teal rectangles with thick bunds in 5-3-2, where algae make an NDVI cycle |
| "The trough must be seen" would **not** be a safe rule | only 28-92 % of known-rice plots (by region) had a clear low observation within 20 days of their trough |
| A harvest guard on the weeks before the drop would **not** be safe | in one region rice is flooded straight after the previous crop, so the canopy there was still 0.7 |

## The v2 water test

The v1 test (dip of 3 dB at the optical trough, against the field's own level 40-15 days before) is
kept. v2 adds a second way to confirm the water and a new class:

* **Flood anywhere in the bare period** — for each Sentinel-1 pass from 100 to 5 days before the
  optical **climb** (a sharper event than the trough), the drop below the field's own median of the
  12-45 days before that pass (`local_drops`). The best pass over tracks and polarisations must:
  - drop at least **4 dB** (`FLOOD_DROP_MIN`);
  - be a **monsoon** pass, on or after **15 May** (`FLOOD_EARLIEST`): 99 %+ of the plots' floods came
    after it, and a dry-season pass on a harvested, drying field also falls to about -22 dB VH;
  - be **dark in VH on that pass**, at or below **-19 dB** (`FLOOD_VH_MAX`): 90 % of the plots' flood
    passes were at or below -19.4 dB; trees and bare ground were never below -18;
  - have **no canopy on the field** that day, fitted NDVI at most 0.5 (`FLOOD_NDVI_MAX`): a harvest is
    also a radar drop;
  - be **supported by a second pass** (any track) within 14 days that is also 2 dB below its own
    earlier level (`SUPPORT_MIN`): standing water lasts; one speckled pass does not.
* **Class 5, never bare** — a rice-like optical curve whose radar shows a canopy or buildings all
  season: VH within 20 days of the trough above -15 dB, and the second-darkest VH pass before the
  climb above -17 dB (the second, so one outlier pass cannot decide). The optical trough there is
  haze. Class 5 is not rice.

Class 1 = rice-like, standing, and water confirmed by v1 **or** v2. Class 3 = rice-like, standing,
no water by either test, and not class 5.

## Validation (the field plots and negatives of docs/10)

| Plot interior, share called rice | v1 | v2 |
|---|---|---|
| delta | 91.6 % | 94.9 % |
| region B | 86.2 % | 89.6 % |
| capital area | 87.9 % | 92.6 % |
| region Y (late transplanting, crop still young) | 12.7 % | 13.9 % |

* Class 3 on the plots fell from 4.6-6.6 % to 1.1-2.0 %.
* Class 5 took 0.1-0.6 % of plot interiors and 0.5-2.1 % of plot edges.
* Evergreen, bare, water and cut-before-map-date negatives: **0 %** rice under both versions; the
  control AOI stays at 0 acres.
* **Limit:** the negatives rarely pass the optical rule at all, so they test the water test only
  weakly. That is why every change above was also checked by eye on field-interior blocks, and why
  the flood test carries four independent conditions instead of one threshold.

## Effect over the first 89 AOIs

| acres | v1 | v2 |
|---|---|---|
| rice, water confirmed | 33,258 | 38,038 |
| standing rice-like, water unconfirmed | 18,906 | 8,937 |
| never bare (trees, houses; not rice) | — | 5,189 |

Young, harvested and not-rice acres do not change (the water test only splits standing rice-like
pixels). What remains in class 3 is concentrated in a few northern AOIs whose blocks look like
dry-land crops on fields that were bare and dry until the rains, plus floodplain fields whose water
came only after the optical climb; naming those crops needs a field check.

`water_investigation.monsoon_flood_share` asks the remaining class 3 a plainer question: did the
radar see standing water at *any* monsoon date (4 dB below the field's own earlier level, on two
passes)? Over the fourteen AOIs with the most class 3 the answer splits them cleanly: in some,
45-87 % of the field interiors flooded, but only in August, weeks after the crop had closed its
canopy (a river or rain flood over a standing crop, not transplanting water); in others 0-10 % ever
flooded. Neither is evidence of transplanted rice, which is why class 3 stays a separate class.

## How to run

```python
from sar_pipeline.analysis import monsoon_rule as mr
mr.run_aoi(20)                         # v2 is the default (mr.WATER_DEFAULT)
mr.run_aoi(20, water="v1", suffix="_v1")   # the old test, written next to it for comparison

from sar_pipeline.analysis import validation as va
va.compare_versions([20, 24], plots, suffixes=("_v1", ""))   # plots and negatives, per version

from sar_pipeline.analysis import water_investigation as wi
wi.group_sheet(20, center_pid, out_dir="processed/_batch/s2_2026/figures/groups")   # look at a block
```
