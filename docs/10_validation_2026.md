# 10 — Validation of the standing-rice rule against field plots (September 2026)

This page records how the rule in `analysis/monsoon_rule.py` was checked against 2,950 surveyed
plots reported as standing rice, what it got wrong, and what was changed. Every number comes from
`analysis/validation.py`, `analysis/crop_signature.py` and `analysis/geography.py`; the figures and
CSVs live under `processed/_batch/s2_2026/` (not in the repository). Regions are named neutrally
here: **delta** (four AOIs, 951 plots), **region B** (one AOI, 1,133 plots), **capital area** (two
AOIs, 245 plots) and **region Y** (five AOIs, 475 plots, late transplanting); plus one large AOI
without plots and one control AOI with no monsoon crop.

## 1. What is being validated

The deliverable is rice **standing on the map date** (21 September 2026). The rule, per pixel:

1. a **trough** in the last 110 days, fitted NDVI <= 0.40, that lasted at least **8 windows
   (about 40 days)** without a canopy (`LOW_WINDOWS_MIN`);
2. a **climb** of >= 0.30 after it, reaching a canopy >= 0.50;
3. the canopy **still standing** on the last window (>= 0.50 and not more than 0.25 below its peak);
4. **water at the trough confirmed by radar**: a drop of >= 3 dB in VV or VH below the field's own
   dry level 40-15 days earlier, on at least one track.

Pixels passing 1-3 but not 4 are *rice by phenology, water unconfirmed* (class 3); a canopy that
was cut before the map date is *harvested* (4); a climb that has not reached a canopy is *young* (2).

## 2. The ground data

* 3,533 plots in five deliveries; 2,950 inside AOIs; 108 records without geometry; median plot
  0.16-0.72 acres, i.e. 6-30 pixels. No attribute names the crop: "standing rice" is the survey's
  statement for the whole delivery, dated 2 September 2026.
* **Registration.** Shifting every plot outline by 10, 20 or 30 m in four directions made the
  transplant dates inside the outlines *less* coherent in every AOI tested (median within-plot
  spread 7.4 -> 9-11 days in the capital area, 1.9 -> 2.3-2.4 days in the delta). The real outline
  is the most coherent one: the polygons sit on the fields they mean.
* **Interior against edge.** Plot-interior pixels (all four neighbours in the same plot) and edge
  pixels score alike (88-91 % against 85-90 % called rice), so the plots' size does not drive
  the result. 28 of 1,735 plots with >= 6 pixels disagree as a whole; the two looked at were a
  field still under water on the last date and a field whose new crop had just started.

## 3. Negative references

Only rice had ever been measured. Negative sets were built from the same AOIs, from the series
alone: **evergreen** (NDVI >= 0.5 in >= 90 % of windows; 3,736 px), **permanent water**,
**bare or built** (max NDVI < 0.3), and **cut before the map date** (a canopy in June-August, gone on
the last window; 4,345 px). The rule must call none of them standing rice.

## 4. What the first version got wrong

| Check | Finding | Change |
|---|---|---|
| Evergreen false positives | 27 % of evergreen pixels in region Y and 4 % in the large AOI were called rice. A haze dip that the light cloud mask lets through makes a fake trough on a tree canopy, and radar noise then passes 6-18 % of them (VH null at 3 dB). | The 40-day no-canopy criterion: rice plots are without canopy for 55 days at the 5th percentile, a haze dip lasts one or two windows. Evergreen false positives 7.1 % -> **0.0 %**, recall unchanged. |
| Cloud leakage of the mask | 9-16 % of the observations QA60 kept are called cloudy by Cloud Score+ (30-46 % in the monsoon); they sit 0.13-0.22 NDVI below nearby clear observations on rice plots and 0.49 below on evergreen. | Kept as a known limit; the stricter mask (QA60 + Cloud Score+ >= 60) also removes the evergreen false positives but costs 5 points of recall and opens 65-105-day gaps. Available as `ndvi_5day.build(..., cs_min=60)`. |
| Trough at the wrong moment | The season minimum was often May's bare dry field, weeks before a July transplanting; the radar then looked for water where there was none. | Trough searched in the last 110 days only. Radar-wet share in the delta and capital area rose from 90-92 % to 93-95 %. |
| Standing crop undetected | The cycle detector needs a descent after a peak; a crop still climbing on the last date had no cycle at all (every capital-area plot). | The last sample counts as a peak when the series has risen by the prominence since its last minimum. |
| Harvest looks like a dip | On fields cut before the map date, 14-30 % show a >= 3 dB "dip" at a random date: a harvest drops VV too. | The standing criterion excludes them (0 % called rice); the dip alone is not proof of water. |

## 5. Results (final rule, plot-interior pixels)

| Group | Called standing rice | Water unconfirmed | Not standing rice |
|---|---|---|---|
| Delta plots | 89-93 % (one AOI 73 %) | 4-16 % | 1-10 % |
| Region B plots | 80 % | 8 % | 1 % (+7 % harvested) |
| Capital-area plots | 87-89 % | 6-8 % | 1 % |
| Region Y plots | 0-15 % | 2-6 % | 50-82 % (young or under water) |
| Evergreen (3,736 px) | **0.0 %** | | |
| Cut before map date (4,345 px) | **0.0 %** | | |
| Bare / built, permanent water | **0.0 %** | | |
| Control AOI | 0.0 acres | | |

Region Y is not a failure of the rule: its rice was transplanted in late August and had not built a
canopy by the map date, which is outside the deliverable; it will be mapped when new imagery is
added.

**Sensitivity.** Recall is flat around every threshold: trough 0.25-0.50 gives 89-91 %, climb
0.15-0.40 gives 90 %, canopy 0.40-0.55 gives 89-92 %, dip 2-4 dB gives 87-93 %. Thresholds derived
mechanically as the plots' own 10th/90th percentiles on two regions and tested on the third give
59-78 % recall with 0 % false positives: the chosen values are looser than that, and hold in all
three regions.

**Water.** Without the radar requirement recall is 95.7 %; with it 90.5 % (any track), 68.8 %
(both tracks). The radar dip's null on evergreen ground at a random date is 2-10 % (VV) and 6-18 %
(VH) at 3 dB, 0.3-3 % / 3-7 % at 4 dB; on bare dark ground VH is noisy (25 % at 3 dB) and VV is
not (4 %).

**Interpolation.** The trough window itself was observed on only 17 % of plot pixels (62 % within
10 days, 18 % more than 15 days away). Recall is not lower where the trough was filled (96.9 %
against 86.8 % where observed), and the fitted trough sits 0.01 NDVI from the lowest real
observation within 15 days. This is the honest limit: dates are ±1-2 weeks in the monsoon.

## 6. The large AOI without plots

Its phenology-rice area was compared with the plots descriptor by descriptor
(`crop_signature`): the plots were still standing on the last date (90 %) with a radar dip at
transplanting in 71 %; the AOI's candidates had been cut before the last date in 77 %, after
cycles of about 45 days from green-up, with a dip in 29 %. Short crops cut in July-August are not
monsoon rice. Under the final rule the AOI carries 585 acres of standing rice with water confirmed,
808 acres standing with water unconfirmed, 576 acres harvested and 123 young; the unconfirmed part
needs ground truth to be resolved either way.

## 7. Geography

`geography.shift_table` lays every AOI out north to south with its transplant date, green-up,
canopy, water signature and the previous crop's harvest. On this delivery the transplant date
moves from late June in the delta to mid-July in region B and the capital area, and to late
August in region Y; the water shows as deep open water in the delta (optical and radar agree) and
as a shallow, short wetting in the capital area (radar only). The previous crop was cut in
November-December in the north and in January-March in region Y.
