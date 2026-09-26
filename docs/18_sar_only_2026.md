# 18. Rice from radar alone: how far Sentinel-1 gets without an optical series (September 2026)

*Status: E1 and E2 measured on 26 September 2026; E3 (field level) below.*

## The question

Some AOIs lie under cloud so heavy that no optical time series can be built. The validated
method (docs/17) needs one: it reads the crop cycle from the 5-day Sentinel-2 series and only
confirms the water with Sentinel-1. Could rice be mapped there from the radar alone, and how well?
Rice first; other crops follow the same path once rice works (user decision).

## How it is tested honestly

Radar-only features, but labels from wherever they are best:

* **surveyed plots** (3,533 plots reported as standing rice on 2 Sep 2026, 2,950 inside twelve AOIs
  of four regions; plot-interior pixels are the recall set; one region's plots are late transplants,
  young on the map date, and score low for every method);
* **negatives** defined from the optical series (`validation.reference_sets`: evergreen cover,
  permanent water, bare / built ground, fields cut before the map date);
* the **teacher**: the validated optical+radar map of all 132 AOIs (classes 0-8). It gives training
  labels (rice = its classes 1 and 6; young = 2 and 7; not rice = 0 and 4; its unsure classes 3, 5
  and 8 are not used for training) and the agreement reference on AOIs a model never saw.

Cross-validation leaves **whole AOIs** out (five folds) or **whole regions** out; pixels of one AOI
are near-copies of each other and a random split would flatter every number. Scores: plot recall
(interior), negative false-positive rate, agreement with the teacher in acres on held-out AOIs.

## Data and features (`analysis/sar_only.py`)

Both tracks (ascending and descending, 17-18 passes each from mid-March to 22 September, 12-day
repeat per track) are read as 5 x 5 power means in dB and merged by date into one series with a
pass every ~6 days. Per pixel, 131 features:

* the merged series resampled to a fixed 6-day grid (32 steps) in VH, VV and VH-VV;
* season summaries per polarisation: pre-monsoon mean and minimum (15 Mar-10 May), the monsoon
  minimum and its date, the maximum and its date, the standard deviation, the rise after the
  minimum, the end level (last 20 days), the late-season maximum and the fall from it;
* the flood, by the per-pass water test of the validated rule but with no optical anchor: among
  the monsoon passes dark enough for water (VH <= -19 dB with a drop >= 4 dB below the pass's own
  previous 12-45 days, or the shallow variant <= -18 dB / >= 5 dB / seen on >= 4 passes), the
  largest drop; its date, VH and VV on it, the VV drop, the number of supporting passes within
  14 days, whether it was the shallow variant, the VH rise from it to the season end, the VV rise;
* how many monsoon passes sat at or below -19 / -18 dB and the longest run of consecutive dark
  passes.

## E1. The radar rule (no learning)

Rice standing = not permanent cover before the monsoon (pre-monsoon VH minimum <= -14 dB) + a
flood (the per-pass test, at least two passes) + a canopy at the end (VH >= -18 dB and >= 3 dB
above the water) + no harvest fall (VH and VV both < 3 dB below their late-season maximum).
Young = the flood and an end between -19 and -18 dB; flooded, not green = the flood and an end
still at or below -19 dB; harvested = the canopy, then both polarisations fell.

| reference set | delta | region B | region N | fourth region (young plots) |
|---|---|---|---|---|
| plot interiors called rice | 94.9 % | 84.4 % | 70.7 % | 27.6 % |
| plot edges | 88.6 | 83.3 | 63.8 | 27.8 |
| evergreen called rice | 12.8 | 12.9 | 0.0 | 3.2 |
| cut before the map date called rice | 6.7 (15 px) | 66.7 (12 px) | - | 17.7 |

Against the teacher on all 132 AOIs: 79 % of the teacher's rice acres are rule rice (37,458 of
47,208), 5.3 % of its not-rice acres are rule rice (1,836 of 34,388), and its class 3 (rice-like
cycle, water not confirmed: 11,872 ac) is 96 % "no cycle" for the radar as well, which is the
radar agreeing with the teacher's doubt.

What the rule misses (a 32,649-pixel sample of the teacher's rice the rule calls "no cycle"):
87 % never show a pass that passes the water test (median VH minimum -19.5 dB: shallow water,
small drops), 13 % have the flood on one pass only; the pre-monsoon cover guard and the canopy
conditions cost nothing. The first version of the rule required a 5 dB rise from the water and
lost half of region N (shallow -21 dB floods ending at -16 dB); 3 dB with the end at a canopy
level was the fix. Evergreen false alarms are tree lines beside paddies whose 5 x 5 box floods on
one or two passes (median dark run 1 pass); requiring two consecutive dark passes would remove
them but also a quarter of region N's plots, so the rule leaves that to the model.

## E2. Gradient boosting on teacher labels, whole AOIs held out

XGBoost (400 trees, depth 6) on 800,016 pixels: up to 3,000 per class per AOI from the teacher's
rice (classes 1 + 6), young / flooded (2 + 7) and not rice (0 + 4), all 132 AOIs; five folds, each
holding out 26-27 whole AOIs; every held-out AOI is scored by the model that never saw it.

| reference set | delta | region B | region N | fourth region (young plots) |
|---|---|---|---|---|
| plot interiors called rice | **97.6 %** | **97.1 %** | **95.6 %** | 29.9 % |
| plot edges | 91.2 | 94.3 | 91.7 | 29.6 |
| evergreen called rice | 14.2 | 19.4 | 12.1 | 3.4 |
| cut before the map date called rice | 6.7 (15 px) | 83.3 (12 px) | - | 13.7 |
| *validated optical+radar map, plot interiors (field labels)* | *99.0* | *98.0* | *96.8* | *29.1* |

Against the teacher on the held-out AOIs (acres): 92 % of the teacher's rice is model rice
(43,548 of 47,208; 3,042 to "not rice", 618 to "young"); 6.8 % of the teacher's not-rice is model
rice (2,339 of 34,388); the teacher's young / flooded is only 44 % recalled as such (the rest
splits between rice and not rice: the radar cannot place a canopy that is not there yet); the
teacher's class 3 (rice-like cycle, water unconfirmed) is 23 % model rice (2,709 of 11,872 ac),
its class 5 (never bare) 0.6 %, its class 8 (cut, no water) 4 %.

Per AOI (median over 132): 91 % of the teacher's rice recalled, 7 % of its not-rice called rice.
The tails say where the two disagree, and both sides are worth knowing:

* high "false positives": the AOI the round-2 review found to be double-cropped (31 % of the
  teacher's not-rice is model rice: the teacher itself misses part of that rice), two region-N
  plot AOIs (48 % and 36 % on a few hundred not-rice pixels between paddies whose water the
  optical rule could not confirm), and three AOIs at 44-55 % that need a look before anything is
  concluded (`oof_aoi.parquet`);
* low recall: the AOI whose rice rests on the user's class-3 relabel (33 %: the teacher's rice
  there has no confirmed water, so the radar cannot see it), the fourth region's late transplants
  (28-62 %), and two AOIs at 45-58 %.

Feature importance: the VH rise from the flood to the season end (0.15), the number of passes at
or below -18 dB (0.07), the VH minimum (0.06), VH at the end (0.05), the flood date (0.05), the
flood found at all, the longest dark run, and the September VH values. The season's second half
carries the decision: a map from radar alone is a map made after the canopy closed.

## E3. Field level

*(filled after E2)*

## Recommendation for the cloudy AOIs

*(filled after E2 and E3)*
