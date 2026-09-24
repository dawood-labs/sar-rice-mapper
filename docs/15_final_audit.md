# 15 — Final end-to-end audit, at field level

Code: `analysis/final_audit.py` (acquisition, artefact passes, missed rice), `analysis/field_level.py`
(the rule re-run on each field's mean curves), `analysis/field_rice.py` (registration check,
confidence flag), class 6 in `analysis/monsoon_rule.py`.

## 1. Acquisition

* **Sentinel-1**: 132 AOIs, 266 tracks, 12-18 passes each (15 Mar - 24 Sep), longest gap 12 days on
  every track, valid pixels >= 92 % on every pass, no failed export chunk.
* **Sentinel-2**: 93-172 dates per AOI, files on disk match the export manifests; in the monsoon a
  date shows on average 18 % of an AOI clear (QA60), 2-17 dates per AOI are more than half clear.
* **Grids**: optical and radar grids are identical in all 132 AOIs.
* **Artefact passes**: a pass is an artefact when ground that cannot change within a week (evergreen
  pixels, class 5) jumps 3 dB or more against the passes around it. 15 of 8,356 passes (0.2 %) in
  121 AOIs with such ground; the largest a descending VH pass in mid-June (-7.5 dB on trees, VV
  unchanged: a VH-only processing artefact) and an ascending pass in mid-July after 27 mm of rain.
  Other AOI-wide jumps were real (fields flooding while trees stayed flat). The artefact passes are
  now dropped when the radar is read (`sar_curve.bad_pass_indices`); confirmed rice changed by
  -100 acres.

## 2. Field polygons against the map

Moving the delineation 10-30 m lowers field purity in 7 of 8 AOIs tested; in two delta AOIs a 10-20
m north/east shift raises it by up to 1 point (a small offset between the tracing and the 10 m
grid). A 10 m shift changes the rice acres of those AOIs by at most 1.1 %.

## 3. The rule re-run on each field's mean curves

Radar averaged in linear power over the field's pixels and NDVI averaged over the field, then the
same rule. It agrees with the pixel vote on 89.5 % of the field area. On the plots the two are
equally good (plot interiors 96.6/91.9/96.5 % by vote, 96.7/93.1/93.7 % by the mean rule), so the
vote stays the label and the second opinion becomes the field's `label_confidence` (`high` where
both agree, `mixed` otherwise; `not checked` for fields under 4 pixels).

## 4. Every class

* **Rice (1)**: 97 % of its field acres were last seen clear within 30 days of the map date: standing
  is observed, not extrapolated.
* **Young (2) -> young rice (6)**: the young class held 11,000 acres, 74 % with radar water; its
  galleries showed flooded fields (VH -22 to -27 dB for weeks) with a crop rising in September, and
  field-plot pixels (reported as standing rice) sat there too. User decision: young rice whose water
  is confirmed and whose canopy is visible goes to standing rice. Threshold: last NDVI >= 0.30, which
  95 % of the plots' young pixels reach (bare soil 0.15-0.25, open water 0-0.1). For young crops the
  v2 flood is searched up to the map date. Class 6, delivered as rice.
* **Rice-like, water unconfirmed (3)**: docs/12 stands; at field level 10,428 acres.
* **Harvested (4)**: 54 % had radar water: early monsoon rice already cut, correctly not standing.
* **Not rice (0) and harvested (4), searched for hidden rice**: pixels there with radar water and a
  standing canopy total 2,559 acres (upper bound), failing the 40-day bare period (1,137), the trough
  <= 0.40 (1,003) or standing (412). Inspected cases in the late-flooded region were tree lines
  along canals inside plot polygons, flat radar all season: the 40-day criterion was right.
* **Late-flooded region's plots**: 71 % of their pixels are still open water on the map date (NDVI
  median 0.03, radar wet): transplanted too late to be visible; not the target until new imagery.

## 5. Final numbers (inside the AOIs, field labels, acres)

| class | acres |
|---|---|
| rice, standing, water confirmed (1) | 44,261 |
| rice, standing, young (6) | 5,325 |
| rice-like, water not confirmed (3) | 10,428 |
| young, not yet visible or no water (2) | 3,115 |
| harvested (4) | 6,795 |
| never bare (5) | 3,684 |
| not rice (0) | 39,819 |

Plot interiors called rice (1 + 6) by field label: 96.9 % delta, 94.8 % region B, 98.3 % capital
area, 20.3 % region Y (still under water); negatives 0-0.6 %.
