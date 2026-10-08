# aoi39: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[39]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
- `AGE_LOW_SHARE` = `0.05`
- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`
- `SOWING_FROM_RADAR` = `True`
- `TREE_NEEDS_NO_WATER` = `True`
- `WATER_BOTTOM` = `0.3`
- `WATER_END_TRACKS` = `earliest_unless_wet_view`
- `WATER_FALL_LOOKBACK_DAYS` = `45`
- `WATER_FALL_POLS` = `VV`
- `WATER_SPELL_DAYS` = `30`
- `YOUNG_AFTER_LAST_WATER` = `True`
- `YOUNG_NEEDS_AGE` = `True`
- `YOUNG_WHILE_RADAR_LOW` = `True`

## History





### 2026-10-05 (back-filled from the handover table) - rules and reasons until 5 Oct

the aoi160 defaults + `YOUNG_WHILE_RADAR_LOW True`, `YOUNG_RADAR_LOW_NEEDS_RISE True`, `AGE_FROM_WATER True`, `WATER_BOTTOM 0.3`, `WATER_FALL_LOOKBACK_DAYS 45`, `WATER_SPELL_MIN_PASSES 1`, `GROWN_NOT_IF_WATER_VIEW True`, `POND_IF_DRY_WATER True`, `SECOND_CROP_BY_RADAR True` (`WATER_DEPTH_K 3`), `RADAR_DECIDES_WITHOUT_OPTICAL True`, `YOUNG_AFTER_LAST_WATER True`, `HARVEST_NOT_IF_RADAR_RISING True`, `YOUNG_MIN_RISE_DAYS 40`, `YOUNG_NEEDS_WATER True`, `SIEVE_ACRES 0.15` (aoi39 only; the aoi72 set was tried and rejected by the user, its map kept in `rice_fresh/aoi39/rule_trials/rules_aoi72/`); NEW inputs: radar v002_20261002 (to 25 Sep), series hyb40m1late

Why: 13 Sep clear view: ~70 of 483 ac still under water. 8391 (watered late Aug, radar rising, NDVI 0.74 still climbing: young, not flooded / bare). 30425 = fish pond (label flooded; map still direct seeded, a dry-season-water rule is proposed, not applied). 11226 (transplanted, sown 30 Jun: age from the water spell; a dense canopy lowering the radar is not young). 8384 (no view 6 May - 6 Aug, gradual VV fall into water mid July, one wet pass: transplanted, 61 days). 21405 (under water from mid July, newest view open water: flooded, not harvested); 30425 (fish pond: water on the March-April views -> flooded / bare); 21856 (user: transplanted rice; map still tree, TREE_NEEDS_NO_WATER not applied: it would move ~70 ac of real trees too, decision pending); 13054 left as the rule says (user). 24716 (first crop May-Jun, deep water from 15 Jul, radar up in Sep: second crop = transplanted, not tree). 21618 / 21856 (no optical view during a deep radar water spell: the radar alone decides -> transplanted). 21635 (under water to the end: flooded; radar-only needs VV and VH both up to call a crop while the radar is low). 25885 (last view water, radar up since: young via YOUNG_AFTER_LAST_WATER, later relabelled); 37570 / 37571 / 39894 / 39895 (one hazy 18 Sep view + the April crop's peak read as harvested while the radar was at its top: standing transplanted); user rule 5 Oct: young rice only after the VH has risen for >= 40 days, else flooded / bare where the field ever held water (43450, and 8391 / 15496 / 25885 relabelled flooded); a short-rise green field that never went below its dry radar level stays a crop (29260). 29972 / 26437 / 13545 (never held water, green on 13 Sep: young -> direct seeded; young rice only after water or a cut). Raw now: transplanted 173.4, direct seeded 70.1, young 6.9, flooded/bare 118.3, harvested 5.8, tree 102.4, other veg 3.4 ac. Labels 21/22 agree (29260: user transplanted, map direct seeded - no radar water). **Locked 5 Oct** at sieve 0.15 / sliver 0.15.

### 2026-10-06 05:41:00 - rule set chosen

`aoi13` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi13+young_while_radar_low | 43.5 | 54.0 |
| aoi13+long_flood | 43.5 | 54.0 |
| aoi13+sowing_from_radar | 43.5 | 54.0 |
| aoi13+any_water_transplanted | 43.5 | 54.0 |
| aoi13 | 43.5 | 54.0 |
| aoi116 | 40.5 | 54.0 |
| aoi39 | 33.5 | 74.5 |
| aoi83 | 32.5 | 56.0 |
| aoi125 | 30.0 | 54.5 |
| aoi160 | 29.0 | 54.0 |
| aoi28 | 26.0 | 55.0 |
| aoi118 | 26.0 | 58.0 |
| aoi20 | 24.5 | 55.5 |
| aoi72 | 23.0 | 52.0 |
| aoi33 | 14.5 | 49.5 |

### 2026-10-06 05:41:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi39/ (12 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-13) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 34 of 492 rice fields, 19.44 of 294.02 ac (rice standing transplanted 0.71 ac, young rice 18.73 ac).

### 2026-10-08 11:09:00 - v2 rules (weak AOI)

`aoi39` -> `aoi39_fields_v2.gpkg` (delivered files unchanged)

Why: rice vs not right on reviewed fields 63.0 -> 85.5 % (held-out half 65.4 -> 85.6 %); user, 7 Oct 2026
