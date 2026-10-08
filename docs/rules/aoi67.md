# aoi67: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[67]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
- `AGE_LOW_SHARE` = `0.05`
- `ANY_WATER_TRANSPLANTED` = `True`
- `LONG_WATER_DAYS` = `30`
- `LONG_WATER_GREEN_LEAD` = `0`
- `LONG_WATER_LEVEL` = `0.4`
- `LONG_WATER_WET_VIEW` = `True`
- `NEVER_EMPTY_OTHER_VEG` = `0.35`
- `SIEVE_ACRES` = `0.15`
- `SOWING_FROM_RADAR` = `False`
- `TREE_LOW_BEFORE` = `2026-07-01`
- `TREE_NEEDS_NO_WATER` = `True`
- `VV_JUMP` = `inf`
- `WATER_BOTTOM` = `0.3`
- `WATER_END_TRACKS` = `earliest_unless_wet_view`
- `WATER_FALL_LOOKBACK_DAYS` = `45`
- `WATER_FALL_POLS` = `VV`
- `WATER_SPELL_DAYS` = `30`
- `YOUNG_AFTER_LAST_WATER` = `True`
- `YOUNG_NEEDS_AGE` = `True`
- `YOUNG_WHILE_RADAR_LOW` = `True`

## History







### 2026-10-06 05:41:00 - rule set chosen

`aoi20` (switches below)

Why: best of 15 sets / switches on 23 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi20 | 26.1 | 39.1 |
| aoi72 | 26.1 | 39.1 |
| aoi33 | 26.1 | 39.1 |
| aoi116 | 26.1 | 39.1 |
| aoi39 | 26.1 | 39.1 |
| aoi20+sowing_from_radar | 26.1 | 39.1 |
| aoi118 | 26.1 | 39.1 |
| aoi20+long_flood | 26.1 | 39.1 |
| aoi20+young_while_radar_low | 26.1 | 39.1 |
| aoi83 | 17.4 | 43.5 |
| aoi28 | 13.0 | 39.1 |
| aoi160 | 13.0 | 39.1 |
| aoi13 | 13.0 | 39.1 |
| aoi125 | 13.0 | 39.1 |
| aoi20+any_water_transplanted | 13.0 | 39.1 |

### 2026-10-06 05:41:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi67/ (17 S2 dates >= 80 % clear)

### 2026-10-06 06:02:00 - rule set chosen

`aoi83` (switches below)

Why: best of 15 sets / switches on 60 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi83+young_while_radar_low | 38.3 | 50.0 |
| aoi83+long_flood | 38.3 | 50.0 |
| aoi83 | 38.3 | 50.0 |
| aoi83+any_water_transplanted | 38.3 | 50.0 |
| aoi160 | 36.7 | 48.3 |
| aoi28 | 36.7 | 48.3 |
| aoi125 | 36.7 | 48.3 |
| aoi83+sowing_from_radar | 36.7 | 50.0 |
| aoi13 | 35.0 | 48.3 |
| aoi39 | 21.7 | 53.3 |
| aoi20 | 16.7 | 51.7 |
| aoi33 | 16.7 | 51.7 |
| aoi72 | 15.0 | 48.3 |
| aoi116 | 15.0 | 48.3 |
| aoi118 | 15.0 | 48.3 |

### 2026-10-06 06:03:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi67/ (17 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-18) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 50 rice fields, 0.0 of 23.99 ac (none).

### 2026-10-08 11:10:00 - v2 rules (weak AOI)

`aoi33+any_water_transplanted` -> `aoi67_fields_v2.gpkg` (delivered files unchanged)

Why: rice vs not right on reviewed fields 75.0 -> 81.7 % (held-out half 71.4 -> 78.6 %); user, 7 Oct 2026

### 2026-10-08 16:28:00 - v3 rules (weak AOI)

`aoi33+rice_needs_empty_field` -> `aoi67_fields_v3.gpkg` (delivered files unchanged)

Why: rice vs not right on reviewed fields 81.7 -> 85.0 % (held-out half 78.6 -> 85.7 %, vs v2)
