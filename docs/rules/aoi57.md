# aoi57: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[57]` (code comments name the pixel or field behind each).

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





### 2026-10-06 07:40:00 - rule set chosen

`aoi13` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi13+young_while_radar_low | 44.5 | 60.5 |
| aoi13+long_flood | 44.5 | 60.5 |
| aoi13+sowing_from_radar | 44.5 | 60.5 |
| aoi13+any_water_transplanted | 44.5 | 60.5 |
| aoi13 | 44.5 | 60.5 |
| aoi83 | 41.5 | 64.5 |
| aoi116 | 41.0 | 60.0 |
| aoi39 | 36.5 | 52.0 |
| aoi118 | 35.5 | 61.5 |
| aoi72 | 35.0 | 61.5 |
| aoi28 | 33.5 | 53.0 |
| aoi125 | 31.0 | 48.5 |
| aoi160 | 28.5 | 44.0 |
| aoi20 | 16.0 | 53.5 |
| aoi33 | 14.0 | 64.0 |

### 2026-10-06 07:41:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi57/ (30 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-10-01) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 14 of 1544 rice fields, 4.93 of 666.54 ac (young rice 4.93 ac).

### 2026-10-08 11:09:00 - v2 rules (weak AOI)

`aoi33` -> `aoi57_fields_v2.gpkg` (delivered files unchanged)

Why: rice vs not right on reviewed fields 68.0 -> 72.5 % (held-out half 75.8 -> 76.8 %); user, 7 Oct 2026

### 2026-10-08 16:27:00 - v3 rules (weak AOI)

`aoi33+harvest_not_if_radar_top+rice_needs_empty_field` -> `aoi57_fields_v3.gpkg` (delivered files unchanged)

Why: rice vs not right on reviewed fields 72.5 -> 79.0 % (held-out half 76.8 -> 81.1 %, vs v2)
