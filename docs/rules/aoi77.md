# aoi77: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[77]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
- `GROWN_NOT_IF_WATER_VIEW` = `True`
- `HARVEST_NOT_IF_RADAR_RISING` = `True`
- `POND_IF_DRY_WATER` = `True`
- `RADAR_DECIDES_WITHOUT_OPTICAL` = `True`
- `SECOND_CROP_BY_RADAR` = `True`
- `SIEVE_ACRES` = `0.15`
- `SOWING_FROM_RADAR` = `True`
- `WATER_BOTTOM` = `0.3`
- `WATER_FALL_LOOKBACK_DAYS` = `45`
- `WATER_SPELL_MIN_PASSES` = `1`
- `YOUNG_AFTER_LAST_WATER` = `True`
- `YOUNG_MIN_RISE_DAYS` = `40`
- `YOUNG_NEEDS_WATER` = `True`
- `YOUNG_RADAR_LOW_NEEDS_RISE` = `True`
- `YOUNG_WHILE_RADAR_LOW` = `True`

## History





### 2026-10-06 05:48:00 - rule set chosen

`aoi39+sowing_from_radar` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi39+sowing_from_radar | 48.5 | 66.5 |
| aoi39+any_water_transplanted | 48.0 | 65.5 |
| aoi39 | 46.5 | 65.5 |
| aoi39+young_while_radar_low | 46.5 | 65.5 |
| aoi39+long_flood | 46.5 | 66.5 |
| aoi160 | 42.0 | 57.0 |
| aoi125 | 42.0 | 57.0 |
| aoi83 | 42.0 | 54.0 |
| aoi28 | 40.5 | 57.0 |
| aoi13 | 40.5 | 55.5 |
| aoi118 | 40.0 | 55.5 |
| aoi116 | 39.5 | 55.5 |
| aoi72 | 35.5 | 55.0 |
| aoi20 | 33.0 | 57.5 |
| aoi33 | 27.5 | 53.0 |

### 2026-10-06 05:49:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi77/ (15 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-13) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 23 of 1372 rice fields, 7.2 of 555.14 ac (rice standing direct seeded 0.13 ac, rice standing transplanted 0.46 ac, young rice 6.61 ac).

### 2026-10-08 11:11:00 - v2 rules (weak AOI)

`aoi160+sowing_from_radar` -> `aoi77_fields_v2.gpkg` (delivered files unchanged)

Why: rice vs not right on reviewed fields 77.0 -> 80.5 % (held-out half 75.0 -> 78.8 %); user, 7 Oct 2026

### 2026-10-08 16:30:00 - v3 rules (weak AOI)

`aoi160+rice_needs_empty_field` -> `aoi77_fields_v3.gpkg` (delivered files unchanged)

Why: rice vs not right on reviewed fields 80.5 -> 81.5 % (held-out half 78.8 -> 79.8 %, vs v2)
