# aoi25: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[25]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
- `ANY_WATER_TRANSPLANTED` = `True`
- `GROWN_NOT_IF_WATER_VIEW` = `True`
- `HARVEST_NOT_IF_RADAR_RISING` = `True`
- `POND_IF_DRY_WATER` = `True`
- `RADAR_DECIDES_WITHOUT_OPTICAL` = `True`
- `SECOND_CROP_BY_RADAR` = `True`
- `SIEVE_ACRES` = `0.15`
- `WATER_BOTTOM` = `0.3`
- `WATER_FALL_LOOKBACK_DAYS` = `45`
- `WATER_SPELL_MIN_PASSES` = `1`
- `YOUNG_AFTER_LAST_WATER` = `True`
- `YOUNG_MIN_RISE_DAYS` = `40`
- `YOUNG_NEEDS_WATER` = `True`
- `YOUNG_RADAR_LOW_NEEDS_RISE` = `True`
- `YOUNG_WHILE_RADAR_LOW` = `True`

## History



### 2026-10-06 06:07:00 - rule set chosen

`aoi39+any_water_transplanted` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi39+any_water_transplanted | 98.5 | 98.5 |
| aoi39+young_while_radar_low | 97.0 | 98.5 |
| aoi39 | 97.0 | 98.5 |
| aoi118 | 96.5 | 99.0 |
| aoi13 | 96.0 | 99.0 |
| aoi39+sowing_from_radar | 96.0 | 98.5 |
| aoi116 | 93.5 | 99.0 |
| aoi28 | 90.0 | 90.0 |
| aoi125 | 88.5 | 89.5 |
| aoi160 | 88.5 | 89.5 |
| aoi72 | 87.5 | 90.0 |
| aoi83 | 86.5 | 99.0 |
| aoi39+long_flood | 83.5 | 98.5 |
| aoi33 | 69.5 | 89.5 |
| aoi20 | 69.5 | 89.5 |

### 2026-10-06 06:07:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi25/ (14 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-13) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 418 rice fields, 0.0 of 240.44 ac (none).
