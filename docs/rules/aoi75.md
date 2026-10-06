# aoi75: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[75]` (code comments name the pixel or field behind each).

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



### 2026-10-06 05:50:00 - rule set chosen

`aoi39+any_water_transplanted` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi39+sowing_from_radar | 80.0 | 83.0 |
| aoi39+any_water_transplanted | 80.0 | 83.0 |
| aoi39 | 79.5 | 83.0 |
| aoi39+young_while_radar_low | 79.5 | 83.0 |
| aoi28 | 76.5 | 80.5 |
| aoi160 | 76.5 | 80.5 |
| aoi118 | 76.0 | 80.0 |
| aoi72 | 74.5 | 79.0 |
| aoi125 | 73.5 | 80.0 |
| aoi116 | 73.0 | 79.5 |
| aoi13 | 73.0 | 79.0 |
| aoi39+long_flood | 70.5 | 83.0 |
| aoi83 | 66.5 | 78.0 |
| aoi20 | 50.5 | 78.0 |
| aoi33 | 49.0 | 77.0 |

### 2026-10-06 05:50:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi75/ (13 S2 dates >= 80 % clear)
