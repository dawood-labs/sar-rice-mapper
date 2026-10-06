# aoi79: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[79]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
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



### 2026-10-06 06:04:00 - rule set chosen

`aoi39` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi39 | 57.5 | 66.0 |
| aoi39+young_while_radar_low | 57.5 | 66.0 |
| aoi39+long_flood | 57.5 | 66.0 |
| aoi39+any_water_transplanted | 56.5 | 65.0 |
| aoi39+sowing_from_radar | 56.0 | 65.0 |
| aoi125 | 52.5 | 60.5 |
| aoi160 | 51.5 | 60.0 |
| aoi28 | 51.0 | 59.5 |
| aoi116 | 51.0 | 61.0 |
| aoi72 | 51.0 | 60.5 |
| aoi13 | 50.5 | 59.0 |
| aoi118 | 49.5 | 60.5 |
| aoi83 | 49.0 | 57.5 |
| aoi20 | 32.0 | 59.5 |
| aoi33 | 29.5 | 57.0 |

### 2026-10-06 06:05:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi79/ (12 S2 dates >= 80 % clear)
