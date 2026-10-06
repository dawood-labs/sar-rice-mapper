# aoi78: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[78]` (code comments name the pixel or field behind each).

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



### 2026-10-06 05:46:00 - rule set chosen

`aoi39+sowing_from_radar` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi39+sowing_from_radar | 61.0 | 72.5 |
| aoi39+long_flood | 59.5 | 71.0 |
| aoi39+any_water_transplanted | 40.0 | 52.0 |
| aoi39+young_while_radar_low | 39.0 | 51.5 |
| aoi39 | 39.0 | 51.5 |
| aoi13 | 38.5 | 40.5 |
| aoi116 | 36.5 | 41.5 |
| aoi83 | 33.5 | 41.0 |
| aoi72 | 30.5 | 41.5 |
| aoi118 | 30.5 | 41.5 |
| aoi125 | 25.0 | 36.0 |
| aoi160 | 24.5 | 35.5 |
| aoi28 | 18.5 | 36.0 |
| aoi20 | 17.5 | 42.0 |
| aoi33 | 9.0 | 34.0 |

### 2026-10-06 05:48:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi78/ (15 S2 dates >= 80 % clear)
