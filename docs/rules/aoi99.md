# aoi99: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[99]` (code comments name the pixel or field behind each).

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



### 2026-10-06 07:05:00 - rule set chosen

`aoi39+any_water_transplanted` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi39+any_water_transplanted | 57.5 | 67.0 |
| aoi39+sowing_from_radar | 56.5 | 66.5 |
| aoi39 | 55.0 | 66.5 |
| aoi39+young_while_radar_low | 55.0 | 66.5 |
| aoi39+long_flood | 53.5 | 66.0 |
| aoi13 | 35.0 | 46.5 |
| aoi160 | 34.5 | 50.0 |
| aoi125 | 34.5 | 49.5 |
| aoi83 | 32.5 | 46.5 |
| aoi116 | 31.0 | 46.0 |
| aoi72 | 30.0 | 46.0 |
| aoi28 | 30.0 | 46.5 |
| aoi118 | 30.0 | 46.0 |
| aoi20 | 18.5 | 49.0 |
| aoi33 | 18.0 | 48.5 |

### 2026-10-06 07:05:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi99/ (15 S2 dates >= 80 % clear)
