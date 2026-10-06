# aoi60: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[60]` (code comments name the pixel or field behind each).

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



### 2026-10-06 07:11:00 - rule set chosen

`aoi39+any_water_transplanted` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi39+any_water_transplanted | 51.5 | 63.0 |
| aoi39+young_while_radar_low | 48.5 | 63.0 |
| aoi39 | 48.5 | 63.0 |
| aoi39+sowing_from_radar | 48.0 | 63.5 |
| aoi39+long_flood | 46.5 | 62.0 |
| aoi83 | 43.5 | 55.0 |
| aoi13 | 37.5 | 52.0 |
| aoi28 | 31.0 | 51.0 |
| aoi118 | 29.5 | 54.0 |
| aoi116 | 28.5 | 53.0 |
| aoi72 | 25.0 | 51.0 |
| aoi160 | 20.5 | 48.5 |
| aoi125 | 19.5 | 49.5 |
| aoi20 | 12.5 | 61.5 |
| aoi33 | 11.0 | 65.0 |

### 2026-10-06 07:12:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi60/ (14 S2 dates >= 80 % clear)
