# aoi87: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[87]` (code comments name the pixel or field behind each).

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



### 2026-10-06 07:42:00 - rule set chosen

`aoi39+any_water_transplanted` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi39+any_water_transplanted | 54.0 | 71.5 |
| aoi39+sowing_from_radar | 37.5 | 71.5 |
| aoi39 | 37.0 | 71.5 |
| aoi39+young_while_radar_low | 37.0 | 71.5 |
| aoi39+long_flood | 37.0 | 71.5 |
| aoi28 | 35.5 | 45.0 |
| aoi160 | 35.5 | 45.0 |
| aoi125 | 35.5 | 45.0 |
| aoi83 | 35.0 | 46.0 |
| aoi13 | 34.5 | 47.0 |
| aoi33 | 27.0 | 54.0 |
| aoi72 | 25.5 | 47.5 |
| aoi116 | 25.5 | 47.5 |
| aoi118 | 25.5 | 46.5 |
| aoi20 | 25.0 | 45.5 |

### 2026-10-06 07:42:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi87/ (26 S2 dates >= 80 % clear)
