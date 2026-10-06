# aoi68: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[68]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
- `GROWN_NOT_IF_WATER_VIEW` = `True`
- `HARVEST_NOT_IF_RADAR_RISING` = `True`
- `LONG_WATER_DAYS` = `30`
- `LONG_WATER_GREEN_LEAD` = `0`
- `LONG_WATER_LEVEL` = `0.4`
- `LONG_WATER_WET_VIEW` = `True`
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



### 2026-10-06 06:01:00 - rule set chosen

`aoi39+long_flood` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi39+long_flood | 51.5 | 72.0 |
| aoi39+sowing_from_radar | 51.0 | 72.5 |
| aoi39 | 50.0 | 72.5 |
| aoi39+young_while_radar_low | 50.0 | 72.5 |
| aoi39+any_water_transplanted | 49.0 | 73.0 |
| aoi13 | 48.5 | 65.0 |
| aoi72 | 48.0 | 66.0 |
| aoi125 | 48.0 | 67.5 |
| aoi160 | 47.5 | 67.5 |
| aoi118 | 47.5 | 66.5 |
| aoi28 | 47.5 | 67.5 |
| aoi116 | 47.0 | 65.0 |
| aoi83 | 45.5 | 61.5 |
| aoi20 | 43.0 | 74.0 |
| aoi33 | 20.5 | 57.5 |

### 2026-10-06 06:01:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi68/ (17 S2 dates >= 80 % clear)
