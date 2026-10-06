# aoi61: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[61]` (code comments name the pixel or field behind each).

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



### 2026-10-06 07:12:00 - rule set chosen

`aoi13` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi13+young_while_radar_low | 43.5 | 62.5 |
| aoi13+long_flood | 43.5 | 62.5 |
| aoi13+sowing_from_radar | 43.5 | 62.5 |
| aoi13+any_water_transplanted | 43.5 | 62.5 |
| aoi13 | 43.5 | 62.5 |
| aoi83 | 41.5 | 61.5 |
| aoi116 | 41.0 | 62.0 |
| aoi72 | 38.0 | 62.5 |
| aoi118 | 37.5 | 61.5 |
| aoi160 | 28.0 | 33.0 |
| aoi125 | 28.0 | 33.0 |
| aoi39 | 23.5 | 36.0 |
| aoi28 | 23.0 | 35.0 |
| aoi20 | 10.5 | 29.5 |
| aoi33 | 9.0 | 58.5 |

### 2026-10-06 07:13:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi61/ (19 S2 dates >= 80 % clear)
