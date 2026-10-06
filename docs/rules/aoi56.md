# aoi56: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[56]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
- `AGE_LOW_SHARE` = `0.05`
- `SIEVE_ACRES` = `0.15`
- `SOWING_FROM_RADAR` = `True`
- `TREE_NEEDS_NO_WATER` = `True`
- `VV_JUMP` = `inf`
- `WATER_BOTTOM` = `0.3`
- `WATER_END_TRACKS` = `earliest_unless_wet_view`
- `WATER_FALL_LOOKBACK_DAYS` = `45`
- `WATER_FALL_POLS` = `VV`
- `WATER_SPELL_DAYS` = `30`
- `YOUNG_AFTER_LAST_WATER` = `True`
- `YOUNG_NEEDS_AGE` = `True`
- `YOUNG_WHILE_RADAR_LOW` = `True`

## History



### 2026-10-06 06:28:00 - rule set chosen

`aoi118+sowing_from_radar` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi118+sowing_from_radar | 62.5 | 70.5 |
| aoi118+any_water_transplanted | 56.0 | 70.5 |
| aoi118+long_flood | 56.0 | 70.5 |
| aoi118 | 55.0 | 70.0 |
| aoi118+young_while_radar_low | 55.0 | 70.0 |
| aoi125 | 54.5 | 60.5 |
| aoi83 | 54.0 | 68.0 |
| aoi160 | 50.0 | 57.0 |
| aoi13 | 49.0 | 56.0 |
| aoi39 | 48.5 | 70.5 |
| aoi116 | 48.0 | 56.0 |
| aoi28 | 46.5 | 61.5 |
| aoi20 | 42.5 | 63.5 |
| aoi72 | 37.5 | 52.5 |
| aoi33 | 23.5 | 44.5 |

### 2026-10-06 06:29:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi56/ (14 S2 dates >= 80 % clear)
