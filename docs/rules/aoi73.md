# aoi73: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[73]` (code comments name the pixel or field behind each).

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



### 2026-10-06 06:30:00 - rule set chosen

`aoi118+sowing_from_radar` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi118+sowing_from_radar | 48.0 | 61.0 |
| aoi118 | 47.5 | 60.5 |
| aoi118+young_while_radar_low | 47.5 | 60.5 |
| aoi125 | 47.0 | 54.5 |
| aoi39 | 47.0 | 70.5 |
| aoi118+any_water_transplanted | 47.0 | 60.5 |
| aoi28 | 46.5 | 57.0 |
| aoi160 | 46.0 | 53.0 |
| aoi72 | 46.0 | 57.5 |
| aoi116 | 42.0 | 55.0 |
| aoi118+long_flood | 42.0 | 60.5 |
| aoi13 | 41.5 | 55.0 |
| aoi83 | 41.0 | 60.0 |
| aoi20 | 40.5 | 64.0 |
| aoi33 | 26.0 | 49.5 |

### 2026-10-06 06:31:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi73/ (16 S2 dates >= 80 % clear)
