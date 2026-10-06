# aoi69: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[69]` (code comments name the pixel or field behind each).

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



### 2026-10-06 07:08:00 - rule set chosen

`aoi13` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi13+young_while_radar_low | 58.5 | 72.5 |
| aoi13+long_flood | 58.5 | 72.5 |
| aoi13+sowing_from_radar | 58.5 | 72.5 |
| aoi13+any_water_transplanted | 58.5 | 72.5 |
| aoi13 | 58.5 | 72.5 |
| aoi39 | 57.5 | 71.0 |
| aoi116 | 55.5 | 73.0 |
| aoi118 | 54.0 | 73.0 |
| aoi28 | 53.5 | 71.0 |
| aoi72 | 52.0 | 72.0 |
| aoi125 | 49.0 | 69.0 |
| aoi160 | 48.5 | 68.5 |
| aoi83 | 43.0 | 71.5 |
| aoi33 | 20.5 | 74.5 |
| aoi20 | 20.5 | 74.5 |

### 2026-10-06 07:09:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi69/ (16 S2 dates >= 80 % clear)
