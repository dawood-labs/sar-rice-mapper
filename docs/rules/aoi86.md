# aoi86: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[86]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
- `AGE_LOW_SHARE` = `0.05`
- `ANY_WATER_TRANSPLANTED` = `True`
- `LONG_WATER_DAYS` = `30`
- `LONG_WATER_GREEN_LEAD` = `0`
- `LONG_WATER_LEVEL` = `0.4`
- `LONG_WATER_WET_VIEW` = `True`
- `NEVER_EMPTY_OTHER_VEG` = `0.35`
- `SIEVE_ACRES` = `0.15`
- `SOWING_FROM_RADAR` = `False`
- `TREE_LOW_BEFORE` = `2026-07-01`
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



### 2026-10-06 07:32:00 - rule set chosen

`aoi83` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi83+young_while_radar_low | 57.5 | 68.0 |
| aoi83+long_flood | 57.5 | 68.0 |
| aoi83 | 57.5 | 68.0 |
| aoi83+any_water_transplanted | 57.5 | 68.0 |
| aoi83+sowing_from_radar | 56.5 | 67.5 |
| aoi13 | 48.5 | 60.5 |
| aoi160 | 48.0 | 59.5 |
| aoi125 | 48.0 | 59.5 |
| aoi28 | 47.5 | 60.5 |
| aoi116 | 31.0 | 60.5 |
| aoi118 | 30.5 | 61.5 |
| aoi72 | 29.0 | 60.0 |
| aoi20 | 26.0 | 61.0 |
| aoi39 | 25.0 | 56.0 |
| aoi33 | 23.5 | 64.5 |

### 2026-10-06 07:33:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi86/ (23 S2 dates >= 80 % clear)
