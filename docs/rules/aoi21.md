# aoi21: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[21]` (code comments name the pixel or field behind each).

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
- `SOWING_FROM_RADAR` = `True`
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



### 2026-10-06 07:47:00 - rule set chosen

`aoi83+sowing_from_radar` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi83+sowing_from_radar | 55.0 | 59.5 |
| aoi83 | 53.5 | 60.5 |
| aoi83+any_water_transplanted | 53.5 | 60.5 |
| aoi83+young_while_radar_low | 53.5 | 60.5 |
| aoi83+long_flood | 53.5 | 60.5 |
| aoi13 | 37.5 | 43.5 |
| aoi39 | 36.5 | 42.0 |
| aoi118 | 34.5 | 43.5 |
| aoi116 | 34.5 | 43.5 |
| aoi28 | 34.0 | 41.5 |
| aoi72 | 34.0 | 43.0 |
| aoi125 | 31.0 | 40.0 |
| aoi160 | 29.5 | 36.5 |
| aoi20 | 28.0 | 45.0 |
| aoi33 | 27.0 | 47.5 |

### 2026-10-06 07:48:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi21/ (19 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-28) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 726 rice fields, 0.0 of 265.8 ac (none).
