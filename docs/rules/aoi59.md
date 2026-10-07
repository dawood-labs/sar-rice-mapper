# aoi59: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[59]` (code comments name the pixel or field behind each).

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



### 2026-10-06 07:09:00 - rule set chosen

`aoi83` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi83+young_while_radar_low | 72.0 | 81.5 |
| aoi83+long_flood | 72.0 | 81.5 |
| aoi83 | 72.0 | 81.5 |
| aoi83+any_water_transplanted | 72.0 | 81.5 |
| aoi28 | 71.0 | 82.5 |
| aoi160 | 69.0 | 81.5 |
| aoi125 | 68.5 | 81.5 |
| aoi83+sowing_from_radar | 66.5 | 82.0 |
| aoi13 | 66.5 | 82.0 |
| aoi39 | 42.5 | 79.5 |
| aoi72 | 29.0 | 84.0 |
| aoi118 | 29.0 | 83.5 |
| aoi116 | 28.0 | 83.5 |
| aoi20 | 17.0 | 84.0 |
| aoi33 | 15.5 | 83.0 |

### 2026-10-06 07:09:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi59/ (34 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-10-01) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 993 rice fields, 0.0 of 444.61 ac (none).
