# aoi24: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[24]` (code comments name the pixel or field behind each).

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



### 2026-10-06 06:21:00 - rule set chosen

`aoi118+sowing_from_radar` (switches below)

Why: best of 15 sets / switches on 181 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi118+sowing_from_radar | 94.5 | 94.5 |
| aoi118 | 93.9 | 95.6 |
| aoi118+young_while_radar_low | 93.9 | 95.6 |
| aoi118+any_water_transplanted | 93.4 | 95.6 |
| aoi116 | 91.7 | 94.5 |
| aoi13 | 91.7 | 94.5 |
| aoi83 | 89.0 | 95.0 |
| aoi118+long_flood | 89.0 | 95.0 |
| aoi39 | 84.5 | 85.1 |
| aoi160 | 70.7 | 71.8 |
| aoi28 | 70.7 | 72.9 |
| aoi72 | 69.6 | 73.5 |
| aoi125 | 69.1 | 71.8 |
| aoi33 | 56.9 | 74.0 |
| aoi20 | 56.9 | 74.0 |

### 2026-10-06 06:22:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi24/ (15 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-13) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 167 rice fields, 0.0 of 72.75 ac (none).
