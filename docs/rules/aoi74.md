# aoi74: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[74]` (code comments name the pixel or field behind each).

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



### 2026-10-06 05:42:00 - rule set chosen

`aoi13` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi13+young_while_radar_low | 57.0 | 69.0 |
| aoi13+long_flood | 57.0 | 69.0 |
| aoi13+sowing_from_radar | 57.0 | 69.0 |
| aoi13+any_water_transplanted | 57.0 | 69.0 |
| aoi13 | 57.0 | 69.0 |
| aoi125 | 54.5 | 61.5 |
| aoi160 | 54.5 | 61.0 |
| aoi39 | 52.5 | 70.0 |
| aoi116 | 50.5 | 69.5 |
| aoi28 | 50.0 | 61.5 |
| aoi118 | 47.5 | 70.0 |
| aoi83 | 45.0 | 65.5 |
| aoi72 | 39.0 | 60.5 |
| aoi33 | 27.5 | 66.5 |
| aoi20 | 26.5 | 60.0 |

### 2026-10-06 05:42:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi74/ (13 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-13) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 1396 rice fields, 0.0 of 675.02 ac (none).
