# aoi96: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[96]` (code comments name the pixel or field behind each).

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



### 2026-10-06 07:49:00 - rule set chosen

`aoi13` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi13+young_while_radar_low | 48.5 | 60.0 |
| aoi13+long_flood | 48.5 | 60.0 |
| aoi13+sowing_from_radar | 48.5 | 60.0 |
| aoi13+any_water_transplanted | 48.5 | 60.0 |
| aoi13 | 48.5 | 60.0 |
| aoi28 | 46.0 | 60.0 |
| aoi39 | 43.0 | 53.5 |
| aoi116 | 35.5 | 58.5 |
| aoi83 | 35.0 | 60.0 |
| aoi118 | 34.5 | 60.0 |
| aoi72 | 32.0 | 59.0 |
| aoi125 | 21.0 | 47.0 |
| aoi160 | 20.5 | 45.0 |
| aoi33 | 15.0 | 87.0 |
| aoi20 | 14.0 | 82.0 |

### 2026-10-06 07:50:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi96/ (17 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-26) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 1639 rice fields, 0.0 of 539.87 ac (none).
