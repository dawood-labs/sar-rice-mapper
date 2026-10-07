# aoi63: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[63]` (code comments name the pixel or field behind each).

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



### 2026-10-06 07:15:00 - rule set chosen

`aoi13` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi13+young_while_radar_low | 45.5 | 53.5 |
| aoi13+long_flood | 45.5 | 53.5 |
| aoi13+sowing_from_radar | 45.5 | 53.5 |
| aoi13+any_water_transplanted | 45.5 | 53.5 |
| aoi13 | 45.5 | 53.5 |
| aoi83 | 41.5 | 56.5 |
| aoi28 | 41.0 | 49.5 |
| aoi125 | 23.0 | 37.5 |
| aoi160 | 23.0 | 35.5 |
| aoi116 | 20.5 | 55.0 |
| aoi118 | 20.0 | 56.5 |
| aoi72 | 17.0 | 52.0 |
| aoi39 | 17.0 | 34.0 |
| aoi33 | 9.0 | 61.0 |
| aoi20 | 6.5 | 56.5 |

### 2026-10-06 07:15:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi63/ (18 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-10-01) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 826 rice fields, 0.0 of 477.35 ac (none).
