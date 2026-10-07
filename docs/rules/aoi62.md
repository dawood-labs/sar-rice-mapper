# aoi62: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[62]` (code comments name the pixel or field behind each).

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



### 2026-10-06 07:44:00 - rule set chosen

`aoi13` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi13+young_while_radar_low | 47.0 | 56.5 |
| aoi13+long_flood | 47.0 | 56.5 |
| aoi13+sowing_from_radar | 47.0 | 56.5 |
| aoi13+any_water_transplanted | 47.0 | 56.5 |
| aoi13 | 47.0 | 56.5 |
| aoi28 | 45.5 | 52.0 |
| aoi118 | 33.0 | 57.5 |
| aoi33 | 32.0 | 71.0 |
| aoi116 | 31.5 | 57.5 |
| aoi83 | 31.5 | 55.5 |
| aoi39 | 31.5 | 48.5 |
| aoi20 | 31.0 | 69.5 |
| aoi160 | 29.0 | 42.5 |
| aoi125 | 29.0 | 44.0 |
| aoi72 | 28.0 | 51.5 |

### 2026-10-06 07:44:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi62/ (18 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-10-01) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 1209 rice fields, 0.0 of 464.21 ac (none).
