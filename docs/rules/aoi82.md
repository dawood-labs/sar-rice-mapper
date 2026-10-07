# aoi82: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[82]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_FROM_WATER` = `True`
- `AGE_LOW_SHARE` = `0.05`
- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`
- `WATER_BOTTOM` = `0.3`
- `WATER_FALL_LOOKBACK_DAYS` = `45`
- `WATER_SPELL_DAYS` = `30`
- `YOUNG_AFTER_LAST_WATER` = `True`
- `YOUNG_NEEDS_AGE` = `True`
- `YOUNG_WHILE_RADAR_LOW` = `True`

## History



### 2026-10-06 06:21:00 - rule set chosen

`aoi72+any_water_transplanted` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi72+any_water_transplanted | 66.0 | 70.5 |
| aoi72+young_while_radar_low | 64.5 | 70.0 |
| aoi72 | 64.5 | 70.0 |
| aoi72+long_flood | 64.5 | 70.0 |
| aoi72+sowing_from_radar | 64.0 | 70.0 |
| aoi125 | 62.5 | 70.5 |
| aoi39 | 62.0 | 68.5 |
| aoi13 | 59.5 | 64.5 |
| aoi118 | 59.5 | 65.5 |
| aoi28 | 59.0 | 70.0 |
| aoi116 | 59.0 | 65.5 |
| aoi160 | 58.5 | 69.5 |
| aoi20 | 56.0 | 69.0 |
| aoi83 | 56.0 | 61.0 |
| aoi33 | 55.5 | 68.5 |

### 2026-10-06 06:23:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi82/ (12 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-23) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 237 rice fields, 0.0 of 124.45 ac (none).
