# aoi30: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[30]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_LOW_SHARE` = `0.05`
- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`
- `SOWING_FROM_RADAR` = `True`
- `YOUNG_NEEDS_AGE` = `True`

## History



### 2026-10-06 07:40:00 - rule set chosen

`aoi28+sowing_from_radar` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi28+sowing_from_radar | 55.0 | 71.5 |
| aoi28+any_water_transplanted | 53.0 | 73.0 |
| aoi28 | 53.0 | 73.0 |
| aoi13 | 51.5 | 69.0 |
| aoi28+long_flood | 50.0 | 70.0 |
| aoi28+young_while_radar_low | 49.0 | 72.5 |
| aoi116 | 48.5 | 69.0 |
| aoi118 | 47.5 | 73.0 |
| aoi72 | 46.5 | 73.5 |
| aoi83 | 36.0 | 65.5 |
| aoi125 | 34.5 | 54.5 |
| aoi39 | 33.0 | 39.0 |
| aoi160 | 32.5 | 47.0 |
| aoi33 | 19.5 | 80.5 |
| aoi20 | 19.5 | 80.0 |

### 2026-10-06 07:40:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi30/ (20 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-26) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 789 rice fields, 0.0 of 359.57 ac (none).
