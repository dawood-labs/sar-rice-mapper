# aoi7: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[7]` (code comments name the pixel or field behind each).

## Switches now

- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`
- `YOUNG_WHILE_RADAR_LOW` = `True`

## History



### 2026-10-06 14:24:00 - rule set chosen

`aoi125` (switches below)

Why: best of 18 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi125+young_while_radar_low | 53.5 | 64.5 |
| aoi125+any_water_transplanted | 53.5 | 64.5 |
| aoi125 | 53.5 | 64.5 |
| aoi160 | 53.0 | 64.0 |
| aoi125+long_flood | 51.0 | 66.0 |
| aoi13 | 50.0 | 67.0 |
| aoi125+sowing_from_radar | 50.0 | 65.0 |
| aoi116 | 48.5 | 67.0 |
| aoi125+universal | 48.0 | 71.0 |
| aoi28 | 45.0 | 66.5 |
| aoi83 | 43.5 | 66.0 |
| aoi72 | 41.5 | 67.0 |
| aoi39 | 40.0 | 71.0 |
| aoi118 | 39.0 | 68.0 |
| universal | 38.5 | 70.5 |
| aoi63 | 33.0 | 69.0 |
| aoi33 | 28.5 | 68.5 |
| aoi20 | 28.5 | 68.5 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-10-01) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 432 rice fields, 0.0 of 199.77 ac (none).

### 2026-10-06 14:25:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi7/ (13 S2 dates >= 80 % clear)
