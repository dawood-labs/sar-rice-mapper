# aoi10: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[10]` (code comments name the pixel or field behind each).

## Switches now

- `ANY_WATER_TRANSPLANTED` = `True`
- `LONG_WATER_DAYS` = `30`
- `LONG_WATER_GREEN_LEAD` = `0`
- `LONG_WATER_LEVEL` = `0.4`
- `LONG_WATER_WET_VIEW` = `True`
- `SIEVE_ACRES` = `0.15`

## History



### 2026-10-06 13:40:00 - rule set chosen

`aoi160+long_flood` (switches below)

Why: best of 18 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi160+long_flood | 58.5 | 69.5 |
| aoi160 | 57.5 | 69.5 |
| aoi160+any_water_transplanted | 57.5 | 69.5 |
| aoi160+sowing_from_radar | 57.0 | 70.0 |
| aoi160+universal | 56.5 | 69.0 |
| aoi125 | 56.5 | 69.5 |
| aoi160+young_while_radar_low | 56.5 | 69.5 |
| aoi116 | 53.0 | 67.0 |
| aoi13 | 53.0 | 66.5 |
| aoi28 | 48.0 | 72.5 |
| aoi72 | 46.5 | 73.5 |
| aoi39 | 41.5 | 65.0 |
| aoi118 | 40.5 | 67.5 |
| aoi83 | 38.0 | 63.5 |
| aoi63 | 37.5 | 67.5 |
| universal | 36.0 | 68.5 |
| aoi33 | 15.5 | 72.5 |
| aoi20 | 15.5 | 72.5 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-10-01) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 924 rice fields, 0.0 of 457.08 ac (none).

### 2026-10-06 13:40:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi10/ (14 S2 dates >= 80 % clear)
