# aoi9: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[9]` (code comments name the pixel or field behind each).

## Switches now

- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`
- `SOWING_FROM_RADAR` = `True`

## History



### 2026-10-06 12:56:00 - rule set chosen

`aoi160+sowing_from_radar` (switches below)

Why: best of 18 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi160+sowing_from_radar | 69.0 | 82.0 |
| aoi160 | 66.5 | 81.5 |
| aoi160+any_water_transplanted | 66.5 | 81.5 |
| aoi125 | 66.0 | 81.5 |
| aoi160+young_while_radar_low | 66.0 | 81.5 |
| aoi160+universal | 64.5 | 80.0 |
| aoi28 | 63.5 | 81.5 |
| aoi160+long_flood | 62.0 | 81.5 |
| aoi39 | 60.5 | 80.5 |
| aoi72 | 59.0 | 80.5 |
| aoi63 | 58.0 | 80.0 |
| aoi118 | 52.5 | 75.0 |
| aoi116 | 51.5 | 75.5 |
| aoi13 | 51.5 | 75.5 |
| aoi83 | 43.5 | 73.0 |
| universal | 29.5 | 79.5 |
| aoi33 | 24.5 | 82.0 |
| aoi20 | 24.5 | 82.0 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-10-01) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 462 rice fields, 0.0 of 255.09 ac (none).

### 2026-10-06 12:56:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi9/ (15 S2 dates >= 80 % clear)
