# aoi76: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[76]` (code comments name the pixel or field behind each).

## Switches now

- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`

## History



### 2026-10-06 06:25:00 - rule set chosen

`aoi160` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi160 | 74.5 | 76.0 |
| aoi28 | 74.5 | 76.0 |
| aoi160+any_water_transplanted | 74.5 | 76.0 |
| aoi13 | 73.5 | 78.0 |
| aoi160+young_while_radar_low | 73.5 | 77.0 |
| aoi125 | 73.5 | 77.0 |
| aoi160+sowing_from_radar | 73.5 | 76.0 |
| aoi160+long_flood | 66.5 | 76.0 |
| aoi39 | 65.0 | 78.5 |
| aoi83 | 59.0 | 80.0 |
| aoi118 | 57.5 | 78.0 |
| aoi72 | 55.5 | 75.5 |
| aoi116 | 55.5 | 77.0 |
| aoi20 | 28.5 | 76.5 |
| aoi33 | 27.5 | 78.5 |

### 2026-10-06 06:25:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi76/ (13 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-13) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 187 rice fields, 0.0 of 165.69 ac (none).
