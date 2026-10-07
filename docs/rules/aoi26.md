# aoi26: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[26]` (code comments name the pixel or field behind each).

## Switches now

- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`

## History



### 2026-10-06 06:30:00 - rule set chosen

`aoi160` (switches below)

Why: best of 15 sets / switches on 27 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi160 | 92.6 | 92.6 |
| aoi28 | 92.6 | 92.6 |
| aoi72 | 92.6 | 92.6 |
| aoi39 | 92.6 | 92.6 |
| aoi116 | 92.6 | 92.6 |
| aoi13 | 92.6 | 92.6 |
| aoi118 | 92.6 | 92.6 |
| aoi160+long_flood | 92.6 | 92.6 |
| aoi160+young_while_radar_low | 92.6 | 92.6 |
| aoi125 | 92.6 | 92.6 |
| aoi160+any_water_transplanted | 92.6 | 92.6 |
| aoi160+sowing_from_radar | 92.6 | 92.6 |
| aoi83 | 88.9 | 92.6 |
| aoi33 | 59.3 | 92.6 |
| aoi20 | 59.3 | 92.6 |

### 2026-10-06 06:30:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi26/ (15 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-13) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 26 rice fields, 0.0 of 15.0 ac (none).
