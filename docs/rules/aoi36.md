# aoi36: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[36]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_LOW_SHARE` = `0.05`
- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`
- `YOUNG_NEEDS_AGE` = `True`

## History



### 2026-10-06 07:34:00 - rule set chosen

`aoi28` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi28 | 59.0 | 74.0 |
| aoi28+any_water_transplanted | 59.0 | 74.0 |
| aoi28+long_flood | 59.0 | 74.0 |
| aoi160 | 57.5 | 71.5 |
| aoi28+sowing_from_radar | 50.5 | 73.5 |
| aoi83 | 48.0 | 73.5 |
| aoi28+young_while_radar_low | 46.0 | 74.0 |
| aoi125 | 44.5 | 71.5 |
| aoi33 | 32.0 | 74.0 |
| aoi20 | 31.5 | 72.5 |
| aoi13 | 30.0 | 72.5 |
| aoi72 | 27.0 | 73.0 |
| aoi118 | 26.5 | 72.5 |
| aoi116 | 22.0 | 72.0 |
| aoi39 | 21.0 | 58.0 |

### 2026-10-06 07:35:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi36/ (17 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-26) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 1269 rice fields, 0.0 of 582.33 ac (none).
