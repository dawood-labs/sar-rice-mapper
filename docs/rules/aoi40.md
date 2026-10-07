# aoi40: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[40]` (code comments name the pixel or field behind each).

## Switches now

- `AGE_LOW_SHARE` = `0.05`
- `ANY_WATER_TRANSPLANTED` = `True`
- `SIEVE_ACRES` = `0.15`
- `YOUNG_NEEDS_AGE` = `True`

## History



### 2026-10-06 06:25:00 - rule set chosen

`aoi28` (switches below)

Why: best of 15 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| aoi28 | 61.5 | 88.0 |
| aoi39 | 61.5 | 85.0 |
| aoi28+any_water_transplanted | 61.5 | 88.0 |
| aoi160 | 61.0 | 88.0 |
| aoi28+sowing_from_radar | 60.5 | 88.0 |
| aoi13 | 57.5 | 87.0 |
| aoi28+long_flood | 57.0 | 88.0 |
| aoi28+young_while_radar_low | 53.5 | 88.0 |
| aoi125 | 53.0 | 88.0 |
| aoi72 | 47.0 | 87.5 |
| aoi118 | 46.0 | 86.0 |
| aoi116 | 44.5 | 86.5 |
| aoi83 | 31.0 | 87.0 |
| aoi33 | 12.5 | 87.5 |
| aoi20 | 12.5 | 87.5 |

### 2026-10-06 06:26:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi40/ (12 S2 dates >= 80 % clear)

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-09-13) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 1506 rice fields, 0.0 of 1140.56 ac (none).
