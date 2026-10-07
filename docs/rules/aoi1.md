# aoi1: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[1]` (code comments name the pixel or field behind each).

## Switches now

- `ANY_WATER_TRANSPLANTED` = `True`
- `HARVEST_NEEDS_BOTH_POLS` = `True`
- `RADAR_DECIDES_WITHOUT_OPTICAL` = `True`
- `RADAR_STORY_IN_GAP` = `True`
- `SIEVE_ACRES` = `0.15`
- `TONE_AND_CURVE` = `True`

## History



### 2026-10-06 18:28:00 - rule set chosen

`universal+any_water_transplanted` (switches below)

Why: best of 18 sets / switches on 264 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| universal+any_water_transplanted | 50.8 | 61.7 |
| universal+young_while_radar_low | 42.4 | 61.7 |
| universal | 42.4 | 61.7 |
| universal+universal | 42.4 | 61.7 |
| universal+sowing_from_radar | 41.7 | 62.1 |
| universal+long_flood | 40.5 | 59.8 |
| aoi39 | 37.1 | 49.2 |
| aoi125 | 36.0 | 43.6 |
| aoi63 | 36.0 | 54.2 |
| aoi160 | 34.8 | 41.7 |
| aoi13 | 31.1 | 40.2 |
| aoi116 | 29.5 | 40.9 |
| aoi72 | 28.8 | 45.5 |
| aoi28 | 27.7 | 43.6 |
| aoi118 | 27.3 | 44.3 |
| aoi83 | 26.9 | 41.3 |
| aoi20 | 14.4 | 45.1 |
| aoi33 | 12.5 | 43.9 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-10-01) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 4187 rice fields, 0.0 of 2391.42 ac (none).

### 2026-10-06 18:30:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi1/ (14 S2 dates >= 80 % clear)
