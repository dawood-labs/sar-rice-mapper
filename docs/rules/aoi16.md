# aoi16: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[16]` (code comments name the pixel or field behind each).

## Switches now

- `ANY_WATER_TRANSPLANTED` = `True`
- `HARVEST_NEEDS_BOTH_POLS` = `True`
- `RADAR_DECIDES_WITHOUT_OPTICAL` = `True`
- `RADAR_STORY_IN_GAP` = `True`
- `SIEVE_ACRES` = `0.15`
- `TONE_AND_CURVE` = `True`

## History



### 2026-10-06 18:59:00 - rule set chosen

`universal+any_water_transplanted` (switches below)

Why: best of 18 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| universal+any_water_transplanted | 61.5 | 70.5 |
| universal+long_flood | 57.5 | 70.5 |
| universal+sowing_from_radar | 56.5 | 70.5 |
| universal | 56.0 | 70.5 |
| universal+universal | 56.0 | 70.5 |
| universal+young_while_radar_low | 55.5 | 70.0 |
| aoi83 | 55.0 | 66.0 |
| aoi13 | 52.5 | 60.0 |
| aoi116 | 51.0 | 60.5 |
| aoi160 | 50.5 | 62.0 |
| aoi125 | 49.5 | 61.0 |
| aoi39 | 32.0 | 59.5 |
| aoi63 | 30.0 | 64.5 |
| aoi118 | 29.5 | 64.5 |
| aoi20 | 25.5 | 70.5 |
| aoi72 | 24.5 | 60.5 |
| aoi28 | 23.0 | 63.5 |
| aoi33 | 17.0 | 62.5 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-10-01) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 198 rice fields, 0.0 of 83.34 ac (none).

### 2026-10-06 18:59:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi16/ (14 S2 dates >= 80 % clear)
