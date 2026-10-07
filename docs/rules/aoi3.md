# aoi3: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[3]` (code comments name the pixel or field behind each).

## Switches now

- `HARVEST_NEEDS_BOTH_POLS` = `True`
- `RADAR_DECIDES_WITHOUT_OPTICAL` = `True`
- `RADAR_STORY_IN_GAP` = `True`
- `SIEVE_ACRES` = `0.15`
- `TONE_AND_CURVE` = `True`
- `YOUNG_WHILE_RADAR_LOW` = `True`

## History



### 2026-10-06 18:24:00 - rule set chosen

`universal+young_while_radar_low` (switches below)

Why: best of 18 sets / switches on 33 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| universal+young_while_radar_low | 57.6 | 60.6 |
| universal+long_flood | 51.5 | 60.6 |
| universal+sowing_from_radar | 51.5 | 60.6 |
| universal | 48.5 | 60.6 |
| universal+any_water_transplanted | 48.5 | 57.6 |
| aoi83 | 48.5 | 51.5 |
| universal+universal | 48.5 | 60.6 |
| aoi13 | 45.5 | 48.5 |
| aoi118 | 42.4 | 51.5 |
| aoi116 | 42.4 | 51.5 |
| aoi125 | 42.4 | 42.4 |
| aoi72 | 39.4 | 45.5 |
| aoi33 | 27.3 | 42.4 |
| aoi160 | 27.3 | 36.4 |
| aoi20 | 27.3 | 39.4 |
| aoi28 | 21.2 | 42.4 |
| aoi39 | 18.2 | 24.2 |
| aoi63 | 15.2 | 30.3 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-10-01) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 14 rice fields, 0.0 of 4.86 ac (none).

### 2026-10-06 18:24:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi3/ (13 S2 dates >= 80 % clear)
