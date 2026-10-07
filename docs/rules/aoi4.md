# aoi4: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[4]` (code comments name the pixel or field behind each).

## Switches now

- `HARVEST_NEEDS_BOTH_POLS` = `True`
- `RADAR_DECIDES_WITHOUT_OPTICAL` = `True`
- `RADAR_STORY_IN_GAP` = `True`
- `SIEVE_ACRES` = `0.15`
- `TONE_AND_CURVE` = `True`

## History



### 2026-10-06 14:21:00 - rule set chosen

`universal` (switches below)

Why: best of 18 sets / switches on 25 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| universal+long_flood | 52.0 | 88.0 |
| universal+young_while_radar_low | 52.0 | 88.0 |
| universal | 52.0 | 88.0 |
| universal+any_water_transplanted | 52.0 | 88.0 |
| universal+universal | 52.0 | 88.0 |
| universal+sowing_from_radar | 52.0 | 88.0 |
| aoi116 | 36.0 | 84.0 |
| aoi13 | 36.0 | 80.0 |
| aoi83 | 32.0 | 80.0 |
| aoi118 | 32.0 | 84.0 |
| aoi28 | 28.0 | 76.0 |
| aoi160 | 28.0 | 76.0 |
| aoi39 | 28.0 | 80.0 |
| aoi72 | 28.0 | 76.0 |
| aoi125 | 28.0 | 76.0 |
| aoi63 | 24.0 | 80.0 |
| aoi33 | 12.0 | 76.0 |
| aoi20 | 12.0 | 76.0 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-10-01) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 21 rice fields, 0.0 of 7.25 ac (none).

### 2026-10-06 14:21:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi4/ (4 S2 dates >= 80 % clear)
