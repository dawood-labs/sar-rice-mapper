# aoi12: rules and the reasons behind them

Written by `sar_pipeline.analysis.rule_records`; the switches are in `curve_rules.AOI_OVERRIDES[12]` (code comments name the pixel or field behind each).

## Switches now

- `HARVEST_NEEDS_BOTH_POLS` = `True`
- `LONG_WATER_DAYS` = `30`
- `LONG_WATER_GREEN_LEAD` = `0`
- `LONG_WATER_LEVEL` = `0.4`
- `LONG_WATER_WET_VIEW` = `True`
- `RADAR_DECIDES_WITHOUT_OPTICAL` = `True`
- `RADAR_STORY_IN_GAP` = `True`
- `SIEVE_ACRES` = `0.15`
- `TONE_AND_CURVE` = `True`

## History



### 2026-10-06 14:29:00 - rule set chosen

`universal+long_flood` (switches below)

Why: best of 18 sets / switches on 200 fields judged by reviewers

| rule set | fields right % | standing/young merged % |
|---|---|---|
| universal+long_flood | 68.5 | 74.0 |
| universal+any_water_transplanted | 63.0 | 73.5 |
| universal+young_while_radar_low | 63.0 | 73.0 |
| universal+sowing_from_radar | 62.5 | 73.5 |
| universal+universal | 62.0 | 73.0 |
| universal | 62.0 | 73.0 |
| aoi116 | 61.0 | 63.5 |
| aoi13 | 61.0 | 63.5 |
| aoi125 | 60.0 | 64.5 |
| aoi83 | 59.0 | 63.5 |
| aoi160 | 57.0 | 62.5 |
| aoi20 | 48.5 | 73.0 |
| aoi33 | 45.5 | 70.5 |
| aoi72 | 42.5 | 75.0 |
| aoi28 | 42.0 | 71.0 |
| aoi118 | 37.0 | 69.5 |
| aoi39 | 22.0 | 52.0 |
| aoi63 | 14.5 | 65.0 |

## Too young rice (not delivered)

Manager's rule (6 Oct 2026): only rice older than about 40 days goes to the client; a field still under open water on the newest clear Sentinel-2 date (2026-10-01) is too young. A rice field is relabelled non-rice when fewer than 5 % of its clear pixels are not open water that day. Result: 0 of 159 rice fields, 0.0 of 69.77 ac (none).

### 2026-10-06 14:29:00 - locked and delivered

<bucket>/rice_map_2026-10-05/aoi12/ (16 S2 dates >= 80 % clear)
