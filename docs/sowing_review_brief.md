# Sowing review brief (for reviewers judging a field's planting date by eye)

Why this file: the rule gives every rice field a sowing date (`analysis.qc_sowing`), and its crop age on 15 September
goes to the client. Before that, reviewers who never see the rule's date read the same curves and say when they see
the water come, the radar rise and the crop planted. The same brief for every reviewer keeps their answers comparable.

## Blind test

Use only the three sheet files of each field: `<id>_curve.png`, `<id>_chips.png`, `<id>.txt`. Never open any other
file of the review folder, any field layer, map or rule output, and never list, glob or print the `verdicts/` folder.
Do not run analysis code. Open all three files for every field.

## What the sheets show

- `_curve.png` top: field-median NDVI (filled dots clear views, hollow hazy, line fitted), dashed LSWI.
  Bottom: Sentinel-1 VV (solid) and VH (dotted) per track, dB.
- `_chips.png`: Sentinel-2 5-3-2 chips around the field, field outline in magenta.
- `.txt`: the numbers: clear-view median NDVI / LSWI per date, radar VV / VH per pass.

Every field shown is rice in the delivered layer. Optical views are rare in the monsoon; the radar sees through clouds.

## What to read (all dates YYYY-MM-DD, or "" when not visible)

- **water_start**: the first radar pass of the season's main flooding before the crop, when VH (and usually VV)
  falls toward the bottom of its own season range and stays there for at least two passes; or a clear view with
  LSWI above NDVI / NDVI near 0. "" when the field shows no water at all.
- **rise_start**: the last radar pass at the bottom before VH climbs steadily out of it over several passes (the
  young crop growing out of the water). For a field without water: when NDVI leaves its bare level and keeps rising.
- **planting**: your best date for when this season's rice was put into the field (transplanted or sown). For a
  transplanted field this lies between water_start and rise_start; say in the reason which one you followed and why.
- **confidence**: high / medium / low. Low when the radar is noisy, there are two candidate floodings, or the
  optical never saw the field grow.
- **reason**: one or two sentences citing dates and values.

## Write

One JSON per field: `<review folder>/verdicts/<id>.json` with exactly the keys `field_id`, `water_start`,
`rise_start`, `planting`, `confidence`, `reason`. Be honest: "" and low confidence where the evidence is thin.
