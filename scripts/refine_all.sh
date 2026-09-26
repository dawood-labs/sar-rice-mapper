#!/usr/bin/env bash
# Re-run everything downstream of the field delineation refinement for every AOI, in the order the
# data flows: geometry stage (from the RAW delineation) -> field-level audit -> field labels (with
# the label stage: crumbs and small pieces merged, ponds flagged) -> delivery -> comparison with
# the frozen baseline -> geometry audit of every delivered fields file.
#
# Why a script: the refinement changed after the user's QGIS review of aoi116 (holes, crumbs,
# overlaps, pieces), and the maps themselves did not; re-running the rule would waste an hour.
# The previous refined files are moved to fields_refined/prev/ (run_geometry skips existing files).
#
# Usage: [JOBS=n] [KEEP_GEOMETRY=1] scripts/refine_all.sh <stage_name> [<id> ...]
set -eu
STAGE=$1; shift 1
PY=${PY:-.venv/bin/python}
JOBS=${JOBS:-4}
IDS="$*"
if [ -z "$IDS" ]; then
  IDS=$($PY -c "from sar_pipeline.prep import batch; print(' '.join(str(a) for a in batch.aoi_index()['aoi']))")
fi
mkdir -p logs processed/_batch/s2_2026/fields_refined/prev
t0=$(date +%s); stamp() { echo "[$(( ($(date +%s) - t0) / 60 )) min] $*"; }

stamp "geometry stage (raw delineation -> fields_refined/)"
if [ "${KEEP_GEOMETRY:-0}" != "1" ]; then       # KEEP_GEOMETRY=1: the refined files on disk are current, only the later steps run
  for i in $IDS; do f=processed/_batch/s2_2026/fields_refined/aoi${i}_delineation_refined.gpkg; [ -f "$f" ] && mv "$f" processed/_batch/s2_2026/fields_refined/prev/; done
fi
echo $IDS | tr ' ' '\n' | xargs -P "$JOBS" -I{} sh -c "$PY -m sar_pipeline.analysis.field_refine geometry --ids {} > logs/refine_${STAGE}_geom_aoi{}.log 2>&1 || echo 'aoi{} geometry FAILED'"
stamp "field-level audit"
echo $IDS | tr ' ' '\n' | xargs -P "$JOBS" -I{} sh -c "$PY -m sar_pipeline.analysis.field_level --ids {} > logs/refine_${STAGE}_fl_aoi{}.log 2>&1 || echo 'aoi{} field_level FAILED'"
stamp "field labels (label stage: merges, ponds)"
$PY -m sar_pipeline.analysis.field_rice --ids $IDS > logs/refine_${STAGE}_fields.log 2>&1; tail -12 logs/refine_${STAGE}_fields.log
stamp "delivery"
$PY -m sar_pipeline.delivery --out processed/_batch/s2_2026/delivery > logs/refine_${STAGE}_delivery.log 2>&1; tail -3 logs/refine_${STAGE}_delivery.log
stamp "comparison with the baseline"
$PY -m sar_pipeline.analysis.compare_runs --stage "$STAGE" > logs/refine_${STAGE}_compare.log 2>&1; grep -vE "RuntimeWarning|nanmax" logs/refine_${STAGE}_compare.log
stamp "geometry audit of the delivered fields"
$PY -m sar_pipeline.analysis.field_refine audit --ids $IDS > processed/_batch/s2_2026/report/compare/${STAGE}_fields_audit.txt 2>&1
$PY - <<PYEOF
import pandas as pd, io, re
txt = open("processed/_batch/s2_2026/report/compare/${STAGE}_fields_audit.txt").read()
lines = [l for l in txt.splitlines() if l.strip() and "Warning" not in l and "warnings.warn" not in l and not l.startswith("/")]
t = pd.read_csv(io.StringIO("\n".join(lines)), sep=r"\s+")
print(t.drop(columns=["aoi"]).sum().to_string())
print("AOIs with invalid or overlaps > 1 px or non-polygons:", t[(t.invalid > 0) | (t.overlap_pairs_over_1px > 0) | (t.non_polygon_features > 0)]["aoi"].tolist())
PYEOF
stamp "DONE"
