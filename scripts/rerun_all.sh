#!/usr/bin/env bash
# Re-run the standing-rice map for every AOI after the fix plan (stage 4), in the order the data
# flows: 5-day series with the chosen cloud mask -> rule -> finalize (relabels + sieve) -> field-level
# audit -> field labels on the refined delineation -> delivery -> comparison with the frozen baseline.
#
# Why a script: nine steps in a fixed order for 132 AOIs; each step skips or overwrites cleanly, so
# the script can be re-run after an interruption. Nothing here starts an Earth Engine export.
#
# Usage: scripts/rerun_all.sh <mask> <stage_name> [<id> ...]
#   <mask>: a variant name of analysis/mask_experiment.VARIANTS (e.g. hyb60); the series are built
#           straight into the standard folder processed/_batch/s2_2026/<aoi>/ (the baseline series
#           are not needed any more: the map baseline is baseline_v3/).
#   <stage_name>: label for the comparison files (report/compare/<stage_name>_*.csv).
#   ids: default every AOI in the AOI index.
set -eu
MASK=$1; STAGE=$2; shift 2
PY=${PY:-.venv/bin/python}
JOBS=${JOBS:-3}
IDS="$*"
if [ -z "$IDS" ]; then
  IDS=$($PY -c "from sar_pipeline.prep import batch; print(' '.join(str(a) for a in batch.aoi_index()['aoi']))")
fi
mkdir -p logs
t0=$(date +%s); stamp() { echo "[$(( ($(date +%s) - t0) / 60 )) min] $*"; }

stamp "series ($MASK mask, $JOBS at a time)"
$PY -m sar_pipeline.analysis.mask_experiment build --ids $IDS --variants "$MASK" --jobs "$JOBS" --into-standard \
    > logs/rerun_${STAGE}_series.log 2>&1; tail -1 logs/rerun_${STAGE}_series.log
stamp "rule"
echo $IDS | tr ' ' '\n' | xargs -P "$JOBS" -I{} sh -c "$PY -u -m sar_pipeline.monsoon_batch rule --ids {} --out processed/_batch/s2_2026/${STAGE}_parts/aoi{}.csv > logs/rerun_${STAGE}_rule_aoi{}.log 2>&1 || echo 'aoi{} rule FAILED'"
stamp "finalize (relabels + sieve)"
$PY -m sar_pipeline.analysis.finalize --ids $IDS > logs/rerun_${STAGE}_finalize.log 2>&1; tail -8 logs/rerun_${STAGE}_finalize.log
stamp "field-level audit"
echo $IDS | tr ' ' '\n' | xargs -P "$JOBS" -I{} sh -c "$PY -m sar_pipeline.analysis.field_level --ids {} > logs/rerun_${STAGE}_fl_aoi{}.log 2>&1 || echo 'aoi{} field_level FAILED'"
stamp "field labels (refined delineation)"
$PY -m sar_pipeline.analysis.field_rice --ids $IDS > logs/rerun_${STAGE}_fields.log 2>&1; tail -12 logs/rerun_${STAGE}_fields.log
stamp "delivery"
$PY -m sar_pipeline.delivery --out processed/_batch/s2_2026/delivery > logs/rerun_${STAGE}_delivery.log 2>&1; tail -3 logs/rerun_${STAGE}_delivery.log
stamp "comparison with the baseline"
$PY -m sar_pipeline.analysis.compare_runs --stage "$STAGE" > logs/rerun_${STAGE}_compare.log 2>&1; cat logs/rerun_${STAGE}_compare.log
stamp "DONE"
