#!/usr/bin/env bash
# The revised rule (docs/17, 28 Sep 2026) on chosen AOIs, in data order: rule with the 140-day
# fallback -> class-3 phenology check on the rule maps -> generated overrides (class 3 -> rice where
# the AOI's class 3 is its rice) -> finalize -> field-level audit -> field labels -> preview delivery
# of those AOIs -> comparison with the frozen baseline.
#
# Why a script: seven steps in a fixed order; JOBS defaults to 2 because the pod now has 4 CPUs and
# 20 GB (one rule process peaks at ~4-6 GB on a large AOI).
#
# Usage: [JOBS=n] scripts/run_new_rules.sh <stage_name> <out_dir> <id> [<id> ...]
set -eu
STAGE=$1; OUT=$2; shift 2
IDS="$*"
PY=${PY:-.venv/bin/python}
JOBS=${JOBS:-2}
mkdir -p logs
t0=$(date +%s); stamp() { echo "[$(( ($(date +%s) - t0) / 60 )) min] $*"; }

stamp "rule (110-day window, 140-day fallback), $JOBS at a time"
echo $IDS | tr ' ' '\n' | xargs -P "$JOBS" -I{} sh -c "$PY -u -m sar_pipeline.monsoon_batch rule --ids {} --out processed/_batch/s2_2026/${STAGE}_parts/aoi{}.csv > logs/${STAGE}_rule_aoi{}.log 2>&1 || echo 'aoi{} rule FAILED'"
stamp "class-3 phenology on the rule maps -> overrides"
rm -f processed/_batch/s2_2026/report/class3_phenology.csv
$PY -m sar_pipeline.analysis.class3_phenology --ids $IDS --map-suffix "" --write-overrides > logs/${STAGE}_phenology.log 2>&1; tail -3 logs/${STAGE}_phenology.log
stamp "finalize (overrides + sieve)"
$PY -m sar_pipeline.analysis.finalize --ids $IDS > logs/${STAGE}_finalize.log 2>&1; tail -4 logs/${STAGE}_finalize.log
stamp "field-level audit"
echo $IDS | tr ' ' '\n' | xargs -P "$JOBS" -I{} sh -c "$PY -m sar_pipeline.analysis.field_level --ids {} > logs/${STAGE}_fl_aoi{}.log 2>&1 || echo 'aoi{} field_level FAILED'"
stamp "field labels"
$PY -m sar_pipeline.analysis.field_rice --ids $IDS > logs/${STAGE}_fields.log 2>&1; tail -3 logs/${STAGE}_fields.log
stamp "preview delivery -> $OUT"
$PY -m sar_pipeline.delivery --out "$OUT" --ids $IDS > logs/${STAGE}_delivery.log 2>&1; tail -2 logs/${STAGE}_delivery.log
stamp "comparison with the baseline"
$PY -m sar_pipeline.analysis.compare_runs --stage "$STAGE" --ids $IDS > logs/${STAGE}_compare.log 2>&1; grep -vE "RuntimeWarning|nanmax" logs/${STAGE}_compare.log
stamp "DONE"
