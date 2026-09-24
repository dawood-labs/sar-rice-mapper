#!/usr/bin/env bash
# Review many AOIs (sar_pipeline.review), a few at a time, without running the machine out of memory.
#
# Why: one AOI needs up to ~7 GB of RAM (the largest measured, 7.3 GB); on a machine with ~20 GB
# usable, more than two at once risks an out-of-memory kill of the whole instance. Each AOI waits
# until MIN_FREE_GB are available, and AOIs already reviewed (review/<aoi>/rendered.txt) are skipped,
# so the script can simply be run again after an interruption.
#
# Usage: scripts/review_batch.sh <id> [<id> ...]      (env: JOBS=2, MIN_FREE_GB=10)
set -u
JOBS=${JOBS:-2}
export MIN_FREE_GB=${MIN_FREE_GB:-10}
echo "$@" | tr ' ' '\n' | xargs -P "$JOBS" -I{} sh -c '
  if [ -f processed/_batch/s2_2026/review/aoi{}/rendered.txt ]; then echo "aoi{} skipped (done)"; exit 0; fi
  while [ "$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo)" -lt "$MIN_FREE_GB" ]; do sleep 20; done
  MPLBACKEND=Agg .venv/bin/python -m sar_pipeline.review --ids {} > logs/review_aoi{}.log 2>&1 \
    && echo "aoi{} $(tail -1 logs/review_aoi{}.log)" || echo "aoi{} FAILED"'
echo "REVIEW BATCH DONE"
