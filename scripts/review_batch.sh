#!/usr/bin/env bash
# Review many AOIs (sar_pipeline.review), a few at a time, without running the machine out of memory.
#
# Why: one AOI needs up to ~7.6 GB of RAM (largest measured). Each AOI therefore
#   * is claimed first (an atomic mkdir), so two batches never review the same AOI;
#   * is skipped when already reviewed (review/<aoi>/rendered.txt) - the script can be re-run;
#   * starts only when MIN_FREE_GB of RAM are available, and starts are staggered (a shared lock held
#     for RAMP_S seconds) so several AOIs never allocate their memory at the same moment.
#
# Usage: scripts/review_batch.sh <id> [<id> ...]      (env: JOBS=3, MIN_FREE_GB=10, RAMP_S=90)
set -u
JOBS=${JOBS:-3}
export MIN_FREE_GB=${MIN_FREE_GB:-10} RAMP_S=${RAMP_S:-90}
echo "$@" | tr ' ' '\n' | xargs -P "$JOBS" -I{} sh -c '
  d=processed/_batch/s2_2026/review/aoi{}
  if [ -f $d/rendered.txt ]; then echo "aoi{} skipped (done)"; exit 0; fi
  mkdir -p processed/_batch/s2_2026/review
  if ! mkdir $d.claim 2>/dev/null; then echo "aoi{} skipped (claimed by another batch)"; exit 0; fi
  (
    flock 8
    while [ "$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo)" -lt "$MIN_FREE_GB" ]; do sleep 20; done
    (sleep "$RAMP_S"; flock -u 8) &
    MPLBACKEND=Agg .venv/bin/python -m sar_pipeline.review --ids {} > logs/review_aoi{}.log 2>&1 \
      && echo "aoi{} $(tail -1 logs/review_aoi{}.log)" || echo "aoi{} FAILED"
  ) 8>/tmp/review_start.lock
  rmdir $d.claim'
echo "REVIEW BATCH DONE"
