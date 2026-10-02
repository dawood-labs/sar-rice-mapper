#!/usr/bin/env bash
# Fresh-start chain for one or more AOIs (30 Sep 2026): September first-clear image -> vegetation mask -> step 1
# (crop ground vs trees) -> step 2 (sowing) -> step 3t (learned rice traits) -> the AOI's own curve groups for labelling.
# One AOI at a time and under a memory cap: the pod has 20 GB and a crash once killed the instance.
# Usage: SERIES_ROOT=processed/_batch/s2_2026_hyb40m1late scripts/fresh_start_aoi.sh 13 28 39
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
SERIES_ROOT=${SERIES_ROOT:-processed/_batch/s2_2026_hyb40m1late}
ulimit -v 14000000
export GDAL_NUM_THREADS=ALL_CPUS   # one AOI at a time, every core inside it (user, 1 Oct)
for aoi in "$@"; do
  echo "== aoi$aoi $(date +%H:%M:%S)"
  python -m sar_pipeline.analysis.first_clear --aoi "$aoi" --start 2026-09-01 --end 2026-09-30 > /dev/null
  python -m sar_pipeline.analysis.first_clear --aoi "$aoi" --vegetation
  python -m sar_pipeline.analysis.vegetation_types --aoi "$aoi" --series-root "$SERIES_ROOT" | cut -c1-110
  python -m sar_pipeline.analysis.sowing_fresh --aoi "$aoi" --series-root "$SERIES_ROOT"
  python -m sar_pipeline.analysis.rice_features run --aoi "$aoi"
  python -m sar_pipeline.analysis.plot_clusters --aoi-groups "$aoi"
done
echo "== done $(date +%H:%M:%S)"
