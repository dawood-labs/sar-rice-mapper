"""`python -m sar_pipeline.analysis <step> --config <yaml> [--run RUN] ...` (no Earth Engine exports).

Steps, in order:
  gcp-qc      vector QC of the ground-truth fields; with --write-copy also the QC-labelled copy (needs gcp-eda)
  gcp-eda     per-field time series, class curves, outlier flags, review list
  features    per-pixel features for one or more feature sets (bins x start x window)
  evaluate    spatial CV of the feature sets (optionally saving out-of-fold predictions)
  cv-layers   QGIS layers of the out-of-fold predictions for one set
  class-map   class map + target probability map of a run's AOI
  model       holdout split, RF/XGBoost tuning, target threshold, one holdout test, validated map
  final-models  refit the model winner on all fields as variants for missing data + a field-level model
  predict     class map of a (large) AOI with the final models, best available model per pixel
  field-labels  label a delineation's field polygons from a class map (geometry cleanup + two labels)
"""
from __future__ import annotations

import argparse
import logging
import sys

from .. import config as config_mod


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)

    def add(name, run=True):
        s = sub.add_parser(name)
        s.add_argument("--config", required=True, help="your config (config/<aoi>_<season>.yaml)")
        if run:
            s.add_argument("--run", default=None, help="run id (default: newest run)")
        return s

    s = add("gcp-qc", run=True)
    s.add_argument("--write-copy", action="store_true", help="write <gcps>_qc.gpkg from the gcp-eda review list")
    s.add_argument("--force", action="store_true", help="rebuild an existing QC copy")
    add("gcp-eda")
    s = add("features")
    s.add_argument("--bins", nargs="+", default=None, help="half_month and/or bin lengths in days, e.g. half_month 8 12")
    s.add_argument("--starts", nargs="+", default=None, help="first dates YYYY-MM-DD (default: analysis.bins.start)")
    s.add_argument("--windows", nargs="+", type=int, default=None, help="window sizes in pixels (default: analysis.window_px)")
    s.add_argument("--interior-only", action="store_true", help="only pixels at least analysis.border_m inside the edge")
    s.add_argument("--name", default="features", help="output file stem (default: features)")
    s = add("evaluate")
    s.add_argument("--name", default="features")
    s.add_argument("--sets", nargs="+", default=None, help="feature set names (default: all in the file)")
    s.add_argument("--save-oof", action="store_true", help="save out-of-fold predictions (needed by cv-layers)")
    s = add("cv-layers")
    s.add_argument("--set", required=True)
    s = add("class-map")
    s.add_argument("--set", required=True)
    s.add_argument("--name", default="features")
    s.add_argument("--map-config", required=True, help="config of the AOI to map")
    s.add_argument("--map-run", default=None)
    s = add("model")
    s.add_argument("--name", default="features", help="features file stem to reuse when its set matches the config")
    s.add_argument("--map-config", default=None, help="config of the AOI to map (optional)")
    s.add_argument("--map-run", default=None)
    s = add("final-models")
    s.add_argument("--name", default="features", help="features file stem (as used by the model step)")
    s.add_argument("--without-bin", nargs="+", default=[], metavar="TRACK:MMDD",
                   help="extra variant without one time bin of one track, e.g. RO123_ASC:0801")
    s = add("predict")
    s.add_argument("--map-config", required=True, help="config of the AOI to map")
    s.add_argument("--map-run", default=None)
    s.add_argument("--models", default=None, help="folder of .joblib models (default: the training run's final_models)")
    s.add_argument("--name", default=None, help="output file stem (default: final_<model>_<set>)")
    s = add("field-labels")
    s.add_argument("--delineation", required=True, help="field polygons traced on imagery (any vector file)")
    s.add_argument("--class-raster", required=True, help="class map of this run's AOI (from class-map or model)")
    s.add_argument("--prob-raster", default=None, help="target-probability map (default: the class raster's sibling)")
    s.add_argument("--model", default=None, help="field-level model (.joblib) for the second label")
    s.add_argument("--train-config", default=None, help="config of the run the model was trained on (default: --config)")
    s.add_argument("--train-run", default=None, help="run id of that training run")
    s.add_argument("--name", default="fields", help="output folder name under processed/<aoi>/<season>/fields/")
    s.add_argument("--simplify-m", type=float, default=None,
                   help="simplify outlines by this many metres first (default 0.5; 0 = off)")
    return p


def _sets(cfg: dict, args):
    from .pixel_features import FeatureSet, set_from_config

    base = set_from_config(cfg)
    bins = args.bins or [base.kind if base.kind == "half_month" else str(base.days)]
    starts = args.starts or [base.start]
    windows = args.windows or [base.window]
    out = []
    for b in bins:
        for s in starts:
            for w in windows:
                out.append(FeatureSet(kind="half_month", window=w, start=s) if b == "half_month"
                           else FeatureSet(kind="days", days=int(b), window=w, start=s))
    return out


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parser().parse_args(argv)
    cfg = config_mod.load_config(args.config)
    run = config_mod.run_dir(cfg, args.run)
    if args.step == "gcp-qc":
        from . import gcp_qc
        from .pixel_features import analysis_dir, run_context

        _, gd, *_ = run_context(run)
        out_dir = analysis_dir(run)
        gcp_qc.write_vector_qc(cfg, out_dir, gd["crs"], gd["res"], float((cfg.get("analysis") or {}).get("border_m", 10)))
        if args.write_copy:
            gcp_qc.write_qc_copy(cfg, out_dir / "gcp_review.csv", force=args.force)
    elif args.step == "gcp-eda":
        from . import gcp_eda
        gcp_eda.run(cfg, run)
    elif args.step == "features":
        from . import pixel_features
        pixel_features.extract(cfg, run, _sets(cfg, args), interior_only=args.interior_only, name=args.name)
    elif args.step == "evaluate":
        from . import evaluate_cv
        from .pixel_features import analysis_dir
        print(evaluate_cv.evaluate(cfg, analysis_dir(run) / f"{args.name}.parquet", args.sets, args.save_oof).to_string(index=False))
    elif args.step == "cv-layers":
        from . import maps
        maps.cv_layers(cfg, run, args.set)
    elif args.step == "class-map":
        from . import maps
        map_cfg = config_mod.load_config(args.map_config)
        maps.class_map(cfg, run, args.set, config_mod.run_dir(map_cfg, args.map_run), features_name=args.name)
    elif args.step == "model":
        from . import model
        map_run = config_mod.run_dir(config_mod.load_config(args.map_config), args.map_run) if args.map_config else None
        model.run(cfg, run, map_run, name=args.name)
    elif args.step == "final-models":
        from . import final_models
        final_models.save_final_models(cfg, run, name=args.name, without_bins=args.without_bin)
    elif args.step == "predict":
        from pathlib import Path

        from . import final_models
        map_run = config_mod.run_dir(config_mod.load_config(args.map_config), args.map_run)
        final_models.predict(cfg, run, map_run, models_dir=Path(args.models) if args.models else None, name=args.name)
    elif args.step == "field-labels":
        from pathlib import Path

        from . import field_labels
        train_cfg = config_mod.load_config(args.train_config) if args.train_config else cfg
        train_run = config_mod.run_dir(train_cfg, args.train_run) if args.train_config or args.train_run else run
        field_labels.run(cfg, run, Path(args.delineation), Path(args.class_raster),
                         prob_raster=Path(args.prob_raster) if args.prob_raster else None,
                         model_path=Path(args.model) if args.model else None,
                         train_cfg=train_cfg, train_run=train_run, name=args.name,
                         simplify_m=field_labels.SIMPLIFY_M if args.simplify_m is None else args.simplify_m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
