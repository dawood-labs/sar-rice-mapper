"""Ground-truth analysis on top of a finished run: field QC, EDA, pixel features, spatial CV, maps.

Run it with `python -m sar_pipeline.analysis <step> --config <yaml> --run <run_id>`; see docs/08_analysis.md.
Nothing here starts Earth Engine exports: every step reads local run outputs only.
"""
