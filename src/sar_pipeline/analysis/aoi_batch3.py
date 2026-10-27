"""Third AOI set (27 Oct 2026): the client's late AOIs, one file per polygon, ready for the field delineation run.

Why: after the delivery the client sent 7 more AOI polygons (one zip) and a pilot site (another zip). The delineation
team runs on one GeoPackage per AOI in its own folder (``AOI_<N>/AOI_<N>.gpkg``), as for the second set, so these
continue the second set's numbering (:data:`FIRST_NUMBER` onward), the client's main zip first (files by name, polygons
in file order), the pilot site last. Each file keeps the client's columns as given and adds ``layer`` (source file name), ``path`` (zip
folder / file) and ``AOI_NUMBER``, like the second set's files. Delivered names will be ``b3_aoi<N>``.

Inputs: the zips unpacked under :data:`CLIENT_DIR` (``<zip name>/<folder>/<file>.shp``; a zip found again inside
another zip is read once) and the gitignored :data:`CONFIG` (``{"zips": [...in numbering order], "s3_prefix": ...}``: both
name the client, so they stay out of the repository). Outputs: ``<OUT_DIR>/AOI_<N>/AOI_<N>.gpkg`` + ``index.csv``;
``upload`` copies them to the S3 prefix with the AWS CLI (credentials from the environment) and checks each object's size.

Run::

    python -m sar_pipeline.analysis.aoi_batch3 prepare
    source secrets/aws_env.sh && python -m sar_pipeline.analysis.aoi_batch3 upload
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

CLIENT_DIR = Path("../data/client_aoi_files_2026-10-26/unzipped")
OUT_DIR = Path("../data/aoi_batch3_2026-10-27/split")
FIRST_NUMBER = 239
CONFIG = Path("config/aoi_batch3.json")                        # gitignored: zip names and bucket path name the client


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def shapefiles(root: Path = CLIENT_DIR, zips=None) -> list[Path]:
    """The client shapefiles in numbering order (``zips`` order, files by name), each (folder, file) once."""
    seen, out = set(), []
    for z in zips or _config()["zips"]:
        for p in sorted((root / z).glob("*/*.shp"), key=lambda p: p.name.lower()):
            if (p.parent.name, p.name) not in seen:
                seen.add((p.parent.name, p.name))
                out.append(p)
    return out


def prepare(root: Path = CLIENT_DIR, out: Path = OUT_DIR, first: int = FIRST_NUMBER, zips=None) -> pd.DataFrame:
    """One GeoPackage per polygon (``AOI_<N>/AOI_<N>.gpkg``, layer ``AOI_<N>``) and ``index.csv``."""
    import geopandas as gpd

    rows, n = [], first
    for p in shapefiles(root, zips):
        g = gpd.read_file(p)
        for i in range(len(g)):
            one = g.iloc[[i]].copy()
            one["layer"] = p.stem
            one["path"] = f"{p.parent.name}/{p.name}"
            one["AOI_NUMBER"] = n
            d = out / f"AOI_{n}"
            d.mkdir(parents=True, exist_ok=True)
            f = d / f"AOI_{n}.gpkg"
            one.to_file(f, driver="GPKG", layer=f"AOI_{n}")
            rows.append({"AOI_NUMBER": n, "delivered_name": f"b3_aoi{n}", "source_file": one["path"].iloc[0],
                         "row_in_file": i, "company": one.get("Company", pd.Series([None])).iloc[0],
                         "acres": round(float(one.to_crs(32646).area.iloc[0]) / 4046.8564224, 3)})
            n += 1
    idx = pd.DataFrame(rows)
    idx.to_csv(out / "index.csv", index=False)
    return idx


def upload(out: Path = OUT_DIR, prefix: str | None = None) -> dict:
    """``aws s3 cp`` of every prepared file to the prefix, then a size check of each object."""
    prefix = (prefix or _config()["s3_prefix"]).rstrip("/")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    subprocess.run(["aws", "s3", "cp", "--recursive", "--only-show-errors", str(out), prefix + "/"], check=True)
    ls = subprocess.run(["aws", "s3", "ls", "--recursive", prefix + "/"], check=True, capture_output=True, text=True)
    remote = {}
    for line in ls.stdout.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) == 4:
            remote[parts[3].split(prefix.split("/", 3)[-1] + "/", 1)[-1]] = int(parts[2])
    ok = [remote.get(str(p.relative_to(out))) == p.stat().st_size for p in files]
    return {"files": len(files), "size_ok": int(sum(ok)), "s3": prefix + "/"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["prepare", "upload"])
    a = p.parse_args(argv)
    if a.step == "prepare":
        print(prepare().to_string(index=False))
    else:
        print(json.dumps(upload(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
