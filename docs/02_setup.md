# 02 — Setup

This page takes you from an empty machine to a working pipeline, both on a **laptop/WSL** and on a
**JupyterHub** server (e.g. on AWS).

---

## 1. What you need

| Item | Why |
|---|---|
| Python ≥ 3.10 | the package uses modern type hints |
| A Google Cloud **service account key** (JSON) | authenticates Earth Engine and Cloud Storage without a browser login |
| The service account **registered for Earth Engine** on a Cloud project | otherwise every EE call fails with "not registered" |
| **Write access** for that service account on the export bucket | Earth Engine writes exported GeoTIFFs there; the pipeline lists and downloads them |
| An AOI polygon file (SHP or GPKG) | defines the area |
| Free disk space | the `download` step checks this before starting |

Ask the project owner for the key file and the bucket/project names. **Never** send keys through chat
or commit them.

---

## 2. Install

```bash
git clone git@github.com:<org>/sar-rice-mapper.git
cd sar-rice-mapper

python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -e ".[dev]"            # installs the package in editable mode + pytest
```

Everything installs from pip. `rasterio` wheels bundle GDAL, and the pipeline writes VRT files itself, so
**no system GDAL or `osgeo` Python bindings are needed**. `psutil` is installed as a dependency so memory
detection also works outside Linux.

---

## 3. Local folders (all gitignored)

```
secrets/     service-account key(s) and private_terms.txt
data/aoi/    AOI polygons
data/gcps/   ground-truth polygons, if any ever exist (gcps.path in the config)
processed/   everything the pipeline writes
logs/        log files of long-running commands
```

```bash
mkdir -p secrets data/aoi logs
cp /path/to/service-account.json secrets/
chmod 600 secrets/*.json
```

### `secrets/private_terms.txt`

A plain text file, one word per line, listing words that must never appear in tracked files (client
name, place names of the project area, and similar). `tests/test_repo_hygiene.py` reads it and fails if
any file git would commit contains one of those words. It also checks your local config values (bucket,
project id, AOI file name) and the key's service-account email automatically.

- Keep it in `secrets/`, which is gitignored, so the list itself is never published.
- Add a word whenever you notice a new sensitive term, then run `pytest tests/test_repo_hygiene.py`.
- Words shorter than 4 characters are ignored (too many false alarms).

---

## 4. Config

```bash
cp config/pipeline.example.yaml config/my_aoi_season.yaml
```

The config **must live in `config/`**: every path inside it (AOI, key file) is relative to the project root,
which is the parent of `config/`. A config stored elsewhere must set `project_root:` (relative to the config
file's folder); otherwise loading fails instead of silently resolving paths against the wrong folder.

Fill every `<placeholder>`:

- `aoi.key`, `aoi.path`: short folder-safe name and path to the polygon file
- `season`: key, `timezone` (IANA name of the AOI's time zone; only used for a local-date column), start (inclusive), end (exclusive)
- `auth.key_file`, `auth.project`: key path under `secrets/`, Earth Engine Cloud project id
- `gcs.bucket`, `gcs.base_folder`: export bucket and the project folder inside it
- `grid.crs`: one projected CRS for the whole AOI (see [07 Scaling](07_scaling_to_large_aois.md#single-crs))

Leave `s1.tracks: []` for now; you fill it after the audit.

Your config file is gitignored because it contains bucket/project names.

---

## 5. Verify

```bash
CFG=config/my_aoi_season.yaml

# Machine resources (CPU/RAM/disk) as the pipeline sees them
python -m sar_pipeline --config $CFG resources

# Read-only pre-flight checks: is the AOI file sound, and does Sentinel-1 cover it?
# They write nothing and start nothing. See docs/04_runbook.md step 0.
python -m sar_pipeline.prep aoi-qc          --config $CFG
python -m sar_pipeline.prep s1-availability --config $CFG

# Unit tests (no network) — includes the public-repo hygiene test
pytest

# Live Earth Engine tests (read-only, never start exports). They read the config named by
# SAR_PIPELINE_CONFIG (default: config/live.yaml) and skip if it or the key is missing.
SAR_PIPELINE_CONFIG=$CFG pytest -m gee
```

If `resources` shows sensible numbers and both test runs pass, you are ready for the
[runbook](04_runbook.md).

---

## 6. JupyterHub (e.g. on AWS)

JupyterHub runs your notebook server **inside a container**. Things that differ from a laptop:

1. **Resources are limited by the container, not the host.** `os.cpu_count()` may report 96 cores while
   your container may use only 8. `sar_pipeline.resources` reads the container (cgroup) limits:
   - it finds the process's **own cgroup** (spawners often nest users, e.g. `/kubepods/pod…/container`) and
     also checks every **parent** cgroup, taking the tightest limit;
   - it supports cgroup v1 and v2;
   - it does not count the kernel's reclaimable **page cache** as used memory, so right after a large
     download the available memory is not wrongly reported as ~0.

   All thread and worker counts adapt automatically. Check with the `resources` command. You can cap usage
   in the config (`resources.cpus`, `resources.memory_fraction`), but you never need to hard-code numbers.
2. **Home directories are often small.** Put the repository (and so `processed/`) on the large data
   volume (e.g. an attached EBS volume) and check free space with `resources`.
3. **Uploading the key:** upload the JSON through the JupyterHub file browser directly into `secrets/`,
   then in a terminal run `chmod 600 secrets/*.json`. Do not paste the key into a notebook cell,
   because notebook files keep cell contents.
4. **Continuing a run from another machine** is supported: `auth.key_file` and `resources` are taken from the
   config of the machine you are on. `auth.project` must stay the same as when the run was exported (tasks and
   queue counts live in that project; see [runbook step 3](04_runbook.md#step-3--create-a-run--notebook-02)).

Register the virtual environment as a notebook kernel:

```bash
source .venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name sar-rice-mapper
```

Then choose the `sar-rice-mapper` kernel in the notebooks.

---

## 7. Updating

```bash
git pull
pip install -e ".[dev]"   # only needed when dependencies changed
pytest
```
