# OpenStyleLab Workflows

This guide walks through end-to-end usage of OpenStyleLab (OSL): ingesting a Lightroom catalog, training a model, and applying predicted edits back to a new catalog bundle.

OSL operates locally and never modifies Adobe databases directly. Predicted edits are written to XMP sidecars and integrated via a Lightroom plugin.


## 1) System requirements

- OS: macOS or Windows (Linux supported for training; Lightroom integration requires macOS/Windows)
- Python: 3.10+
- Optional GPU: NVIDIA CUDA (Windows/Linux) or Apple Silicon (MPS) for faster training
- Disk space: depends on catalog size (previews and XMP corpus can be several GB)


## 2) Installation

Because of PEP 668, macOS Python installs are often externally-managed. Create a virtual environment and install OSL editable.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Training extras (does not install PyTorch; see below):
```
pip install -e ".[train]"
```

Install PyTorch separately for your platform:
- Apple Silicon (CPU/MPS):
  - https://pytorch.org/get-started/locally/
- NVIDIA CUDA (Windows/Linux):
  - https://pytorch.org/get-started/locally/

Verify CLI:
```
osl --help
```

If you prefer module invocation:
```
python -m osl.cli --help
```


## 3) Ingest a Lightroom catalog

Command:
```
osl ingest --catalog /path/to/src.lrcat --out ./corpus \
  --max-dim 1536 \
  --jpeg-quality 92 \
  --limit 2000
```

- --catalog: Path to Lightroom Classic catalog (.lrcat)
- --out: Output corpus directory (will be created if missing)
- --max-dim: Longest edge size for previews (default 1536)
- --jpeg-quality: JPEG quality for previews (default 92)
- --limit: Optional limit for debugging (first N photos)

What ingest does:
- Reads the catalog read-only (SQLite) and iterates photos
- Generates a neutral BEFORE XMP for each image (Adobe-like neutral baseline; sliders zeroed)
- Locates AFTER XMP from sidecar next to the RAW if present; else writes a neutral placeholder
- Creates scaled JPEG previews (tries rawpy/libraw for RAWs; falls back to Pillow when possible)
- Writes a manifest.csv and ingest_report.json

Example corpus structure:
```
corpus/
  previews/
    123.jpg
    124.jpg
    ...
  xmp_before/
    123.xmp
    124.xmp
    ...
  xmp_after/
    123.xmp
    124.xmp
    ...
  manifest.csv
  ingest_report.json
```

Manifest columns:
```
id, raw_path, preview_path, before_xmp, after_xmp, camera, lens, iso, shutter, aperture, session_id
```

Notes:
- AFTER XMP is read from the sidecar if available (we do not read Adobe DB develop settings directly)
- Missing files or corrupt RAWs are skipped and logged into ingest_report.json


## 4) Train a model

Command:
```
osl train --corpus ./corpus --ckpt ./ckpt/best.pt \
  --backbone mobilenet_v3_large \
  --img-size 512 \
  --batch-size 64 \
  --epochs 30 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --workers 4 \
  --val-frac 0.1 \
  --seed 1337 \
  --huber-delta 1.0 \
  --warmup-epochs 2
```

Outputs:
- Checkpoint: ./ckpt/best.pt
- Metrics: ./ckpt/metrics.json (rolling per-epoch logs; best metrics included)

Model:
- Backbone: MobileNetV3-Large or ConvNeXt-Tiny (select with --backbone)
- Heads:
  - Regression for global sliders (WB, Basic, Presence, Color, HSL, VignetteAmount)
  - Classification for toggles (EnableProfileCorrections)
- Loss:
  - Huber for regression (per-dim weights planned)
  - BCEWithLogits for binary toggles
- Scheduler:
  - Cosine decay with warmup
- Normalization:
  - Bounded sliders mapped to [-1, 1]
  - Temperature mapped on log scale 2000–50000K
  - Toggle targets in {-1, +1} (converted internally to {0,1} for BCE)

Tips:
- Start with --backbone mobilenet_v3_large for speed and stability
- Increase --img-size and --batch-size if you have GPU memory
- Consider --val-frac 0.2 for larger corpora
- Apple Silicon: set `PYTORCH_ENABLE_MPS_FALLBACK=1` to allow CPU fallback where MPS ops are missing


## 5) Apply predictions to a new catalog bundle

Command:
```
osl apply --catalog /path/to/src.lrcat --ckpt ./ckpt/best.pt --out-catalog ./dst_catalog_bundle/dst.lrcat \
  --img-size 512 \
  --batch-size 32
```

Outputs a bundle directory:
```
dst_catalog_bundle/
  dst.lrcat
  xmp_pred/
    123.xmp
    124.xmp
    apply_manifest.csv
    originals/
      123.xmp   # backups if originals existed
      124.xmp
  OSL.lrplugin/
    Info.lua
    ImportPredictedXMP.lua
  README_FIRST.txt
  apply_report.json
```

How to import predictions:
1) Open dst_catalog_bundle/dst.lrcat in Lightroom Classic
2) Plug-in Manager → Add → select OSL.lrplugin (in the bundle)
3) Run Plug-in Extras → OpenStyleLab: Import Predicted XMP
   - The plugin matches photos by original absolute RAW path and copies predicted XMP sidecars next to them
   - If an original sidecar exists, it is backed up as .bak
   - Then the plugin invokes photo:readMetadataFromFile() for each

Notes:
- The plugin never modifies Adobe’s database schema
- If you want to preserve originals as-is, a v1.1 “virtual copies” mode is planned


## 6) Troubleshooting

PEP 668 “externally-managed environment” on macOS:
- Always use a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```

rawpy/libraw issues:
- Prebuilt wheels are included for common platforms
- For macOS, ensure command line tools are installed
- If needed, install libraw: `brew install libraw`

PyTorch install:
- Install via https://pytorch.org/get-started/locally/ for your OS/GPU
- Apple Silicon:
  - MPS backend can accelerate training; some ops may fall back to CPU

Memory errors during training:
- Reduce --batch-size
- Use --backbone mobilenet_v3_large
- Reduce --img-size to 448 or 384


## 7) Safety, privacy, and backups

- OSL writes only XMP sidecars and never edits Lightroom DB schema
- OSL backs up existing sidecars as .bak on write
- Keep backups of your catalogs and originals
- Everything runs locally; no data is uploaded


## 8) References

- CLI entry point: src/osl/cli.py
- Catalog readers: src/osl/lrcat.py
- XMP utilities: src/osl/xmp.py
- Ingest pipeline: src/osl/ingest.py
- Training pipeline: src/osl/train.py
- Apply pipeline and plugin bundling: src/osl/apply.py