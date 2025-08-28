# OpenStyleLightRoom (OSLR)

OpenStyleLightRoom (OSLR) is a local, open-source AI tool that learns a photographer’s personal editing style from Lightroom catalogs and applies it automatically to new catalogs. The core workflow is CLI-first: start with a Lightroom catalog (`.lrcat`) and produce a new catalog (`.lrcat`) with predicted edits applied.

Target parity: ImagenAI / Aftershoot style transfer — fully local, no SaaS, no artificial pricing.

Status: Alpha (v0.1.0). Ingest, train, and apply commands implemented end-to-end; use with backups and caution.


## Core workflow

1) **Ingest:** `osl ingest` reads a Lightroom catalog, finds RAWs and sidecar XMPs, and builds a self-contained training corpus with JPEG previews and before/after edit pairs.
2) **Train:** `osl train` trains a model on the corpus to predict global slider adjustments and toggles from image previews.
3) **Apply:** `osl apply` clones a catalog, runs the trained model on its images, and bundles the predicted XMP edits with a Lightroom plugin for easy import.


## Installation

Requirements:
- Python 3.10+
- macOS or Windows (Linux for training is supported; Lightroom integration requires macOS/Windows)
- Optional GPU for training (PyTorch with CUDA or Apple Silicon MPS)

Because of PEP 668, macOS Python installs are often externally-managed. Create a virtual environment and install OSLR editable.

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


## Usage guide

### Step 1: Ingest a Lightroom catalog

**Prerequisite:** In Lightroom, enable `Catalog Settings → Metadata → Automatically write changes into XMP`. This ensures your edits are saved as sidecar `.xmp` files next to your RAWs, which OSLR needs for ground truth.

**Command:**
```
osl ingest --catalog /path/to/your/catalog.lrcat --out ./corpus/my-style \
  --max-dim 1536 \
  --jpeg-quality 92
```

- `--catalog`: Path to your Lightroom Classic catalog (`.lrcat`).
- `--out`: Output directory for the training corpus.
- `--max-dim`: Longest edge for JPEG previews (default 1536).
- `--jpeg-quality`: Preview quality (default 92).
- `--limit`: Optional, process only the first N photos for a quick test.

**What to expect:**
- This is a long-running, CPU-heavy process that generates previews and XMPs. A progress bar will show ETA.
- A corpus folder is created with `previews/`, `xmp_before/`, `xmp_after/`, `manifest.csv`, and `ingest_report.json`.
- Check `ingest_report.json` for skipped files or errors. Common issues are missing RAWs or sidecar XMPs.

### Step 2: Train your style model

**Command:**
```
osl train --corpus ./corpus/my-style --ckpt ./ckpt/my-style.pt \
  --backbone mobilenet_v3_large \
  --img-size 512 \
  --batch-size 64 \
  --epochs 30
```

- `--corpus`: Path to the corpus folder from the ingest step.
- `--ckpt`: Path to save the final model checkpoint (`.pt`).
- `--backbone`: `mobilenet_v3_large` (faster, good baseline) or `convnext_tiny` (slower, potentially more accurate).
- `--img-size`: Training image resolution (default 512).
- `--batch-size`: Adjust based on GPU memory (e.g., 16-64).
- `--epochs`: Number of training passes (default 30).

**What to expect:**
- Training is GPU-accelerated if available (CUDA/MPS). A progress bar shows per-epoch mini-batch progress.
- The progress bar resets each epoch (e.g., `1/16` → `16/16` repeats).
- `ckpt/metrics.json` is updated each epoch with train/validation loss and accuracy. You can monitor this file to track progress.
- The best model checkpoint is saved to the `--ckpt` path based on validation performance.

### Step 3: Apply your model to a catalog

**Command:**
```
osl apply --catalog /path/to/target/catalog.lrcat --ckpt ./ckpt/my-style.pt \
  --out-catalog ./output_bundles/wedding-A/wedding-A.lrcat
```

- `--catalog`: The catalog you want to apply edits to (will be cloned, not modified).
- `--ckpt`: Your trained model checkpoint.
- `--out-catalog`: Path for the new catalog inside an output bundle.

**What to expect:**
- A new "bundle" directory is created (e.g., `./output_bundles/wedding-A/`).
- The bundle contains:
  - `wedding-A.lrcat`: A safe copy of your target catalog.
  - `xmp_pred/`: Predicted XMP files for each image.
  - `OSL.lrplugin/`: A minimal Lightroom plugin to import the predictions.
  - `README_FIRST.txt`: Instructions for the final import step.
  - `apply_report.json`: Log of predictions and any errors.

### Step 4: Import predictions into Lightroom

1) Open the new catalog (e.g., `wedding-A.lrcat`) in Lightroom Classic.
2) Go to `File → Plug-in Manager → Add`.
3) Select the `OSL.lrplugin` folder inside your output bundle.
4) Run `File → Plug-in Extras → OpenStyleLab: Import Predicted XMP`.
5) The plugin will copy predicted XMPs next to your RAWs and instruct Lightroom to read them. Original sidecars are backed up as `.bak`.


## Training data guidance

- **How many images?**
  - **1k–3k:** Good for initial tests and learning a rough style.
  - **5k–15k:** Solid baseline for a consistent global look.
  - **20k–50k:** Robust generalization. Diminishing returns for global sliders start after ~50k.
- **What matters more than count?**
  - **Label quality:** Ensure your edits are saved to XMP sidecars.
  - **Diversity:** Include a representative spread of cameras, lenses, lighting, and scenes.
  - **Balance:** Avoid huge clusters of near-duplicate images.


## Features (v1 scope)

- **Global sliders:** Exposure, Contrast, Highlights, Shadows, Whites, Blacks
- **Presence:** Clarity, Texture, Dehaze
- **Color:** Vibrance, Saturation
- **WB:** Temperature (log scale), Tint
- **HSL:** Hue/Sat/Lum per channel
- **Toggles:** EnableProfileCorrections, VignetteAmount
- **ToneCurve:** Passthrough only for now.


## Roadmap

- **v1:** Global sliders end-to-end, plugin-based catalog sync.
- **v1.1:** Virtual copies option, per-camera conditioning.
- **v2:** Region-aware masks (subject/sky) via lightweight segmentation.
- **v3:** Continual training on user corrections.


## Development

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for architecture details, testing strategy, and contribution guidelines.


## License

AGPL-3.0-only. See [LICENSE](LICENSE) for full text.


## Disclaimer

This project does not modify Adobe databases directly. Predicted edits are integrated via XMP and a Lightroom plugin using Lightroom’s public APIs. Users are advised to back up catalogs and originals. The project is not affiliated with Adobe.