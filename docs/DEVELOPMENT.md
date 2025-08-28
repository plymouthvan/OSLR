# OpenStyleLab Development Guide

This document explains the codebase layout, local development setup, testing strategy, and contribution guidelines.


## Repository structure

- [src/osl/cli.py](../src/osl/cli.py) — CLI entry points (Click group with `ingest`, `train`, `apply`)
- [src/osl/lrcat.py](../src/osl/lrcat.py) — Lightroom catalog (SQLite) readers (read-only)
- [src/osl/xmp.py](../src/osl/xmp.py) — XMP parsing and write helpers, slider normalization
- [src/osl/ingest.py](../src/osl/ingest.py) — Corpus builder (previews, before/after XMPs, manifest, report)
- [src/osl/train.py](../src/osl/train.py) — PyTorch dataset/model/training loop, checkpoint + metrics
- [src/osl/apply.py](../src/osl/apply.py) — Inference pipeline and Lightroom plugin bundle
- [README.md](../README.md) — User-facing overview
- [docs/WORKFLOWS.md](./WORKFLOWS.md) — End-to-end workflows and troubleshooting


## Environment setup

Use a virtual environment (PEP 668-safe) and install OSL in editable mode.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Training extras (adds TQDM, TorchVision dep pins, etc.; install PyTorch separately for your platform):
```
pip install -e ".[train]"
```

Install PyTorch from https://pytorch.org/get-started/locally/ according to your OS and GPU/MPS availability.


## Running the CLI locally

- Global help:
  ```
  osl --help
  ```

- Ingest:
  ```
  osl ingest --catalog /path/to/src.lrcat --out ./corpus --limit 1000
  ```

- Train (ensure PyTorch is installed):
  ```
  osl train --corpus ./corpus --ckpt ./ckpt/best.pt --backbone mobilenet_v3_large
  ```

- Apply:
  ```
  osl apply --catalog /path/to/src.lrcat --ckpt ./ckpt/best.pt --out-catalog ./dst_bundle/dst.lrcat
  ```


## Data contracts

Manifest columns (written by ingest):
```
id, raw_path, preview_path, before_xmp, after_xmp, camera, lens, iso, shutter, aperture, session_id
```

- `preview_path`, `before_xmp`, `after_xmp` are stored relative to the corpus root for portability.
- `raw_path` is absolute (follows Lightroom root folder + relative path join).

Normalization rules (XMP develop sliders):
- Bounded sliders (e.g., Contrast2012, Vibrance) map linearly to [-1, 1]
- Temperature (Kelvin) maps via log scale in [2000, 50000] → [-1, 1]
- Binary toggles (EnableProfileCorrections) use {-1, +1} internally for training, clamped to {0,1} on write


## Architectural notes

- Catalog access is read-only via SQLite URI `?mode=ro` to avoid accidental writes.
- Ingest pulls AFTER edits from sidecar XMPs adjacent to RAWs when present. If users keep edits only in the catalog, we recommend enabling “Automatically write changes into XMP” before ingest to ensure AFTER ground truth exists as sidecars.
- Apply writes predicted XMPs into a standalone bundle (`xmp_pred/`) and provides a Lightroom plugin (`OSL.lrplugin`) that imports them via `photo:readMetadataFromFile()` to avoid touching Adobe DB schema.


## Code style and tooling

- Python 3.10+; Black + isort config in pyproject
- Keep functions small and pure where possible (facilitates testing)
- Prefer defensive SQL queries; Lightroom schemas vary across LR versions

Recommendations:
- Type hints throughout (checked by your editor or `pyright` optionally)
- Use `rich` for CLI messages (already integrated)
- Where possible lazy-import heavy deps inside commands (e.g., torch/torchvision)


## Testing

Planned structure:
- Unit tests for:
  - XMP normalization + clamping (+ round-trip for representative values)
  - XMP read/write idempotence for known sample snippets
  - Catalog path composition and exposure time normalization
- Integration tests:
  - Ingest against a tiny fixture catalog (dozens of rows) with synthetic folder structure
  - Apply pipeline on a small set of RAW/JPEGs to ensure XMPs are produced and plugin manifest is valid

Proposed layout:
```
tests/
  test_xmp.py
  test_lrcat.py
  test_ingest.py
  test_apply.py
```

Notes:
- Avoid shipping large binaries to the repo; use tiny RAW samples or generated JPEGs for unit tests.
- If including RAW test files, ensure licensing/attribution is compatible.


## Contributing

- Fork and create feature branches; open PRs against `main`
- Write clear commit messages and PR descriptions (what/why)
- Include tests when adding functionality; keep coverage reasonable for critical logic
- Keep public APIs (CLI flags, manifest schema) backward compatible when possible


## Releasing (future)

- Tag releases (e.g., `v1.0.0`) and attach changelog
- Consider publishing to PyPI when interfaces stabilize
- Provide sample corpus and fixture catalogs for reproducible benchmarks


## Known limitations

- AFTER edits are sourced from sidecar XMPs; if the catalog contains edits but sidecars are missing, ingest will fall back to neutral AFTER XMP. Users should enable LR’s “Automatically write changes into XMP” before ingest for best results.
- Tone curve support is pass-through; advanced parametric/point-curves will be added incrementally.
- Region-aware masks are out-of-scope for v1 (see roadmap).
## Apple Silicon (MPS) development tips

These defaults are implemented in the training pipeline, but you can verify and tune locally for best performance on Apple Silicon.

- Verify MPS build/availability:
  ```python
  import torch
  print("MPS is_built:", torch.backends.mps.is_built())
  print("MPS is_available:", torch.backends.mps.is_available())
  ```
- Device and mixed precision:
  - The trainer prefers the "mps" device when both built and available, otherwise falls back to CUDA/CPU.
  - Training and validation wrap forward/backward in autocast for MPS/CUDA using FP16 compute, while keeping master weights in FP32.
- Explicit CPU fallbacks:
  - The trainer sets the environment variable PYTORCH_ENABLE_MPS_FALLBACK=1 when using MPS so unsupported operations fall back explicitly (visible in logs).
  - If you see frequent fallbacks in logs, replace the offending ops with MPS‑supported equivalents (e.g., prefer supported interpolation modes).
- DataLoader tuning on MPS:
  - Use more workers (recommend 8–12).
  - Enable persistent_workers=True and prefetch_factor=4 (when num_workers > 0).
  - Do not use pinned memory on MPS (pin_memory=False). This is configured automatically by the trainer.
- Image size while prototyping:
  - Use smaller images like --img-size 256–384 to speed up iterations by 2–3×.
- Faster JPEG decoding (optional):
  - Install a libjpeg‑turbo backed Pillow for faster JPEG decode: `pip install pillow-jpegturbo`.

Note: The CLI exposes --img-size, --batch-size, and --workers (default workers is 8). Adjust based on your hardware and dataset.