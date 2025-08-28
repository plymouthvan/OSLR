from __future__ import annotations

import csv
import json
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from .lrcat import open_catalog_readonly, iter_photo_records, PhotoRecord, count_photos
from .xmp import DevelopSettings, denormalize, write_xmp

# Keep keys consistent with training
# If checkpoint contains explicit keys, we will use those; otherwise fall back to these defaults.
FALLBACK_REG_KEYS: List[str] = [
    # WB
    "Temperature",
    "Tint",
    # Basic
    "Exposure2012",
    "Contrast2012",
    "Highlights2012",
    "Shadows2012",
    "Whites2012",
    "Blacks2012",
    # Presence
    "Clarity2012",
    "Texture",
    "Dehaze",
    # Color
    "Vibrance",
    "Saturation",
    # HSL
    "HueAdjustmentRed",
    "HueAdjustmentOrange",
    "HueAdjustmentYellow",
    "HueAdjustmentGreen",
    "HueAdjustmentAqua",
    "HueAdjustmentBlue",
    "HueAdjustmentPurple",
    "HueAdjustmentMagenta",
    "SaturationAdjustmentRed",
    "SaturationAdjustmentOrange",
    "SaturationAdjustmentYellow",
    "SaturationAdjustmentGreen",
    "SaturationAdjustmentAqua",
    "SaturationAdjustmentBlue",
    "SaturationAdjustmentPurple",
    "SaturationAdjustmentMagenta",
    "LuminanceAdjustmentRed",
    "LuminanceAdjustmentOrange",
    "LuminanceAdjustmentYellow",
    "LuminanceAdjustmentGreen",
    "LuminanceAdjustmentAqua",
    "LuminanceAdjustmentBlue",
    "LuminanceAdjustmentPurple",
    "LuminanceAdjustmentMagenta",
    # Vignette
    "VignetteAmount",
]
FALLBACK_CLS_KEYS: List[str] = ["EnableProfileCorrections"]

# Default normalization (ImageNet)
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

RAW_EXTENSIONS = {
    ".3fr", ".ari", ".arw", ".bay", ".crw", ".cr2", ".cr3", ".cap", ".data", ".dcs", ".dcr",
    ".dng", ".drf", ".eip", ".erf", ".fff", ".gpr", ".iiq", ".k25", ".kdc", ".mdc", ".mef",
    ".mos", ".mrw", ".nef", ".nrw", ".obm", ".orf", ".pef", ".ptx", ".pxn", ".r3d", ".raf",
    ".raw", ".rw2", ".rwl", ".rwz", ".sr2", ".srf", ".srw", ".x3f",
}


@dataclass
class ApplyConfig:
    catalog_path: Path
    ckpt_path: Path
    dst_catalog: Path  # destination .lrcat path inside bundle
    img_size: int = 512
    batch_size: int = 32
    device: Optional[str] = None  # "cuda" | "cpu" | None->auto


def _pil_from_raw_or_file(raw_path: Path) -> Image.Image:
    # Load RAW with rawpy if possible; fallback to PIL
    try:
        if raw_path.suffix.lower() in RAW_EXTENSIONS:
            try:
                import rawpy  # type: ignore

                with rawpy.imread(str(raw_path)) as raw:
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        no_auto_bright=True,
                        output_bps=8,
                        gamma=(2.2, 4.5),
                        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                    )
                return Image.fromarray(rgb)
            except Exception:
                # Fallback (may fail for proprietary RAWs)
                return Image.open(raw_path)
        else:
            return Image.open(raw_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load preview for {raw_path}: {e}")


def _to_tensor_normalized(img: Image.Image, size: int, mean: List[float], std: List[float]) -> torch.Tensor:
    # Resize to square for backbone
    img = img.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0  # H,W,3
    arr = np.transpose(arr, (2, 0, 1))  # 3,H,W
    for c in range(3):
        arr[c] = (arr[c] - mean[c]) / std[c]
    return torch.from_numpy(arr)  # 3,H,W


class HeadedRegClsModel(torch.nn.Module):
    """Backbone + heads. Should match the architecture used in training."""
    def __init__(self, backbone: str, n_reg: int, n_cls: int, hidden: int = 512):
        super().__init__()
        from torchvision import models  # Lazy import to avoid heavy deps when unused

        self.backbone_name = backbone.lower()
        if self.backbone_name in ("mobilenet_v3_large", "mobilenetv3"):
            m = models.mobilenet_v3_large(weights=None)
            feat_dim = m.features[-1].out_channels
            self.features = m.features
        elif self.backbone_name in ("convnext_tiny", "convnext-t", "convnext_t"):
            m = models.convnext_tiny(weights=None)
            feat_dim = m.features[-1][-1].out_channels
            self.features = m.features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.pool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(1)
        )
        self.neck = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, hidden, bias=False),
            torch.nn.BatchNorm1d(hidden),
            torch.nn.SiLU(inplace=True),
        )
        self.head_reg = torch.nn.Linear(hidden, n_reg) if n_reg > 0 else None
        self.head_cls = torch.nn.Linear(hidden, n_cls) if n_cls > 0 else None

    def forward(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        feats = self.features(x)
        feats = self.pool(feats)
        feats = self.neck(feats)
        y_reg = self.head_reg(feats) if self.head_reg is not None else None
        y_cls = self.head_cls(feats) if self.head_cls is not None else None
        return y_reg, y_cls


def _load_checkpoint(ckpt_path: Path, device: torch.device):
    payload = torch.load(str(ckpt_path), map_location=device)
    cfg = payload.get("config", {})
    reg_keys = cfg.get("reg_keys", FALLBACK_REG_KEYS)
    cls_keys = cfg.get("cls_keys", FALLBACK_CLS_KEYS)
    mean = cfg.get("mean", DEFAULT_MEAN)
    std = cfg.get("std", DEFAULT_STD)
    backbone = cfg.get("backbone", "mobilenet_v3_large")
    n_reg = cfg.get("n_reg", len(reg_keys))
    n_cls = cfg.get("n_cls", len(cls_keys))

    model = HeadedRegClsModel(backbone=backbone, n_reg=n_reg, n_cls=n_cls)
    model.load_state_dict(payload["state_dict"], strict=False)

    return model, reg_keys, cls_keys, mean, std, backbone


def _write_readme(bundle_dir: Path) -> None:
    (bundle_dir / "README_FIRST.txt").write_text(
        """OpenStyleLab (OSL) — Predicted XMP Bundle

Steps:
1) Copy the XMP files from the 'xmp_pred' directory into the same folder as your original RAW files.
2) Open your original catalog in Lightroom Classic.
3) Select the photos you want to update.
4) In the menu, go to Metadata → Read Metadata from Files.

Notes:
- This process will overwrite any existing edits in your Lightroom catalog for the selected photos.
- Original sidecar XMPs are backed up in the 'originals' subfolder inside 'xmp_pred'.
- No Adobe database schema is modified.
""",
        encoding="utf-8",
    )


def run(
    catalog_path: Path,
    ckpt_path: Path,
    dst_catalog: Path,
    img_size: int = 512,
    batch_size: int = 32,
) -> None:
    """Apply trained model to a catalog and produce a bundle with predicted XMP."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle_dir = dst_catalog.parent.resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Clone catalog into bundle
    shutil.copy2(catalog_path, dst_catalog)

    # Prepare output dirs
    xmp_pred_dir = bundle_dir / "xmp_pred"
    bak_dir = xmp_pred_dir / "originals"
    xmp_pred_dir.mkdir(parents=True, exist_ok=True)
    bak_dir.mkdir(parents=True, exist_ok=True)

    # Load model and config
    model, reg_keys, cls_keys, mean, std, backbone = _load_checkpoint(ckpt_path, device)
    model.eval().to(device)

    # Iterate catalog
    rows_manifest: List[Tuple[str, str]] = []
    report: Dict[str, object] = {
        "catalog": str(catalog_path),
        "bundle_dir": str(bundle_dir),
        "dst_catalog": str(dst_catalog),
        "ckpt": str(ckpt_path),
        "img_size": img_size,
        "batch_size": batch_size,
        "device": str(device),
        "processed": 0,
        "predicted": 0,
        "skipped": 0,
        "errors": [],
        "backed_up_sidecars": 0,
    }

    def flush_batch(batch_imgs: List[torch.Tensor], batch_records: List[PhotoRecord], xmp_pred_dir: Path) -> None:
        nonlocal rows_manifest, report
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            y_reg, y_cls = model(x)
        # Convert tensors to CPU numpy for processing
        reg_np = y_reg.detach().cpu().numpy() if y_reg is not None else None
        cls_np = y_cls.detach().cpu().numpy() if y_cls is not None else None

        for i, rec in enumerate(batch_records):
            try:
                # Build normalized dict for regression
                reg_norm_map: Dict[str, float] = {}
                if reg_np is not None:
                    for k_idx, key in enumerate(reg_keys):
                        v = float(reg_np[i, k_idx])
                        # clamp to [-1,1] to be safe
                        reg_norm_map[key] = max(-1.0, min(1.0, v))
                # Denormalize to LR scale
                ds = denormalize(reg_norm_map)

                # Toggles from logits
                if cls_np is not None:
                    for k_idx, key in enumerate(cls_keys):
                        logit = float(cls_np[i, k_idx])
                        ds.set(key, 1 if logit >= 0.0 else 0)

                ds.clamp_inplace()

                # Debugging: print the denormalized values
                print(f"ID: {rec.id}, Preds: {ds.values}")

                # Write predicted XMP (preserve unrelated metadata if an existing sidecar exists)
                pred_xmp = xmp_pred_dir / rec.raw_path.with_suffix(".xmp").name
                sidecar = rec.raw_path.with_suffix(".xmp")
                if sidecar.exists():
                    # backup original
                    shutil.copy2(sidecar, bak_dir / pred_xmp.name)
                    report["backed_up_sidecars"] = int(report["backed_up_sidecars"]) + 1  # type: ignore
                write_xmp(pred_xmp, ds, base=sidecar if sidecar.exists() else None, preserve_unrelated=True)

                # Update manifest mapping
                rows_manifest.append((str(rec.raw_path), pred_xmp.name))
                report["predicted"] = int(report["predicted"]) + 1  # type: ignore
            except Exception as e:
                report["errors"].append(  # type: ignore
                    {"id": rec.id, "raw_path": str(rec.raw_path), "error": str(e), "traceback": traceback.format_exc()}
                )
                report["skipped"] = int(report["skipped"]) + 1  # type: ignore

    # Open catalog and stream over photos
    conn = open_catalog_readonly(catalog_path)
    batch_imgs: List[torch.Tensor] = []
    batch_records: List[PhotoRecord] = []
    try:
        total_photos = count_photos(conn)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Applying edits...", total=total_photos)
            for rec in iter_photo_records(conn, limit=None):
                report["processed"] = int(report["processed"]) + 1  # type: ignore
                try:
                    img = _pil_from_raw_or_file(rec.raw_path)
                    t = _to_tensor_normalized(img, size=img_size, mean=mean, std=std)
                    batch_imgs.append(t)
                    batch_records.append(rec)
                    if len(batch_imgs) >= batch_size:
                        flush_batch(batch_imgs, batch_records, xmp_pred_dir)
                        batch_imgs.clear()
                        batch_records.clear()
                except Exception as e:
                    report["errors"].append(  # type: ignore
                        {"id": rec.id, "raw_path": str(rec.raw_path), "error": str(e), "traceback": traceback.format_exc()}
                    )
                    report["skipped"] = int(report["skipped"]) + 1  # type: ignore
                finally:
                    progress.update(task, advance=1)
            # flush remainder
            flush_batch(batch_imgs, batch_records, xmp_pred_dir)
    finally:
        conn.close()

    # Write plugin and README
    _write_readme(bundle_dir)

    # Write apply report
    (bundle_dir / "apply_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")