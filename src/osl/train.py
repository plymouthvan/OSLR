from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from .xmp import (
    DevelopSettings,
    build_neutral_settings,
    normalize,
    read_xmp,
)

# ====== Targets and keys ======
# Regression keys: all continuous sliders we train on
REG_KEYS: List[str] = [
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
    # HSL (Hue/Sat/Lum per channel)
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

# Classification (binary toggle) keys
CLS_KEYS: List[str] = [
    "EnableProfileCorrections",
]


# ====== Dataset ======
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class CorpusItem:
    id: int
    preview_path: Path
    after_xmp: Path


class CorpusDataset(Dataset):
    def __init__(self, corpus_dir: Path, items: List[CorpusItem], img_size: int = 512) -> None:
        self.corpus_dir = corpus_dir
        self.items = items
        self.img_size = img_size
        self.tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        # Image
        img = Image.open(it.preview_path).convert("RGB")
        x = self.tf(img)

        # Targets from AFTER XMP, fallback to neutral for missing keys
        ds_after = read_xmp(it.after_xmp)
        ds_neutral = build_neutral_settings()
        merged_vals: Dict[str, float] = {}
        for k in REG_KEYS:
            v = ds_after.get(k, ds_neutral.get(k))
            merged_vals[k] = float(v if v is not None else ds_neutral.get(k, 0.0))
        # Toggles as ints
        for k in CLS_KEYS:
            v = ds_after.get(k, ds_neutral.get(k))
            merged_vals[k] = int(1 if (v is not None and int(v) != 0) else 0)

        # Normalize to model space [-1, 1] for regression; toggles → {-1, +1}
        ds = DevelopSettings(merged_vals.copy())
        norm = normalize(ds)
        y_reg = np.array([norm[k] for k in REG_KEYS], dtype=np.float32)
        y_cls = np.array([1.0 if merged_vals[k] > 0 else -1.0 for k in CLS_KEYS], dtype=np.float32)

        return {
            "image": x,  # Tensor [3, H, W]
            "y_reg": torch.from_numpy(y_reg),  # [n_reg]
            "y_cls": torch.from_numpy(y_cls),  # [n_cls] in {-1,+1}
            "id": it.id,
        }


def load_manifest(corpus_dir: Path) -> List[CorpusItem]:
    path = corpus_dir / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"manifest.csv not found in {corpus_dir}")

    items: List[CorpusItem] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                _id = int(row["id"])
                preview = corpus_dir / row["preview_path"]
                after_xmp = corpus_dir / row["after_xmp"]
                if not preview.exists() or not after_xmp.exists():
                    continue
                items.append(CorpusItem(id=_id, preview_path=preview, after_xmp=after_xmp))
            except Exception:
                continue
    if not items:
        raise RuntimeError("No valid rows found in manifest.csv")
    return items


# ====== Model ======
class HeadedRegClsModel(nn.Module):
    def __init__(self, backbone: str, img_size: int, n_reg: int, n_cls: int, hidden: int = 512):
        super().__init__()
        self.backbone_name = backbone
        self.img_size = img_size

        feat_dim, feature_extractor = self._build_backbone(backbone)
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        self.neck = nn.Sequential(
            nn.Linear(feat_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.SiLU(inplace=True),
        )
        self.head_reg = nn.Linear(hidden, n_reg) if n_reg > 0 else None
        self.head_cls = nn.Linear(hidden, n_cls) if n_cls > 0 else None

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.feature_extractor = feature_extractor

    def _build_backbone(self, name: str) -> Tuple[int, nn.Module]:
        name = name.lower()
        if name in ("mobilenet_v3_large", "mobilenetv3"):
            m = models.mobilenet_v3_large(weights=None)
            feat_dim = m.features[-1].out_channels  # 960
            feature_extractor = m.features
            return feat_dim, feature_extractor
        elif name in ("convnext_tiny", "convnext-t", "convnext_t"):
            m = models.convnext_tiny(weights=None)
            feat_dim = m.features[-1][-1].out_channels  # final stage channels
            feature_extractor = m.features
            return feat_dim, feature_extractor
        else:
            raise ValueError(f"Unsupported backbone: {name}")

    def forward(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        feats = self.feature_extractor(x)  # [B, C, H', W']
        feats = self.pool(feats)
        feats = self.neck(feats)
        y_reg = self.head_reg(feats) if self.head_reg is not None else None
        y_cls = self.head_cls(feats) if self.head_cls is not None else None
        return y_reg, y_cls


# ====== Training ======
@dataclass
class TrainConfig:
    corpus_dir: Path
    ckpt_path: Path
    backbone: str = "mobilenet_v3_large"
    img_size: int = 512
    batch_size: int = 64
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    workers: int = 4
    val_frac: float = 0.1
    seed: int = 1337
    huber_delta: float = 1.0
    reg_weights: Optional[List[float]] = None  # len == n_reg, defaults to 1.0
    warmup_epochs: int = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_items(items: List[CorpusItem], val_frac: float, seed: int) -> Tuple[List[CorpusItem], List[CorpusItem]]:
    rng = random.Random(seed)
    items_copy = list(items)
    rng.shuffle(items_copy)
    n_val = max(1, int(len(items_copy) * val_frac)) if len(items_copy) > 10 else max(0, int(len(items_copy) * 0.2))
    val = items_copy[:n_val]
    train = items_copy[n_val:]
    if not train:
        # Ensure we always have a training set
        train = val
        val = []
    return train, val


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def bce_logits_accuracy(logits: torch.Tensor, target_pm1: torch.Tensor) -> torch.Tensor:
    # target in {-1, +1}; logits -> probabilities via sigmoid; threshold at 0.0
    pred = (logits >= 0.0).float() * 2.0 - 1.0
    return (pred == target_pm1).float().mean()


def train_one_epoch(
    model: HeadedRegClsModel,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    sched: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    cfg: TrainConfig,
) -> Dict[str, float]:
    model.train()
    loss_reg_fn = nn.HuberLoss(delta=cfg.huber_delta)
    loss_cls_fn = nn.BCEWithLogitsLoss()
    reg_w = None
    if cfg.reg_weights is not None:
        reg_w = torch.tensor(cfg.reg_weights, dtype=torch.float32, device=device)

    tot_loss = 0.0
    tot_reg_mae = 0.0
    tot_cls_acc = 0.0
    count = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        x = batch["image"].to(device, non_blocking=True)
        y_reg = batch["y_reg"].to(device, non_blocking=True)  # [-1,1]
        y_cls = batch["y_cls"].to(device, non_blocking=True)  # {-1,+1}

        opt.zero_grad(set_to_none=True)
        pred_reg, pred_cls = model(x)

        loss = 0.0
        if pred_reg is not None:
            l_reg = loss_reg_fn(pred_reg, y_reg)
            if reg_w is not None:
                # weight per-dimension MAE approx by scaling loss — for Huber use per-dim weighting
                l_reg = (F.huber_loss(pred_reg, y_reg, delta=cfg.huber_delta, reduction="none")).mean(0)
                l_reg = (l_reg * reg_w).mean()
            loss = loss + l_reg
            tot_reg_mae += mae(pred_reg.detach(), y_reg).item()
        if pred_cls is not None and pred_cls.numel() > 0:
            # Convert targets from {-1,+1} to {0,1}
            y01 = (y_cls + 1.0) * 0.5
            l_cls = loss_cls_fn(pred_cls, y01)
            loss = loss + l_cls
            tot_cls_acc += bce_logits_accuracy(pred_cls.detach(), y_cls).item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        if sched is not None:
            sched.step()

        tot_loss += loss.item()
        count += 1

        pbar.set_postfix(loss=f"{loss.item():.3f}")

    logs: Dict[str, float] = {
        "loss": tot_loss / max(count, 1),
        "reg_mae": tot_reg_mae / max(count, 1) if count else 0.0,
        "cls_acc": tot_cls_acc / max(count, 1) if count else 0.0,
    }
    return logs


@torch.no_grad()
def validate(
    model: HeadedRegClsModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    tot_reg_mae = 0.0
    tot_cls_acc = 0.0
    count = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y_reg = batch["y_reg"].to(device, non_blocking=True)
        y_cls = batch["y_cls"].to(device, non_blocking=True)

        pred_reg, pred_cls = model(x)
        if pred_reg is not None:
            tot_reg_mae += mae(pred_reg, y_reg).item()
        if pred_cls is not None and pred_cls.numel() > 0:
            tot_cls_acc += bce_logits_accuracy(pred_cls, y_cls).item()
        count += 1

    return {
        "val_reg_mae": tot_reg_mae / max(count, 1) if count else 0.0,
        "val_cls_acc": tot_cls_acc / max(count, 1) if count else 0.0,
    }


def build_loaders(
    corpus_dir: Path,
    img_size: int,
    batch_size: int,
    workers: int,
    val_frac: float,
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    items = load_manifest(corpus_dir)
    train_items, val_items = split_items(items, val_frac=val_frac, seed=seed)
    ds_train = CorpusDataset(corpus_dir, train_items, img_size=img_size)
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True if len(ds_train) >= batch_size else False,
    )
    dl_val = None
    if val_items:
        ds_val = CorpusDataset(corpus_dir, val_items, img_size=img_size)
        dl_val = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )
    return dl_train, dl_val


def cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: Path,
    model: HeadedRegClsModel,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    best_metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {
            "backbone": cfg.backbone,
            "img_size": cfg.img_size,
            "n_reg": len(REG_KEYS),
            "n_cls": len(CLS_KEYS),
            "reg_keys": REG_KEYS,
            "cls_keys": CLS_KEYS,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "metrics": best_metrics,
    }
    torch.save(payload, str(path))


def write_metrics_json(path: Path, logs: Dict[str, float]) -> None:
    path.write_text(json.dumps(logs, indent=2), encoding="utf-8")


def run(
    corpus: Path,
    ckpt: Path,
    backbone: str = "mobilenet_v3_large",
    img_size: int = 512,
    batch_size: int = 64,
    epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    workers: int = 4,
    val_frac: float = 0.1,
    seed: int = 1337,
    huber_delta: float = 1.0,
    warmup_epochs: int = 2,
) -> None:
    """Train the OSL model from corpus and save a checkpoint + metrics.json."""
    cfg = TrainConfig(
        corpus_dir=corpus,
        ckpt_path=ckpt,
        backbone=backbone,
        img_size=img_size,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        workers=workers,
        val_frac=val_frac,
        seed=seed,
        huber_delta=huber_delta,
        warmup_epochs=warmup_epochs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # Data
    dl_train, dl_val = build_loaders(
        corpus_dir=cfg.corpus_dir,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        workers=cfg.workers,
        val_frac=cfg.val_frac,
        seed=cfg.seed,
    )

    # Model
    model = HeadedRegClsModel(
        backbone=cfg.backbone,
        img_size=cfg.img_size,
        n_reg=len(REG_KEYS),
        n_cls=len(CLS_KEYS),
        hidden=512,
    ).to(device)

    # Optimizer + Scheduler (cosine with warmup over total steps)
    total_steps = cfg.epochs * max(1, len(dl_train))
    warmup_steps = cfg.warmup_epochs * max(1, len(dl_train))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = cosine_with_warmup(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    best = {
        "val_reg_mae": float("inf"),
        "val_cls_acc": 0.0,
        "epoch": -1,
    }

    metrics_path = cfg.ckpt_path.parent / "metrics.json"

    for epoch in range(cfg.epochs):
        logs_tr = train_one_epoch(model, dl_train, optimizer, scheduler, device, cfg)
        logs = {f"train_{k}": v for k, v in logs_tr.items()}

        if dl_val is not None:
            logs_val = validate(model, dl_val, device)
            logs.update(logs_val)
            # Track best by val_reg_mae (lower is better)
            improved = logs_val["val_reg_mae"] < best["val_reg_mae"]
        else:
            # If no validation, track best by train loss
            improved = logs_tr["loss"] < best["val_reg_mae"]  # using same key for convenience
            logs["val_reg_mae"] = logs_tr["loss"]
            logs["val_cls_acc"] = logs_tr.get("cls_acc", 0.0)

        if improved:
            best["val_reg_mae"] = logs["val_reg_mae"]
            best["val_cls_acc"] = logs["val_cls_acc"]
            best["epoch"] = epoch
            save_checkpoint(cfg.ckpt_path, model, optimizer, cfg, best)

        # Write rolling metrics each epoch
        write_metrics_json(metrics_path, {"epoch": epoch, **logs, "best": best})

    # If never saved (e.g., tiny dataset), save final
    if not cfg.ckpt_path.exists():
        save_checkpoint(cfg.ckpt_path, model, optimizer, cfg, best)
        write_metrics_json(metrics_path, {"final_best": best})