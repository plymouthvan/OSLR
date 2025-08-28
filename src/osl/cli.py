from __future__ import annotations

import sys
import pathlib
import json
import shutil
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel

console = Console()

VERSION = "0.1.0"


def _echo_header() -> None:
    console.print(Panel.fit(f"OpenStyleLab (OSL) v{VERSION}", style="bold cyan"))


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(version=VERSION, prog_name="osl")
def main() -> None:
    """OpenStyleLab CLI.

    Commands:
      - osl ingest --catalog src.lrcat --out corpus/
      - osl train  --corpus corpus/ --ckpt ckpt/best.pt
      - osl apply  --catalog src.lrcat --ckpt ckpt/best.pt --out-catalog dst.lrcat
    """
    pass


@main.command("ingest")
@click.option("--catalog", "catalog_path", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), required=True, help="Source Lightroom catalog (.lrcat)")
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=pathlib.Path), required=True, help="Output corpus directory")
@click.option("--max-dim", type=int, default=1536, show_default=True, help="Max preview long-edge in pixels")
@click.option("--jpeg-quality", type=int, default=92, show_default=True, help="Preview JPEG quality")
@click.option("--limit", type=int, default=None, help="Process only the first N photos (debug)")
def ingest_cmd(catalog_path: pathlib.Path, out_dir: pathlib.Path, max_dim: int, jpeg_quality: int, limit: Optional[int]) -> None:
    """Build corpus from a Lightroom catalog."""
    _echo_header()
    from osl.ingest import run as ingest_run
    try:
        ingest_run(catalog_path, out_dir, max_dim=max_dim, jpeg_quality=jpeg_quality, limit=limit)
        console.print(f"[green]Ingest completed[/] → {out_dir}")
    except Exception as e:
        console.print(f"[red]Ingest failed:[/] {e}")
        raise SystemExit(1)


@main.command("train")
@click.option("--corpus", type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path), required=True, help="Corpus directory from 'osl ingest'")
@click.option("--ckpt", type=click.Path(dir_okay=False, path_type=pathlib.Path), required=True, help="Path to save model checkpoint (.pt)")
@click.option("--backbone", type=click.Choice(["mobilenet_v3_large", "convnext_tiny"]), default="mobilenet_v3_large", show_default=True, help="Backbone architecture")
@click.option("--img-size", type=int, default=512, show_default=True, help="Training image size (square)")
@click.option("--batch-size", type=int, default=64, show_default=True, help="Batch size")
@click.option("--epochs", type=int, default=30, show_default=True, help="Number of epochs")
@click.option("--lr", type=float, default=3e-4, show_default=True, help="Learning rate")
@click.option("--weight-decay", type=float, default=1e-4, show_default=True, help="Weight decay")
@click.option("--workers", type=int, default=4, show_default=True, help="DataLoader workers")
@click.option("--val-frac", type=float, default=0.1, show_default=True, help="Validation fraction")
@click.option("--seed", type=int, default=1337, show_default=True, help="Random seed")
@click.option("--huber-delta", type=float, default=1.0, show_default=True, help="Huber delta")
@click.option("--warmup-epochs", type=int, default=2, show_default=True, help="Warmup epochs for cosine scheduler")
def train_cmd(
    corpus: pathlib.Path,
    ckpt: pathlib.Path,
    backbone: str,
    img_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    workers: int,
    val_frac: float,
    seed: int,
    huber_delta: float,
    warmup_epochs: int,
) -> None:
    """Train model from corpus."""
    _echo_header()
    try:
        from osl.train import run as train_run
    except Exception as e:
        console.print("[red]Training dependencies missing.[/] Install extras: [bold]pip install -e .[train][/]")
        console.print(f"Details: {e}")
        raise SystemExit(1)
    try:
        train_run(
            corpus=corpus,
            ckpt=ckpt,
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
        console.print(f"[green]Training completed[/] → {ckpt} (metrics.json alongside)")
    except Exception as e:
        console.print(f"[red]Training failed:[/] {e}")
        raise SystemExit(1)


@main.command("apply")
@click.option("--catalog", "catalog_path", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), required=True, help="Source catalog to clone")
@click.option("--ckpt", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path), required=True, help="Trained model checkpoint (.pt)")
@click.option("--out-catalog", "dst_catalog", type=click.Path(dir_okay=False, path_type=pathlib.Path), required=True, help="Destination catalog path (.lrcat)")
@click.option("--img-size", type=int, default=512, show_default=True, help="Inference image size (square)")
@click.option("--batch-size", type=int, default=32, show_default=True, help="Inference batch size")
def apply_cmd(catalog_path: pathlib.Path, ckpt: pathlib.Path, dst_catalog: pathlib.Path, img_size: int, batch_size: int) -> None:
    """Apply trained model to catalog and bundle plugin for XMP import."""
    _echo_header()
    try:
        from osl.apply import run as apply_run
    except Exception as e:
        console.print("[red]Inference dependencies missing.[/] Install extras: [bold]pip install -e .[train][/]")
        console.print(f"Details: {e}")
        raise SystemExit(1)
    try:
        apply_run(catalog_path=catalog_path, ckpt_path=ckpt, dst_catalog=dst_catalog, img_size=img_size, batch_size=batch_size)
        console.print(f"[green]Apply completed[/] → Bundle at: {dst_catalog.parent}")
    except Exception as e:
        console.print(f"[red]Apply failed:[/] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()