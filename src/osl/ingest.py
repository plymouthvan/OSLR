from __future__ import annotations

import csv
import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from .lrcat import open_catalog_readonly, iter_photo_records, PhotoRecord, count_photos
from .xmp import (
    build_neutral_settings,
    write_xmp,
)

# Preview size target (longest edge)
DEFAULT_MAX_DIM = 1536
DEFAULT_JPEG_QUALITY = 92

RAW_EXTENSIONS = {
    ".3fr",
    ".ari",
    ".arw",
    ".bay",
    ".crw",
    ".cr2",
    ".cr3",
    ".cap",
    ".data",
    ".dcs",
    ".dcr",
    ".dng",
    ".drf",
    ".eip",
    ".erf",
    ".fff",
    ".gpr",
    ".iiq",
    ".k25",
    ".kdc",
    ".mdc",
    ".mef",
    ".mos",
    ".mrw",
    ".nef",
    ".nrw",
    ".obm",
    ".orf",
    ".pef",
    ".ptx",
    ".pxn",
    ".r3d",
    ".raf",
    ".raw",
    ".rw2",
    ".rwl",
    ".rwz",
    ".sr2",
    ".srf",
    ".srw",
    ".x3f",
}


class IngestError(Exception):
    pass


def run(
    catalog_path: Path,
    out_dir: Path,
    max_dim: int = DEFAULT_MAX_DIM,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    limit: Optional[int] = None,
) -> None:
    """Execute ingestion pipeline; produces corpus directory.

    Layout:
      out_dir/
        previews/{id}.jpg
        xmp_before/{id}.xmp
        xmp_after/{id}.xmp
        manifest.csv
        ingest_report.json
    """
    out_dir = out_dir.resolve()
    previews_dir = out_dir / "previews"
    before_dir = out_dir / "xmp_before"
    after_dir = out_dir / "xmp_after"
    out_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, object] = {
        "catalog": str(catalog_path),
        "out_dir": str(out_dir),
        "max_dim": max_dim,
        "jpeg_quality": jpeg_quality,
        "total_catalog_photos": 0,
        "processed": 0,
        "skipped": 0,
        "errors": [],
        "missing": {
            "raw_file": 0,
            "after_xmp": 0,
            "preview": 0,
        },
        "notes": [
            "after_xmp is sourced from existing sidecar XMP if present; "
            "Lightroom-stored develop settings are not extracted directly in this version."
        ],
    }

    rows: List[Dict[str, object]] = []

    conn = open_catalog_readonly(catalog_path)
    try:
        total_photos = count_photos(conn)
        if limit is not None:
            total_photos = min(total_photos, limit)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Ingesting...", total=total_photos)
            for rec in iter_photo_records(conn, limit=limit):
                report["total_catalog_photos"] = int(report["total_catalog_photos"]) + 1  # type: ignore
                try:
                    row = _process_record(
                        rec,
                        previews_dir,
                        before_dir,
                        after_dir,
                        max_dim=max_dim,
                        jpeg_quality=jpeg_quality,
                        report=report,
                    )
                    if row is None:
                        report["skipped"] = int(report["skipped"]) + 1  # type: ignore
                        continue
                    rows.append(row)
                    report["processed"] = int(report["processed"]) + 1  # type: ignore
                except Exception as e:
                    report["errors"].append(  # type: ignore
                        {
                            "id": rec.id,
                            "raw_path": str(rec.raw_path),
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    report["skipped"] = int(report["skipped"]) + 1  # type: ignore
                finally:
                    progress.update(task, advance=1)
    finally:
        conn.close()

    # Write manifest.csv
    manifest_path = out_dir / "manifest.csv"
    _write_manifest(manifest_path, rows, base_dir=out_dir)

    # Write report
    report_path = out_dir / "ingest_report.json"
    report_json = json.dumps(report, indent=2)
    report_path.write_text(report_json, encoding="utf-8")


def _process_record(
    rec: PhotoRecord,
    previews_dir: Path,
    before_dir: Path,
    after_dir: Path,
    max_dim: int,
    jpeg_quality: int,
    report: Dict[str, object],
) -> Optional[Dict[str, object]]:
    raw_path = rec.raw_path
    if not raw_path.exists():
        report["missing"]["raw_file"] = int(report["missing"]["raw_file"]) + 1  # type: ignore
        return None

    # Paths
    preview_path = previews_dir / f"{rec.id}.jpg"
    before_xmp_path = before_dir / f"{rec.id}.xmp"
    after_xmp_path = after_dir / f"{rec.id}.xmp"

    # Preview
    ok_preview = _ensure_preview(raw_path, preview_path, max_dim=max_dim, quality=jpeg_quality)
    if not ok_preview:
        report["missing"]["preview"] = int(report["missing"]["preview"]) + 1  # type: ignore

    # BEFORE XMP (neutral baseline)
    neutral = build_neutral_settings()
    write_xmp(before_xmp_path, neutral, preserve_unrelated=False)

    # AFTER XMP (attempt from sidecar next to RAW; else mark missing and create placeholder)
    sidecar = _find_sidecar_xmp(raw_path)
    if sidecar and sidecar.exists():
        # Copy/preserve only develop-related attributes by re-writing via our writer using sidecar as base
        # This ensures unrelated XMP is preserved as-is.
        # We do not parse and re-set values here to avoid losing information (tone curves, unknown keys).
        # Instead, we load the file and write an identical tree to destination.
        after_xmp_path.parent.mkdir(parents=True, exist_ok=True)
        after_xmp_path.write_bytes(sidecar.read_bytes())
    else:
        report["missing"]["after_xmp"] = int(report["missing"]["after_xmp"]) + 1  # type: ignore
        # Create placeholder XMP with only neutral baseline (acts as no-edit target)
        write_xmp(after_xmp_path, neutral, preserve_unrelated=False)

    # Manifest row (paths stored relative to corpus root; caller will relativize)
    row = {
        "id": rec.id,
        "raw_path": str(raw_path),
        "preview_path": str(preview_path),
        "before_xmp": str(before_xmp_path),
        "after_xmp": str(after_xmp_path),
        "camera": rec.camera or "",
        "lens": rec.lens or "",
        "iso": rec.iso if rec.iso is not None else "",
        "shutter": rec.shutter or "",
        "aperture": rec.aperture if rec.aperture is not None else "",
        "session_id": rec.session_id or "",
    }
    return row


def _find_sidecar_xmp(raw_path: Path) -> Optional[Path]:
    # XMP next to raw with same basename
    xmp = raw_path.with_suffix(".xmp")
    if xmp.exists():
        return xmp
    # Try upper-case .XMP
    xmpU = raw_path.with_suffix(".XMP")
    if xmpU.exists():
        return xmpU
    return None


def _ensure_preview(raw_path: Path, out_jpg: Path, max_dim: int, quality: int) -> bool:
    try:
        if raw_path.suffix.lower() in RAW_EXTENSIONS:
            # Try rawpy for RAW files
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
                    img = Image.fromarray(rgb)
            except Exception:
                # Fallback to PIL (in case the file is actually readable, e.g., DNG sometimes)
                img = Image.open(raw_path)
        else:
            img = Image.open(raw_path)

        img = _resize_long_edge(img, max_dim)
        out_jpg.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_jpg, "JPEG", quality=quality, optimize=True, progressive=True)
        return True
    except Exception:
        return False


def _resize_long_edge(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    if w == 0 or h == 0:
        return img
    scale = min(max_dim / float(max(w, h)), 1.0)
    if scale == 1.0:
        return img
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return img.resize(new_size, Image.LANCZOS)


def _write_manifest(path: Path, rows: List[Dict[str, object]], base_dir: Path) -> None:
    # Relativize selected path columns to base_dir for portability
    rel_cols = ("preview_path", "before_xmp", "after_xmp")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "raw_path",
                "preview_path",
                "before_xmp",
                "after_xmp",
                "camera",
                "lens",
                "iso",
                "shutter",
                "aperture",
                "session_id",
            ]
        )
        for r in rows:
            row = dict(r)
            for c in rel_cols:
                try:
                    row[c] = str(Path(row[c]).resolve().relative_to(base_dir))  # type: ignore
                except Exception:
                    row[c] = row[c]
            writer.writerow(
                [
                    row["id"],
                    row["raw_path"],
                    row["preview_path"],
                    row["before_xmp"],
                    row["after_xmp"],
                    row["camera"],
                    row["lens"],
                    row["iso"],
                    row["shutter"],
                    row["aperture"],
                    row["session_id"],
                ]
            )